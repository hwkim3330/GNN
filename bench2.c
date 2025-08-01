// bench_tool.c
// CBS, TAS normalization + latency & throughput benchmark

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <linux/if_packet.h>
#include <linux/if_ether.h>
#include <linux/net_tstamp.h>
#include <netinet/ether.h>

// Interface to test
static const char *IFACE = "enp5s0";
// EtherType for test
#define ETH_TYPE 0x1337
// Benchmark parameters
#define LATENCY_COUNT 1000
#define THROUGHPUT_DURATION_SEC 5
#define THROUGHPUT_RATE_BPS 1000000000L  // 1 Gbps

// === CBS Config ===
typedef struct {int prio, max_frame, bandwidth;} CbsChild;
typedef struct {int sendslope,idleslope,hicredit,locredit;} CbsCredit;
static CbsChild cbs_streams[] = {{3,512,70000},{2,512,30000}};

// === TAS Config ===
typedef struct {int time_ns, prios[16], prio_count;} TasSchedule;
static TasSchedule tas_sched[] = {{300000,{5},1},{300000,{2,3},2},{400000,{-1},1}};

// Utility to get link speed via ethtool
long get_link_speed_bps(const char *ifname) {
    char cmd[128]; snprintf(cmd,sizeof(cmd),"ethtool %s 2>/dev/null",ifname);
    FILE *fp=popen(cmd,"r"); if(!fp) return 0;
    char line[256]; long mbps=0;
    while(fgets(line,sizeof(line),fp)) if(sscanf(line," Speed: %ldMb/s",&mbps)==1) break;
    pclose(fp);
    return mbps*1000000L;
}

// CBS normalization
void normalize_cbs(const char *ifname) {
    long link = get_link_speed_bps(ifname); if(link<=0) link=1000000000L;
    long link_k=link/1000;
    long ia=0,fa=0, ib=0,fb=0;
    for(int i=0;i<2;i++){
        CbsChild *c=&cbs_streams[i];
        if(c->prio==3){ia+=c->bandwidth;fa+=c->max_frame;} else {ib+=c->bandwidth;fb+=c->max_frame;}
    }
    double ida=ia, idb=ib;
    CbsCredit a={(int)((ida-link_k)*fa/link_k),(int)(ida*fa/link_k), (int)ceil(ida*fa/link_k),(int)ceil((ida-link_k)*fa/link_k)};
    CbsCredit b={(int)(((double)ib-link_k)*fb/link_k),(int)(idb*fb/link_k), (int)ceil(idb*((double)fb/(link_k-ia)+ida/link_k)), (int)ceil(((double)ib-link_k)*fb/link_k)};
    printf("[CBS] %s link=%ldMbps\n A: send=%d idle=%d hi=%d lo=%d\n B: send=%d idle=%d hi=%d lo=%d\n",IFACE,link/1000000,
        a.sendslope,a.idleslope,a.hicredit,a.locredit,b.sendslope,b.idleslope,b.hicredit,b.locredit);
}

// TAS normalization
void normalize_tas() {
    int tc_map[16]; int next=0;
    for(int i=0;i<16;i++) tc_map[i]=-1;
    for(int i=0;i<3;i++) for(int j=0;j<tas_sched[i].prio_count;j++){int p=tas_sched[i].prios[j];if(p>=0&&tc_map[p]<0)tc_map[p]=next++;}
    int d=next++;
    for(int i=0;i<16;i++) if(tc_map[i]<0) tc_map[i]=d;
    printf("[TAS] num_tc=%d\n map:",next);
    for(int i=0;i<16;i++) printf(" %d->%d",i,tc_map[i]); printf("\nSched:\n");
    for(int i=0;i<3;i++){int mask=0;for(int j=0;j<tas_sched[i].prio_count;j++){int p=tas_sched[i].prios[j];int t=(p>=0&&p<16)?tc_map[p]:tc_map[0];mask|=1<<t;} printf(" S %d %d\n",mask,tas_sched[i].time_ns);} }

// ===== Raw socket setup =====
int setup_socket(int *ifs) {
    int sock=socket(AF_PACKET,SOCK_RAW,htons(ETH_TYPE)); if(sock<0)return -1;
    struct ifreq ifr; strcpy(ifr.ifr_name,IFACE);
    if(ioctl(sock,SIOCGIFINDEX,&ifr)<0) return -1;
    *ifs=ifr.ifr_ifindex;
    struct sockaddr_ll addr={.sll_family=AF_PACKET,.sll_protocol=htons(ETH_TYPE),.sll_ifindex=ifr.ifr_ifindex};
    bind(sock,(struct sockaddr*)&addr,sizeof(addr));
    // timestamping
    int ts=SOF_TIMESTAMPING_TX_SOFTWARE|SOF_TIMESTAMPING_RX_SOFTWARE|SOF_TIMESTAMPING_RAW_HARDWARE;
    setsockopt(sock,SOL_SOCKET,SO_TIMESTAMPING,&ts,sizeof(ts));
    return sock;
}

// Send & receive latency
void measure_latency() {
    int ifindex; int sock=setup_socket(&ifindex);
    if(sock<0){perror("socket");return;}
    unsigned char buf[ETH_FRAME_LEN];
    struct sockaddr_ll addr={.sll_family=AF_PACKET,.sll_protocol=htons(ETH_TYPE),.sll_ifindex=ifindex,.sll_halen=ETH_ALEN};
    memset(addr.sll_addr,0xff,6);
    // build frame
    for(int i=0;i<6;i++) buf[i]=0xff; // dest
    memcpy(buf+6,\"MYMAC\",6);// src stub
    buf[12]=ETH_TYPE>>8;buf[13]=ETH_TYPE;
    printf("[Latency] sending %d pings...\n",LATENCY_COUNT);
    for(int id=0;id<LATENCY_COUNT;id++){
        buf[14]=id&0xff; buf[15]=id>>8;
        struct timespec t1,t2;
        clock_gettime(CLOCK_MONOTONIC,&t1);
        sendto(sock,buf,64,0,(struct sockaddr*)&addr,sizeof(addr));
        recvfrom(sock,buf,ETH_FRAME_LEN,0,NULL,NULL);
        clock_gettime(CLOCK_MONOTONIC,&t2);
        long dt=(t2.tv_sec-t1.tv_sec)*1e9 + (t2.tv_nsec-t1.tv_nsec);
        printf("%d: %ld ns\n",id,dt);
    }
    close(sock);
}

// Throughput: send continuous for duration
void measure_throughput() {
    int ifindex; int sock=setup_socket(&ifindex);
    if(sock<0){perror("socket");return;}
    unsigned char buf[1500]={0}; // minimal frame
    struct sockaddr_ll addr={.sll_family=AF_PACKET,.sll_protocol=htons(ETH_TYPE),.sll_ifindex=ifindex,.sll_halen=ETH_ALEN};
    memset(addr.sll_addr,0xff,6);
    long bytes_sent=0;
    struct timespec start,cur;
    clock_gettime(CLOCK_MONOTONIC,&start);
    printf("[Throughput] sending for %d s at %ld bps...\n",THROUGHPUT_DURATION_SEC,THROUGHPUT_RATE_BPS);
    while(1){
        clock_gettime(CLOCK_MONOTONIC,&cur);
        double elapsed=cur.tv_sec-start.tv_sec+1e-9*(cur.tv_nsec-start.tv_nsec);
        if(elapsed>THROUGHPUT_DURATION_SEC)break;
        sendto(sock,buf,1500,0,(struct sockaddr*)&addr,sizeof(addr));
        bytes_sent+=1500;
        // throttle
        // busy wait to maintain rate
        double target = bytes_sent*8.0/THROUGHPUT_RATE_BPS;
        while((cur.tv_sec-start.tv_sec+1e-9*(cur.tv_nsec-start.tv_nsec))<target) clock_gettime(CLOCK_MONOTONIC,&cur);
    }
    printf("Sent %.2f MB in %.2f s = %.2f Mbps\n", bytes_sent/1e6, (cur.tv_sec-start.tv_sec)+1e-9*(cur.tv_nsec-start.tv_nsec), bytes_sent*8/1e6/((cur.tv_sec-start.tv_sec)+1e-9*(cur.tv_nsec-start.tv_nsec)));
    close(sock);
}

int main(){
    printf("=== Bench Tool on %s ===\n",IFACE);
    normalize_cbs(IFACE);
    normalize_tas();
    measure_latency();
    measure_throughput();
    return 0;
}

