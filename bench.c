// bench_tool.c
// CBS, TAS normalization + latency & throughput benchmark (fixed)

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
#include <arpa/inet.h>
#include <net/if.h>
#include <linux/if_packet.h>
#include <linux/if_ether.h>
#include <linux/net_tstamp.h>
#include <netinet/ether.h>

static const char *IFACE = "enp5s0";
#define ETH_TYPE 0x1337
#define LATENCY_COUNT 1000
#define THROUGHPUT_DURATION_SEC 5
#define THROUGHPUT_RATE_BPS 1000000000L

// ===== CBS Config =====
typedef struct { int prio, max_frame, bandwidth; } CbsChild;
typedef struct { int sendslope, idleslope, hicredit, locredit; } CbsCredit;
static CbsChild cbs_streams[] = { {3,512,70000}, {2,512,30000} };

// ===== TAS Config =====
typedef struct { int time_ns, prios[16], prio_count; } TasSchedule;
static TasSchedule tas_sched[] = {
    {300000, {5}, 1},
    {300000, {2,3}, 2},
    {400000, {-1}, 1}
};

// Utility: get link speed
tatic long get_link_speed_bps(const char *ifname) {
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "ethtool %s 2>/dev/null", ifname);
    FILE *fp = popen(cmd, "r");
    if (!fp) return 0;
    char line[256]; long mbps = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, " Speed: %ldMb/s", &mbps) == 1) break;
    }
    pclose(fp);
    return mbps * 1000000L;
}

// Normalize CBS
void normalize_cbs(const char *ifname) {
    long link = get_link_speed_bps(ifname);
    if (link <= 0) link = 1000000000L;
    long link_k = link / 1000L;
    long ia = 0, fa = 0, ib = 0, fb = 0;
    for (int i = 0; i < 2; i++) {
        CbsChild *c = &cbs_streams[i];
        if (c->prio == 3) { ia += c->bandwidth; fa += c->max_frame; }
        else              { ib += c->bandwidth; fb += c->max_frame; }
    }
    double idle_a = ia, idle_b = ib;
    double send_a = idle_a - link_k;
    double send_b = idle_b - link_k;
    CbsCredit cred_a = {
        (int)floor(send_a/1000.0),
        (int)floor(idle_a/1000.0),
        (int)ceil(idle_a * fa / link_k),
        (int)ceil(send_a * fa / link_k)
    };
    CbsCredit cred_b = {
        (int)floor(send_b/1000.0),
        (int)floor(idle_b/1000.0),
        (int)ceil(idle_b * ((double)fb/(link_k-ia) + idle_a/link_k)),
        (int)ceil(send_b * fb / link_k)
    };
    printf("[CBS] %s link=%ldMbps\n", ifname, link/1000000L);
    printf(" A: send=%d idle=%d hi=%d lo=%d\n", cred_a.sendslope, cred_a.idleslope, cred_a.hicredit, cred_a.locredit);
    printf(" B: send=%d idle=%d hi=%d lo=%d\n", cred_b.sendslope, cred_b.idleslope, cred_b.hicredit, cred_b.locredit);
}

// Normalize TAS
void normalize_tas() {
    int tc_map[16]; int next = 0;
    for (int i = 0; i < 16; i++) tc_map[i] = -1;
    for (int i = 0; i < 3; i++) {
        TasSchedule *s = &tas_sched[i];
        for (int j = 0; j < s->prio_count; j++) {
            int p = s->prios[j];
            if (p >= 0 && p < 16 && tc_map[p] == -1) tc_map[p] = next++;
        }
    }
    int default_tc = next++;
    for (int i = 0; i < 16; i++) if (tc_map[i] == -1) tc_map[i] = default_tc;
    printf("[TAS] num_tc=%d\n map: ", next);
    for (int i = 0; i < 16; i++) {
        printf("%d->%d ", i, tc_map[i]);
    }
    printf("\nSched:\n");
    for (int i = 0; i < 3; i++) {
        TasSchedule *s = &tas_sched[i];
        int mask = 0;
        for (int j = 0; j < s->prio_count; j++) {
            int p = s->prios[j];
            int t = (p >= 0 && p < 16) ? tc_map[p] : tc_map[0];
            mask |= 1 << t;
        }
        printf(" S %d %d\n", mask, s->time_ns);
    }
}

// Setup raw socket with timestamping and get MAC address
int setup_socket(int *ifindex, unsigned char src_mac[6]) {
    int sock = socket(AF_PACKET, SOCK_RAW, htons(ETH_TYPE));
    if (sock < 0) return -1;
    struct ifreq ifr;
    strncpy(ifr.ifr_name, IFACE, IFNAMSIZ);
    if (ioctl(sock, SIOCGIFINDEX, &ifr) < 0) { close(sock); return -1; }
    *ifindex = ifr.ifr_ifindex;
    if (ioctl(sock, SIOCGIFHWADDR, &ifr) == 0) {
        memcpy(src_mac, ifr.ifr_hwaddr.sa_data, 6);
    } else {
        memset(src_mac, 0, 6);
    }
    struct sockaddr_ll addr = {0};
    addr.sll_family = AF_PACKET;
    addr.sll_protocol = htons(ETH_TYPE);
    addr.sll_ifindex = *ifindex;
    bind(sock, (struct sockaddr*)&addr, sizeof(addr));
    int ts = SOF_TIMESTAMPING_TX_SOFTWARE | SOF_TIMESTAMPING_RX_SOFTWARE;
    setsockopt(sock, SOL_SOCKET, SO_TIMESTAMPING, &ts, sizeof(ts));
    return sock;
}

// Measure latency
void measure_latency() {
    int ifindex;
    unsigned char src_mac[6];
    int sock = setup_socket(&ifindex, src_mac);
    if (sock < 0) { perror("setup_socket"); return; }
    unsigned char buf[ETH_FRAME_LEN] = {0};
    struct sockaddr_ll addr = {0};
    addr.sll_family = AF_PACKET;
    addr.sll_protocol = htons(ETH_TYPE);
    addr.sll_ifindex = ifindex;
    memset(addr.sll_addr, 0xff, 6);

    printf("[Latency] sending %d pings...\n", LATENCY_COUNT);
    for (int id = 0; id < LATENCY_COUNT; id++) {
        memcpy(buf, addr.sll_addr, 6); // dest
        memcpy(buf+6, src_mac, 6);     // src
        buf[12] = ETH_TYPE >> 8; buf[13] = ETH_TYPE;
        buf[14] = id & 0xff; buf[15] = (id >> 8) & 0xff;
        struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        sendto(sock, buf, 64, 0, (struct sockaddr*)&addr, sizeof(addr));
        recvfrom(sock, buf, ETH_FRAME_LEN, 0, NULL, NULL);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        long dt = (t2.tv_sec - t1.tv_sec) * 1000000000L + (t2.tv_nsec - t1.tv_nsec);
        printf("%d: %ld ns\n", id, dt);
    }
    close(sock);
}

// Measure throughput
void measure_throughput() {
    int ifindex;
    unsigned char src_mac[6];
    int sock = setup_socket(&ifindex, src_mac);
    if (sock < 0) { perror("setup_socket"); return; }
    unsigned char buf[1500] = {0};
    struct sockaddr_ll addr = {0};
    addr.sll_family = AF_PACKET;
    addr.sll_protocol = htons(ETH_TYPE);
    addr.sll_ifindex = ifindex;
    memset(addr.sll_addr, 0xff, 6);

    long bytes_sent = 0;
    struct timespec start, cur;
    clock_gettime(CLOCK_MONOTONIC, &start);
    printf("[Throughput] sending for %d s at %ld bps...\n", THROUGHPUT_DURATION_SEC, THROUGHPUT_RATE_BPS);
    while (1) {
        clock_gettime(CLOCK_MONOTONIC, &cur);
        double elapsed = cur.tv_sec - start.tv_sec + (cur.tv_nsec - start.tv_nsec) / 1e9;
        if (elapsed > THROUGHPUT_DURATION_SEC) break;
        sendto(sock, buf, sizeof(buf), 0, (struct sockaddr*)&addr, sizeof(addr));
        bytes_sent += sizeof(buf);
        double target = bytes_sent * 8.0 / THROUGHPUT_RATE_BPS;
        do { clock_gettime(CLOCK_MONOTONIC, &cur);
        } while ((cur.tv_sec - start.tv_sec + (cur.tv_nsec - start.tv_nsec)/1e9) < target);
    }
    double total = (cur.tv_sec - start.tv_sec) + (cur.tv_nsec - start.tv_nsec)/1e9;
    printf("Sent %.2f MB in %.2f s = %.2f Mbps\n", bytes_sent/1e6, total, bytes_sent*8/1e6/total);
    close(sock);
}

int main() {
    printf("=== Bench Tool on %s ===\n", IFACE);
    normalize_cbs(IFACE);
    normalize_tas();
    measure_latency();
    measure_throughput();
    return 0;
}
