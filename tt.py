import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import networkx as nx
import random
import numpy as np
import json
import time
import copy
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import itertools
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, PULP_CBC_CMD

# ==============================================================================
# 0. 설정
# ==============================================================================
IMITATION_ITERATIONS = 5000000; IMITATION_LR = 1e-4; IMITATION_MODEL_PATH = "gnn_imitated_transformer.pth"
ONLINE_EPISODES = 2000000; ONLINE_LR_RL = 1e-6; ONLINE_LR_IMITATION = 5e-6; RL_EXPLORATION_COUNT = 5; ILP_TIME_LIMIT_SEC = 60
PREV_MODEL_PATH = "gnn_transformer.pth" if os.path.exists("gnn_transformer.pth") else None
CHECKPOINT_PATH = "gnn_transformer_checkpoint.pth"
FINAL_MODEL_PATH = "gnn_congestion_aware.pth"; BENCHMARK_SCENARIOS = 10; RESULT_PLOT_PATH = "benchmark_results_congestion.png"
LINK_BANDWIDTH_BPS = 1e9; PROPAGATION_DELAY_NS_PER_METER = 5; LINK_LENGTH_METER = 10; SWITCH_PROC_DELAY_NS = 1000
DEVICE = torch.device("cpu"); VALIDATION_SEED = 99999

# ==============================================================================
# 1. 환경, 모델, 솔버 정의
# ==============================================================================
class TSN_Static_Env_v2:
    def __init__(self, graph, flow_definitions):
        self.graph = graph; self.num_nodes = graph.number_of_nodes(); self.flow_defs = flow_definitions
        self.link_prop_delay_ns = PROPAGATION_DELAY_NS_PER_METER * LINK_LENGTH_METER
    def _calculate_tx_time_ns(self, size_bytes): return (size_bytes * 8 / LINK_BANDWIDTH_BPS) * 1e9
    def _get_queuing_delay_ns(self, link_utilization): return 1000 * (link_utilization ** 2)
    def _evaluate_single_scenario(self, paths, deadlines_ms, failed_link=None):
        eval_graph=self.graph.copy()
        if failed_link and eval_graph.has_edge(*failed_link): eval_graph.remove_edge(*failed_link)
        link_utilization={(u,v): 0.0 for u,v in eval_graph.edges}; link_utilization.update({(v,u): 0.0 for u,v in eval_graph.edges})
        for flow_id, path_pair in paths.items():
            path = path_pair.get('primary')
            if path and nx.is_path(eval_graph, path):
                props = self.flow_defs[flow_id]
                usage_to_add = (props['size_bytes'] * 8) / (props['period_ms'] * 1e-3) / LINK_BANDWIDTH_BPS
                for u,v in zip(path[:-1], path[1:]):
                    link_utilization[(u,v)] += usage_to_add
                    link_utilization[(v,u)] += usage_to_add
        flow_results={}
        for flow_id, path_pair in paths.items():
            props=self.flow_defs[flow_id]; path = path_pair.get('primary')
            if use_backup := (failed_link and path and (failed_link in list(zip(path[:-1], path[1:])) or tuple(reversed(failed_link)) in list(zip(path[:-1], path[1:])))):
                path = path_pair.get('backup')
            if not path or not nx.is_path(eval_graph, path):
                if use_backup: return -1000, {"error":f"Backup path not found or invalid for {flow_id}"}
                continue
            tx_time_ns=self._calculate_tx_time_ns(props['size_bytes']); e2e_d=0
            for u,v in zip(path[:-1], path[1:]):
                queuing_delay = self._get_queuing_delay_ns(link_utilization.get((u,v), 0))
                e2e_d += self.link_prop_delay_ns + SWITCH_PROC_DELAY_NS + tx_time_ns + queuing_delay
            flow_results[flow_id]={'e2e_delay_ms':e2e_d/1e6}
            deadline=deadlines_ms.get(props.get('type'))
            if deadline and (e2e_d/1e6) > deadline: return -1000, {"error":f"Deadline missed for {flow_id}"}
        total_lat=sum(res['e2e_delay_ms'] for res in flow_results.values()); return 1.0/(1.0+total_lat), {"results":flow_results}
    def evaluate_robust_configuration(self, paths, deadlines_ms, contingency_scenarios):
        if not paths: return 0.0, {"error":"No paths provided."}
        p_score, p_details=self._evaluate_single_scenario(paths, deadlines_ms)
        if p_score <= 0: return 0.0, p_details
        c_score_sum=0
        if contingency_scenarios:
            for scenario in contingency_scenarios:
                f_link=tuple(scenario.get('failed_link')) if scenario.get('failed_link') else None
                score, details=self._evaluate_single_scenario(paths, deadlines_ms, failed_link=f_link)
                if score <= 0: return 0.0, details
                c_score_sum+=score
            avg_c_score=c_score_sum/len(contingency_scenarios) if contingency_scenarios else 1.0
        else: avg_c_score=p_score
        final_score=0.7*p_score+0.3*avg_c_score; return final_score, {"paths":paths, "primary_score":p_score, "avg_contingency_score":avg_c_score}

class GraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout_rate=0.2):
        super(GraphTransformer, self).__init__(); self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout_rate)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout_rate)
        self.ln1 = nn.LayerNorm(hidden_dim); self.ln2 = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index; x = F.relu(self.input_layer(x))
        x_res = self.conv1(x, edge_index); x = self.ln1(x + x_res); x = F.relu(x)
        x_res = self.conv2(x, edge_index); x = self.ln2(x + x_res); x = F.relu(x)
        logits = self.output_layer(x).squeeze(-1); return logits

def get_state_tensor(graph, flow_props, partial_paths):
    num_nodes = graph.number_of_nodes(); features = np.zeros((num_nodes, 4))
    if flow_props['src'] < num_nodes and flow_props['dst'] < num_nodes:
        features[flow_props['src'], 0] = 1; features[flow_props['dst'], 1] = 1
    link_usage = {edge: 0 for edge in graph.edges()}
    for path_pair in partial_paths.values():
        for path_type in ['primary', 'backup']:
            path = path_pair.get(path_type)
            if path:
                for u,v in zip(path[:-1], path[1:]):
                    edge = tuple(sorted((u,v)))
                    if edge in link_usage: link_usage[edge] += 1
    for u,v in graph.edges():
        norm_usage = link_usage.get(tuple(sorted((u,v))), 0) / len(partial_paths) if partial_paths else 0
        features[u, 2] += norm_usage; features[v, 2] += norm_usage
    for i in range(num_nodes): features[i, 3] = graph.degree[i] / (num_nodes -1)
    return Data(x=torch.tensor(features, dtype=torch.float), edge_index=torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous(), graph=graph)

class DynamicProfileGenerator:
    def generate(self):
        num_nodes=random.randint(8, 150); m=random.randint(2, 40); graph=nx.barabasi_albert_graph(n=num_nodes, m=m)
        while not nx.is_connected(graph): graph=nx.barabasi_albert_graph(n=num_nodes, m=m)
        flow_defs={}; num_flows=random.randint(5, 100)
        for i in range(num_flows):
            src, dst=random.sample(range(num_nodes), 2); flow_type=random.choice(["TT", "AVB"])
            flow_defs[f"flow_{i}"]={"src":src, "dst":dst, "type":flow_type, "size_bytes":random.randint(200, 4000), "period_ms":random.randint(5, 50)}
        deadlines_ms={"TT":random.uniform(2.0, 10.0), "AVB":random.uniform(10.0, 40.0)}; contingency_scenarios=[]
        if random.random() < 0.7:
            num_failures=random.randint(1, 3); possible_edges=list(graph.edges)
            if possible_edges:
                for _ in range(num_failures): contingency_scenarios.append({"failed_link":list(random.choice(possible_edges))})
        return {"graph":graph, "flow_definitions":flow_defs, "deadlines_ms":deadlines_ms, "contingency_scenarios":contingency_scenarios}

class BaseSolver:
    def __init__(self, name): self.name=name
    def solve(self, env, profile): raise NotImplementedError

class RLAgent:
    def __init__(self, state_dim, lr):
        self.policy_net = GraphTransformer(state_dim).to(DEVICE); self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
    def select_action(self, state, current_path, is_eval=False):
        if is_eval: self.policy_net.eval()
        else: self.policy_net.train()
        with torch.set_grad_enabled(not is_eval): logits=self.policy_net(state.to(DEVICE))
        mask=torch.ones_like(logits)*-1e9
        valid_neighbors=[n for n in state.graph.neighbors(current_path[-1]) if n not in current_path]
        if not valid_neighbors: return None, None
        mask[valid_neighbors]=0; masked_logits=logits+mask
        dist=torch.distributions.Categorical(logits=masked_logits)
        action=dist.sample() if not is_eval else masked_logits.argmax()
        log_prob=dist.log_prob(action) if not is_eval else None
        return action.item(), log_prob
    def find_path(self, graph, start_node, end_node, partial_paths={}, is_eval=False):
        current_path, log_probs=[start_node], []
        while current_path[-1] != end_node:
            state=get_state_tensor(graph, {'src':current_path[-1], 'dst':end_node}, partial_paths)
            action, log_prob=self.select_action(state, current_path, is_eval=is_eval)
            if action is None or len(current_path) > graph.number_of_nodes()*2: return None, None
            current_path.append(action)
            if log_prob is not None: log_probs.append(log_prob)
        return current_path, log_probs
    def update_policy_rl(self, final_reward, all_log_probs):
        if not all_log_probs: return 0
        policy_loss=[]; returns=torch.tensor([final_reward]*len(all_log_probs), dtype=torch.float, device=DEVICE)
        for log_prob, R in zip(all_log_probs, returns): policy_loss.append(-log_prob*R)
        self.optimizer.zero_grad()
        if policy_loss: loss=torch.stack(policy_loss).sum(); loss.backward(); self.optimizer.step(); return loss.item()
        return 0
    def update_policy_imitation(self, expert_path, graph, partial_paths, loss_fn):
        if not expert_path or len(expert_path) < 2: return 0
        total_loss=0
        for j in range(len(expert_path)-1):
            current_node, destination_node=expert_path[j], expert_path[-1]; expert_action=expert_path[j+1]
            state_tensor=get_state_tensor(graph, {'src':current_node, 'dst':destination_node}, partial_paths)
            logits=self.policy_net(state_tensor.to(DEVICE)); self.optimizer.zero_grad()
            loss=loss_fn(logits.unsqueeze(0), torch.tensor([expert_action], device=DEVICE)); loss.backward(); self.optimizer.step(); total_loss+=loss.item()
        return total_loss

class GNN_Solver_Final(BaseSolver):
    def __init__(self, model_path): super().__init__("GNN (Final)"); self.agent=RLAgent(state_dim=4, lr=0); self.agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    def solve(self, env, profile):
        paths, partial_paths={}, {}; graph=env.graph
        for flow_id, props in sorted(profile['flow_definitions'].items(), key=lambda item:item[1]['period_ms']):
            p_path, _=self.agent.find_path(graph, props['src'], props['dst'], partial_paths, is_eval=True)
            if p_path is None: return None
            backup_graph=graph.copy()
            if len(p_path) > 1: backup_graph.remove_edges_from(list(zip(p_path[:-1], p_path[1:])))
            b_path, _=self.agent.find_path(backup_graph, props['src'], props['dst'], partial_paths, is_eval=True)
            paths[flow_id]={'primary':p_path, 'backup':b_path}; partial_paths[flow_id]=paths[flow_id]
        return paths

class Greedy_Solver(BaseSolver):
    def __init__(self): super().__init__("Greedy")
    def solve(self, env, profile):
        graph=env.graph; paths, link_usage={}, {tuple(sorted(edge)):0 for edge in graph.edges()}
        def weight_func(u, v, d): return 1+link_usage.get(tuple(sorted((u, v))), 0)*10
        for flow_id, props in sorted(profile['flow_definitions'].items(), key=lambda item:item[1]['period_ms']):
            try:
                primary_path=nx.shortest_path(graph, source=props['src'], target=props['dst'], weight=weight_func)
                for u, v in zip(primary_path[:-1], primary_path[1:]): link_usage[tuple(sorted((u, v)))]+=1
                backup_graph=graph.copy()
                if len(primary_path) > 1: backup_graph.remove_edges_from(list(zip(primary_path[:-1], primary_path[1:])))
                if nx.has_path(backup_graph, source=props['src'], target=props['dst']): backup_path=nx.shortest_path(backup_graph, source=props['src'], target=props['dst'], weight=weight_func)
                else: backup_path=None
                paths[flow_id]={'primary':primary_path, 'backup':backup_path}
            except nx.NetworkXNoPath: return None
        return paths

class ILP_Solver(BaseSolver):
    def __init__(self, time_limit_sec): super().__init__("ILP"); self.time_limit=time_limit_sec
    def solve(self, env, profile):
        prob=LpProblem("TSN_Routing", LpMinimize); G=env.graph; flow_defs=profile['flow_definitions']
        p_vars={}; all_directed_edges=list(G.edges())+[(v,u) for u,v in G.edges()]
        for f in flow_defs: p_vars[f]={(u,v):LpVariable(f"p_{f}_{u}_{v}", 0, 1, 'Binary') for u,v in all_directed_edges}
        prob+=lpSum(p_vars[f][(u,v)] for f in flow_defs for u,v in all_directed_edges), "Minimize_Total_Hops"
        for f, props in flow_defs.items():
            src, dst=props['src'], props['dst']
            for node in G.nodes():
                in_flow=lpSum(p_vars[f][(i,node)] for i in G.neighbors(node) if (i,node) in all_directed_edges)
                out_flow=lpSum(p_vars[f][(node,j)] for j in G.neighbors(node) if (node,j) in all_directed_edges)
                if node == src: prob += out_flow - in_flow == 1
                elif node == dst: prob += out_flow - in_flow == -1
                else: prob += out_flow - in_flow == 0
        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=self.time_limit))
        if LpStatus[prob.status] !='Optimal': return None
        paths={}
        for f, props in flow_defs.items():
            src, dst=props['src'], props['dst']; p_path=[src]; curr_p=src
            while curr_p != dst and len(p_path) <= G.number_of_nodes():
                next_nodes=[v for u,v in all_directed_edges if u==curr_p and p_vars[f][(u,v)].varValue > 0.9]
                if not next_nodes: break
                curr_p=next_nodes[0]; p_path.append(curr_p)
            if not p_path or p_path[-1] != dst: return None
            paths[f]={'primary': p_path, 'backup': None}
        final_paths = {}
        for f, props in flow_defs.items():
            p_path=paths[f]['primary']
            if not p_path: return None
            backup_graph=env.graph.copy()
            if len(p_path) > 1: backup_graph.remove_edges_from(list(zip(p_path[:-1], p_path[1:])))
            b_path=None
            if nx.has_path(backup_graph, props['src'], props['dst']):
                try: b_path=nx.shortest_path(backup_graph, props['src'], props['dst'])
                except nx.NetworkXNoPath: b_path=None
            final_paths[f] = {'primary': p_path, 'backup': b_path}
        return final_paths

def run_imitation_learning(agent, expert_solver, profile_generator, iterations):
    print("\n===== 1단계: 대규모 전문가 모방 학습 시작 ====="); pbar=tqdm(range(iterations), desc="모방 학습", ncols=100)
    optimizer=torch.optim.Adam(agent.policy_net.parameters(), lr=IMITATION_LR)
    agent.optimizer=optimizer; loss_fn=nn.CrossEntropyLoss()
    for i in pbar:
        profile=profile_generator.generate(); env=TSN_Static_Env_v2(profile["graph"], profile['flow_definitions']); expert_paths=expert_solver.solve(env, profile)
        if not expert_paths: continue
        partial_paths={}; flow_ids = sorted(profile['flow_definitions'].keys(), key=lambda k: profile['flow_definitions'][k]['period_ms'])
        for flow_id in flow_ids:
            expert_path=expert_paths.get(flow_id, {}).get('primary')
            agent.update_policy_imitation(expert_path, env.graph, partial_paths, loss_fn); partial_paths[flow_id]=expert_paths[flow_id]
        if i>0 and i%100==0 and expert_paths:
            last_flow_id=flow_ids[-1]; last_expert_path=expert_paths.get(last_flow_id, {}).get('primary')
            if last_expert_path and len(last_expert_path) >= 2:
                state_tensor=get_state_tensor(env.graph,{'src':last_expert_path[-2],'dst':last_expert_path[-1]},partial_paths); logits=agent.policy_net(state_tensor.to(DEVICE))
                pbar.set_postfix(loss=f"{loss_fn(logits.unsqueeze(0), torch.tensor([last_expert_path[-1]], device=DEVICE)):.4f}")
    torch.save(agent.policy_net.state_dict(), IMITATION_MODEL_PATH); print(f"\n모방 학습 완료. 모델 저장: '{IMITATION_MODEL_PATH}'")

def run_ilp_guided_rl(agent, ilp_solver, profile_generator, episodes):
    print("\n===== 2단계: ILP-앵커 온라인 학습 시작 (단일 프로세스) =====")
    start_episode=0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"체크포인트 '{CHECKPOINT_PATH}' 발견. 학습 재개."); checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        agent.policy_net.load_state_dict(checkpoint['model_state_dict']); start_episode = checkpoint['episode']
    elif PREV_MODEL_PATH:
        print(f"이전 모델 '{PREV_MODEL_PATH}'에서 학습 이어가기."); agent.policy_net.load_state_dict(torch.load(PREV_MODEL_PATH, map_location=DEVICE))
    
    pbar = tqdm(range(start_episode, episodes), desc="온라인 학습", ncols=100, initial=start_episode, total=episodes); il_loss_fn = nn.CrossEntropyLoss()
    for ep in pbar:
        profile=profile_generator.generate(); env=TSN_Static_Env_v2(profile["graph"], profile['flow_definitions'])
        ilp_paths=ilp_solver.solve(env, profile)
        if not ilp_paths: continue
        score_ilp, _=env.evaluate_robust_configuration(ilp_paths, profile['deadlines_ms'], profile['contingency_scenarios'])
        
        best_rl_score, best_rl_paths = -1.0, None
        for _ in range(RL_EXPLORATION_COUNT):
            paths, partial_paths, failed={}, {}, False
            flow_ids = sorted(profile['flow_definitions'].keys(), key=lambda k: profile['flow_definitions'][k]['period_ms'])
            for flow_id in flow_ids:
                props = profile['flow_definitions'][flow_id]
                p_path, _=agent.find_path(env.graph, props['src'], props['dst'], partial_paths, is_eval=False)
                if p_path is None: failed=True; break
                b_path, _=None, None
                if not failed:
                    backup_graph=env.graph.copy()
                    if len(p_path) > 1: backup_graph.remove_edges_from(list(zip(p_path[:-1], p_path[1:])))
                    b_path, _=agent.find_path(backup_graph, props['src'], props['dst'], partial_paths, is_eval=False)
                paths[flow_id]={'primary':p_path, 'backup':b_path}; partial_paths[flow_id]=paths[flow_id]
            if not failed:
                score_rl, _=env.evaluate_robust_configuration(paths, profile['deadlines_ms'], profile['contingency_scenarios'])
                if score_rl > best_rl_score: best_rl_score, best_rl_paths = score_rl, paths
        
        if best_rl_paths and best_rl_score >= score_ilp:
            agent.optimizer=torch.optim.Adam(agent.policy_net.parameters(), lr=ONLINE_LR_RL)
            all_log_probs, total_path_len = [], 0; partial_paths = {}
            for flow_id in sorted(best_rl_paths.keys()):
                props = profile['flow_definitions'][flow_id]
                p_path_sol = best_rl_paths[flow_id]['primary']; b_path_sol = best_rl_paths[flow_id]['backup']
                _, p_logs = agent.find_path(env.graph, props['src'], props['dst'], partial_paths); all_log_probs.extend(p_logs if p_logs else []); total_path_len += len(p_path_sol)
                if b_path_sol:
                     backup_graph=env.graph.copy(); backup_graph.remove_edges_from(list(zip(p_path_sol[:-1], p_path_sol[1:])))
                     _, b_logs = agent.find_path(backup_graph, props['src'], props['dst'], partial_paths); all_log_probs.extend(b_logs if b_logs else []); total_path_len += len(b_path_sol)
                partial_paths[flow_id] = best_rl_paths[flow_id]
            final_reward = best_rl_score - (0.001 * total_path_len)
            loss = agent.update_policy_rl(final_reward, all_log_probs)
            pbar.set_postfix(Action="RL_Update", Score=f"{best_rl_score:.3f}>{score_ilp:.3f}")
        elif ilp_paths:
            agent.optimizer=torch.optim.Adam(agent.policy_net.parameters(), lr=ONLINE_LR_IMITATION)
            loss=0; partial_paths={}
            for flow_id in sorted(ilp_paths.keys()):
                primary_path = ilp_paths.get(flow_id, {}).get('primary')
                loss+=agent.update_policy_imitation(primary_path, env.graph, partial_paths, il_loss_fn)
                partial_paths[flow_id]=ilp_paths[flow_id]
            pbar.set_postfix(Action="IL_Correction", Score=f"{best_rl_score:.3f}<{score_ilp:.3f}", Loss=f"{loss:.3f}")
        
        if (ep + 1) % 100 == 0:
            torch.save({'episode': ep + 1, 'model_state_dict': agent.policy_net.state_dict()}, CHECKPOINT_PATH)

    torch.save(agent.policy_net.state_dict(), FINAL_MODEL_PATH); print(f"\n온라인 학습 완료. 최종 모델 저장: '{FINAL_MODEL_PATH}'")

def run_full_procedure_and_benchmark():
    agent=RLAgent(state_dim=4, lr=IMITATION_LR); expert_solver=ILP_Solver(time_limit_sec=60); profile_gen=DynamicProfileGenerator()
    
    if not os.path.exists(IMITATION_MODEL_PATH):
        run_imitation_learning(agent, expert_solver, profile_gen, iterations=IMITATION_ITERATIONS)
    
    run_ilp_guided_rl(agent, expert_solver, profile_gen, episodes=ONLINE_EPISODES)
    
    print("\n===== 최종 랜덤 벤치마크 실행 ====="); solvers=[GNN_Solver_Final(FINAL_MODEL_PATH), Greedy_Solver(), ILP_Solver(time_limit_sec=60)]
    results=[]; pbar_bench=tqdm(range(BENCHMARK_SCENARIOS), desc="랜덤 벤치마크 진행", ncols=100)
    for i in pbar_bench:
        profile=profile_gen.generate(); env=TSN_Static_Env_v2(profile["graph"], profile['flow_definitions']); row={"Scenario":i}
        for solver in solvers:
            start_time=time.time(); paths=solver.solve(env, profile); end_time=time.time(); computation_time=end_time-start_time
            score, _=env.evaluate_robust_configuration(paths, profile['deadlines_ms'], profile['contingency_scenarios'])
            row[f"{solver.name}_Score"] = score; row[f"{solver.name}_Time"] = computation_time
        results.append(row)
    
    df=pd.DataFrame(results)
    avg_scores={"GNN (Final)":df["GNN (Final)_Score"].mean(), "Greedy":df["Greedy_Score"].mean(), "ILP":df["ILP_Score"].mean()}
    avg_times={"GNN (Final)":df["GNN (Final)_Time"].mean(), "Greedy":df["Greedy_Time"].mean(), "ILP":df["ILP_Time"].mean()}
    print("\n\n===== 최종 벤치마크 결과 요약 (평균) ====="); summary_df=pd.DataFrame([avg_scores, avg_times], index=["Average Score", "Average Time (s)"]); print(summary_df.to_string())
    
    fig, axes=plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True); fig.suptitle(f'Congestion-Aware Benchmark Results ({BENCHMARK_SCENARIOS} Scenarios)', fontsize=16)
    summary_df.T.plot(kind='bar', y='Average Score', ax=axes[0], rot=0, legend=False); axes[0].set_title('Average Performance Score (Higher is Better)'); axes[0].set_ylabel('Avg. Robustness Score'); axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    summary_df.T.plot(kind='bar', y='Average Time (s)', ax=axes[1], rot=0, legend=False); axes[1].set_title('Average Computation Time (Lower is Better)'); axes[1].set_ylabel('Avg. Time (seconds)'); axes[1].set_yscale('log'); axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(RESULT_PLOT_PATH); print(f"\n결과 그래프가 '{RESULT_PLOT_PATH}'에 저장되었습니다.")

if __name__ == '__main__':
    run_full_procedure_and_benchmark()

