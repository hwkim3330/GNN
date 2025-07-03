import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import networkx as nx
import random
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, PULP_CBC_CMD
import multiprocessing as mp

# ==============================================================================
# 0. 설정 (v2 - 성능 및 안정성 최적화)
# ==============================================================================
# --- 비동기 + 배치 파이프라인 ---
NUM_PRODUCERS = max(1, mp.cpu_count() - 2)  # 시스템 CPU 코어 수에 맞춰 동적 할당
TASK_QUEUE_SIZE = NUM_PRODUCERS * 4

# --- 학습 파라미터 ---
# 이전 모델이 있다면 불러오고, 없다면 처음부터 학습
IMITATION_MODEL_PATH = "gnn_imitated_v3.pth"
ONLINE_MODEL_PATH = "gnn_online_v3.pth"
FINAL_MODEL_PATH = "gnn_ultimate_v3.pth"
CHECKPOINT_PATH = "gnn_checkpoint_v3.pth"

# 모방 학습을 건너뛰고 싶으면 IMITATION_ITERATIONS = 0 으로 설정
IMITATION_ITERATIONS = 5000; IMITATION_LR = 3e-4
ONLINE_EPISODES = 10000; ONLINE_LR_RL = 5e-6; ONLINE_LR_IMITATION = 1e-5; RL_EXPLORATION_COUNT = 8; ILP_TIME_LIMIT_SEC = 20; CHECKPOINT_INTERVAL = 200
BATCH_SIZE = 32

# --- 벤치마크 ---
BENCHMARK_SCENARIOS = 20; RESULT_PLOT_PATH = "benchmark_results_v3.png"

# --- 환경 상수 ---
LINK_BANDWIDTH_BPS = 1e9; PROPAGATION_DELAY_NS_PER_METER = 5; LINK_LENGTH_METER = 10; SWITCH_PROC_DELAY_NS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. 환경 및 기본 솔버 (이전과 거의 동일, 안정성 강화)
# ==============================================================================
class TSN_Static_Env:
    def __init__(self, graph, flow_definitions):
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.flow_defs = flow_definitions
        self.link_prop_delay_ns = PROPAGATION_DELAY_NS_PER_METER * LINK_LENGTH_METER
        self.flow_tx_times_ns = {
            fid: (props['size_bytes'] * 8 / LINK_BANDWIDTH_BPS) * 1e9
            for fid, props in flow_definitions.items()
        }

    def _evaluate_single_scenario(self, paths, deadlines_ms, failed_link=None):
        eval_graph = self.graph.copy()
        if failed_link and eval_graph.has_edge(*failed_link):
            eval_graph.remove_edge(*failed_link)

        link_schedules = {e: [] for e in eval_graph.edges}
        link_schedules.update({(v, u): [] for u, v in eval_graph.edges})
        flow_results = {}

        # 주기가 짧은 플로우(더 엄격한)부터 스케줄링
        sorted_flow_ids = sorted(self.flow_defs.keys(), key=lambda fid: self.flow_defs[fid]['period_ms'])

        for flow_id in sorted_flow_ids:
            props = self.flow_defs[flow_id]
            path_pair = paths.get(flow_id)

            if not path_pair: return -1.0, {"error": f"Path pair not found for {flow_id}"}

            p_edges = list(zip(path_pair['primary'][:-1], path_pair['primary'][1:])) if path_pair.get('primary') else []
            use_backup = failed_link and any(e == failed_link or e == tuple(reversed(failed_link)) for e in p_edges)

            path = path_pair['backup'] if use_backup else path_pair['primary']

            if not path:
                if use_backup: return -1.0, {"error": f"Backup path needed but not available for {flow_id}"}
                continue # 정상 상황에서 Primary 경로가 없는 경우는 스킵 (라우팅 실패)

            if not nx.is_path(eval_graph, path):
                 return -1.0, {"error": f"Path for {flow_id} with nodes {path} invalid in this scenario (link failure)"}

            tx_time_ns = self.flow_tx_times_ns[flow_id]
            e2e_delay_ns = -1
            
            # Simple greedy scheduling
            current_time_ns = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                link = (u, v)
                
                # Find the earliest possible transmission time on this link
                earliest_start = current_time_ns
                slot_found = False
                while not slot_found:
                    conflict = False
                    for start, end in link_schedules[link]:
                        if not (earliest_start + tx_time_ns <= start or earliest_start >= end):
                            conflict = True
                            earliest_start = end # Jump to the end of the conflicting schedule
                            break
                    if not conflict:
                        slot_found = True
                
                link_schedules[link].append((earliest_start, earliest_start + tx_time_ns))
                link_schedules[link].sort()
                
                # Update current time to arrival time at the next node
                current_time_ns = earliest_start + tx_time_ns + self.link_prop_delay_ns + SWITCH_PROC_DELAY_NS
            
            e2e_delay_ns = current_time_ns - (self.link_prop_delay_ns + SWITCH_PROC_DELAY_NS)
            flow_results[flow_id] = {'e2e_delay_ms': e2e_delay_ns / 1e6}
            deadline = deadlines_ms.get(props.get('type'))

            if deadline and (e2e_delay_ns / 1e6) > deadline:
                return -1.0, {"error": f"Deadline missed for {flow_id}: {e2e_delay_ns / 1e6:.4f} > {deadline} ms"}

        total_latency_ms = sum(res['e2e_delay_ms'] for res in flow_results.values())
        # 보상 점수: 0~1 사이, 낮을수록 좋음 (지연시간이 0에 가까울수록 1에 수렴)
        return 1.0 / (1.0 + total_latency_ms), {"results": flow_results}

    def evaluate_robust_configuration(self, paths, deadlines_ms, contingency_scenarios):
        if not paths: return 0.0, {"error": "No paths provided."}
        
        primary_score, primary_details = self._evaluate_single_scenario(paths, deadlines_ms)
        if primary_score < 0: return 0.0, primary_details

        total_contingency_score = 0
        if contingency_scenarios:
            for scenario in contingency_scenarios:
                f_link = tuple(scenario.get('failed_link')) if scenario.get('failed_link') else None
                score, details = self._evaluate_single_scenario(paths, deadlines_ms, failed_link=f_link)
                if score < 0: return 0.0, details # 시나리오 하나라도 실패하면 0점
                total_contingency_score += score
            avg_contingency_score = total_contingency_score / len(contingency_scenarios)
        else:
            avg_contingency_score = primary_score # 고장 시나리오가 없으면 Primary 점수를 그대로 사용

        # 최종 점수: Primary 70%, Contingency 30%
        final_score = 0.7 * primary_score + 0.3 * avg_contingency_score
        return final_score, {"paths": paths, "primary_score": primary_score, "avg_contingency_score": avg_contingency_score}

class DynamicProfileGenerator:
    def generate(self):
        num_nodes = random.randint(8, 15)
        m = random.randint(2, 4)
        graph = nx.barabasi_albert_graph(n=num_nodes, m=m, seed=random.randint(0, 100000))
        while not nx.is_connected(graph):
            graph = nx.barabasi_albert_graph(n=num_nodes, m=m, seed=random.randint(0, 100000))

        flow_defs = {}
        num_flows = random.randint(5, 10)
        for i in range(num_flows):
            src, dst = random.sample(range(num_nodes), 2)
            flow_type = random.choice(["TT", "AVB"])
            # 대역폭 요구량을 명시적으로 추가
            size_bytes = random.randint(100, 1500)
            period_ms = random.randint(10, 100)
            bw_req_bps = (size_bytes * 8) / (period_ms / 1000)
            flow_defs[f"flow_{i}"] = {"src": src, "dst": dst, "type": flow_type, "size_bytes": size_bytes, "period_ms": period_ms, "bw_req_bps": bw_req_bps}

        deadlines_ms = {"TT": random.uniform(2, 5), "AVB": random.uniform(10, 30)}
        contingency_scenarios = []
        if random.random() < 0.5:
            num_failures = random.randint(1, 2)
            possible_edges = list(graph.edges)
            if possible_edges:
                for _ in range(num_failures):
                    contingency_scenarios.append({"failed_link": list(random.choice(possible_edges))})
        
        return {"graph": graph, "flow_definitions": flow_defs, "deadlines_ms": deadlines_ms, "contingency_scenarios": contingency_scenarios}

# ==============================================================================
# 2. 강화된 GNN 모델 및 상태 표현 (v3)
# ==============================================================================
class PolicyGNN_v3(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_heads=4, dropout_rate=0.3):
        super(PolicyGNN_v3, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=True, dropout=dropout_rate)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True, dropout=dropout_rate)
        self.conv3 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.leaky_relu(self.input_layer(x))
        
        # GAT 레이어 + Residual Connection
        x_res1 = F.elu(self.conv1(x, edge_index))
        # head-dimension 이슈로 residual connection은 3번째 layer에서만 적용
        
        x_res2 = F.elu(self.conv2(x_res1, edge_index))
        
        x_res3 = self.conv3(x_res2, edge_index)
        x = F.leaky_relu(x_res3) + x # Residual connection
        
        logits = self.output_layer(x).squeeze(-1)
        return logits

def get_state_tensor_v2(graph, all_flow_props, current_flow_id, partial_paths):
    num_nodes = graph.number_of_nodes()
    # Feature: [is_src, is_dst, degree, current_flow_bw_req, aggregated_node_congestion]
    features = np.zeros((num_nodes, 5), dtype=np.float32)

    # 1. Source and Destination
    props = all_flow_props[current_flow_id]
    features[props['src'], 0] = 1.0
    features[props['dst'], 1] = 1.0

    # 2. Node Degree (Normalized)
    degrees = np.array([deg for _, deg in graph.degree()], dtype=np.float32)
    max_degree = np.max(degrees) if np.max(degrees) > 0 else 1
    features[:, 2] = degrees / max_degree

    # 3. Current Flow BW Requirement (Normalized by link capacity)
    features[:, 3] = props['bw_req_bps'] / LINK_BANDWIDTH_BPS
    
    # 4. Aggregated Node Congestion
    link_usage = {e: 0 for e in graph.edges}
    for fid, path_pair in partial_paths.items():
        flow_bw = all_flow_props[fid]['bw_req_bps']
        for path_type in ['primary', 'backup']:
            if path_pair.get(path_type):
                path_edges = list(zip(path_pair[path_type][:-1], path_pair[path_type][1:]))
                for u, v in path_edges:
                    edge = tuple(sorted((u, v)))
                    if edge in link_usage:
                        link_usage[edge] += flow_bw

    for i in range(num_nodes):
        total_congestion = 0
        for neighbor in graph.neighbors(i):
            edge = tuple(sorted((i, neighbor)))
            if edge in link_usage:
                total_congestion += link_usage[edge]
        features[i, 4] = total_congestion / LINK_BANDWIDTH_BPS

    return Data(x=torch.tensor(features, dtype=torch.float),
                edge_index=torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous(),
                graph=graph)

class RLAgent:
    def __init__(self, state_dim, lr):
        self.policy_net = PolicyGNN_v3(state_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
    
    def select_action(self, state, current_path, is_eval=False):
        if is_eval: self.policy_net.eval()
        else: self.policy_net.train()

        with torch.set_grad_enabled(not is_eval):
            logits = self.policy_net(state.to(DEVICE))

        mask = torch.ones_like(logits) * -1e9
        valid_neighbors = [n for n in state.graph.neighbors(current_path[-1]) if n not in current_path]
        
        if not valid_neighbors: return None, None

        mask[valid_neighbors] = 0
        masked_logits = logits + mask

        dist = torch.distributions.Categorical(logits=masked_logits)
        action = masked_logits.argmax() if is_eval else dist.sample()
        log_prob = dist.log_prob(action) if not is_eval else None
        
        return action.item(), log_prob

    def find_path(self, graph, start_node, end_node, all_flow_props, current_flow_id, partial_paths={}, is_eval=False):
        current_path, log_probs = [start_node], []
        while current_path[-1] != end_node:
            state = get_state_tensor_v2(graph, all_flow_props, current_flow_id, partial_paths)
            action, log_prob = self.select_action(state, current_path, is_eval=is_eval)
            
            # 경로가 너무 길어지거나, 다음 노드를 찾지 못하면 실패
            if action is None or len(current_path) > graph.number_of_nodes() * 1.5:
                return None, None
            
            current_path.append(action)
            if log_prob is not None:
                log_probs.append(log_prob)
        
        return current_path, log_probs
    
    def update_policy_imitation(self, expert_path, graph, all_flow_props, current_flow_id, partial_paths, loss_fn):
        if not expert_path or len(expert_path) < 2: return 0.0
        
        total_loss = 0
        temp_partial_paths = partial_paths.copy()

        for j in range(len(expert_path) - 1):
            current_node, destination_node = expert_path[j], expert_path[-1]
            expert_action = expert_path[j+1]
            
            # 모방 학습 시에도 현재 경로 정보를 반영
            current_sub_path = expert_path[:j+1]
            
            state_tensor = get_state_tensor_v2(graph, all_flow_props, current_flow_id, temp_partial_paths)
            logits = self.policy_net(state_tensor.to(DEVICE))
            
            self.optimizer.zero_grad()
            loss = loss_fn(logits.unsqueeze(0), torch.tensor([expert_action], device=DEVICE))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss

# ==============================================================================
# 3. 솔버 정의 (GNN, Greedy, ILP)
# ==============================================================================
class BaseSolver:
    def __init__(self, name): self.name = name
    def solve(self, env, profile): raise NotImplementedError

class GNN_Solver_v3(BaseSolver):
    def __init__(self, model_path, state_dim):
        super().__init__("GNN_v3")
        self.agent = RLAgent(state_dim=state_dim, lr=0)
        self.agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))

    def solve(self, env, profile):
        paths, partial_paths = {}, {}
        graph = env.graph
        all_flow_props = profile['flow_definitions']
        
        # 주기가 짧은 플로우부터 라우팅
        sorted_flow_ids = sorted(all_flow_props.keys(), key=lambda k: all_flow_props[k]['period_ms'])

        for flow_id in sorted_flow_ids:
            props = all_flow_props[flow_id]
            p_path, _ = self.agent.find_path(graph, props['src'], props['dst'], all_flow_props, flow_id, partial_paths, is_eval=True)
            if p_path is None: return None # Primary 경로 못찾으면 실패

            backup_graph = graph.copy()
            if len(p_path) > 1:
                backup_graph.remove_edges_from(list(zip(p_path[:-1], p_path[1:])))

            b_path, _ = self.agent.find_path(backup_graph, props['src'], props['dst'], all_flow_props, flow_id, partial_paths, is_eval=True)
            
            paths[flow_id] = {'primary': p_path, 'backup': b_path}
            partial_paths[flow_id] = paths[flow_id]
            
        return paths

class Greedy_Solver(BaseSolver):
    def __init__(self): super().__init__("Greedy")
    def solve(self, env, profile):
        graph=env.graph; paths, link_usage={}, {tuple(sorted(edge)):0 for edge in graph.edges()}
        def weight_func(u, v, d): return 1 + link_usage.get(tuple(sorted((u, v))), 0) * 10
        sorted_flow_ids = sorted(profile['flow_definitions'].keys(), key=lambda k: profile['flow_definitions'][k]['period_ms'])
        for flow_id in sorted_flow_ids:
            props = profile['flow_definitions'][flow_id]
            try:
                primary_path=nx.shortest_path(graph, source=props['src'], target=props['dst'], weight=weight_func)
                for u, v in zip(primary_path[:-1], primary_path[1:]): link_usage[tuple(sorted((u, v)))]+=1
                backup_graph=graph.copy()
                if len(primary_path) > 1: backup_graph.remove_edges_from(list(zip(primary_path[:-1], primary_path[1:])))
                backup_path = nx.shortest_path(backup_graph, source=props['src'], target=props['dst'], weight=weight_func) if nx.has_path(backup_graph, source=props['src'], target=props['dst']) else None
                paths[flow_id]={'primary':primary_path, 'backup':backup_path}
            except nx.NetworkXNoPath: return None
        return paths

class ILP_Solver(BaseSolver):
    def __init__(self, time_limit_sec): super().__init__("ILP"); self.time_limit=time_limit_sec
    def solve(self, env, profile):
        prob=LpProblem("TSN_Routing", LpMinimize); G=env.graph; flow_defs=profile['flow_definitions']
        p_vars={(f,u,v):LpVariable(f"p_{f}_{u}_{v}", 0, 1, 'Binary') for f in flow_defs for u,v in G.edges}
        prob+=lpSum(p_vars[(f,u,v)] for f in flow_defs for u,v in G.edges), "Minimize_Total_Hops"
        for f, props in flow_defs.items():
            src, dst = props['src'], props['dst']
            for node in G.nodes():
                in_flow = lpSum(p_vars.get((f,i,node)) for i in G.predecessors(node) if (f,i,node) in p_vars)
                out_flow = lpSum(p_vars.get((f,node,j)) for j in G.successors(node) if (f,node,j) in p_vars)
                if node == src: prob += out_flow - in_flow == 1
                elif node == dst: prob += out_flow - in_flow == -1
                else: prob += out_flow - in_flow == 0
        # G를 DiGraph로 변환하여 사용
        DiG = nx.DiGraph(G)
        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=self.time_limit))
        if LpStatus[prob.status] !='Optimal': return None
        paths={}
        for f, props in flow_defs.items():
            src, dst = props['src'], props['dst']; p_path=[src]; curr=src
            while curr != dst and len(p_path) <= DiG.number_of_nodes():
                next_node = [j for i,j in DiG.edges if i==curr and p_vars[(f,i,j)].varValue > 0.9]
                if not next_node: break
                curr = next_node[0]; p_path.append(curr)
            if not p_path or p_path[-1] != dst: return None
            
            backup_graph = env.graph.copy()
            if len(p_path) > 1: backup_graph.remove_edges_from(list(zip(p_path[:-1], p_path[1:])))
            b_path = nx.shortest_path(backup_graph, src, dst) if nx.has_path(backup_graph, src, dst) else None
            paths[f] = {'primary': p_path, 'backup': b_path}
        return paths

# ==============================================================================
# 4. 학습 파이프라인 (v3)
# ==============================================================================
def run_imitation_learning(agent, expert_solver, profile_generator, iterations):
    print("\n===== 1단계: 전문가 모방 학습 (IL) 시작 =====")
    agent.policy_net.train()
    agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=IMITATION_LR)
    loss_fn = nn.CrossEntropyLoss()
    pbar = tqdm(range(iterations), desc="모방 학습", ncols=100)

    for i in pbar:
        profile = profile_generator.generate()
        env = TSN_Static_Env(profile["graph"], profile['flow_definitions'])
        # 모방학습에서는 Greedy를 전문가로 사용 (빠르고 안정적)
        expert_paths = expert_solver.solve(env, profile)
        if not expert_paths: continue

        partial_paths = {}
        sorted_flow_ids = sorted(profile['flow_definitions'].keys(), key=lambda k: profile['flow_definitions'][k]['period_ms'])
        
        total_loss = 0
        for flow_id in sorted_flow_ids:
            expert_path = expert_paths.get(flow_id, {}).get('primary')
            if expert_path and len(expert_path) > 1:
                loss = agent.update_policy_imitation(expert_path, env.graph, profile['flow_definitions'], flow_id, partial_paths, loss_fn)
                total_loss += loss
            # 실제 전문가가 사용한 경로를 다음 플로우 결정에 반영
            if flow_id in expert_paths:
                partial_paths[flow_id] = expert_paths[flow_id]
        
        if i % 50 == 0:
            pbar.set_postfix(avg_loss=f"{total_loss / len(sorted_flow_ids):.4f}")

    torch.save(agent.policy_net.state_dict(), IMITATION_MODEL_PATH)
    print(f"\n모방 학습 완료. 모델 저장: '{IMITATION_MODEL_PATH}'")

def producer_task(task_queue, stop_event):
    profile_gen = DynamicProfileGenerator()
    ilp_solver = ILP_Solver(time_limit_sec=ILP_TIME_LIMIT_SEC)
    while not stop_event.is_set():
        if task_queue.full():
            time.sleep(0.1)
            continue
        try:
            profile = profile_gen.generate()
            env = TSN_Static_Env(profile["graph"], profile['flow_definitions'])
            ilp_paths = ilp_solver.solve(env, profile)
            if ilp_paths:
                score_ilp, _ = env.evaluate_robust_configuration(ilp_paths, profile['deadlines_ms'], profile['contingency_scenarios'])
                if score_ilp > 0: # 유효한 해만 큐에 추가
                    task_data = {
                        "graph": nx.node_link_data(env.graph),
                        "flow_definitions": profile['flow_definitions'], "deadlines_ms": profile['deadlines_ms'],
                        "contingency_scenarios": profile['contingency_scenarios'], 
                        "ilp_paths": ilp_paths, "score_ilp": score_ilp
                    }
                    task_queue.put(task_data)
        except Exception:
            continue # ILP 풀다 에러나면 그냥 무시하고 다음 문제 생성

def run_ilp_guided_rl(agent, episodes):
    print("\n===== 2단계: 비동기 ILP-앵커 온라인 학습 (RL) 시작 =====")
    start_episode = 0
    model_loaded = False
    if os.path.exists(CHECKPOINT_PATH):
        print(f"체크포인트 '{CHECKPOINT_PATH}' 발견. 학습 재개.")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        start_episode = checkpoint['episode']
        model_loaded = True
    elif os.path.exists(IMITATION_MODEL_PATH):
        print(f"모방 학습 모델 '{IMITATION_MODEL_PATH}'에서 새로 시작.")
        agent.policy_net.load_state_dict(torch.load(IMITATION_MODEL_PATH, map_location=DEVICE))
        model_loaded = True

    if not model_loaded:
        print("경고: 학습된 모델이 없습니다. 랜덤 가중치로 시작합니다.")

    task_queue = mp.Queue(maxsize=TASK_QUEUE_SIZE)
    stop_event = mp.Event()
    producers = [mp.Process(target=producer_task, args=(task_queue, stop_event)) for _ in range(NUM_PRODUCERS)]
    for p in producers: p.start()
    print(f"{NUM_PRODUCERS}개의 CPU 생산자 프로세스 시작... GPU 학습 대기 중...")

    pbar = tqdm(range(start_episode, episodes), desc="온라인 학습", ncols=100, initial=start_episode, total=episodes)
    il_loss_fn = nn.CrossEntropyLoss()
    rl_wins_count, il_wins_count = 0, 0

    for ep in pbar:
        task_data = task_queue.get() # 큐에서 전문가가 푼 문제 가져오기
        
        G = nx.node_link_graph(task_data["graph"])
        env = TSN_Static_Env(G, task_data['flow_definitions'])
        
        # RL 에이전트의 최적해 탐색
        best_rl_score = -1.0
        best_rl_data = None
        for _ in range(RL_EXPLORATION_COUNT):
            paths, partial_paths, all_log_probs, failed = {}, {}, [], False
            all_flow_props = task_data['flow_definitions']
            sorted_flow_ids = sorted(all_flow_props.keys(), key=lambda k: all_flow_props[k]['period_ms'])

            for flow_id in sorted_flow_ids:
                props = all_flow_props[flow_id]
                p_path, p_logs = agent.find_path(env.graph, props['src'], props['dst'], all_flow_props, flow_id, partial_paths)
                if p_path is None: failed = True; break
                all_log_probs.extend(p_logs)
                
                backup_graph = env.graph.copy()
                if len(p_path) > 1: backup_graph.remove_edges_from(list(zip(p_path[:-1], p_path[1:])))
                b_path, b_logs = agent.find_path(backup_graph, props['src'], props['dst'], all_flow_props, flow_id, partial_paths)
                if b_path: all_log_probs.extend(b_logs)
                
                paths[flow_id] = {'primary': p_path, 'backup': b_path}
                partial_paths[flow_id] = paths[flow_id]
            
            if not failed:
                score_rl, _ = env.evaluate_robust_configuration(paths, task_data['deadlines_ms'], task_data['contingency_scenarios'])
                if score_rl > best_rl_score:
                    best_rl_score = score_rl
                    best_rl_data = {"score": score_rl, "log_probs": all_log_probs, "paths": paths}

        # ILP 앵커와 비교하여 학습 방향 결정
        if best_rl_data and best_rl_data['score'] >= task_data['score_ilp']:
            # RL 승리: 강화학습으로 정책 업데이트
            rl_wins_count += 1
            agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=ONLINE_LR_RL)
            
            # 보상 함수: 점수가 높을수록, 즉 지연시간이 적을수록 큰 보상
            # 점수를 제곱하여 차이를 극대화
            reward = best_rl_data['score'] ** 2
            
            policy_loss = [-log_prob * reward for log_prob in best_rl_data['log_probs']]
            agent.optimizer.zero_grad()
            if policy_loss:
                loss = torch.stack(policy_loss).sum()
                loss.backward()
                agent.optimizer.step()
        else:
            # ILP 승리 (또는 RL 실패): ILP의 해를 모방 학습
            il_wins_count += 1
            agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=ONLINE_LR_IMITATION)
            
            partial_paths = {}
            expert_paths = task_data['ilp_paths']
            sorted_flow_ids = sorted(expert_paths.keys(), key=lambda k: task_data['flow_definitions'][k]['period_ms'])
            
            for flow_id in sorted_flow_ids:
                primary_path = expert_paths.get(flow_id, {}).get('primary')
                agent.update_policy_imitation(primary_path, env.graph, task_data['flow_definitions'], flow_id, partial_paths, il_loss_fn)
                if flow_id in expert_paths:
                    partial_paths[flow_id] = expert_paths[flow_id]

        if ep > 0 and ep % CHECKPOINT_INTERVAL == 0:
            torch.save({'episode': ep, 'model_state_dict': agent.policy_net.state_dict()}, CHECKPOINT_PATH)
        
        pbar.set_postfix(RL_Wins=rl_wins_count, IL_Wins=il_wins_count, Q_Size=task_queue.qsize())

    # Final save and cleanup
    torch.save(agent.policy_net.state_dict(), FINAL_MODEL_PATH)
    print(f"\n온라인 학습 완료. 최종 모델 저장: '{FINAL_MODEL_PATH}'")
    stop_event.set()
    for p in producers:
        p.join(timeout=5)
        if p.is_alive(): p.terminate()
    print("모든 생산자 프로세스 종료.")

# ==============================================================================
# 5. 최종 벤치마크 실행
# ==============================================================================
def run_full_procedure_and_benchmark():
    if DEVICE.type == 'cuda':
        try: mp.set_start_method('spawn', force=True)
        except RuntimeError: pass

    # 1. 모방 학습 (선택적)
    if IMITATION_ITERATIONS > 0 and not os.path.exists(IMITATION_MODEL_PATH):
        agent_for_imitation = RLAgent(state_dim=5, lr=IMITATION_LR)
        greedy_expert = Greedy_Solver()
        profile_gen = DynamicProfileGenerator()
        run_imitation_learning(agent_for_imitation, greedy_expert, profile_gen, iterations=IMITATION_ITERATIONS)
    else:
        print("모방 학습을 건너뛰거나, 이미 완료된 모델을 사용합니다.")

    # 2. 온라인 학습
    agent_for_rl = RLAgent(state_dim=5, lr=ONLINE_LR_RL)
    run_ilp_guided_rl(agent_for_rl, episodes=ONLINE_EPISODES)
    
    # 3. 최종 벤치마크
    print("\n===== 최종 벤치마크 실행 =====")
    solvers = [
        GNN_Solver_v3(FINAL_MODEL_PATH, state_dim=5), 
        Greedy_Solver(), 
        ILP_Solver(time_limit_sec=ILP_TIME_LIMIT_SEC)
    ]
    results = []
    profile_gen = DynamicProfileGenerator()
    pbar_bench = tqdm(range(BENCHMARK_SCENARIOS), desc="최종 벤치마크", ncols=100)

    for i in pbar_bench:
        profile = profile_gen.generate()
        env = TSN_Static_Env(profile["graph"], profile['flow_definitions'])
        row = {"Scenario": i}
        for solver in solvers:
            start_time = time.time()
            paths = solver.solve(env, profile)
            end_time = time.time()
            computation_time = end_time - start_time
            
            score, _ = env.evaluate_robust_configuration(paths, profile['deadlines_ms'], profile['contingency_scenarios'])
            
            row[f"{solver.name}_Score"] = score
            row[f"{solver.name}_Time"] = computation_time
        results.append(row)
        
    df = pd.DataFrame(results)
    avg_scores = {solver.name: df[f"{solver.name}_Score"].mean() for solver in solvers}
    avg_times = {solver.name: df[f"{solver.name}_Time"].mean() for solver in solvers}

    print("\n\n===== 최종 벤치마크 결과 요약 (평균) =====")
    summary_df = pd.DataFrame([avg_scores, avg_times], index=["Average Score", "Average Time (s)"])
    print(summary_df.to_string())

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    fig.suptitle(f'Benchmark Results (v3 GNN vs. Baselines, {BENCHMARK_SCENARIOS} Scenarios)', fontsize=16)
    
    summary_df.T.plot(kind='bar', y='Average Score', ax=axes[0], rot=0, legend=False, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_title('Average Performance Score (Higher is Better)')
    axes[0].set_ylabel('Avg. Robustness Score')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    summary_df.T.plot(kind='bar', y='Average Time (s)', ax=axes[1], rot=0, legend=False, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_title('Average Computation Time (Lower is Better)')
    axes[1].set_ylabel('Avg. Time (seconds)')
    axes[1].set_yscale('log')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(RESULT_PLOT_PATH)
    print(f"\n결과 그래프가 '{RESULT_PLOT_PATH}'에 저장되었습니다.")


if __name__ == '__main__':
    run_full_procedure_and_benchmark()
