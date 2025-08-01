import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch
import networkx as nx
import random
import numpy as np
import time
import copy
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, PULP_CBC_CMD
import itertools

# ==============================================================================
# 0. 설정
# ==============================================================================
os.environ['OMP_NUM_THREADS'] = '16'; os.environ['MKL_NUM_THREADS'] = '16'; torch.set_num_threads(16)

IMITATION_EPOCHS = 50
IMITATION_DATASET_SIZE = 500
IMITATION_LR = 1e-4
BENCHMARK_SCENARIOS = 100
DEVICE = torch.device("cpu")

MODEL_PATH = "gnn_path_scorer_final.pth"
RESULT_PLOT_PATH = "benchmark_results_path_scorer.png"
K_SHORTEST_PATHS = 5

CORE_AGG_BANDWIDTH_BPS = 10e9
AGG_EDGE_BANDWIDTH_BPS = 5e9
LINK_LENGTH_METER = 10
SWITCH_PROC_DELAY_NS = 1000
PROPAGATION_DELAY_NS_PER_METER = 5

# ==============================================================================
# 1. 환경 정의 (이전과 동일)
# ==============================================================================
class RealisticProfileGenerator:
    def _generate_fat_tree(self, k):
        if k % 2 != 0: raise ValueError("k must be an even number.")
        num_pods = k; num_core_switches = (k // 2) ** 2
        num_agg_switches = num_edge_switches = k * (k // 2)
        G = nx.Graph()
        core_switches = range(num_core_switches)
        agg_switches = range(num_core_switches, num_core_switches + num_agg_switches)
        edge_switches = range(agg_switches.stop, agg_switches.stop + num_edge_switches)
        for i in range(num_core_switches):
            for j in range(num_pods):
                agg_switch_idx = j * (k // 2) + (i // (k // 2))
                G.add_edge(core_switches[i], agg_switches[agg_switch_idx], bandwidth=CORE_AGG_BANDWIDTH_BPS)
        for j in range(num_pods):
            for i in range(k // 2):
                agg_switch_idx = j * (k // 2) + i
                for l in range(k // 2):
                    edge_switch_idx = j * (k // 2) + l
                    G.add_edge(agg_switches[agg_switch_idx], edge_switches[edge_switch_idx], bandwidth=AGG_EDGE_BANDWIDTH_BPS)
        G.graph.update({'core_switches': list(core_switches), 'agg_switches': list(agg_switches), 'edge_switches': list(edge_switches)})
        return G
    def generate(self):
        k = random.choice([4, 6]); graph = self._generate_fat_tree(k)
        edge_switches = graph.graph['edge_switches']; flow_defs = {}
        num_flows = random.randint(20, 70)
        num_servers = max(1, len(edge_switches) // 10)
        server_nodes = random.sample(edge_switches, num_servers)
        for i in range(num_flows):
            flow_type = random.choices(["TT", "AVB", "BE"], weights=[0.2, 0.4, 0.4], k=1)[0]
            if random.random() < 0.6 and server_nodes: src, dst = random.choice(edge_switches), random.choice(server_nodes)
            else: src, dst = random.sample(edge_switches, 2)
            if src == dst: continue
            if flow_type == "TT": flow_defs[f"flow_{i}"] = {"src": src, "dst": dst, "type": "TT", "size_bytes": random.randint(100, 1000), "period_ms": random.choice([5, 10, 20])}
            elif flow_type == "AVB": flow_defs[f"flow_{i}"] = {"src": src, "dst": dst, "type": "AVB", "size_bytes": random.randint(500, 2000), "period_ms": random.randint(10, 50)}
            else: flow_defs[f"flow_{i}"] = {"src": src, "dst": dst, "type": "BE", "size_bytes": random.randint(100, 1500), "period_ms": random.randint(20, 100)}
        deadlines_ms = {"TT": 5.0, "AVB": 20.0}; contingency_scenarios = []
        if random.random() < 0.5:
            for _ in range(random.randint(1, 2)):
                if random.random() < 0.3: contingency_scenarios.append({"failed_node": random.choice(graph.graph['agg_switches'])})
                else: contingency_scenarios.append({"failed_link": list(random.choice(list(graph.edges())))})
        return {"graph": graph, "flow_definitions": flow_defs, "deadlines_ms": deadlines_ms, "contingency_scenarios": contingency_scenarios}

class RealisticTSNEnv:
    def __init__(self, graph, flow_definitions):
        self.graph = graph; self.flow_defs = flow_definitions
        self.link_prop_delay_ns = PROPAGATION_DELAY_NS_PER_METER * LINK_LENGTH_METER
    def _calculate_tx_time_ns(self, size_bytes, bandwidth_bps): return (size_bytes * 8 / bandwidth_bps) * 1e9 if bandwidth_bps > 0 else 1e9
    def _get_queuing_delay_ns(self, link_utilization): return 1e9 if link_utilization >= 1.0 else 1000 * (link_utilization / (1.0 - link_utilization + 1e-9))
    def _evaluate_single_scenario(self, paths, deadlines_ms, failed_link=None, failed_node=None):
        eval_graph = self.graph.copy()
        if failed_node is not None and failed_node in eval_graph: eval_graph.remove_node(failed_node)
        if failed_link is not None and eval_graph.has_edge(*failed_link): eval_graph.remove_edge(*failed_link)
        link_data_rate = {tuple(sorted(edge)): 0.0 for edge in eval_graph.edges}
        for flow_id, path_pair in paths.items():
            path = path_pair.get('primary')
            if path and nx.is_path(eval_graph, path):
                rate_to_add = (self.flow_defs[flow_id]['size_bytes'] * 8) / (self.flow_defs[flow_id]['period_ms'] * 1e-3)
                for u, v in zip(path[:-1], path[1:]):
                    edge = tuple(sorted((u,v)))
                    if edge in link_data_rate: link_data_rate[edge] += rate_to_add
        link_utilization = {edge: rate / eval_graph.edges[edge]['bandwidth'] for edge, rate in link_data_rate.items() if eval_graph.has_edge(*edge)}
        for edge, util in link_utilization.items():
            if util >= 1.0: return -1000, {"error": f"Bandwidth exceeded on link {edge} (util: {util:.2f})"}
        flow_results, total_path_len = {}, 0
        for flow_id, path_pair in paths.items():
            props, path = self.flow_defs[flow_id], path_pair.get('primary')
            use_backup = (failed_node is not None and path and failed_node in path) or (failed_link is not None and path and (tuple(sorted(failed_link)) in [tuple(sorted(e)) for e in zip(path[:-1], path[1:])]))
            if use_backup: path = path_pair.get('backup')
            if not path or not nx.is_path(eval_graph, path):
                if props['type'] != 'BE': return -1000, {"error": f"Path for critical flow {flow_id} is invalid."}
                continue
            e2e_d, total_path_len = 0, total_path_len + len(path) - 1
            for u, v in zip(path[:-1], path[1:]):
                edge = tuple(sorted((u,v)))
                tx_time_ns = self._calculate_tx_time_ns(props['size_bytes'], eval_graph.edges[edge]['bandwidth'])
                queuing_delay = self._get_queuing_delay_ns(link_utilization.get(edge, 0))
                e2e_d += self.link_prop_delay_ns + SWITCH_PROC_DELAY_NS + tx_time_ns + queuing_delay
            flow_results[flow_id] = {'e2e_delay_ms': e2e_d / 1e6}
            if (deadline := deadlines_ms.get(props['type'])) and (e2e_d / 1e6) > deadline:
                return -1000, {"error": f"Deadline of {deadline}ms missed for {flow_id} (delay: {e2e_d/1e6:.3f}ms)"}
        if not flow_results: return 0.0, {"error": "No flows were routed successfully."}
        latencies = [res['e2e_delay_ms'] for res in flow_results.values()]
        avg_latency, max_latency = (np.mean(latencies), np.max(latencies)) if latencies else (0, 0)
        score = 1.0 / (1.0 + 0.8 * avg_latency + 0.2 * max_latency)
        return score, {"total_path_len": total_path_len}
    def evaluate_robust_configuration(self, paths, deadlines_ms, contingency_scenarios):
        if not paths: return 0.0, {"error": "No paths provided."}
        p_score, p_details = self._evaluate_single_scenario(paths, deadlines_ms)
        if p_score <= 0: return 0.0, p_details
        c_scores = [self._evaluate_single_scenario(paths, deadlines_ms, failed_link=tuple(s.get('failed_link')) if s.get('failed_link') else None, failed_node=s.get('failed_node'))[0] for s in contingency_scenarios]
        if any(s <= 0 for s in c_scores): return 0.0, {"error": "Failed in a contingency scenario."}
        avg_c_score = np.mean(c_scores) if c_scores else p_score
        resource_penalty = 1 + 0.001 * p_details.get("total_path_len", 0)
        return (0.7 * p_score + 0.3 * avg_c_score) / resource_penalty, {}

# ==============================================================================
# 2. 새로운 아키텍처: 경로 채점 GNN
# ==============================================================================
class PathScorerGNN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(node_feature_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.path_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.scorer_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, network_batch_data, path_node_indices_list):
        x, edge_index, batch = network_batch_data.x, network_batch_data.edge_index, network_batch_data.batch
        
        # GNN Encoder
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        node_embeddings = x

        # Graph-level embedding
        graph_embedding = global_mean_pool(node_embeddings, batch)

        path_scores = []
        for i, path_nodes in enumerate(path_node_indices_list):
            # Select node embeddings for the current path
            path_node_embeddings = node_embeddings[path_nodes]
            
            # Path Encoder (LSTM)
            _, (hn, _) = self.path_encoder(path_node_embeddings.unsqueeze(0))
            path_embedding = hn.squeeze(0)
            
            # Combine graph and path embeddings
            current_graph_embedding = graph_embedding[i]
            combined_embedding = torch.cat([current_graph_embedding, path_embedding.squeeze(0)], dim=-1)
            
            # Score Prediction
            score = self.scorer_mlp(combined_embedding)
            path_scores.append(score)
            
        return torch.cat(path_scores)

def get_network_state_tensor(graph, partial_paths, all_flow_defs):
    num_nodes = graph.number_of_nodes(); features = np.zeros((num_nodes, 4))
    max_degree = max(d for _, d in graph.degree) if num_nodes > 1 else 1
    if max_degree > 0: features[:, 0] = [d/max_degree for _, d in graph.degree]
    for i in range(num_nodes):
        if i in graph.graph.get('core_switches', []): features[i, 1] = 3
        elif i in graph.graph.get('agg_switches', []): features[i, 1] = 2
        elif i in graph.graph.get('edge_switches', []): features[i, 1] = 1
    max_bw = max(d['bandwidth'] for _, _, d in graph.edges(data=True)) if graph.edges else 1
    if max_bw > 0:
        for i in range(num_nodes):
            if graph.degree(i) > 0: features[i, 2] = np.mean([graph.edges[i, nbr]['bandwidth'] for nbr in graph.neighbors(i)]) / max_bw
    link_data_rate = {tuple(sorted(e)): 0.0 for e in graph.edges}
    if partial_paths:
        for flow_id, path_pair in partial_paths.items():
            if (flow_info := all_flow_defs.get(flow_id)):
                rate = (flow_info['size_bytes'] * 8) / (flow_info['period_ms'] * 1e-3)
                if (path := path_pair.get('primary')):
                    for u, v in zip(path[:-1], path[1:]): link_data_rate[tuple(sorted((u,v)))] += rate
    node_utilization = np.zeros(num_nodes)
    for u, v in graph.edges:
        edge, bw = tuple(sorted((u,v))), graph.edges[u,v]['bandwidth']
        util = link_data_rate.get(edge, 0) / bw if bw > 0 else 0
        node_utilization[u] += util; node_utilization[v] += util
    features[:, 3] = node_utilization
    return Data(x=torch.tensor(features, dtype=torch.float), edge_index=torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous())

# ==============================================================================
# 3. 솔버 정의
# ==============================================================================
class BaseSolver:
    def __init__(self, name): self.name = name; self.solve_time = 0
    def solve(self, env, profile): raise NotImplementedError

class GNN_Path_Scorer_Solver(BaseSolver):
    def __init__(self, model_path):
        super().__init__("GNN-PathScorer")
        self.model = PathScorerGNN(node_feature_dim=4).to(DEVICE)
        if os.path.exists(model_path): self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
    def solve(self, env, profile):
        paths, partial_paths = {}, {}
        graph, all_flow_defs = env.graph, profile['flow_definitions']
        for flow_id, props in sorted(all_flow_defs.items(), key=lambda item: item[1]['period_ms']):
            src, dst = props['src'], props['dst']
            try: candidate_paths = list(itertools.islice(nx.shortest_simple_paths(graph, src, dst), K_SHORTEST_PATHS))
            except nx.NetworkXNoPath: return None
            if not candidate_paths: continue
            net_state = get_network_state_tensor(graph, partial_paths, all_flow_defs)
            path_node_indices_list = [torch.tensor(p, dtype=torch.long).to(DEVICE) for p in candidate_paths]
            net_batch = Batch.from_data_list([net_state] * len(candidate_paths)).to(DEVICE)
            with torch.no_grad():
                scores = self.model(net_batch, path_node_indices_list)
            best_path_idx = torch.argmax(scores).item()
            p_path = candidate_paths[best_path_idx]
            backup_graph = graph.copy()
            if len(p_path) > 1: backup_graph.remove_edges_from(list(zip(p_path[:-1], p_path[1:])))
            b_path = next(itertools.islice(nx.shortest_simple_paths(backup_graph, src, dst), 1), None) if nx.has_path(backup_graph, src, dst) else None
            paths[flow_id] = {'primary': p_path, 'backup': b_path}; partial_paths[flow_id] = paths[flow_id]
        return paths

class Greedy_Solver(BaseSolver):
    def __init__(self): super().__init__("Greedy")
    def solve(self, env, profile):
        graph, paths, link_rates = env.graph, {}, {tuple(sorted(e)): 0 for e in env.graph.edges}
        for flow_id, props in sorted(profile['flow_definitions'].items(), key=lambda item: item[1]['period_ms']):
            try:
                flow_rate = (props['size_bytes'] * 8) / (props['period_ms'] * 1e-3)
                def weight_func(u, v, d):
                    edge, bw = tuple(sorted((u,v))), graph.edges[u,v]['bandwidth']
                    util = (link_rates.get(edge, 0) + flow_rate) / bw if bw > 0 else 1e9
                    return 1e9 if util >= 1.0 else 1 + 100 * (util ** 2)
                p_path = nx.shortest_path(graph, source=props['src'], target=props['dst'], weight=weight_func)
                for u, v in zip(p_path[:-1], p_path[1:]): link_rates[tuple(sorted((u,v)))] += flow_rate
                backup_graph = graph.copy()
                if len(p_path) > 1: backup_graph.remove_edges_from(list(zip(p_path[:-1], p_path[1:])))
                b_path = nx.shortest_path(backup_graph, source=props['src'], target=props['dst']) if nx.has_path(backup_graph, props['src'], props['dst']) else None
                paths[flow_id] = {'primary': p_path, 'backup': b_path}
            except (nx.NetworkXNoPath, nx.NetworkXUnfeasible): return None
        return paths

# ==============================================================================
# 4. 새로운 학습 파이프라인
# ==============================================================================
def generate_imitation_dataset(profile_generator, env_class, num_episodes):
    dataset = []
    print("모방 학습을 위한 전문가 데이터셋 생성 중...")
    for _ in tqdm(range(num_episodes), desc="데이터셋 생성"):
        profile = profile_generator.generate()
        env = env_class(profile['graph'], profile['flow_definitions'])
        partial_paths = {}
        for flow_id, props in sorted(profile['flow_definitions'].items(), key=lambda item: item[1]['period_ms']):
            src, dst = props['src'], props['dst']
            try: candidate_paths = list(itertools.islice(nx.shortest_simple_paths(env.graph, src, dst), K_SHORTEST_PATHS))
            except nx.NetworkXNoPath: continue
            if not candidate_paths: continue
            candidate_scores = []
            for candidate_path in candidate_paths:
                temp_paths = copy.deepcopy(partial_paths)
                temp_paths[flow_id] = {'primary': candidate_path}
                score, _ = env.evaluate_robust_configuration(temp_paths, profile['deadlines_ms'], profile['contingency_scenarios'])
                candidate_scores.append(score)
            if max(candidate_scores) <= 0: continue
            net_state_tensor = get_network_state_tensor(env.graph, partial_paths, profile['flow_definitions'])
            dataset.append({'net_state': net_state_tensor, 'candidates': candidate_paths, 'scores': torch.tensor(candidate_scores, dtype=torch.float)})
            best_path_idx = np.argmax(candidate_scores)
            best_path = candidate_paths[best_path_idx]
            partial_paths[flow_id] = {'primary': best_path}
    return dataset

def run_imitation_learning(model, optimizer, dataset, epochs):
    print("\n===== 경로 채점기(Path Scorer) 모방 학습 시작 =====")
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0; random.shuffle(dataset)
        pbar = tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for data_point in pbar:
            optimizer.zero_grad()
            net_states = [data_point['net_state']] * len(data_point['candidates'])
            path_node_indices = [torch.tensor(p, dtype=torch.long) for p in data_point['candidates']]
            target_scores = data_point['scores'].to(DEVICE)
            net_batch = Batch.from_data_list(net_states).to(DEVICE)
            predicted_scores = model(net_batch, path_node_indices)
            loss = loss_fn(predicted_scores, target_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        print(f"Epoch {epoch+1} 완료, 평균 손실: {total_loss / len(dataset):.6f}")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"학습 완료. 모델 저장: {MODEL_PATH}")

# ==============================================================================
# 5. 메인 실행 함수
# ==============================================================================
def run_full_procedure_and_benchmark():
    profile_gen = RealisticProfileGenerator()
    model = PathScorerGNN(node_feature_dim=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=IMITATION_LR)
    
    if not os.path.exists(MODEL_PATH):
        dataset = generate_imitation_dataset(profile_gen, RealisticTSNEnv, num_episodes=IMITATION_DATASET_SIZE)
        if not dataset:
            print("데이터셋 생성 실패. 유효한 시나리오가 없습니다. 학습을 건너뜁니다.")
        else:
            run_imitation_learning(model, optimizer, dataset, epochs=IMITATION_EPOCHS)
    else:
        print(f"'{MODEL_PATH}' 발견. 학습을 건너뜁니다.")

    print("\n===== 최종 랜덤 벤치마크 실행 (Path Scorer) =====")
    solvers = [GNN_Path_Scorer_Solver(MODEL_PATH), Greedy_Solver()]
    results = []
    for i in tqdm(range(BENCHMARK_SCENARIOS), desc="벤치마크", ncols=100):
        profile = profile_gen.generate(); env = RealisticTSNEnv(profile["graph"], profile['flow_definitions'])
        row = {"Scenario": i}
        for solver in solvers:
            start_time = time.time(); paths = solver.solve(env, profile)
            solver.solve_time = time.time() - start_time
            score, details = env.evaluate_robust_configuration(paths, profile['deadlines_ms'], profile['contingency_scenarios'])
            if score <= 0: print(f"Solver {solver.name} failed on scenario {i}: {details.get('error', 'Unknown')}")
            row[f"{solver.name}_Score"], row[f"{solver.name}_Time"] = max(0, score), solver.solve_time
        results.append(row)
    
    df = pd.DataFrame(results)
    avg_scores = {s.name: df[f"{s.name}_Score"].mean() for s in solvers}; avg_times = {s.name: df[f"{s.name}_Time"].mean() for s in solvers}
    
    print("\n\n===== 최종 벤치마크 결과 요약 (평균) ====="); print(pd.DataFrame([avg_scores, avg_times], index=["Average Score", "Average Time (s)"]).to_string())
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)
    fig.suptitle(f'Path Scorer Benchmark Results ({BENCHMARK_SCENARIOS} Scenarios)', fontsize=16)
    pd.DataFrame.from_dict(avg_scores, orient='index', columns=['Score']).plot(kind='bar', y='Score', ax=axes[0], rot=0, legend=False, color=['#3498db', '#f1c40f'])
    axes[0].set_title('Average Performance Score (Higher is Better)'); axes[0].set_ylabel('Avg. Robustness Score'); axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    for c in axes[0].containers: axes[0].bar_label(c, fmt='%.3f')
    
    pd.DataFrame.from_dict(avg_times, orient='index', columns=['Time']).plot(kind='bar', y='Time', ax=axes[1], rot=0, legend=False, color=['#3498db', '#f1c40f'])
    axes[1].set_title('Average Computation Time (Lower is Better)'); axes[1].set_ylabel('Avg. Time (seconds)'); axes[1].set_yscale('log'); axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    for c in axes[1].containers: axes[1].bar_label(c, fmt='%.3f')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(RESULT_PLOT_PATH); print(f"\n결과 그래프가 '{RESULT_PLOT_PATH}'에 저장되었습니다.")

if __name__ == '__main__':
    run_full_procedure_and_benchmark()
