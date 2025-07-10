import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import networkx as nx
import random
import numpy as np
import time
import copy # NameError 해결
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, PULP_CBC_CMD

# ==============================================================================
# 0. 설정
# ==============================================================================
os.environ['OMP_NUM_THREADS'] = '8'; os.environ['MKL_NUM_THREADS'] = '8'; torch.set_num_threads(8)

IMITATION_EPISODES = 1000
ONLINE_EPISODES = 5000
IMITATION_LR = 1e-4
ONLINE_LR_PPO = 3e-5
ONLINE_LR_IMITATION = 5e-6
ILP_TIME_LIMIT_SEC = 60
BENCHMARK_SCENARIOS = 10
DEVICE = torch.device("cpu")

IMITATION_MODEL_PATH = "gnn_hop_by_hop_imitated.pth"
CHECKPOINT_PATH = "gnn_hop_by_hop_checkpoint.pth"
FINAL_MODEL_PATH = "gnn_hop_by_hop_final.pth"
RESULT_PLOT_PATH = "benchmark_results_hop_by_hop.png"
PREV_MODEL_PATH = IMITATION_MODEL_PATH if os.path.exists(IMITATION_MODEL_PATH) else None

PPO_GAMMA = 0.99; PPO_GAE_LAMBDA = 0.95; PPO_CLIP_EPS = 0.2
PPO_EPOCHS_UPDATE = 4; PPO_CRITIC_COEF = 0.5; PPO_ENTROPY_COEF = 0.01
REWARD_PER_STEP = -0.01
GRADIENT_CLIP_NORM = 1.0

CORE_AGG_BANDWIDTH_BPS = 10e9
AGG_EDGE_BANDWIDTH_BPS = 5e9
LINK_LENGTH_METER = 10
SWITCH_PROC_DELAY_NS = 1000
PROPAGATION_DELAY_NS_PER_METER = 5

# ==============================================================================
# 1. 환경, 모델, 솔버 정의
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
                for u, v in zip(path[:-1], path[1:]): link_data_rate[tuple(sorted((u,v)))] += rate_to_add
        link_utilization = {edge: rate / eval_graph.edges[edge]['bandwidth'] for edge, rate in link_data_rate.items()}
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

class ActorCriticSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.2):
        super().__init__(); self.conv1 = SAGEConv(input_dim, hidden_dim); self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate); self.actor_head = nn.Linear(hidden_dim, 1); self.critic_head = nn.Linear(hidden_dim, 1)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index)); x = self.dropout(x); x = F.relu(self.conv2(x, edge_index))
        return self.actor_head(x).squeeze(-1), self.critic_head(torch.mean(x, dim=0)).squeeze()

def get_state_tensor(graph, flow_props, partial_paths, all_flow_defs):
    num_nodes = graph.number_of_nodes(); features = np.zeros((num_nodes, 6))
    features[flow_props['src'], 0] = 1; features[flow_props['dst'], 1] = 1
    max_degree = max(d for _, d in graph.degree) if num_nodes > 1 else 1
    if max_degree > 0: features[:, 2] = [d/max_degree for _, d in graph.degree]
    for i in range(num_nodes):
        if i in graph.graph.get('core_switches', []): features[i, 3] = 3
        elif i in graph.graph.get('agg_switches', []): features[i, 3] = 2
        elif i in graph.graph.get('edge_switches', []): features[i, 3] = 1
    max_bw = max(d['bandwidth'] for _, _, d in graph.edges(data=True)) if graph.edges else 1
    if max_bw > 0:
        for i in range(num_nodes):
            if graph.degree(i) > 0: features[i, 4] = np.mean([graph.edges[i, nbr]['bandwidth'] for nbr in graph.neighbors(i)]) / max_bw
    link_data_rate = {tuple(sorted(e)): 0.0 for e in graph.edges}
    if partial_paths:
        for flow_id, path_pair in partial_paths.items():
            if (flow_info := all_flow_defs.get(flow_id)):
                rate = (flow_info['size_bytes'] * 8) / (flow_info['period_ms'] * 1e-3)
                for path in [path_pair.get('primary'), path_pair.get('backup')]:
                    if path: [link_data_rate.update({tuple(sorted(e)): link_data_rate.get(tuple(sorted(e)), 0) + rate}) for e in zip(path[:-1], path[1:])]
    node_utilization = np.zeros(num_nodes)
    for u, v in graph.edges:
        edge, bw = tuple(sorted((u,v))), graph.edges[u,v]['bandwidth']
        util = link_data_rate.get(edge, 0) / bw if bw > 0 else 0
        node_utilization[u] += util; node_utilization[v] += util
    features[:, 5] = node_utilization
    return Data(x=torch.tensor(features, dtype=torch.float), edge_index=torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous())

class SAGE_PPOAgent:
    def __init__(self, state_dim, lr):
        self.policy_net = ActorCriticSAGE(state_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = []
    def set_optimizer_lr(self, lr): self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
    def clear_memory(self): self.memory = []
    def store_experience(self, state, action, log_prob, value): self.memory.append({"state": state, "action": action, "log_prob": log_prob, "value": value})
    def _select_action_internal(self, state_data, current_path, graph, is_eval=False):
        with torch.set_grad_enabled(not is_eval):
            logits, value = self.policy_net(state_data.x.to(DEVICE), state_data.edge_index.to(DEVICE))
        mask = torch.ones_like(logits) * -1e9
        valid_neighbors = [n for n in graph.neighbors(current_path[-1]) if n not in current_path]
        if not valid_neighbors: return None, None, None
        mask[valid_neighbors] = 0; masked_logits = logits + mask
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample() if not is_eval else masked_logits.argmax()
        return action.item(), dist.log_prob(action), value
    def find_path(self, graph, props, partial_paths, all_flow_defs, is_eval):
        path, failed = [props['src']], False
        while path[-1] != props['dst']:
            state = get_state_tensor(graph, {'src': path[-1], 'dst': props['dst']}, partial_paths, all_flow_defs)
            action, log_prob, value = self._select_action_internal(state, path, graph, is_eval=is_eval)
            if action is None or len(path) > graph.number_of_nodes(): failed = True; break
            if not is_eval: self.store_experience(state, action, log_prob, value)
            path.append(action)
        return (None, True) if failed else (path, False)
    def update(self, final_reward):
        if not self.memory: return 0, 0, 0
        rewards, advantages, gae, next_value = [REWARD_PER_STEP] * (len(self.memory) - 1) + [final_reward], [], 0, 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + PPO_GAMMA * next_value - self.memory[i]['value']
            gae = delta + PPO_GAMMA * PPO_GAE_LAMBDA * gae; advantages.insert(0, gae); next_value = self.memory[i]['value'].detach()
        advantages = torch.tensor(advantages, dtype=torch.float, device=DEVICE)
        if advantages.numel() > 1: advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_states, old_actions = [m['state'] for m in self.memory], torch.tensor([m['action'] for m in self.memory], dtype=torch.long, device=DEVICE)
        old_log_probs, old_values = torch.stack([m['log_prob'] for m in self.memory]).detach(), torch.stack([m['value'] for m in self.memory]).detach()
        returns = advantages + old_values
        for _ in range(PPO_EPOCHS_UPDATE):
            new_log_probs, new_values, entropies = [], [], []
            for i, state in enumerate(old_states):
                logits, value = self.policy_net(state.x.to(DEVICE), state.edge_index.to(DEVICE))
                dist = torch.distributions.Categorical(logits=logits); new_log_probs.append(dist.log_prob(old_actions[i])); new_values.append(value); entropies.append(dist.entropy())
            new_log_probs, new_values, entropy = torch.stack(new_log_probs), torch.stack(new_values), torch.stack(entropies).mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1, surr2 = ratio * advantages, torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * advantages
            policy_loss, value_loss = -torch.min(surr1, surr2).mean(), F.mse_loss(new_values, returns)
            loss = policy_loss + PPO_CRITIC_COEF * value_loss - PPO_ENTROPY_COEF * entropy
            self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRADIENT_CLIP_NORM); self.optimizer.step()
        self.clear_memory(); return policy_loss.item(), value_loss.item(), entropy.item()
    def update_imitation(self, expert_path, graph, partial_paths, all_flow_defs, loss_fn):
        if not expert_path or len(expert_path) < 2: return 0
        total_loss = 0
        for j in range(len(expert_path) - 1):
            state = get_state_tensor(graph, {'src': expert_path[j], 'dst': expert_path[-1]}, partial_paths, all_flow_defs)
            logits, _ = self.policy_net(state.x.to(DEVICE), state.edge_index.to(DEVICE))
            loss = loss_fn(logits.unsqueeze(0), torch.tensor([expert_path[j+1]], device=DEVICE))
            self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRADIENT_CLIP_NORM); self.optimizer.step()
            total_loss += loss.item()
        return total_loss

class BaseSolver:
    def __init__(self, name): self.name = name; self.solve_time = 0
    def solve(self, env, profile): raise NotImplementedError

class GNN_SAGE_Solver(BaseSolver):
    def __init__(self, model_path):
        super().__init__("GNN-SAGE"); self.agent = SAGE_PPOAgent(state_dim=6, lr=0)
        if os.path.exists(model_path): self.agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    def solve(self, env, profile):
        paths, partial_paths = {}, {}
        for flow_id, props in sorted(profile['flow_definitions'].items(), key=lambda item: item[1]['period_ms']):
            p_path, _ = self.agent.find_path(env.graph, props, partial_paths, profile['flow_definitions'], is_eval=True)
            backup_graph = env.graph.copy()
            if p_path and len(p_path) > 1: backup_graph.remove_edges_from(list(zip(p_path[:-1], p_path[1:])))
            b_path = None
            if nx.has_path(backup_graph, props['src'], props['dst']):
                b_path, _ = self.agent.find_path(backup_graph, props, partial_paths, profile['flow_definitions'], is_eval=True)
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
                backup_graph = graph.copy(); backup_graph.remove_edges_from(list(zip(p_path[:-1], p_path[1:])))
                b_path = nx.shortest_path(backup_graph, source=props['src'], target=props['dst']) if nx.has_path(backup_graph, props['src'], props['dst']) else None
                paths[flow_id] = {'primary': p_path, 'backup': b_path}
            except (nx.NetworkXNoPath, nx.NetworkXUnfeasible): return None
        return paths

class ILP_Solver(BaseSolver):
    def __init__(self, time_limit_sec): super().__init__("ILP"); self.time_limit = time_limit_sec
    def solve(self, env, profile):
        prob, G, flow_defs = LpProblem("TSN_Routing_BW", LpMinimize), env.graph, profile['flow_definitions']
        all_directed_edges = list(G.edges()) + [(v, u) for u, v in G.edges()]
        p_vars = {f: {e: LpVariable(f"p_{f}_{e[0]}_{e[1]}", 0, 1, 'Binary') for e in all_directed_edges} for f in flow_defs}
        prob += lpSum(p_vars[f][e] for f in flow_defs for e in all_directed_edges), "Minimize_Path_Length"
        for u, v in G.edges():
            prob += lpSum((flow_defs[f]['size_bytes'] * 8 / (flow_defs[f]['period_ms'] * 1e-3)) * (p_vars[f][(u,v)] + p_vars[f][(v,u)]) for f in flow_defs) <= G.edges[u,v]['bandwidth']
        for f, props in flow_defs.items():
            src, dst = props['src'], props['dst']
            for node in G.nodes():
                in_flow = lpSum(p_vars[f][(i, node)] for i in G.neighbors(node))
                out_flow = lpSum(p_vars[f][(node, j)] for j in G.neighbors(node))
                if node == src: prob += out_flow - in_flow == 1
                elif node == dst: prob += in_flow - out_flow == 1
                else: prob += out_flow - in_flow == 0
        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=self.time_limit))
        if LpStatus[prob.status] != 'Optimal': return None
        paths = {}
        for f, props in flow_defs.items():
            src, dst = props['src'], props['dst']; p_path, curr_p = [src], src
            while curr_p != dst and len(p_path) <= G.number_of_nodes():
                next_node = next((v for u, v in all_directed_edges if u == curr_p and p_vars[f][(u,v)].varValue > 0.9), None)
                if next_node is None: break
                p_path.append(next_node); curr_p = next_node
            if not p_path or p_path[-1] != dst: return None
            backup_graph = G.copy(); backup_graph.remove_edges_from(list(zip(p_path[:-1], p_path[1:])))
            b_path = nx.shortest_path(backup_graph, src, dst) if nx.has_path(backup_graph, src, dst) else None
            paths[f] = {'primary': p_path, 'backup': b_path}
        return paths

def run_imitation_learning(agent, expert_solver, profile_generator, episodes):
    print("\n===== 1단계: 전문가 모방 학습 시작 ====="); agent.set_optimizer_lr(IMITATION_LR); loss_fn = nn.CrossEntropyLoss()
    for ep in tqdm(range(episodes), desc="모방 학습", ncols=100):
        profile = profile_generator.generate()
        env = RealisticTSNEnv(profile["graph"], profile['flow_definitions'])
        expert_paths = expert_solver.solve(env, profile)
        if not expert_paths: continue
        partial_paths, total_loss = {}, 0
        for flow_id in sorted(profile['flow_definitions'].keys(), key=lambda k: profile['flow_definitions'][k]['period_ms']):
            if (expert_path := expert_paths.get(flow_id, {}).get('primary')):
                loss = agent.update_imitation(expert_path, env.graph, partial_paths, profile['flow_definitions'], loss_fn)
                total_loss += loss
            partial_paths[flow_id] = expert_paths.get(flow_id, {})
        tqdm.write(f"에피소드 {ep+1} 완료, 총 손실: {total_loss:.4f}")
    torch.save(agent.policy_net.state_dict(), IMITATION_MODEL_PATH)

def run_ppo_guided_rl(agent, ilp_solver, profile_generator, episodes):
    print("\n===== 2단계: PPO 온라인 학습 시작 =====")
    start_episode = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        agent.policy_net.load_state_dict(checkpoint['model_state_dict']); start_episode = checkpoint['episode']
    elif PREV_MODEL_PATH: agent.policy_net.load_state_dict(torch.load(PREV_MODEL_PATH, map_location=DEVICE))
    pbar, il_loss_fn = tqdm(range(start_episode, episodes), desc="온라인 학습", ncols=100, initial=start_episode, total=episodes), nn.CrossEntropyLoss()
    for ep in pbar:
        profile = profile_generator.generate(); env = RealisticTSNEnv(profile["graph"], profile['flow_definitions'])
        ilp_paths = ilp_solver.solve(env, profile)
        score_ilp = env.evaluate_robust_configuration(ilp_paths, profile['deadlines_ms'], profile['contingency_scenarios'])[0] if ilp_paths else 0.0
        agent.clear_memory(); paths, partial_paths, failed = {}, {}, False
        for flow_id, props in sorted(profile['flow_definitions'].items(), key=lambda k: profile['flow_definitions'][k]['period_ms']):
            p_path, p_failed = agent.find_path(env.graph, props, partial_paths, profile['flow_definitions'], is_eval=False)
            if p_failed: failed=True; break
            backup_graph = env.graph.copy()
            if p_path and len(p_path) > 1: backup_graph.remove_edges_from(list(zip(p_path[:-1], p_path[1:])))
            b_path, _ = (agent.find_path(backup_graph, props, partial_paths, profile['flow_definitions'], is_eval=False)) if nx.has_path(backup_graph, props['src'], props['dst']) else (None, False)
            paths[flow_id] = {'primary': p_path, 'backup': b_path}; partial_paths[flow_id] = paths[flow_id]
        if not failed:
            score_rl, _ = env.evaluate_robust_configuration(paths, profile['deadlines_ms'], profile['contingency_scenarios'])
            if score_rl > 0:
                agent.set_optimizer_lr(ONLINE_LR_PPO); agent.update(final_reward=score_rl)
                pbar.set_postfix(Action="PPO", Score=f"{score_rl:.3f}")
                if score_rl < score_ilp and ilp_paths:
                    agent.set_optimizer_lr(ONLINE_LR_IMITATION)
                    for f_id in sorted(ilp_paths.keys()):
                        if (p_path := ilp_paths.get(f_id, {}).get('primary')): agent.update_imitation(p_path, env.graph, {}, profile['flow_definitions'], il_loss_fn)
        else: agent.clear_memory()
        if (ep + 1) % 100 == 0: torch.save({'episode': ep + 1, 'model_state_dict': agent.policy_net.state_dict()}, CHECKPOINT_PATH)

def run_full_procedure_and_benchmark():
    agent = SAGE_PPOAgent(state_dim=6, lr=IMITATION_LR); expert_solver = ILP_Solver(time_limit_sec=ILP_TIME_LIMIT_SEC)
    profile_gen = RealisticProfileGenerator()
    if not os.path.exists(IMITATION_MODEL_PATH): run_imitation_learning(agent, expert_solver, profile_gen, episodes=IMITATION_EPISODES)
    else: agent.policy_net.load_state_dict(torch.load(IMITATION_MODEL_PATH, map_location=DEVICE))
    run_ppo_guided_rl(agent, expert_solver, profile_gen, episodes=ONLINE_EPISODES)
    torch.save(agent.policy_net.state_dict(), FINAL_MODEL_PATH)
    print("\n===== 최종 랜덤 벤치마크 실행 =====")
    solvers = [GNN_SAGE_Solver(FINAL_MODEL_PATH), Greedy_Solver(), ILP_Solver(ILP_TIME_LIMIT_SEC)]
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
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), sharex=True)
    fig.suptitle(f'Realistic Benchmark Results ({BENCHMARK_SCENARIOS} Scenarios)', fontsize=16)
    pd.DataFrame.from_dict(avg_scores, orient='index', columns=['Score']).plot(kind='bar', y='Score', ax=axes[0], rot=0, legend=False, color=['#3498db', '#f1c40f', '#e74c3c']); axes[0].set_title('Average Performance Score (Higher is Better)'); axes[0].set_ylabel('Avg. Robustness Score'); axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    for c in axes[0].containers: axes[0].bar_label(c, fmt='%.3f')
    pd.DataFrame.from_dict(avg_times, orient='index', columns=['Time']).plot(kind='bar', y='Time', ax=axes[1], rot=0, legend=False, color=['#3498db', '#f1c40f', '#e74c3c']); axes[1].set_title('Average Computation Time (Lower is Better)'); axes[1].set_ylabel('Avg. Time (seconds)'); axes[1].set_yscale('log'); axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    for c in axes[1].containers: axes[1].bar_label(c, fmt='%.3f')
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(RESULT_PLOT_PATH); print(f"\n결과 그래프가 '{RESULT_PLOT_PATH}'에 저장되었습니다.")

if __name__ == '__main__':
    run_full_procedure_and_benchmark()
