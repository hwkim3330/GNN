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
# --- 공통 설정 ---
IMITATION_ITERATIONS = 50000    # 모방 학습 반복 횟수 (시간 관계상 줄임, 원래 값: 5000000)
ONLINE_EPISODES = 20000         # 온라인 학습 에피소드 수 (시간 관계상 줄임, 원래 값: 2000000)
IMITATION_LR = 1e-4
ONLINE_LR_IMITATION = 5e-6
RL_EXPLORATION_COUNT = 3        # PPO는 더 안정적이므로 탐색 횟수를 줄여도 괜찮음
ILP_TIME_LIMIT_SEC = 60
BENCHMARK_SCENARIOS = 10
DEVICE = torch.device("cpu")

# --- 모델 경로 ---
IMITATION_MODEL_PATH = "gnn_imitated_actor_critic.pth"
PREV_MODEL_PATH = "gnn_imitated_actor_critic.pth" if os.path.exists("gnn_imitated_actor_critic.pth") else None
CHECKPOINT_PATH = "gnn_ppo_checkpoint.pth"
FINAL_MODEL_PATH = "gnn_congestion_aware_ppo.pth"
RESULT_PLOT_PATH = "benchmark_results_congestion_ppo.png"

# --- PPO 하이퍼파라미터 ---
ONLINE_LR_PPO = 3e-5            # PPO 학습률
PPO_GAMMA = 0.99                # 할인 계수 (Discount factor)
PPO_GAE_LAMBDA = 0.95           # GAE 람다 (Generalized Advantage Estimation)
PPO_CLIP_EPS = 0.2              # PPO 클리핑 엡실론
PPO_EPOCHS = 4                  # 한 번 수집한 데이터로 학습하는 횟수
PPO_CRITIC_COEF = 0.5           # Value loss 가중치
PPO_ENTROPY_COEF = 0.01         # Entropy loss 가중치 (탐험 장려)
REWARD_PER_STEP = -0.01         # 경로 탐색 시 스텝별 보상 (짧은 경로 유도)

# --- 시뮬레이션 환경 상수 ---
LINK_BANDWIDTH_BPS = 1e9
PROPAGATION_DELAY_NS_PER_METER = 5
LINK_LENGTH_METER = 10
SWITCH_PROC_DELAY_NS = 1000

# ==============================================================================
# 1. 환경, 모델, 솔버 정의
# ==============================================================================
# TSN_Static_Env_v2, DynamicProfileGenerator, BaseSolver, Greedy_Solver, ILP_Solver 클래스는 이전과 동일하므로 유지합니다.
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
                    if (u,v) in link_utilization: link_utilization[(u,v)] += usage_to_add
                    if (v,u) in link_utilization: link_utilization[(v,u)] += usage_to_add
        flow_results={}
        total_e2e_delay = 0
        for flow_id, path_pair in paths.items():
            props=self.flow_defs[flow_id]; path = path_pair.get('primary')
            use_backup = (failed_link and path and (failed_link in list(zip(path[:-1], path[1:])) or tuple(reversed(failed_link)) in list(zip(path[:-1], path[1:]))))
            if use_backup:
                path = path_pair.get('backup')
            if not path or not nx.is_path(eval_graph, path):
                if use_backup: return -1000, {"error":f"Backup path not found or invalid for {flow_id} on failure {failed_link}"}
                continue # Skip if primary path is invalid in non-failure scenario
            tx_time_ns=self._calculate_tx_time_ns(props['size_bytes']); e2e_d=0
            for u,v in zip(path[:-1], path[1:]):
                queuing_delay = self._get_queuing_delay_ns(link_utilization.get((u,v), 0))
                e2e_d += self.link_prop_delay_ns + SWITCH_PROC_DELAY_NS + tx_time_ns + queuing_delay
            flow_results[flow_id]={'e2e_delay_ms':e2e_d/1e6}
            deadline=deadlines_ms.get(props.get('type'))
            if deadline and (e2e_d/1e6) > deadline: return -1000, {"error":f"Deadline missed for {flow_id}"}
            total_e2e_delay += e2e_d/1e6
        # Reward: inverse of total latency. Higher is better.
        return 1.0/(1.0 + total_e2e_delay), {"results":flow_results}
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

# --- 모델 변경: Actor-Critic 구조로 변경 ---
class ActorCriticGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout_rate=0.2):
        super(ActorCriticGNN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout_rate)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout_rate)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Actor Head: 정책(어떤 행동을 할지)을 결정
        self.actor_head = nn.Linear(hidden_dim, 1)
        
        # Critic Head: 상태의 가치(얼마나 좋은 상황인지)를 평가
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.input_layer(x))
        
        x_res = self.conv1(x, edge_index)
        x = self.ln1(x + x_res); x = F.relu(x)
        
        x_res = self.conv2(x, edge_index)
        x = self.ln2(x + x_res)
        shared_features = F.relu(x)

        # Actor: 각 노드에 대한 정책 로짓 반환
        logits = self.actor_head(shared_features).squeeze(-1)
        
        # Critic: 그래프 전체의 상태 가치를 표현하기 위해 노드 특징을 평균냄
        # detach()를 사용하여 Critic 학습이 Actor의 GNN 부분에 영향을 주지 않도록 할 수 있으나,
        # 공유하는 것이 일반적이므로 여기서는 분리하지 않음.
        graph_embedding = torch.mean(shared_features, dim=0)
        value = self.critic_head(graph_embedding)
        
        return logits, value.squeeze()

def get_state_tensor(graph, flow_props, partial_paths):
    num_nodes = graph.number_of_nodes(); features = np.zeros((num_nodes, 4))
    if flow_props['src'] < num_nodes and flow_props['dst'] < num_nodes:
        features[flow_props['src'], 0] = 1; features[flow_props['dst'], 1] = 1
    
    link_usage = {tuple(sorted(edge)): 0 for edge in graph.edges()}
    total_flows = len(partial_paths) if partial_paths else 1

    for path_pair in partial_paths.values():
        for path_type in ['primary', 'backup']:
            path = path_pair.get(path_type)
            if path:
                for u,v in zip(path[:-1], path[1:]):
                    edge = tuple(sorted((u,v)))
                    if edge in link_usage: link_usage[edge] += 1
    
    node_usage = np.zeros(num_nodes)
    for u,v in graph.edges():
        norm_usage = link_usage.get(tuple(sorted((u,v))), 0) / total_flows
        node_usage[u] += norm_usage
        node_usage[v] += norm_usage
    
    features[:, 2] = node_usage
    max_degree = max(dict(graph.degree).values()) if graph.number_of_nodes() > 0 else 1
    for i in range(num_nodes): 
        features[i, 3] = graph.degree[i] / max_degree if max_degree > 0 else 0
        
    return Data(x=torch.tensor(features, dtype=torch.float), edge_index=torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous(), graph=graph)

# --- 에이전트 변경: REINFORCE -> PPO ---
class PPOAgent:
    def __init__(self, state_dim, lr):
        self.policy_net = ActorCriticGNN(state_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = []

    def set_optimizer_lr(self, lr):
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def clear_memory(self):
        self.memory = []

    def store_experience(self, state, action, log_prob, value):
        self.memory.append({"state": state, "action": action, "log_prob": log_prob, "value": value})

    def select_action(self, state, current_path, is_eval=False):
        if is_eval: self.policy_net.eval()
        else: self.policy_net.train()

        with torch.set_grad_enabled(not is_eval):
            logits, value = self.policy_net(state.to(DEVICE))
        
        mask = torch.ones_like(logits) * -1e9
        valid_neighbors = [n for n in state.graph.neighbors(current_path[-1]) if n not in current_path]
        if not valid_neighbors: return None, None, None

        mask[valid_neighbors] = 0
        masked_logits = logits + mask
        dist = torch.distributions.Categorical(logits=masked_logits)
        
        action = dist.sample() if not is_eval else masked_logits.argmax()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value

    def update(self, final_reward):
        if not self.memory: return 0, 0, 0
        
        # 1. 보상 계산: 마지막 스텝에 최종 보상, 나머지는 스텝별 보상
        rewards = []
        for i in range(len(self.memory)):
            if i == len(self.memory) - 1:
                rewards.append(final_reward)
            else:
                rewards.append(REWARD_PER_STEP)

        # 2. GAE (Generalized Advantage Estimation) 계산
        advantages = []
        gae = 0
        # 마지막 스텝의 value는 0으로 가정 (에피소드 종료)
        next_value = 0
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            value = self.memory[i]['value']
            # TD-error
            delta = reward + PPO_GAMMA * next_value - value
            # GAE
            gae = delta + PPO_GAMMA * PPO_GAE_LAMBDA * gae
            advantages.insert(0, gae)
            next_value = value
        
        advantages = torch.tensor(advantages, dtype=torch.float, device=DEVICE)
        # advantages 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        old_states = [mem['state'] for mem in self.memory]
        old_actions = torch.tensor([mem['action'] for mem in self.memory], dtype=torch.long, device=DEVICE)
        old_log_probs = torch.stack([mem['log_prob'] for mem in self.memory]).detach()
        old_values = torch.stack([mem['value'] for mem in self.memory]).detach()
        returns = advantages + old_values

        # 3. PPO 업데이트 (여러 에포크 동안)
        for _ in range(PPO_EPOCHS):
            # 이 작은 문제에서는 미니배치 없이 전체 배치를 사용
            new_log_probs, new_values, entropies = [], [], []
            for i, state in enumerate(old_states):
                logits, value = self.policy_net(state.to(DEVICE))
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs.append(dist.log_prob(old_actions[i]))
                new_values.append(value)
                entropies.append(dist.entropy())
            
            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values)
            entropy = torch.stack(entropies).mean()

            # 비율 계산
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 정책 손실 (Clipped Surrogate Objective)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 가치 손실 (Value Loss)
            value_loss = F.mse_loss(new_values, returns)

            # 총 손실
            loss = policy_loss + PPO_CRITIC_COEF * value_loss - PPO_ENTROPY_COEF * entropy

            # 업데이트
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.clear_memory()
        return policy_loss.item(), value_loss.item(), entropy.item()

    def update_policy_imitation(self, expert_path, graph, partial_paths, loss_fn):
        if not expert_path or len(expert_path) < 2: return 0
        total_loss=0
        for j in range(len(expert_path)-1):
            current_node, destination_node = expert_path[j], expert_path[-1]
            expert_action = expert_path[j+1]
            state_tensor=get_state_tensor(graph, {'src':current_node, 'dst':destination_node}, partial_paths)
            
            self.policy_net.train()
            logits, _ = self.policy_net(state_tensor.to(DEVICE))
            
            self.optimizer.zero_grad()
            loss = loss_fn(logits.unsqueeze(0), torch.tensor([expert_action], device=DEVICE))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

class GNN_PPO_Solver(BaseSolver):
    def __init__(self, model_path):
        super().__init__("GNN-PPO (Final)")
        self.agent = PPOAgent(state_dim=4, lr=0)
        self.agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))

    def solve(self, env, profile):
        paths, partial_paths = {}, {}
        graph = env.graph
        flow_ids = sorted(profile['flow_definitions'].keys(), key=lambda k: profile['flow_definitions'][k]['period_ms'])

        for flow_id in flow_ids:
            props = profile['flow_definitions'][flow_id]
            
            # 기본 경로 찾기
            current_path = [props['src']]
            while current_path[-1] != props['dst']:
                state = get_state_tensor(graph, {'src': current_path[-1], 'dst': props['dst']}, partial_paths)
                action, _, _ = self.agent.select_action(state, current_path, is_eval=True)
                if action is None or len(current_path) > graph.number_of_nodes() * 2:
                    return None # 경로 찾기 실패
                current_path.append(action)
            p_path = current_path
            
            # 백업 경로 찾기
            backup_graph = graph.copy()
            if len(p_path) > 1:
                backup_graph.remove_edges_from(list(zip(p_path[:-1], p_path[1:])))
            
            b_path = None
            if nx.has_path(backup_graph, props['src'], props['dst']):
                current_path_b = [props['src']]
                while current_path_b[-1] != props['dst']:
                    state_b = get_state_tensor(backup_graph, {'src': current_path_b[-1], 'dst': props['dst']}, partial_paths)
                    action_b, _, _ = self.agent.select_action(state_b, current_path_b, is_eval=True)
                    if action_b is None or len(current_path_b) > backup_graph.number_of_nodes() * 2:
                        b_path = None # 백업 경로 찾기 실패 시 None으로 처리
                        break
                    current_path_b.append(action_b)
                if current_path_b[-1] == props['dst']:
                    b_path = current_path_b

            paths[flow_id] = {'primary': p_path, 'backup': b_path}
            partial_paths[flow_id] = paths[flow_id]
        
        return paths

# Greedy_Solver와 ILP_Solver는 변경 없음
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

# ==============================================================================
# 2. 학습 파이프라인
# ==============================================================================
def run_imitation_learning(agent, expert_solver, profile_generator, iterations):
    print("\n===== 1단계: 대규모 전문가 모방 학습 시작 =====")
    agent.set_optimizer_lr(IMITATION_LR)
    loss_fn = nn.CrossEntropyLoss()
    pbar = tqdm(range(iterations), desc="모방 학습", ncols=100)
    
    for i in pbar:
        profile = profile_generator.generate()
        env = TSN_Static_Env_v2(profile["graph"], profile['flow_definitions'])
        expert_paths = expert_solver.solve(env, profile)
        if not expert_paths: continue

        partial_paths = {}
        flow_ids = sorted(profile['flow_definitions'].keys(), key=lambda k: profile['flow_definitions'][k]['period_ms'])
        total_loss = 0
        num_updates = 0
        for flow_id in flow_ids:
            expert_path = expert_paths.get(flow_id, {}).get('primary')
            loss = agent.update_policy_imitation(expert_path, env.graph, partial_paths, loss_fn)
            total_loss += loss
            if loss > 0: num_updates += 1
            partial_paths[flow_id] = expert_paths[flow_id]
        
        if i > 0 and i % 100 == 0 and num_updates > 0:
            avg_loss = total_loss / num_updates
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

    torch.save(agent.policy_net.state_dict(), IMITATION_MODEL_PATH)
    print(f"\n모방 학습 완료. 모델 저장: '{IMITATION_MODEL_PATH}'")

# --- 학습 로직 변경: ILP-Guided PPO ---
def run_ppo_guided_rl(agent, ilp_solver, profile_generator, episodes):
    print("\n===== 2단계: PPO-앵커 온라인 학습 시작 (단일 프로세스) =====")
    start_episode = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"체크포인트 '{CHECKPOINT_PATH}' 발견. 학습 재개.")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        start_episode = checkpoint['episode']
    elif PREV_MODEL_PATH:
        print(f"이전 모델 '{PREV_MODEL_PATH}'에서 학습 이어가기.")
        agent.policy_net.load_state_dict(torch.load(PREV_MODEL_PATH, map_location=DEVICE))

    pbar = tqdm(range(start_episode, episodes), desc="온라인 학습 (PPO)", ncols=100, initial=start_episode, total=episodes)
    il_loss_fn = nn.CrossEntropyLoss()

    for ep in pbar:
        profile = profile_generator.generate()
        env = TSN_Static_Env_v2(profile["graph"], profile['flow_definitions'])
        
        ilp_paths = ilp_solver.solve(env, profile)
        if not ilp_paths: continue
        score_ilp, _ = env.evaluate_robust_configuration(ilp_paths, profile['deadlines_ms'], profile['contingency_scenarios'])

        # PPO 에이전트로 탐색 및 경험 수집
        best_rl_score = -1.0
        best_trajectory_memory = None

        for _ in range(RL_EXPLORATION_COUNT):
            current_memory = []
            paths, partial_paths, failed = {}, {}, False
            flow_ids = sorted(profile['flow_definitions'].keys(), key=lambda k: profile['flow_definitions'][k]['period_ms'])

            for flow_id in flow_ids:
                props = profile['flow_definitions'][flow_id]
                # 기본 경로
                current_path_p = [props['src']]
                while current_path_p[-1] != props['dst']:
                    state = get_state_tensor(env.graph, {'src': current_path_p[-1], 'dst': props['dst']}, partial_paths)
                    action, log_prob, value = agent.select_action(state, current_path_p)
                    if action is None or len(current_path_p) > env.num_nodes*2: failed=True; break
                    current_memory.append({"state": state, "action": action, "log_prob": log_prob, "value": value})
                    current_path_p.append(action)
                if failed: break
                
                p_path = current_path_p
                # 백업 경로
                b_path = None
                backup_graph = env.graph.copy()
                if len(p_path) > 1: backup_graph.remove_edges_from(list(zip(p_path[:-1], p_path[1:])))
                if nx.has_path(backup_graph, props['src'], props['dst']):
                    current_path_b = [props['src']]
                    while current_path_b[-1] != props['dst']:
                        state = get_state_tensor(backup_graph, {'src': current_path_b[-1], 'dst': props['dst']}, partial_paths)
                        action, log_prob, value = agent.select_action(state, current_path_b)
                        if action is None or len(current_path_b) > env.num_nodes*2: break
                        current_memory.append({"state": state, "action": action, "log_prob": log_prob, "value": value})
                        current_path_b.append(action)
                    if current_path_b[-1] == props['dst']:
                        b_path = current_path_b

                paths[flow_id] = {'primary': p_path, 'backup': b_path}
                partial_paths[flow_id] = paths[flow_id]
            
            if not failed:
                score_rl, _ = env.evaluate_robust_configuration(paths, profile['deadlines_ms'], profile['contingency_scenarios'])
                if score_rl > best_rl_score:
                    best_rl_score = score_rl
                    best_trajectory_memory = current_memory
        
        # PPO 업데이트 또는 모방 학습
        if best_trajectory_memory and best_rl_score >= score_ilp:
            agent.set_optimizer_lr(ONLINE_LR_PPO)
            agent.memory = best_trajectory_memory # 최고의 경험을 에이전트 메모리로 전달
            p_loss, v_loss, entropy = agent.update(final_reward=best_rl_score)
            pbar.set_postfix(Action="PPO_Update", Score=f"{best_rl_score:.3f}>{score_ilp:.3f}", P_Loss=f"{p_loss:.2f}", V_Loss=f"{v_loss:.2f}")
        elif ilp_paths:
            agent.set_optimizer_lr(ONLINE_LR_IMITATION)
            loss, partial_paths = 0, {}
            for flow_id in sorted(ilp_paths.keys()):
                primary_path = ilp_paths.get(flow_id, {}).get('primary')
                loss += agent.update_policy_imitation(primary_path, env.graph, partial_paths, il_loss_fn)
                partial_paths[flow_id] = ilp_paths[flow_id]
            pbar.set_postfix(Action="IL_Correction", Score=f"{best_rl_score:.3f}<{score_ilp:.3f}", IL_Loss=f"{loss:.3f}")

        if (ep + 1) % 100 == 0:
            torch.save({'episode': ep + 1, 'model_state_dict': agent.policy_net.state_dict()}, CHECKPOINT_PATH)

    torch.save(agent.policy_net.state_dict(), FINAL_MODEL_PATH)
    print(f"\n온라인 학습 완료. 최종 모델 저장: '{FINAL_MODEL_PATH}'")

# ==============================================================================
# 3. 전체 실행 및 벤치마크
# ==============================================================================
def run_full_procedure_and_benchmark():
    agent = PPOAgent(state_dim=4, lr=IMITATION_LR)
    expert_solver = ILP_Solver(time_limit_sec=60)
    profile_gen = DynamicProfileGenerator()

    # 1단계: 모방 학습
    if not os.path.exists(IMITATION_MODEL_PATH):
        run_imitation_learning(agent, expert_solver, profile_gen, iterations=IMITATION_ITERATIONS)

    # 2단계: PPO 기반 온라인 학습
    run_ppo_guided_rl(agent, expert_solver, profile_gen, episodes=ONLINE_EPISODES)

    # 3단계: 최종 벤치마크
    print("\n===== 최종 랜덤 벤치마크 실행 =====")
    solvers = [GNN_PPO_Solver(FINAL_MODEL_PATH), Greedy_Solver(), ILP_Solver(time_limit_sec=ILP_TIME_LIMIT_SEC)]
    results = []
    pbar_bench = tqdm(range(BENCHMARK_SCENARIOS), desc="랜덤 벤치마크 진행", ncols=100)

    for i in pbar_bench:
        profile = profile_gen.generate()
        env = TSN_Static_Env_v2(profile["graph"], profile['flow_definitions'])
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

    # 결과 시각화
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)
    fig.suptitle(f'Congestion-Aware PPO Benchmark Results ({BENCHMARK_SCENARIOS} Scenarios)', fontsize=16)
    
    summary_df.T.plot(kind='bar', y='Average Score', ax=axes[0], rot=0, legend=False, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_title('Average Performance Score (Higher is Better)')
    axes[0].set_ylabel('Avg. Robustness Score')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    summary_df.T.plot(kind='bar', y='Average Time (s)', ax=axes[1], rot=0, legend=False, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_title('Average Computation Time (Lower is Better)')
    axes[1].set_ylabel('Avg. Time (seconds)')
    axes[1].set_yscale('log')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(RESULT_PLOT_PATH)
    print(f"\n결과 그래프가 '{RESULT_PLOT_PATH}'에 저장되었습니다.")

if __name__ == '__main__':
    run_full_procedure_and_benchmark()
