import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
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

# ==============================================================================
# 0. 설정
# ==============================================================================
# --- 데이터 생성 ---
DATASET_PATH = "ilp_dataset.pt"
NUM_DATA_SAMPLES = 10000  # CPU 단일 실행이므로 샘플 수 조정
ILP_TIME_LIMIT_SEC = 30
# --- 1단계: 모방학습 ---
IMITATION_EPOCHS = 30
IMITATION_BATCH_SIZE = 64
IMITATION_LR = 1e-4
IMITATION_MODEL_PATH = "ppo_imitator.pth"
# --- 2단계: PPO 강화학습 ---
RL_EPISODES = 5000
RL_LR = 3e-5
GAMMA = 0.99
EPS_CLIP = 0.2
UPDATE_TIMESTEP = 1000 # PPO 업데이트 주기 (스텝 수 기준)
PPO_UPDATES = 5
# --- 모델 및 벤치마크 ---
FINAL_MODEL_PATH = "gnn_ppo_final_cpu.pth"
BENCHMARK_SCENARIOS = 20
RESULT_PLOT_PATH = "benchmark_results_ppo_cpu.png"
DEVICE = torch.device("cpu")

# ==============================================================================
# 1. 환경 및 모델 정의
# ==============================================================================
class TSN_Static_Env:
    def __init__(self, graph, flow_definitions): self.graph = graph; self.flow_defs = flow_definitions
    def evaluate_score(self, paths):
        if not paths: return 0.0
        total_hops = 0; link_usage = {}
        for path_pair in paths.values():
            for path_type in ['primary', 'backup']:
                path = path_pair.get(path_type)
                if path and len(path) > 1:
                    total_hops += len(path) - 1
                    for u, v in zip(path[:-1], path[1:]):
                        edge = tuple(sorted((u,v))); link_usage[edge] = link_usage.get(edge, 0) + 1
        max_congestion = max(link_usage.values()) if link_usage else 0
        score = 1.0 / (1.0 + total_hops + max_congestion**2); return score

class DynamicProfileGenerator:
    def generate(self):
        num_nodes=random.randint(8, 20); m=random.randint(2, 4); graph=nx.barabasi_albert_graph(n=num_nodes, m=m)
        while not nx.is_connected(graph): graph=nx.barabasi_albert_graph(n=num_nodes, m=m)
        flow_defs={}; num_flows=random.randint(5, 15)
        for i in range(num_flows): src, dst=random.sample(range(num_nodes), 2); flow_defs[f"flow_{i}"]={"src":src, "dst":dst}
        return {"graph":graph, "flow_definitions":flow_defs}

class ActorCriticGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ActorCriticGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, 1)
        self.critic_head = nn.Linear(hidden_dim, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_shared = F.relu(self.conv1(x, edge_index))
        x_shared = F.relu(self.conv2(x_shared, edge_index))
        action_logits = self.actor_head(x_shared).squeeze(-1)
        state_value = self.critic_head(x_shared)
        return action_logits, state_value
    def evaluate(self, state_batch, action):
        action_logits, state_values_all_nodes = self.forward(state_batch)
        node_slices = torch.cat([state_batch.ptr, torch.tensor([state_batch.num_nodes], device=DEVICE)])
        action_logprobs = []; dist_entropy = []; state_values = []
        for i in range(state_batch.num_graphs):
            start, end = node_slices[i], node_slices[i+1]
            graph_logits = action_logits[start:end]
            dist = torch.distributions.Categorical(logits=graph_logits)
            action_in_graph = action[i]
            action_in_graph_local = action_in_graph - start
            action_logprobs.append(dist.log_prob(action_in_graph_local))
            dist_entropy.append(dist.entropy().mean())
            state_values.append(state_values_all_nodes[start:end].mean())
        return torch.stack(action_logprobs), torch.stack(state_values), torch.stack(dist_entropy)

def get_state_tensor(graph, current_node, dest_node, partial_paths):
    num_nodes = graph.number_of_nodes(); features = np.zeros((num_nodes, 3))
    features[current_node, 0] = 1; features[dest_node, 1] = 1
    link_usage = {tuple(sorted(e)): 0 for e in graph.edges}
    for p in partial_paths.values():
        path = p.get('primary');
        if path:
            for u,v in zip(path[:-1], path[1:]): link_usage[tuple(sorted((u,v)))] += 1
    for u,v in graph.edges():
        usage = link_usage[tuple(sorted((u,v)))]
        features[u, 2] += usage; features[v, 2] += usage
    return Data(x=torch.tensor(features, dtype=torch.float), edge_index=torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous())

class Memory:
    def __init__(self): self.actions = []; self.states = []; self.logprobs = []; self.rewards = []; self.is_terminals = []
    def clear_memory(self): self.actions.clear(); self.states.clear(); self.logprobs.clear(); self.rewards.clear(); self.is_terminals.clear()

class PPOAgent:
    def __init__(self, state_dim, hidden_dim, lr, gamma, eps_clip, updates):
        self.lr = lr; self.gamma = gamma; self.eps_clip = eps_clip; self.ppo_updates = updates
        self.policy = ActorCriticGNN(state_dim, hidden_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCriticGNN(state_dim, hidden_dim).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
    def select_action(self, state, memory, current_path):
        with torch.no_grad():
            action_logits, _ = self.policy_old.forward(state.to(DEVICE))
            mask = torch.ones_like(action_logits) * -1e9
            valid_neighbors = [n for n in state.graph.neighbors(current_path[-1]) if n not in current_path]
            if not valid_neighbors: return None
            mask[valid_neighbors] = 0; masked_logits = action_logits + mask
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
        if memory is not None:
            memory.states.append(state); memory.actions.append(action); memory.logprobs.append(dist.log_prob(action))
        return action.item()
    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        old_states = Batch.from_data_list(memory.states).to(DEVICE)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).to(DEVICE)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).to(DEVICE)
        for _ in range(self.ppo_updates):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())

class BaseSolver:
    def __init__(self, name): self.name=name
    def solve(self, env, profile): raise NotImplementedError

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

class GNN_PPOSolver(BaseSolver):
    def __init__(self, model_path):
        super().__init__("GNN (PPO)")
        self.agent = PPOAgent(3, 64, 0, 0, 0, 0)
        self.agent.policy.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.agent.policy_old.load_state_dict(self.agent.policy.state_dict())
    def solve(self, env, profile):
        paths, partial_paths = {}, {}
        graph = env.graph
        for flow_id, props in sorted(profile['flow_definitions'].items()):
            current_path = [props['src']]
            while current_path[-1] != props['dst']:
                state = get_state_tensor(graph, current_path[-1], props['dst'], partial_paths)
                action = self.agent.select_action(state, None, current_path)
                if action is None or len(current_path) > graph.number_of_nodes(): return None
                current_path.append(action)
            primary_path = current_path
            backup_graph = graph.copy()
            if len(primary_path) > 1: backup_graph.remove_edges_from(list(zip(primary_path[:-1], primary_path[1:])))
            backup_path = None
            if nx.has_path(backup_graph, props['src'], props['dst']):
                current_path_b = [props['src']]
                while current_path_b[-1] != props['dst']:
                    state = get_state_tensor(backup_graph, current_path_b[-1], props['dst'], partial_paths)
                    action = self.agent.select_action(state, None, current_path_b)
                    if action is None or len(current_path_b) > graph.number_of_nodes(): break
                    current_path_b.append(action)
                if current_path_b[-1] == props['dst']: backup_path = current_path_b
            paths[flow_id] = {'primary': primary_path, 'backup': backup_path}
            partial_paths[flow_id] = paths[flow_id]
        return paths

class Greedy_Solver(BaseSolver):
    def __init__(self): super().__init__("Greedy")
    def solve(self, env, profile):
        graph=env.graph; paths, link_usage={}, {tuple(sorted(edge)):0 for edge in graph.edges()}
        def weight_func(u, v, d): return 1+link_usage.get(tuple(sorted((u, v))), 0)*10
        for flow_id, props in sorted(profile['flow_definitions'].items()):
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

def generate_dataset_sequential(num_samples):
    print(f"\n===== 순차 데이터 생성 시작 =====")
    dataset = []
    pbar = tqdm(total=num_samples, desc="학습 데이터 생성 중")
    profile_gen = DynamicProfileGenerator(); ilp_solver = ILP_Solver(ILP_TIME_LIMIT_SEC)
    while len(dataset) < num_samples:
        profile = profile_gen.generate()
        env = TSN_Static_Env(profile["graph"], profile['flow_definitions'])
        expert_paths = ilp_solver.solve(env, profile)
        if not expert_paths: continue
        partial_paths = {}
        for flow_id in sorted(profile['flow_definitions'].keys()):
            path = expert_paths.get(flow_id, {}).get('primary')
            if path and len(path) > 1:
                for j in range(len(path) - 1):
                    current_node, dest_node = path[j], path[-1]
                    expert_action = path[j+1]
                    state_tensor = get_state_tensor(env.graph, current_node, dest_node, partial_paths)
                    data = Data(x=state_tensor.x, edge_index=state_tensor.edge_index, y=torch.tensor([expert_action], dtype=torch.long))
                    dataset.append(data)
                    pbar.update(1)
                    if len(dataset) >= num_samples: break
                partial_paths[flow_id] = {'primary': path}
            if len(dataset) >= num_samples: break
    pbar.close(); print(f"\n총 {len(dataset)}개의 고품질 학습 데이터 생성 완료.")
    return dataset

def train_imitator(model, dataset, epochs):
    print("\n===== GNN 모방 학습 시작 =====")
    optimizer = torch.optim.Adam(model.parameters(), lr=IMITATION_LR); loss_fn = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=IMITATION_BATCH_SIZE, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out, _ = model.forward(batch)
            loss = loss_fn(out, batch.y.squeeze())
            loss.backward(); optimizer.step(); total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), IMITATION_MODEL_PATH)
    print(f"\n모방 학습 완료. 최종 모델 저장: '{IMITATION_MODEL_PATH}'")

def run_ppo_training(agent, profile_generator, episodes):
    print("\n===== 2단계: PPO 강화학습 시작 =====")
    if not os.path.exists(IMITATION_MODEL_PATH):
        print(f"오류: 모방학습 모델 '{IMITATION_MODEL_PATH}'이 없습니다. 1단계를 먼저 실행하세요.")
        return
    agent.policy.load_state_dict(torch.load(IMITATION_MODEL_PATH, map_location=DEVICE))
    agent.policy_old.load_state_dict(agent.policy.state_dict())
    
    memory = Memory(); time_step = 0; running_reward = 0
    pbar = tqdm(range(1, episodes + 1), desc="PPO 학습", ncols=100)
    for ep in pbar:
        profile = profile_generator.generate(); env = TSN_Static_Env(profile["graph"], profile['flow_definitions'])
        paths, partial_paths = {}, {}
        ep_reward = 0
        
        for flow_id, props in sorted(profile['flow_definitions'].items()):
            current_path = [props['src']]; terminal = False
            while not terminal:
                time_step += 1
                state = get_state_tensor(env.graph, current_path[-1], props['dst'], partial_paths)
                action = agent.select_action(state, memory, current_path)
                if action is None or len(current_path) > env.graph.number_of_nodes():
                    memory.rewards.append(-1); memory.is_terminals.append(True); terminal = True
                else:
                    current_path.append(action)
                    if action == props['dst']:
                        memory.rewards.append(0); memory.is_terminals.append(True); terminal = True
                    else:
                        memory.rewards.append(-0.01); memory.is_terminals.append(False)
            paths[flow_id] = {'primary': current_path if terminal and current_path[-1] == props['dst'] else None, 'backup':None}
            partial_paths[flow_id] = paths[flow_id]
        
        final_score = env.evaluate_score(paths)
        if memory.rewards: memory.rewards[-1] += final_score
        ep_reward = final_score

        if time_step >= UPDATE_TIMESTEP:
            agent.update(memory); memory.clear_memory(); time_step = 0
        
        running_reward += ep_reward
        if ep % 100 == 0: pbar.set_postfix(avg_reward=f"{running_reward/100:.4f}"); running_reward = 0

    torch.save(agent.policy.state_dict(), FINAL_MODEL_PATH)
    print(f"\nPPO 학습 완료. 최종 모델 저장: '{FINAL_MODEL_PATH}'")

def run_full_procedure_and_benchmark():
    if not os.path.exists(DATASET_PATH):
        dataset = generate_dataset_sequential(NUM_DATA_SAMPLES)
        torch.save(dataset, DATASET_PATH)
    else:
        print(f"기존 데이터셋 '{DATASET_PATH}'를 불러옵니다.")
        dataset = torch.load(DATASET_PATH)

    ppo_agent = PPOAgent(3, 64, RL_LR, GAMMA, EPS_CLIP, PPO_UPDATES)
    
    if not os.path.exists(IMITATION_MODEL_PATH):
        train_imitator(ppo_agent.policy, dataset, epochs=IMITATION_EPOCHS)
    
    run_ppo_training(ppo_agent, DynamicProfileGenerator(), episodes=RL_EPISODES)
    
    print("\n===== 최종 랜덤 벤치마크 실행 =====")
    solvers=[GNN_PPOSolver(FINAL_MODEL_PATH), Greedy_Solver(), ILP_Solver(ILP_TIME_LIMIT_SEC)]
    results=[]; pbar_bench=tqdm(range(BENCHMARK_SCENARIOS), desc="랜덤 벤치마크 진행", ncols=100)
    profile_gen = DynamicProfileGenerator()
    for i in pbar_bench:
        profile=profile_gen.generate(); env=TSN_Static_Env(profile["graph"], profile['flow_definitions']); row={"Scenario": i}
        for solver in solvers:
            start_time=time.time(); paths=solver.solve(env, profile); end_time=time.time(); computation_time=end_time-start_time
            score = env.evaluate_score(paths)
            row[f"{solver.name}_Score"] = score; row[f"{solver.name}_Time"] = computation_time
        results.append(row)
    
    df=pd.DataFrame(results)
    avg_scores={solver.name: df[f"{solver.name}_Score"].mean() for solver in solvers}
    avg_times={solver.name: df[f"{solver.name}_Time"].mean() for solver in solvers}
    print("\n\n===== 최종 벤치마크 결과 요약 (평균) ====="); summary_df=pd.DataFrame([avg_scores, avg_times], index=["Average Score", "Average Time (s)"]); print(summary_df.to_string())
    
    fig, axes=plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True); fig.suptitle(f'Final Benchmark: PPO vs. Others ({BENCHMARK_SCENARIOS} Scenarios)', fontsize=16)
    summary_df.T.sort_values("Average Score", ascending=False).plot(kind='bar', y='Average Score', ax=axes[0], rot=0, legend=False); axes[0].set_title('Average Performance Score (Higher is Better)'); axes[0].set_ylabel('Avg. Score'); axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    summary_df.T.sort_values("Average Time (s)").plot(kind='bar', y='Average Time (s)', ax=axes[1], rot=0, legend=False); axes[1].set_title('Average Computation Time (Lower is Better)'); axes[1].set_ylabel('Avg. Time (seconds)'); axes[1].set_yscale('log'); axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(RESULT_PLOT_PATH); print(f"\n결과 그래프가 '{RESULT_PLOT_PATH}'에 저장되었습니다.")

if __name__ == '__main__':
    run_full_procedure_and_benchmark()

