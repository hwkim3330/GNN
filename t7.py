import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
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
DEVICE = torch.device("cpu")
DATASET_PATH = "ilp_dataset_congestion.pt"
NUM_DATA_SAMPLES = 20000; ILP_TIME_LIMIT_SEC = 60
IMITATION_EPOCHS = 40; IMITATION_BATCH_SIZE = 128; IMITATION_LR = 2e-4
IMITATION_MODEL_PATH = "transformer_imitator_congestion.pth"
RL_EPISODES = 8000; RL_LR = 3e-5; GAMMA = 0.99; EPS_CLIP = 0.2
K_EPOCHS = 10; UPDATE_TIMESTEP = 2000
FINAL_MODEL_PATH = "gnn_transformer_ppo_congestion.pth"
BENCHMARK_SCENARIOS = 20; RESULT_PLOT_PATH = "benchmark_results_congestion_ppo.png"
LINK_BANDWIDTH_BPS = 1e9; PROPAGATION_DELAY_NS_PER_METER = 5; LINK_LENGTH_METER = 10; SWITCH_PROC_DELAY_NS = 1000

# ==============================================================================
# 1. 환경, 모델, 솔버 정의
# ==============================================================================
class TSN_Static_Env_v2:
    def __init__(self, graph, flow_definitions):
        self.graph = graph; self.flow_defs = flow_definitions
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
        for flow_id, path_pair in paths.items():
            props=self.flow_defs[flow_id]; path = path_pair.get('primary')
            if use_backup := (failed_link and path and (tuple(sorted(failed_link)) in [tuple(sorted(e)) for e in zip(path[:-1], path[1:])])):
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
        c_score_sum=0; num_scenarios = 0
        if contingency_scenarios:
            num_scenarios = len(contingency_scenarios)
            for scenario in contingency_scenarios:
                f_link=tuple(scenario.get('failed_link')) if scenario.get('failed_link') else None
                score, details=self._evaluate_single_scenario(paths, deadlines_ms, failed_link=f_link)
                if score <= 0: return 0.0, details
                c_score_sum+=score
            avg_c_score=c_score_sum/num_scenarios if num_scenarios > 0 else 1.0
        else: avg_c_score=p_score
        final_score=0.7*p_score+0.3*avg_c_score; return final_score, {"paths":paths, "primary_score":p_score, "avg_contingency_score":avg_c_score}

class ActorCriticTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout_rate=0.2):
        super(ActorCriticTransformer, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout_rate)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.actor_conv = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout_rate)
        self.actor_ln = nn.LayerNorm(hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, 1)
        self.critic_conv = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout_rate)
        self.critic_ln = nn.LayerNorm(hidden_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.input_layer(x))
        x_res = self.conv1(x, edge_index); x = self.ln1(x + x_res); x = F.relu(x)
        x_actor = self.actor_conv(x, edge_index); x_actor = self.actor_ln(x_actor + x); x_actor = F.relu(x_actor)
        action_logits = self.actor_head(x_actor).squeeze(-1)
        x_critic = self.critic_conv(x, edge_index); x_critic = self.critic_ln(x_critic + x); x_critic = F.relu(x_critic)
        state_value = self.critic_head(x_critic)
        return action_logits, state_value
    def evaluate(self, state_batch, action):
        action_logits, state_values_all_nodes = self.forward(state_batch)
        node_slices = torch.cat([state_batch.ptr, torch.tensor([state_batch.num_nodes], device=DEVICE)])
        action_logprobs = []; dist_entropy = []; state_values = []
        for i in range(state_batch.num_graphs):
            start, end = node_slices[i], node_slices[i+1]
            graph_logits = action_logits[start:end]
            dist = torch.distributions.Categorical(logits=graph_logits)
            action_in_graph = action[i]; action_in_graph_local = action_in_graph - start
            action_logprobs.append(dist.log_prob(action_in_graph_local))
            dist_entropy.append(dist.entropy().mean())
            state_values.append(state_values_all_nodes[start:end].mean())
        return torch.stack(action_logprobs), torch.stack(state_values), torch.stack(dist_entropy)

def get_state_tensor(graph, current_node, dest_node, partial_paths):
    num_nodes = graph.number_of_nodes(); features = np.zeros((num_nodes, 4))
    if current_node < num_nodes and dest_node < num_nodes:
        features[current_node, 0] = 1; features[dest_node, 1] = 1
    link_usage = {tuple(sorted(e)): 0 for e in graph.edges}
    for p in partial_paths.values():
        for path_type in ['primary', 'backup']:
            path = p.get(path_type)
            if path:
                for u,v in zip(path[:-1], path[1:]):
                    edge = tuple(sorted((u,v)))
                    if edge in link_usage: link_usage[edge] += 1
    for u,v in graph.edges():
        usage = link_usage.get(tuple(sorted((u,v))), 0)
        norm_usage = usage / (len(partial_paths) + 1)
        features[u, 2] += norm_usage; features[v, 2] += norm_usage
    for i in range(num_nodes): features[i, 3] = graph.degree[i] / (num_nodes -1) if num_nodes > 1 else 0
    return Data(x=torch.tensor(features, dtype=torch.float), edge_index=torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous())

class DynamicProfileGenerator:
    def generate(self):
        num_nodes=random.randint(8, 15); m=random.randint(2, 4); graph=nx.barabasi_albert_graph(n=num_nodes, m=m)
        while not nx.is_connected(graph): graph=nx.barabasi_albert_graph(n=num_nodes, m=m)
        flow_defs={}; num_flows=random.randint(5, 10)
        for i in range(num_flows):
            src, dst=random.sample(range(num_nodes), 2); flow_type=random.choice(["TT", "AVB"])
            flow_defs[f"flow_{i}"]={"src":src, "dst":dst, "type":flow_type, "size_bytes":random.randint(200, 4000), "period_ms":random.randint(5, 50)}
        deadlines_ms={"TT":random.uniform(5.0, 15.0), "AVB":random.uniform(20.0, 50.0)}; contingency_scenarios=[]
        if random.random() < 0.7:
            num_failures=random.randint(1, 2); possible_edges=list(graph.edges)
            if possible_edges:
                for _ in range(num_failures): contingency_scenarios.append({"failed_link":list(random.choice(possible_edges))})
        return {"graph":graph, "flow_definitions":flow_defs, "deadlines_ms":deadlines_ms, "contingency_scenarios":contingency_scenarios}

class Memory:
    def __init__(self): self.states = []; self.actions = []; self.logprobs = []; self.rewards = []; self.is_terminals = []
    def clear(self): self.__init__()

class PPOAgent:
    def __init__(self, state_dim, hidden_dim, lr, gamma, eps_clip, k_epochs):
        self.gamma = gamma; self.eps_clip = eps_clip; self.k_epochs = k_epochs
        self.policy = ActorCriticTransformer(state_dim, hidden_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCriticTransformer(state_dim, hidden_dim).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
    def select_action(self, state, current_path):
        with torch.no_grad():
            action_logits, _ = self.policy_old.forward(state.to(DEVICE))
            mask = torch.ones_like(action_logits) * -1e9
            valid_neighbors = [n for n in state.graph.neighbors(current_path[-1]) if n not in current_path]
            if not valid_neighbors: return None, None
            mask[valid_neighbors] = 0; masked_logits = action_logits + mask
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
        return action.item(), logprob
    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        old_states_batch = Batch.from_data_list(memory.states).to(DEVICE)
        old_actions = torch.tensor(memory.actions, dtype=torch.long).to(DEVICE)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(DEVICE)
        
        for _ in range(self.k_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_batch, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy.mean()
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
        self.agent = PPOAgent(4, 128, 0, 0, 0, 0)
        self.agent.policy_old.load_state_dict(torch.load(model_path, map_location=DEVICE))
    def solve(self, env, profile):
        paths, partial_paths = {}, {}
        graph = env.graph
        with torch.no_grad():
            for flow_id, props in sorted(profile['flow_definitions'].items()):
                current_path = [props['src']]
                while current_path[-1] != props['dst']:
                    state = get_state_tensor(graph, current_path[-1], props['dst'], partial_paths)
                    action, _ = self.agent.select_action(state, current_path)
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
                        action, _ = self.agent.select_action(state, current_path_b)
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
        env = TSN_Static_Env_v2(profile["graph"], profile['flow_definitions'])
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
                    dataset.append(data); pbar.update(1)
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
        print(f"오류: 모방학습 모델 '{IMITATION_MODEL_PATH}'이 없습니다. 1단계를 먼저 실행하세요."); return
    agent.policy.load_state_dict(torch.load(IMITATION_MODEL_PATH, map_location=DEVICE))
    agent.policy_old.load_state_dict(agent.policy.state_dict())
    
    memory = Memory(); time_step = 0; running_reward = 0
    pbar = tqdm(range(1, episodes + 1), desc="PPO 학습", ncols=100)
    for ep in pbar:
        profile = profile_generator.generate(); env = TSN_Static_Env_v2(profile["graph"], profile['flow_definitions'])
        paths, partial_paths = {}, {}; ep_reward = 0
        
        for flow_id, props in sorted(profile['flow_definitions'].items()):
            current_path = [props['src']]; terminal = False
            while not terminal:
                time_step += 1
                state = get_state_tensor(env.graph, current_path[-1], props['dst'], partial_paths)
                action, log_prob = agent.select_action(state, current_path)
                if action is None or len(current_path) > env.graph.number_of_nodes():
                    memory.rewards.append(-1); memory.is_terminals.append(True); terminal = True
                else:
                    memory.states.append(state); memory.actions.append(torch.tensor(action)); memory.logprobs.append(log_prob)
                    current_path.append(action)
                    if action == props['dst']:
                        memory.rewards.append(0); memory.is_terminals.append(True); terminal = True
                    else:
                        memory.rewards.append(-0.01); memory.is_terminals.append(False)
            paths[flow_id] = {'primary': current_path if terminal and current_path[-1] == props['dst'] else None, 'backup':None}
            partial_paths[flow_id] = paths[flow_id]
        
        final_score, _ = env.evaluate_robust_configuration(paths, profile['deadlines_ms'], profile['contingency_scenarios'])
        if memory.rewards: memory.rewards[-1] += final_score
        ep_reward = final_score

        if ep % UPDATE_TIMESTEP == 0 and ep > 0:
            agent.update(memory); memory.clear()
        
        running_reward += ep_reward
        if ep % 100 == 0: pbar.set_postfix(avg_reward=f"{running_reward/100:.4f}"); running_reward = 0

    torch.save(agent.policy.state_dict(), FINAL_MODEL_PATH)
    print(f"\nPPO 학습 완료. 최종 모델 저장: '{FINAL_MODEL_PATH}'")

def run_full_procedure_and_benchmark():
    if not os.path.exists(DATASET_PATH):
        generate_dataset_sequential(NUM_DATA_SAMPLES)
    
    print(f"기존 데이터셋 '{DATASET_PATH}'를 불러옵니다.")
    dataset = torch.load(DATASET_PATH)

    ppo_agent = PPOAgent(4, 128, RL_LR, GAMMA, EPS_CLIP, K_EPOCHS)
    
    if not os.path.exists(IMITATION_MODEL_PATH):
        train_imitator(ppo_agent.policy, dataset, epochs=IMITATION_EPOCHS)
    
    run_ppo_training(ppo_agent, DynamicProfileGenerator(), episodes=RL_EPISODES)
    
    print("\n===== 최종 랜덤 벤치마크 실행 =====")
    solvers=[GNN_PPOSolver(FINAL_MODEL_PATH), Greedy_Solver(), ILP_Solver(ILP_TIME_LIMIT_SEC)]
    results=[]; pbar_bench=tqdm(range(BENCHMARK_SCENARIOS), desc="랜덤 벤치마크 진행", ncols=100)
    profile_gen = DynamicProfileGenerator()
    for i in pbar_bench:
        profile=profile_gen.generate(); env=TSN_Static_Env_v2(profile["graph"], profile['flow_definitions']); row={"Scenario": i}
        for solver in solvers:
            start_time=time.time(); paths=solver.solve(env, profile); end_time=time.time(); computation_time=end_time-start_time
            score, _=env.evaluate_robust_configuration(paths, profile['deadlines_ms'], profile['contingency_scenarios'])
            row[f"{solver.name}_Score"] = score; row[f"{solver.name}_Time"] = computation_time
        results.append(row)
    
    df=pd.DataFrame(results)
    avg_scores={solver.name: df[f"{solver.name}_Score"].mean() for solver in solvers}
    avg_times={solver.name: df[f"{solver.name}_Time"].mean() for solver in solvers}
    print("\n\n===== 최종 벤치마크 결과 요약 (평균) ====="); summary_df=pd.DataFrame([avg_scores, avg_times], index=["Average Score", "Average Time (s)"]); print(summary_df.to_string())
    
    fig, axes=plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True); fig.suptitle(f'Final Benchmark: Transformer PPO vs. Others ({BENCHMARK_SCENARIOS} Scenarios)', fontsize=16)
    summary_df.T.sort_values("Average Score", ascending=False).plot(kind='bar', y='Average Score', ax=axes[0], rot=0, legend=False); axes[0].set_title('Average Performance Score (Higher is Better)'); axes[0].set_ylabel('Avg. Score'); axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    summary_df.T.sort_values("Average Time (s)").plot(kind='bar', y='Average Time (s)', ax=axes[1], rot=0, legend=False); axes[1].set_title('Average Computation Time (Lower is Better)'); axes[1].set_ylabel('Avg. Time (seconds)'); axes[1].set_yscale('log'); axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(RESULT_PLOT_PATH); print(f"\n결과 그래프가 '{RESULT_PLOT_PATH}'에 저장되었습니다.")

if __name__ == '__main__':
    run_full_procedure_and_benchmark()


