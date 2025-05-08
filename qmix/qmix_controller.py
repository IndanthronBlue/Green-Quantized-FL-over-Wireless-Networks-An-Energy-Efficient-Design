import torch
import os
import numpy as np
from qmix.qmix_module import QMixNet, RNNAgent
import random
from collections import deque

class QMIXController:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.n_agents = args.num_clients
        self.n_actions = len(args.quant_budget_options)
        self.state_shape = 4 * args.num_clients  # 全局状态大小
        self.obs_shape = 4  # 局部观察大小
        
        # 配置QMIX网络参数
        qmix_args = type('QMixArgs', (), {
            'n_agents': self.n_agents,
            'n_actions': self.n_actions,
            'state_shape': self.state_shape,
            'obs_shape': self.obs_shape,
            'rnn_hidden_dim': 64,
            'qmix_hidden_dim': 32,
            'hyper_hidden_dim': 64,
            'two_hyper_layers': True,
            'gamma': 0.99,
            'lr': 0.001,
            'grad_norm_clip': 10,
        })
        
        # 初始化智能体网络
        self.agents = [RNNAgent(self.obs_shape, qmix_args) for _ in range(self.n_agents)]
        self.target_agents = [RNNAgent(self.obs_shape, qmix_args) for _ in range(self.n_agents)]
        
        # 初始化混合网络
        self.qmix_net = QMixNet(qmix_args)
        self.target_qmix_net = QMixNet(qmix_args)
        
        # 移动到指定设备
        if device.type == 'cuda':
            for agent in self.agents:
                agent.to(device)
            for target_agent in self.target_agents:
                target_agent.to(device)
            self.qmix_net.to(device)
            self.target_qmix_net.to(device)
        
        # 复制参数到目标网络
        for i in range(self.n_agents):
            self.target_agents[i].load_state_dict(self.agents[i].state_dict())
        self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())
        
        # 创建优化器
        self.params = list(self.qmix_net.parameters())
        for agent in self.agents:
            self.params += list(agent.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=qmix_args.lr)
        
        # 创建经验回放缓冲区
        self.buffer = deque(maxlen=10000)
        
        # 探索参数
        self.epsilon = args.qmix_epsilon
        self.epsilon_decay = args.qmix_epsilon_decay
        self.epsilon_min = args.qmix_epsilon_min
        
        # 训练参数
        self.gamma = qmix_args.gamma
        self.grad_norm_clip = qmix_args.grad_norm_clip
        self.target_update_cycle = 10
        self.train_step = 0
        
        # 维护隐藏状态
        self.hidden_states = [agent.init_hidden() for agent in self.agents]
        
        print("QMIX controller initialized")
    
    def select_actions(self, obs_list, explore=True):
        """为所有智能体选择动作"""
        actions = []
        
        with torch.no_grad():
            for i, obs in enumerate(obs_list):
                # 将观察转换为tensor
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # ε-greedy策略
                if explore and random.random() < self.epsilon:
                    action = random.randint(0, self.n_actions - 1)
                else:
                    # 获取Q值
                    q, self.hidden_states[i] = self.agents[i](obs_tensor, self.hidden_states[i])
                    action = q.argmax(dim=1).item()
                
                actions.append(action)
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return actions
    
    def store_transition(self, obs, state, actions, reward, next_obs, next_state, done):
        """存储经验"""
        transition = {
            'obs': obs,
            'state': state,
            'actions': actions,
            'reward': reward,
            'next_obs': next_obs,
            'next_state': next_state,
            'done': done
        }
        self.buffer.append(transition)
    
    def train(self, batch_size=32):
        """训练QMIX网络"""
        if len(self.buffer) < batch_size:
            return None
        
        # 采样经验
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]
        
        # 准备批次数据
        obs = torch.tensor(np.array([s['obs'] for s in samples]), dtype=torch.float32).to(self.device)
        state = torch.tensor(np.array([s['state'] for s in samples]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array([s['actions'] for s in samples]), dtype=torch.long).to(self.device)
        reward = torch.tensor(np.array([s['reward'] for s in samples]), dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_obs = torch.tensor(np.array([s['next_obs'] for s in samples]), dtype=torch.float32).to(self.device)
        next_state = torch.tensor(np.array([s['next_state'] for s in samples]), dtype=torch.float32).to(self.device)
        done = torch.tensor(np.array([s['done'] for s in samples]), dtype=torch.float32).unsqueeze(-1).to(self.device)
        
        # 计算当前Q值
        q_values = []
        for i in range(self.n_agents):
            agent_obs = obs[:, i]
            hidden = torch.zeros(batch_size, self.args.rnn_hidden_dim).to(self.device)
            q, _ = self.agents[i](agent_obs, hidden)
            q_values.append(q)
        
        # 堆叠所有智能体的Q值
        q_values = torch.stack(q_values, dim=1)
        
        # 选择动作的Q值
        chosen_action_qvals = torch.gather(q_values, dim=2, 
                                         index=actions.unsqueeze(-1)).squeeze(-1)
        
        # 计算目标Q值
        target_q_values = []
        for i in range(self.n_agents):
            agent_next_obs = next_obs[:, i]
            hidden = torch.zeros(batch_size, self.args.rnn_hidden_dim).to(self.device)
            next_q, _ = self.target_agents[i](agent_next_obs, hidden)
            target_q_values.append(next_q)
        
        # 堆叠所有智能体的目标Q值
        target_q_values = torch.stack(target_q_values, dim=1)
        
        # 计算最大Q值
        max_action_qvals, _ = target_q_values.max(dim=2)
        
        # 混合Q值
        chosen_action_qvals = self.qmix_net(chosen_action_qvals, state)
        max_action_qvals = self.target_qmix_net(max_action_qvals, next_state)
        
        # 计算目标
        targets = reward + self.gamma * (1 - done) * max_action_qvals
        
        # 计算TD误差和损失
        td_error = (chosen_action_qvals - targets.detach())
        loss = (td_error ** 2).mean()
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.grad_norm_clip)
        self.optimizer.step()
        
        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_cycle == 0:
            for i in range(self.n_agents):
                self.target_agents[i].load_state_dict(self.agents[i].state_dict())
            self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())
        
        return loss.item()
    
    def reset_hidden_states(self):
        """重置所有智能体的隐藏状态"""
        self.hidden_states = [agent.init_hidden() for agent in self.agents]
    
    def save_model(self, path):
        """保存模型"""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        
        model_dict = {
            'qmix_net': self.qmix_net.state_dict(),
            'agents': [agent.state_dict() for agent in self.agents],
            'train_step': self.train_step,
            'epsilon': self.epsilon
        }
        torch.save(model_dict, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """加载模型"""
        if not os.path.exists(path):
            print(f"Model file {path} does not exist")
            return False
        
        model_dict = torch.load(path, map_location=self.device)
        self.qmix_net.load_state_dict(model_dict['qmix_net'])
        self.target_qmix_net.load_state_dict(model_dict['qmix_net'])
        
        for i, agent_state_dict in enumerate(model_dict['agents']):
            self.agents[i].load_state_dict(agent_state_dict)
            self.target_agents[i].load_state_dict(agent_state_dict)
        
        self.train_step = model_dict.get('train_step', 0)
        self.epsilon = model_dict.get('epsilon', self.epsilon)
        print(f"Model loaded from {path}")
        return True