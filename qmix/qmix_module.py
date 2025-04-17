import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        
        # 超网络结构
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        self.hyper_b2 = nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.qmix_hidden_dim, 1)
                                     )

    def forward(self, q_values, states):
        # 传入的q_values是三维的，shape为(episode_num, max_episode_len, n_agents)
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)  # (batch_size, 1, n_agents)
        states = states.reshape(-1, self.args.state_shape)  # (batch_size, state_shape)

        w1 = torch.abs(self.hyper_w1(states))  # 权重必须为正
        b1 = self.hyper_b1(states)

        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # 第一层

        w2 = torch.abs(self.hyper_w2(states))  # 权重必须为正
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2  # 输出层
        q_total = q_total.view(episode_num, -1, 1)  # 恢复原始形状
        return q_total

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        
        # 输入层
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # RNN层
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # 输出层
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
    
    def init_hidden(self):
        # 初始化隐藏状态
        return torch.zeros(1, self.args.rnn_hidden_dim)
    
    def forward(self, inputs, hidden_state):
        # 前向传播
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h