from collections import deque

import numpy as np
import random
import torch
from torch import nn as nn, optim as optim


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = torch.Tensor([_[0].numpy() for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = torch.Tensor([_[4].numpy() for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.LayerNorm(400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 300),
            nn.LayerNorm(300),
            nn.ReLU(inplace=True),
            nn.Linear(300, action_dim),
            nn.Sigmoid()
        )
        self.net[6].weight.data.mul_(0.1)
        self.net[6].bias.data.mul_(0.1)

    def forward(self, state):
        return self.net(state)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.embed_state = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.LayerNorm(400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 300),
        )
        self.embed_action = nn.Sequential(
            nn.Linear(action_dim, 300),
        )
        self.joint = nn.Sequential(
            nn.LayerNorm(300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 1)
        )
        self.joint[2].weight.data.mul_(0.1)
        self.joint[2].bias.data.mul_(0.1)

    def forward(self, state, action):
        state = self.embed_state(state)
        action = self.embed_action(action)
        value = self.joint(state + action)
        return value


class Actor(object):
    def __init__(self, state_dim, action_dim, learning_rate, tau):
        self.net = ActorNetwork(state_dim, action_dim)
        self.target_net = ActorNetwork(state_dim, action_dim)
        self.tau = tau
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.update_target_network(1)

    def train_step(self, policy_loss):
        self.net.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def predict(self, state):
        return self.net(state)

    def predict_target(self, state):
        return self.target_net(state)

    def update_target_network(self, custom_tau=-1):
        if custom_tau >= 0:
            tau = custom_tau
        else:
            tau = self.tau
        target_params = self.target_net.named_parameters()
        params = self.net.named_parameters()

        dict_target_params = dict(target_params)

        for name, param in params:
            if name in dict_target_params:
                dict_target_params[name].data.copy_(tau * param.data + (1 - tau) * dict_target_params[name].data)


class Critic(object):
    def __init__(self, state_dim, action_dim, learning_rate, tau):
        self.net = CriticNetwork(state_dim, action_dim)
        self.target_net = CriticNetwork(state_dim, action_dim)
        self.tau = tau
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_network(1)

    def train_step(self, state, action, target):
        self.net.zero_grad()
        pred = self.net(state, action)
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
        return pred

    def predict(self, state, action):
        return self.net(state, action)

    def predict_target(self, state, action):
        return self.target_net(state, action)

    def update_target_network(self, custom_tau=-1):
        if custom_tau >= 0:
            tau = custom_tau
        else:
            tau = self.tau
        target_params = self.target_net.named_parameters()
        params = self.net.named_parameters()

        dict_target_params = dict(target_params)

        for name, param in params:
            if name in dict_target_params:
                dict_target_params[name].data.copy_(tau * param.data + (1 - tau) * dict_target_params[name].data)