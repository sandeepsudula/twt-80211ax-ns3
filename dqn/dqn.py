"""
A small Deep Q-Network: a neural network that predicts the reward of every action.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """state (4 numbers) -> one predicted reward (Q-value) per action"""

    def __init__(self, state_size, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_size, n_actions, lr=1e-3, gamma=0.0, buffer_size=20000,
                 batch_size=64, eps_start=1.0, eps_end=0.05, eps_decay_steps=1500,
                 target_update=200, seed=1):
        random.seed(seed)
        torch.manual_seed(seed)
        self.n_actions = n_actions
        self.gamma = gamma                    # 0: each decision stands alone (contextual bandit)
        self.q = QNetwork(state_size, n_actions)
        self.q_target = QNetwork(state_size, n_actions)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)  # replay memory
        self.batch_size = batch_size
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.target_update = target_update
        self.steps = 0

    def epsilon(self):
        frac = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state, greedy=False):
        """epsilon-greedy: sometimes explore a random action, otherwise take the best one"""
        if not greedy and random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            return int(self.q(torch.as_tensor(state).unsqueeze(0)).argmax(dim=1).item())

    def remember(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))
        self.steps += 1

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        s = torch.as_tensor(s, dtype=torch.float32)
        a = torch.as_tensor(a, dtype=torch.int64)
        r = torch.as_tensor(r, dtype=torch.float32)
        s2 = torch.as_tensor(s2, dtype=torch.float32)
        d = torch.as_tensor(d, dtype=torch.float32)

        q_taken = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(s2).max(dim=1).values
            target = r + self.gamma * (1.0 - d) * q_next      # Bellman target
        loss = nn.functional.smooth_l1_loss(q_taken, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        if self.steps % self.target_update < self.batch_size:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path))
        self.q_target.load_state_dict(self.q.state_dict())
