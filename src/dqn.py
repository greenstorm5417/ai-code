from typing import Deque, Tuple
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.memory: Deque[Tuple] = deque(maxlen=self.capacity)

    def push(self, *transition):
        self.memory.append(tuple(transition))

    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        return list(zip(*batch))

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_capacity: int = 200000,
        batch_size: int = 64,
        target_sync: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 20000,
        device: str | None = None,
    ):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = max(1, eps_decay_steps)
        self.step_count = 0

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = 1
        for d in state_shape:
            state_dim *= d
        self.policy_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.loss_fn = nn.SmoothL1Loss()

    def epsilon(self) -> float:
        frac = min(1.0, self.step_count / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state) -> int:
        self.step_count += 1
        if random.random() < self.epsilon():
            return random.randrange(self.num_actions)
        with torch.no_grad():
            s_np = np.asarray(state, dtype=np.float32)
            s = torch.from_numpy(s_np).to(self.device).view(1, -1)
            q = self.policy_net(s)
            a = int(torch.argmax(q, dim=1).item())
            return a

    def push(self, *transition):
        self.buffer.push(*transition)

    def can_optimize(self):
        return len(self.buffer) >= self.batch_size

    def optimize(self):
        if not self.can_optimize():
            return None
        s, a, r, s2, done = self.buffer.sample(self.batch_size)
        s_np = np.asarray(s, dtype=np.float32)
        s = torch.from_numpy(s_np).to(self.device).view(self.batch_size, -1)
        a = torch.tensor(a, dtype=torch.long, device=self.device).view(self.batch_size, 1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).view(self.batch_size, 1)
        s2_np = np.asarray(s2, dtype=np.float32)
        s2 = torch.from_numpy(s2_np).to(self.device).view(self.batch_size, -1)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).view(self.batch_size, 1)

        q = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target_net(s2).max(1, keepdim=True)[0]
            target = r + (1.0 - done) * self.gamma * q_next
        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        if self.step_count % self.target_sync == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())


class MultiHeadQNetwork(nn.Module):
    def __init__(self, input_dim: int, n_stocks: int, size_bins: int = 5):
        super().__init__()
        self.n_stocks = n_stocks
        self.size_bins = int(max(2, size_bins))
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.trade_head = nn.Linear(256, n_stocks * 3)
        self.select_head = nn.Linear(256, n_stocks * 2)
        self.size_head = nn.Linear(256, n_stocks * self.size_bins)

    def forward(self, x):
        h = self.shared(x)
        trade = self.trade_head(h)
        select = self.select_head(h)
        size = self.size_head(h)
        return trade, select, size


class MultiHeadDQNAgent:
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        n_stocks: int,
        size_bins: int = 5,
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_capacity: int = 200000,
        batch_size: int = 64,
        target_sync: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 20000,
        device: str | None = None,
    ):
        self.state_shape = state_shape
        self.n_stocks = int(n_stocks)
        self.size_bins = int(max(2, size_bins))
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = max(1, eps_decay_steps)
        self.step_count = 0
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = 1
        for d in state_shape:
            state_dim *= d
        self.policy_net = MultiHeadQNetwork(state_dim, self.n_stocks, self.size_bins).to(self.device)
        self.target_net = MultiHeadQNetwork(state_dim, self.n_stocks, self.size_bins).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.loss_fn = nn.SmoothL1Loss()

    def epsilon(self) -> float:
        frac = min(1.0, self.step_count / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state):
        self.step_count += 1
        if random.random() < self.epsilon():
            trade = np.random.randint(0, 3, size=(self.n_stocks,), dtype=np.int32)
            select = np.random.randint(0, 2, size=(self.n_stocks,), dtype=np.int32)
            size = np.random.randint(0, self.size_bins, size=(self.n_stocks,), dtype=np.int32)
            return {"trade": trade, "select": select, "size": size}
        with torch.no_grad():
            s_np = np.asarray(state, dtype=np.float32)
            s = torch.from_numpy(s_np).to(self.device).view(1, -1)
            trade_q, sel_q, size_q = self.policy_net(s)
            trade_q = trade_q.view(1, self.n_stocks, 3)
            sel_q = sel_q.view(1, self.n_stocks, 2)
            size_q = size_q.view(1, self.n_stocks, self.size_bins)
            trade = torch.argmax(trade_q, dim=2).squeeze(0).cpu().numpy().astype(np.int32)
            select = torch.argmax(sel_q, dim=2).squeeze(0).cpu().numpy().astype(np.int32)
            size = torch.argmax(size_q, dim=2).squeeze(0).cpu().numpy().astype(np.int32)
            return {"trade": trade, "select": select, "size": size}

    def push(self, state, trade_actions, select_actions, size_actions, reward, next_state, done):
        self.buffer.push(
            state,
            np.asarray(trade_actions, dtype=np.int64),
            np.asarray(select_actions, dtype=np.int64),
            np.asarray(size_actions, dtype=np.int64),
            float(reward),
            next_state,
            float(done),
        )

    def can_optimize(self):
        return len(self.buffer) >= self.batch_size

    def optimize(self):
        if not self.can_optimize():
            return None
        s, a_trade, a_sel, a_size, r, s2, done = self.buffer.sample(self.batch_size)
        s_np = np.asarray(s, dtype=np.float32)
        s = torch.from_numpy(s_np).to(self.device).view(self.batch_size, -1)
        s2_np = np.asarray(s2, dtype=np.float32)
        s2 = torch.from_numpy(s2_np).to(self.device).view(self.batch_size, -1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).view(self.batch_size, 1)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).view(self.batch_size, 1)
        a_trade = torch.from_numpy(np.asarray(a_trade, dtype=np.int64)).to(self.device).view(self.batch_size, self.n_stocks, 1)
        a_sel = torch.from_numpy(np.asarray(a_sel, dtype=np.int64)).to(self.device).view(self.batch_size, self.n_stocks, 1)
        a_size = torch.from_numpy(np.asarray(a_size, dtype=np.int64)).to(self.device).view(self.batch_size, self.n_stocks, 1)

        trade_q, sel_q, size_q = self.policy_net(s)
        trade_q = trade_q.view(self.batch_size, self.n_stocks, 3)
        sel_q = sel_q.view(self.batch_size, self.n_stocks, 2)
        size_q = size_q.view(self.batch_size, self.n_stocks, self.size_bins)
        picked_trade_q = trade_q.gather(2, a_trade).squeeze(-1).sum(dim=1, keepdim=True)
        picked_sel_q = sel_q.gather(2, a_sel).squeeze(-1).sum(dim=1, keepdim=True)
        picked_size_q = size_q.gather(2, a_size).squeeze(-1).sum(dim=1, keepdim=True)
        q_joint = picked_trade_q + picked_sel_q + picked_size_q

        with torch.no_grad():
            trade_q2, sel_q2, size_q2 = self.target_net(s2)
            trade_q2 = trade_q2.view(self.batch_size, self.n_stocks, 3)
            sel_q2 = sel_q2.view(self.batch_size, self.n_stocks, 2)
            size_q2 = size_q2.view(self.batch_size, self.n_stocks, self.size_bins)
            trade_max = trade_q2.max(dim=2)[0].sum(dim=1, keepdim=True)
            sel_max = sel_q2.max(dim=2)[0].sum(dim=1, keepdim=True)
            size_max = size_q2.max(dim=2)[0].sum(dim=1, keepdim=True)
            target = r + (1.0 - done) * self.gamma * (trade_max + sel_max + size_max)

        loss = self.loss_fn(q_joint, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        if self.step_count % self.target_sync == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())
