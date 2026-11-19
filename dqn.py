
"""DQN em PyTorch: rede Q convolucional, replay buffer e agente."""
import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Entrada (canais, N, N) -> um valor Q por ação."""

    def __init__(self, in_channels=6, size=11, n_actions=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * size * size, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done, mask2=None):
        if mask2 is None:
            mask2 = np.ones(6, dtype=np.float32)
        self.buf.append((s, a, r, s2, done, mask2))

    def sample(self, batch_size):
        s, a, r, s2, d, m = zip(*random.sample(self.buf, batch_size))
        return (
            torch.as_tensor(np.array(s), dtype=torch.float32),
            torch.as_tensor(a, dtype=torch.int64),
            torch.as_tensor(r, dtype=torch.float32),
            torch.as_tensor(np.array(s2), dtype=torch.float32),
            torch.as_tensor(d, dtype=torch.float32),
            torch.as_tensor(np.array(m), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, in_channels=6, size=11, n_actions=6, lr=1e-4, gamma=0.99,
                 buffer_size=50_000, batch_size=64, target_update=1000,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=50_000, double=False, train_every=1, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.double = double  # True = Double DQN
        self.train_every = train_every  # 1 passo de treino a cada N passos do jogo
        self.batch_size = batch_size
        self.target_update = target_update
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps

        self.policy = QNetwork(in_channels, size, n_actions).to(self.device)
        self.target = copy.deepcopy(self.policy).eval()
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    def epsilon(self):
        frac = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs, mask=None, greedy=False):
        """Escolhe a ação. 'mask' (6 flags) limita a escolha às ações permitidas."""
        validas = list(range(self.n_actions))
        if mask is not None:
            validas = [i for i in validas if mask[i] > 0] or validas
        if not greedy and random.random() < self.epsilon():
            return random.choice(validas)
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy(x)[0]
            if mask is not None:
                bloqueada = torch.as_tensor(np.asarray(mask) <= 0, device=self.device)
                if not bool(bloqueada.all()):
                    q = q.masked_fill(bloqueada, -1e9)
            return int(q.argmax().item())

    def update(self, s, a, r, s2, terminated, mask2=None):
        """Guarda a transição e faz um passo de treino. Retorna a perda (ou None)."""
        self.buffer.push(s, a, r, s2, float(terminated), mask2)
        self.steps += 1
        loss = None
        if len(self.buffer) >= self.batch_size and self.steps % self.train_every == 0:
            loss = self._learn()
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return loss

    def _learn(self):
        s, a, r, s2, d, m2 = [t.to(self.device) for t in self.buffer.sample(self.batch_size)]
        q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_alvo = self.target(s2)
            if self.double:
                # Double DQN: a rede principal ESCOLHE a ação, a rede alvo AVALIA
                q_pol = self.policy(s2).masked_fill(m2 <= 0, -1e9)  # só ações permitidas
                a2 = q_pol.argmax(dim=1, keepdim=True)
                q2 = q_alvo.gather(1, a2).squeeze(1)
            else:
                # DQN clássico: a rede alvo escolhe e avalia (pega o máximo)
                q2 = q_alvo.masked_fill(m2 <= 0, -1e9).max(dim=1).values
            y = r + self.gamma * q2 * (1.0 - d)
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.opt.step()
        return loss.item()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.policy.state_dict())
