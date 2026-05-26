"""
ddpg_agent.py — Deep Deterministic Policy Gradient Agent
Python/PyTorch translation of ddpg_agent.js (TensorFlow.js)

Architecture:
  Actor:  state(10) → Dense(256) → Dense(256) → Dense(128) → tanh(2)
  Critic: [state(10) ∥ action(2)] → Dense(256) → Dense(256) → Dense(128) → Q(1)
  + Target networks (Polyak soft-update, τ=0.005)
  + Experience Replay Buffer (5000 transitions)
  + Ornstein-Uhlenbeck Noise (σ decay 0.3→0.05)
"""

import random
import math
import io
import base64
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from environment import DDPG_CONSTANTS

# ── Hyperparameters ──────────────────────────────────────────────────────────
HP = {
    'STATE_DIM':      10,
    'ACTION_DIM':     2,
    'BUFFER_SIZE':    5000,
    'BATCH_SIZE':     64,
    'ACTOR_LR':       1e-4,
    'CRITIC_LR':      1e-3,
    'GAMMA':          0.95,
    'TAU':            0.005,
    'OU_THETA':       0.15,
    'OU_SIGMA_INIT':  0.3,
    'OU_SIGMA_MIN':   0.05,
    'OU_SIGMA_DECAY': 0.9995,
    'MIN_BUFFER':     200,
}


# ─────────────────────────────────────────────────────────────────────────────
# Neural Networks
# ─────────────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),                     nn.ReLU(),
            nn.Linear(256, 128),                     nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, max_size: int = HP['BUFFER_SIZE']):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)

    def push(self, state: list, action: list, reward: float, next_state: list):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    @property
    def size(self) -> int:
        return len(self.buffer)

    @property
    def ready(self) -> bool:
        return len(self.buffer) >= HP['MIN_BUFFER']


# ─────────────────────────────────────────────────────────────────────────────
# Ornstein-Uhlenbeck Noise
# ─────────────────────────────────────────────────────────────────────────────

class OUNoise:
    def __init__(self):
        self.theta     = HP['OU_THETA']
        self.sigma     = HP['OU_SIGMA_INIT']
        self.sigma_min = HP['OU_SIGMA_MIN']
        self.decay     = HP['OU_SIGMA_DECAY']
        self.mu        = [0.0, 0.0]
        self.x         = [0.0, 0.0]
        self.dt        = 0.1

    def sample(self) -> list[float]:
        new_x = []
        for i in range(2):
            dx = (self.theta * (self.mu[i] - self.x[i]) * self.dt
                  + self.sigma * math.sqrt(self.dt) * (random.random() * 2 - 1))
            new_x.append(self.x[i] + dx)
        self.x = new_x
        return list(self.x)

    def decay_(self):
        self.sigma = max(self.sigma_min, self.sigma * self.decay)

    def reset(self):
        self.x     = [0.0, 0.0]
        self.sigma = HP['OU_SIGMA_INIT']

    @property
    def sigma_value(self) -> float:
        return self.sigma


# ─────────────────────────────────────────────────────────────────────────────
# DDPG Agent
# ─────────────────────────────────────────────────────────────────────────────

class DDPGAgent:
    def __init__(self):
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim  = HP['STATE_DIM']
        self.action_dim = HP['ACTION_DIM']

        # Networks
        self.actor         = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_target  = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic        = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)

        # Hard copy to targets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=HP['ACTOR_LR'])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=HP['CRITIC_LR'])

        self.buffer = ReplayBuffer()
        self.noise  = OUNoise()

        self.step_count  = 0
        self.train_count = 0

        self.actor_loss_history:  list[float] = []
        self.critic_loss_history: list[float] = []

        n_actor  = sum(p.numel() for p in self.actor.parameters())
        n_critic = sum(p.numel() for p in self.critic.parameters())
        print(f'[DDPG] Device: {self.device}')
        print(f'  Actor:  {n_actor} params')
        print(f'  Critic: {n_critic} params')

    # ── Action selection ──────────────────────────────────────────────────────
    def select_action(self, state_arr: list[float]) -> dict:
        state = torch.FloatTensor(state_arr).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            raw = self.actor(state).cpu().numpy()[0].tolist()
        self.actor.train()

        noise       = self.noise.sample()
        noisy       = [max(-1.0, min(1.0, raw[i] + noise[i])) for i in range(2)]
        alpha, rho  = self.scale_action(noisy)

        return {'raw_action': noisy, 'alpha': alpha, 'rho': rho}

    def scale_action(self, raw: list[float]) -> tuple[float, float]:
        """Map tanh output [-1,1]² → {alpha, rho} real values."""
        C  = DDPG_CONSTANTS
        a  = C['ALPHA_MIN'] + (raw[0] + 1) / 2 * (C['ALPHA_MAX'] - C['ALPHA_MIN'])
        r  = C['RHO_MIN']   + (raw[1] + 1) / 2 * (C['RHO_MAX']   - C['RHO_MIN'])
        return (
            max(C['ALPHA_MIN'], min(C['ALPHA_MAX'], a)),
            max(C['RHO_MIN'],   min(C['RHO_MAX'],   r)),
        )

    # ── Memory ────────────────────────────────────────────────────────────────
    def remember(self, state: list, raw_action: list,
                 reward: float, next_state: list):
        self.buffer.push(list(state), raw_action, reward, list(next_state))

    # ── Training ──────────────────────────────────────────────────────────────
    def train_step(self) -> dict | None:
        if not self.buffer.ready:
            return None

        batch = self.buffer.sample(HP['BATCH_SIZE'])
        states, actions, rewards, next_states = zip(*batch)

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.FloatTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)

        # ── Critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            next_a  = self.actor_target(ns)
            next_q  = self.critic_target(ns, next_a)
            target  = r + HP['GAMMA'] * next_q

        curr_q       = self.critic(s, a)
        critic_loss  = nn.MSELoss()(curr_q, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor update ──────────────────────────────────────────────────────
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── Soft update targets ───────────────────────────────────────────────
        self._soft_update(self.actor,  self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        self.noise.decay_()
        self.train_count += 1
        self.step_count  += 1

        al = actor_loss.item()
        cl = critic_loss.item()

        self.actor_loss_history.append(al)
        self.critic_loss_history.append(cl)
        if len(self.actor_loss_history) > 500:
            self.actor_loss_history.pop(0)
            self.critic_loss_history.pop(0)

        return {'actor_loss': al, 'critic_loss': cl}

    # ── Soft update ───────────────────────────────────────────────────────────
    def _soft_update(self, src: nn.Module, dst: nn.Module):
        tau = HP['TAU']
        for sp, dp in zip(src.parameters(), dst.parameters()):
            dp.data.copy_(tau * sp.data + (1.0 - tau) * dp.data)

    # ── Noise control ─────────────────────────────────────────────────────────
    def reset_noise(self):
        self.noise.reset()

    # ── Model persistence ─────────────────────────────────────────────────────
    def save_model(self) -> str:
        """Serialize model weights → base64 string (for export)."""
        buf = io.BytesIO()
        torch.save({
            'actor':         self.actor.state_dict(),
            'actor_target':  self.actor_target.state_dict(),
            'critic':        self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, buf)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def load_model(self, data_b64: str):
        """Load model weights from base64 string."""
        buf = io.BytesIO(base64.b64decode(data_b64))
        ckpt = torch.load(buf, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])

    # ── Getters ───────────────────────────────────────────────────────────────
    @property
    def buffer_size(self) -> int:
        return self.buffer.size

    @property
    def buffer_ready(self) -> bool:
        return self.buffer.ready

    @property
    def noise_level(self) -> float:
        return self.noise.sigma_value

    @property
    def last_actor_loss(self) -> float | None:
        return self.actor_loss_history[-1] if self.actor_loss_history else None

    @property
    def last_critic_loss(self) -> float | None:
        return self.critic_loss_history[-1] if self.critic_loss_history else None
