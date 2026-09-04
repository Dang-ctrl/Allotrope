"""DQN for the discrete commitment layer: which gensets are turning.

With `n` gensets there are `2**n` commitment patterns -- 8 at Maitri or Bharati,
both three-set plants -- small enough to enumerate rather than requiring a
factored or autoregressive action representation. Each pattern is one output of
a single Q-network, so this is plain DQN: no distributional head, no dueling
architecture, no rainbow of extensions the problem does not need. The two-layer
split with SDDPG is where this project's novelty is meant to live, not in the
sophistication of either half.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from torch import nn, optim

from allotrope.agents.networks import QNetwork, soft_update
from allotrope.agents.replay import Batch, ReplayBuffer


def enumerate_commitments(n_gensets: int) -> list[tuple[bool, ...]]:
    """Every commitment pattern, in a fixed, reproducible order."""
    return list(product([False, True], repeat=n_gensets))


@dataclass(frozen=True)
class DQNConfig:
    hidden: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.997
    """Per-episode multiplicative decay of the exploration rate."""
    buffer_capacity: int = 200_000
    batch_size: int = 256
    warmup_steps: int = 1_000


class DQNAgent:
    """A Q-network over the enumerated commitment patterns."""

    def __init__(self, obs_dim: int, n_gensets: int, config: DQNConfig | None = None) -> None:
        self.obs_dim = obs_dim
        self.n_gensets = n_gensets
        self.commitments = enumerate_commitments(n_gensets)
        self.n_actions = len(self.commitments)
        self.cfg = config or DQNConfig()

        self.q = QNetwork(obs_dim, self.n_actions, self.cfg.hidden)
        self.q_target = QNetwork(obs_dim, self.n_actions, self.cfg.hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=self.cfg.lr)

        self.buffer = ReplayBuffer(self.cfg.buffer_capacity, obs_dim, 1)
        self.epsilon = self.cfg.epsilon_start
        self.total_steps = 0

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, explore: bool = True, rng: np.random.Generator | None = None
    ) -> tuple[int, tuple[bool, ...]]:
        """The chosen action index and the commitment pattern it names."""
        rng = rng or np.random.default_rng()
        if explore and (self.total_steps < self.cfg.warmup_steps or rng.random() < self.epsilon):
            index = int(rng.integers(0, self.n_actions))
        else:
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            index = int(self.q(x).argmax(dim=-1).item())
        return index, self.commitments[index]

    def observe(
        self, obs: np.ndarray, action_index: int, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        self.buffer.add(obs, np.array([action_index], dtype=np.float32), reward, next_obs, done)
        self.total_steps += 1

    def end_episode(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def update(self, rng: np.random.Generator) -> dict[str, float] | None:
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None
        batch = self.buffer.sample(self.cfg.batch_size, rng)
        return self._update(batch)

    def _update(self, batch: Batch) -> dict[str, float]:
        obs = torch.as_tensor(batch.obs)
        action = torch.as_tensor(batch.action, dtype=torch.long).squeeze(-1)
        reward = torch.as_tensor(batch.reward)
        next_obs = torch.as_tensor(batch.next_obs)
        done = torch.as_tensor(batch.done)

        with torch.no_grad():
            # Double DQN: the online network picks the next action, the target
            # network values it. Plain DQN's shared max is a well known source
            # of overestimation, and it costs nothing extra to avoid here.
            next_action = self.q(next_obs).argmax(dim=-1, keepdim=True)
            next_q = self.q_target(next_obs).gather(-1, next_action).squeeze(-1)
            y = reward + self.cfg.gamma * (1.0 - done) * next_q

        q = self.q(obs).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        loss = nn.functional.smooth_l1_loss(q, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        soft_update(self.q_target, self.q, self.cfg.tau)

        return {"q_loss": float(loss.detach())}

    def state_dict(self) -> dict:
        return {"q": self.q.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.q.load_state_dict(state["q"])
        self.q_target.load_state_dict(state["q"])


__all__ = ["DQNAgent", "DQNConfig", "enumerate_commitments"]
