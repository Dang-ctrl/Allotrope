"""SDDPG: DDPG for the continuous dispatch layer, made safe by where it trains.

The deck's "SDDPG" is Safe DDPG -- standard deep deterministic policy gradient,
made safe not by a modified loss but by training entirely inside the guarded
environment. `PolarMicrogridEnv` applies the safety projection to every action
before it reaches the plant (`allotrope/safety/projection.py`), so the actor
never sees the consequence of an unsafe action because there is no such
consequence to see: exploration is safe from the first random action, in
training exactly as in deployment.

This module owns only the continuous half of the action: per-set loading
fraction, per-pack charge/discharge fraction, and the melting rate. Commitment
is DQN's job (`dqn.py`); `hybrid.py` is where the two meet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim

from allotrope.agents.networks import Actor, Critic, soft_update
from allotrope.agents.replay import Batch, ReplayBuffer


@dataclass(frozen=True)
class SDDPGConfig:
    hidden: int = 128
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    """Polyak coefficient. Small, because the target network's whole job is to
    move slower than the network it bootstraps from."""
    exploration_sigma: float = 0.2
    """Std of the Gaussian action noise added during training rollouts."""
    exploration_sigma_min: float = 0.02
    exploration_decay: float = 0.9995
    """Per-episode multiplicative decay of the exploration noise."""
    buffer_capacity: int = 200_000
    batch_size: int = 256
    warmup_steps: int = 1_000
    """Steps of pure exploration before any gradient update, so the buffer holds
    enough variety that the first updates are not fit to a handful of states."""


class SDDPGAgent:
    """A DDPG actor-critic pair over the continuous dispatch action."""

    def __init__(self, obs_dim: int, action_dim: int, config: SDDPGConfig | None = None) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = config or SDDPGConfig()

        self.actor = Actor(obs_dim, action_dim, self.cfg.hidden)
        self.actor_target = Actor(obs_dim, action_dim, self.cfg.hidden)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, action_dim, self.cfg.hidden)
        self.critic_target = Critic(obs_dim, action_dim, self.cfg.hidden)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

        self.buffer = ReplayBuffer(self.cfg.buffer_capacity, obs_dim, action_dim)
        self.sigma = self.cfg.exploration_sigma
        self.total_steps = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = True, rng: np.random.Generator | None = None) -> np.ndarray:
        """A dispatch action in [-1, 1]^d, optionally with exploration noise."""
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = self.actor(x).squeeze(0).numpy()
        if explore:
            rng = rng or np.random.default_rng()
            if self.total_steps < self.cfg.warmup_steps:
                action = rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)
            else:
                action = action + rng.normal(0.0, self.sigma, size=self.action_dim).astype(np.float32)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def observe(
        self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        self.buffer.add(obs, action, reward, next_obs, done)
        self.total_steps += 1

    def end_episode(self) -> None:
        self.sigma = max(self.cfg.exploration_sigma_min, self.sigma * self.cfg.exploration_decay)

    def update(self, rng: np.random.Generator) -> dict[str, float] | None:
        """One gradient step on a sampled minibatch, or None if not ready yet."""
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None
        batch = self.buffer.sample(self.cfg.batch_size, rng)
        return self._update(batch)

    def _update(self, batch: Batch) -> dict[str, float]:
        obs = torch.as_tensor(batch.obs)
        action = torch.as_tensor(batch.action)
        reward = torch.as_tensor(batch.reward)
        next_obs = torch.as_tensor(batch.next_obs)
        done = torch.as_tensor(batch.done)

        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_action)
            y = reward + self.cfg.gamma * (1.0 - done) * target_q

        q = self.critic(obs, action)
        critic_loss = nn.functional.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        soft_update(self.actor_target, self.actor, self.cfg.tau)
        soft_update(self.critic_target, self.critic, self.cfg.tau)

        return {"critic_loss": float(critic_loss.detach()), "actor_loss": float(actor_loss.detach())}

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_target.load_state_dict(state["actor"])
        self.critic_target.load_state_dict(state["critic"])


__all__ = ["SDDPGAgent", "SDDPGConfig"]
