"""Branching DQN: which gensets to commit.

The commitment decision is binary per genset and there is no way to flatten
`n` independent binaries into a single discrete action space without either
enumerating `2**n` joint actions (intractable as the fleet grows) or losing
the fact that these really are separate decisions. Action branching solves
this the way `allotrope.envs.polar_microgrid` already frames the problem:
one Q-head per genset, a shared trunk between them, and a joint target that
lets the branches coordinate without an exponential action space.

Double DQN target (evaluate the greedy action under the online network,
value it under the target network) is used throughout, because an ordinary
max-based target is a known source of overestimation bias that compounds
across branches.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from allotrope.agents.networks import DuelingBranchingQNetwork, hard_update, soft_update


@dataclass
class DQNConfig:
    hidden: int = 128
    gamma: float = 0.99
    lr: float = 3e-4
    tau: float = 0.005
    batch_size: int = 128
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    grad_clip: float = 10.0
    seed: int = 0


class BranchingDQN:
    """One binary Q-head per genset; epsilon-greedy exploration per branch."""

    def __init__(self, obs_dim: int, n_gensets: int, config: DQNConfig | None = None) -> None:
        self.obs_dim = obs_dim
        self.n_gensets = n_gensets
        self.cfg = config or DQNConfig()

        torch.manual_seed(self.cfg.seed)
        self.online = DuelingBranchingQNetwork(obs_dim, n_gensets, 2, self.cfg.hidden)
        self.target = copy.deepcopy(self.online)
        hard_update(self.target, self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.cfg.lr)

        self._rng = np.random.default_rng(self.cfg.seed)
        self.train_steps = 0
        self.env_steps = 0

    # -- acting -------------------------------------------------------------

    def epsilon(self) -> float:
        frac = min(self.env_steps / max(self.cfg.eps_decay_steps, 1), 1.0)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return a length-`n_gensets` array of 0/1 commitment decisions."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        q = self.online(obs_t).squeeze(0)  # (n_gensets, 2)
        greedy = q.argmax(dim=-1).numpy().astype(np.float32)

        if deterministic:
            self.env_steps += 1
            return greedy

        eps = self.epsilon()
        random_mask = self._rng.random(self.n_gensets) < eps
        random_actions = self._rng.integers(0, 2, self.n_gensets).astype(np.float32)
        self.env_steps += 1
        return np.where(random_mask, random_actions, greedy)

    # -- learning -------------------------------------------------------------

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["genset_on"], dtype=torch.long)  # (B, n_gensets)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32)
        done = torch.as_tensor(batch["done"], dtype=torch.float32)

        q = self.online(obs)  # (B, n_gensets, 2)
        q_taken = q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # (B, n_gensets)

        with torch.no_grad():
            online_next_q = self.online(next_obs)
            greedy_next = online_next_q.argmax(dim=-1)  # (B, n_gensets)
            target_next_q = self.target(next_obs)
            next_q = target_next_q.gather(-1, greedy_next.unsqueeze(-1)).squeeze(-1)
            # Branches share one scalar reward and coordinate through the mean
            # branch value, following the action-branching architecture's
            # target construction rather than n independent Bellman targets.
            mean_next_q = next_q.mean(dim=-1, keepdim=True)
            target = reward.unsqueeze(-1) + self.cfg.gamma * (1.0 - done.unsqueeze(-1)) * mean_next_q
            target = target.expand_as(q_taken)

        loss = F.smooth_l1_loss(q_taken, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        soft_update(self.target, self.online, self.cfg.tau)
        self.train_steps += 1
        return {"dqn_loss": float(loss.item()), "dqn_q_mean": float(q_taken.mean().item())}

    # -- persistence ----------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "env_steps": self.env_steps,
            "train_steps": self.train_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.env_steps = state.get("env_steps", 0)
        self.train_steps = state.get("train_steps", 0)


__all__ = ["BranchingDQN", "DQNConfig"]
