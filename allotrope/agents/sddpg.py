"""SDDPG: how hard to work what is committed.

A stochastic (squashed-Gaussian) actor gives the exploration DDPG's
deterministic policy needs an explicit noise process for; twin critics and
delayed, soft-updated target networks are TD3's fix for the value
overestimation that plain DDPG is prone to. Combining them is what "SDDPG"
names here: a stochastic actor over DDPG-style off-policy actor-critic
learning, not a specific published algorithm with that acronym.

This agent only ever proposes the continuous dispatch vector -- loading
fraction per genset, power fraction per storage pack, melt rate -- in
[-1, 1]. `allotrope.envs.polar_microgrid.PolarMicrogridEnv.decode_action` and
then `allotrope.safety.projection.SafetyProjection` are what turn that
proposal into something the plant is allowed to execute.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from allotrope.agents.networks import StochasticActor, TwinCritic, hard_update, soft_update


@dataclass
class SDDPGConfig:
    hidden: int = 128
    gamma: float = 0.99
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    tau: float = 0.005
    batch_size: int = 128
    policy_delay: int = 2
    grad_clip: float = 10.0
    seed: int = 0


class SDDPG:
    def __init__(self, obs_dim: int, act_dim: int, config: SDDPGConfig | None = None) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = config or SDDPGConfig()

        torch.manual_seed(self.cfg.seed)
        self.actor = StochasticActor(obs_dim, act_dim, self.cfg.hidden)
        self.actor_target = copy.deepcopy(self.actor)
        hard_update(self.actor_target, self.actor)

        self.critic = TwinCritic(obs_dim, act_dim, self.cfg.hidden)
        self.critic_target = copy.deepcopy(self.critic)
        hard_update(self.critic_target, self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

        self._torch_gen = torch.Generator().manual_seed(self.cfg.seed)
        self.train_steps = 0

    # -- acting -------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = self.actor.act(obs_t, deterministic, generator=self._torch_gen)
        return action.squeeze(0).numpy().astype(np.float32)

    # -- learning -------------------------------------------------------------

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        action = torch.as_tensor(batch["dispatch"], dtype=torch.float32)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32)
        done = torch.as_tensor(batch["done"], dtype=torch.float32)

        with torch.no_grad():
            next_action = self.actor_target.act(next_obs, deterministic=False, generator=self._torch_gen)
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.minimum(target_q1, target_q2)
            target = reward + self.cfg.gamma * (1.0 - done) * target_q

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_optimizer.step()

        metrics = {"sddpg_critic_loss": float(critic_loss.item())}

        self.train_steps += 1
        if self.train_steps % self.cfg.policy_delay == 0:
            proposed = self.actor.act(obs, deterministic=False, generator=self._torch_gen)
            actor_q1, _ = self.critic(obs, proposed)
            actor_loss = -actor_q1.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
            self.actor_optimizer.step()

            soft_update(self.actor_target, self.actor, self.cfg.tau)
            soft_update(self.critic_target, self.critic, self.cfg.tau)
            metrics["sddpg_actor_loss"] = float(actor_loss.item())

        return metrics

    # -- persistence ----------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "train_steps": self.train_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.train_steps = state.get("train_steps", 0)


__all__ = ["SDDPG", "SDDPGConfig"]
