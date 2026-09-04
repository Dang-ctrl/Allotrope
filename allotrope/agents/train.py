"""The training loop: drive the guarded environment, learn from what happens.

Training runs entirely inside `PolarMicrogridEnv` with `apply_safety=True`, its
default. This is the load-bearing decision of this module: the agent explores
behind the same projection layer it will be deployed behind, so it never
observes -- and therefore never has to unlearn -- the consequence of an unsafe
action. What it learns is how to be efficient inside bounds that were never its
own to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from allotrope.agents.hybrid import HybridAgent
from allotrope.config import StationConfig
from allotrope.envs.polar_microgrid import PolarMicrogridEnv


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 200
    episode_steps: int = 24 * 30
    """One episode is a month by default: long enough to see multi-day weather
    systems and burn-off cycles, short enough that many episodes fit in a run."""
    update_every: int = 4
    updates_per_step: int = 1
    seed: int = 0
    randomise_start: bool = True
    log_every: int = 10


@dataclass
class EpisodeLog:
    episode: int
    reward: float
    fuel_l: float
    black_carbon_g: float
    mean_genset_load_frac: float
    genset_starts: int
    critical_unserved_kwh: float
    freeze_violation_steps: float
    dqn_epsilon: float
    sddpg_sigma: float
    last_losses: dict[str, float] = field(default_factory=dict)


def make_training_env(
    station: str | StationConfig, cfg: TrainConfig, periods: int = 8760
) -> PolarMicrogridEnv:
    return PolarMicrogridEnv(
        station=station,
        periods=periods,
        seed=cfg.seed,
        episode_steps=cfg.episode_steps,
        randomise_start=cfg.randomise_start,
        apply_safety=True,
    )


def train(
    agent: HybridAgent,
    env: PolarMicrogridEnv,
    cfg: TrainConfig,
    on_episode: Callable[[EpisodeLog], None] | None = None,
) -> list[EpisodeLog]:
    """Run `cfg.episodes` training episodes and return their logs.

    `on_episode`, if given, is called with each `EpisodeLog` as it completes --
    the hook a caller uses to checkpoint the agent or print progress without
    this function needing to know about either.
    """
    rng = np.random.default_rng(cfg.seed)
    logs: list[EpisodeLog] = []

    for episode in range(cfg.episodes):
        obs, _ = env.reset(seed=cfg.seed + episode)
        total_reward = 0.0
        last_losses: dict[str, float] = {}
        step = 0

        while True:
            action, commitment_index = agent.act_training(obs, explore=True, rng=rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.observe(obs, action, reward, next_obs, terminated or truncated, commitment_index)
            total_reward += reward
            obs = next_obs
            step += 1

            if step % cfg.update_every == 0:
                for _ in range(cfg.updates_per_step):
                    stats = agent.update(rng)
                    if stats:
                        last_losses = stats

            if terminated or truncated:
                break

        agent.end_episode()
        summary = env.summary()
        log = EpisodeLog(
            episode=episode,
            reward=total_reward,
            fuel_l=summary["fuel_l"],
            black_carbon_g=summary["black_carbon_g"],
            mean_genset_load_frac=summary["mean_genset_load_frac"],
            genset_starts=int(summary["genset_starts"]),
            critical_unserved_kwh=summary["critical_unserved_kwh"],
            freeze_violation_steps=summary["freeze_violation_steps"],
            dqn_epsilon=agent.dqn.epsilon,
            sddpg_sigma=agent.sddpg.sigma,
            last_losses=last_losses,
        )
        logs.append(log)
        if on_episode is not None:
            on_episode(log)

    return logs


__all__ = ["TrainConfig", "EpisodeLog", "train", "make_training_env"]
