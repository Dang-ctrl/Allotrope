"""Federated averaging across stations: one policy, trained without sharing data.

Maitri and Bharati differ by roughly a factor of two in installed capacity, but
`HybridAgent`'s observation is scaled by each station's own installed capacity
(`allotrope.envs.polar_microgrid.observation_vector`) and its action space is
fixed by asset *count* -- three gensets, two storage packs -- which both
stations share. That agreement is what makes federated averaging meaningful
here rather than merely possible: the two sites can train the same network
architecture on their own local weather and demand, and the weights genuinely
transfer.

This implements FedAvg: each site trains locally for a round, then only the
resulting network *parameters* -- never the weather, demand, or telemetry that
produced them -- are collected and averaged into a new global model, which each
site continues training from. This is the mechanism behind the deck's claim
that "only gradients, never terabytes, cross the satellite link": a round's
parameter delta is bandwidth-equivalent to exchanging the accumulated gradient,
and at no point does raw station data leave the site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from allotrope.agents.dqn import DQNConfig
from allotrope.agents.hybrid import HybridAgent
from allotrope.agents.sddpg import SDDPGConfig
from allotrope.agents.train import TrainConfig, make_training_env, train
from allotrope.config import StationConfig


def average_state_dicts(
    state_dicts: list[dict], weights: list[float] | None = None
) -> dict:
    """Elementwise weighted mean of a list of (possibly nested) state dicts.

    Nesting is handled recursively so this works uniformly on a raw network
    `state_dict()` (tensor leaves) and on `HybridAgent.state_dict()`, which
    nests one level deeper (`{"dqn": {...}, "sddpg": {...}}`).
    """
    if not state_dicts:
        raise ValueError("cannot average an empty list of state dicts")
    n = len(state_dicts)
    w = weights if weights is not None else [1.0 / n] * n
    if len(w) != n:
        raise ValueError("one weight is required per state dict")
    total = sum(w)
    w = [x / total for x in w]

    first = state_dicts[0]
    if isinstance(first, dict) and all(
        isinstance(v, (dict, torch.Tensor)) for v in first.values()
    ):
        result = {}
        for key in first:
            values = [sd[key] for sd in state_dicts]
            if isinstance(first[key], dict):
                result[key] = average_state_dicts(values, w)
            else:
                stacked = torch.stack([v.float() * wi for v, wi in zip(values, w)], dim=0)
                result[key] = stacked.sum(dim=0).to(first[key].dtype)
        return result
    raise TypeError("state dict values must be tensors or nested dicts of tensors")


@dataclass(frozen=True)
class FederatedConfig:
    rounds: int = 20
    local_episodes: int = 10
    episode_steps: int = 24 * 7
    seed: int = 0
    dqn_config: DQNConfig = field(default_factory=DQNConfig)
    sddpg_config: SDDPGConfig = field(default_factory=SDDPGConfig)


@dataclass
class RoundLog:
    round: int
    per_station_mean_reward: dict[str, float]


def run_federated_training(
    stations: dict[str, StationConfig],
    config: FederatedConfig | None = None,
    on_round: Callable[[RoundLog], None] | None = None,
) -> tuple[HybridAgent, list[RoundLog]]:
    """Train one global agent across several stations' local environments.

    Each round: every site starts from the current global weights, trains
    locally for `local_episodes`, and the resulting weights are averaged
    (unweighted -- every site's model counts equally, regardless of installed
    capacity) into the next round's global model. The station whose gensets and
    storage counts define `HybridAgent`'s dimensions is taken from the first
    entry in `stations`; every station passed in must share those counts, or
    their weights cannot be averaged at all.
    """
    cfg = config or FederatedConfig()
    names = list(stations)
    reference = stations[names[0]]
    for name, station_cfg in stations.items():
        if len(station_cfg.gensets) != len(reference.gensets) or len(
            station_cfg.storage
        ) != len(reference.storage):
            raise ValueError(
                f"{name}: asset counts differ from {names[0]}, so their networks "
                "cannot be federated -- the action space itself would differ"
            )

    global_agent = HybridAgent(reference, dqn_config=cfg.dqn_config, sddpg_config=cfg.sddpg_config)
    envs = {
        name: make_training_env(
            station_cfg,
            TrainConfig(episode_steps=cfg.episode_steps, seed=cfg.seed),
        )
        for name, station_cfg in stations.items()
    }

    logs: list[RoundLog] = []
    for round_index in range(cfg.rounds):
        local_states = []
        round_rewards: dict[str, float] = {}

        for name, station_cfg in stations.items():
            local_agent = HybridAgent(
                station_cfg, dqn_config=cfg.dqn_config, sddpg_config=cfg.sddpg_config
            )
            local_agent.load_state_dict(global_agent.state_dict())

            train_cfg = TrainConfig(
                episodes=cfg.local_episodes,
                episode_steps=cfg.episode_steps,
                seed=cfg.seed + round_index * 1000 + hash(name) % 997,
            )
            episode_logs = train(local_agent, envs[name], train_cfg)

            local_states.append(local_agent.state_dict())
            round_rewards[name] = float(
                np.mean([log.reward for log in episode_logs])
            )

        averaged = average_state_dicts(local_states)
        global_agent.load_state_dict(averaged)

        log = RoundLog(round=round_index, per_station_mean_reward=round_rewards)
        logs.append(log)
        if on_round is not None:
            on_round(log)

    return global_agent, logs


__all__ = ["average_state_dicts", "FederatedConfig", "RoundLog", "run_federated_training"]
