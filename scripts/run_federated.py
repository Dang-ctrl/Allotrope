"""Train one global agent across Maitri and Bharati by federated averaging.

    python scripts/run_federated.py --rounds 30 --local-episodes 15 --out checkpoints/federated.pt

Each round, every station trains locally from the current global weights, and
only the resulting network parameters are averaged together -- never the
weather, demand, or telemetry that produced them. This is deliberately the
same mechanism `tests/test_federated.py` exercises at a tiny scale to prove it
is correct; this script runs it at a scale meant to produce an actual policy,
not merely to demonstrate the arithmetic.
"""

from __future__ import annotations

import argparse
import time

from allotrope.agents.checkpoint import save_federated
from allotrope.agents.dqn import DQNConfig
from allotrope.agents.federated import FederatedConfig, run_federated_training
from allotrope.agents.sddpg import SDDPGConfig
from allotrope.config import load_station


def _decay_for(start: float, floor: float, steps: int) -> float:
    if steps <= 1:
        return 1.0
    return (floor / start) ** (1.0 / steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", nargs="+", default=["maitri", "bharati"])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-episodes", type=int, default=15)
    parser.add_argument("--episode-days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="checkpoints/federated.pt")
    args = parser.parse_args()

    stations = {name: load_station(name) for name in args.stations}

    # Exploration must finish decaying within the rounds actually available:
    # local_episodes accumulate across rounds, so the decay horizon is the
    # total number of local-training episodes any one site will see.
    total_local_episodes = args.rounds * args.local_episodes
    horizon = max(int(total_local_episodes * 0.7), 1)
    dqn_defaults, sddpg_defaults = DQNConfig(), SDDPGConfig()
    dqn_config = DQNConfig(
        epsilon_decay=_decay_for(dqn_defaults.epsilon_start, dqn_defaults.epsilon_min, horizon)
    )
    sddpg_config = SDDPGConfig(
        exploration_decay=_decay_for(
            sddpg_defaults.exploration_sigma, sddpg_defaults.exploration_sigma_min, horizon
        )
    )
    print(
        f"federating {list(stations)} over {args.rounds} rounds x "
        f"{args.local_episodes} local episodes ({total_local_episodes} total per site)"
    )

    config = FederatedConfig(
        rounds=args.rounds,
        local_episodes=args.local_episodes,
        episode_steps=24 * args.episode_days,
        seed=args.seed,
        dqn_config=dqn_config,
        sddpg_config=sddpg_config,
    )

    start = time.time()

    def on_round(log) -> None:
        elapsed = time.time() - start
        rewards = "  ".join(f"{name}={r:8.1f}" for name, r in log.per_station_mean_reward.items())
        print(f"round {log.round:3d}/{args.rounds}  {rewards}  [{elapsed:6.1f}s]")

    global_agent, logs = run_federated_training(stations, config, on_round=on_round)
    save_federated(global_agent, list(stations), args.out)
    print(f"\nsaved federated checkpoint to {args.out}")


if __name__ == "__main__":
    main()
