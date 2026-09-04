"""Train the hybrid DQN + SDDPG agent against the guarded environment.

    python scripts/train_agent.py --station maitri --episodes 300 --out checkpoints/maitri.pt

Training runs entirely behind the safety projection (`apply_safety=True`, the
environment default), so the agent explores safely from the first random action
and never has to unlearn an unsafe habit it was never permitted to form.
"""

from __future__ import annotations

import argparse
import time

from allotrope.agents.checkpoint import save
from allotrope.agents.dqn import DQNConfig
from allotrope.agents.hybrid import HybridAgent
from allotrope.agents.sddpg import SDDPGConfig
from allotrope.agents.train import TrainConfig, make_training_env, train
from allotrope.config import load_station


def _decay_for(start: float, floor: float, episodes: int) -> float:
    """The per-episode multiplier that reaches `floor` from `start` in `episodes` steps.

    Exploration has to actually finish decaying within the run it is given, or
    the final checkpoint is evaluated mid-exploration and every reported number
    is measuring random noise as much as the policy. The default schedules on
    DQNConfig and SDDPGConfig assume roughly 1000 episodes; a shorter run needs
    a faster schedule, computed here rather than left to silently undershoot.
    """
    if episodes <= 1:
        return 1.0
    return (floor / start) ** (1.0 / episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="maitri")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--episode-days", type=int, default=30)
    parser.add_argument("--periods", type=int, default=8760)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--out", default="checkpoints/agent.pt")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    cfg = load_station(args.station)
    train_cfg = TrainConfig(
        episodes=args.episodes,
        episode_steps=24 * args.episode_days,
        update_every=args.update_every,
        seed=args.seed,
        log_every=args.log_every,
    )
    env = make_training_env(cfg, train_cfg, periods=args.periods)

    dqn_defaults = DQNConfig()
    sddpg_defaults = SDDPGConfig()
    # Reach the exploration floor at 70% of the run, not at the last episode,
    # so the checkpoint that gets saved and evaluated has had a real stretch of
    # near-greedy training rather than being taken mid-decay.
    decay_horizon = max(int(args.episodes * 0.7), 1)
    dqn_config = DQNConfig(
        epsilon_start=dqn_defaults.epsilon_start,
        epsilon_min=dqn_defaults.epsilon_min,
        epsilon_decay=_decay_for(dqn_defaults.epsilon_start, dqn_defaults.epsilon_min, decay_horizon),
    )
    sddpg_config = SDDPGConfig(
        exploration_sigma=sddpg_defaults.exploration_sigma,
        exploration_sigma_min=sddpg_defaults.exploration_sigma_min,
        exploration_decay=_decay_for(
            sddpg_defaults.exploration_sigma, sddpg_defaults.exploration_sigma_min, decay_horizon
        ),
    )
    print(
        f"exploration schedule for {args.episodes} episodes: "
        f"epsilon decay {dqn_config.epsilon_decay:.4f}, sigma decay {sddpg_config.exploration_decay:.4f}"
    )
    agent = HybridAgent(cfg, dqn_config=dqn_config, sddpg_config=sddpg_config)

    start = time.time()
    best_reward = float("-inf")

    def on_episode(log) -> None:
        nonlocal best_reward
        best_reward = max(best_reward, log.reward)
        if log.episode % train_cfg.log_every == 0 or log.episode == train_cfg.episodes - 1:
            elapsed = time.time() - start
            print(
                f"ep {log.episode:4d}/{train_cfg.episodes}  "
                f"reward {log.reward:10.1f}  best {best_reward:10.1f}  "
                f"fuel {log.fuel_l:7.1f} L  load_frac {log.mean_genset_load_frac:.3f}  "
                f"starts {log.genset_starts:3d}  crit_unserved {log.critical_unserved_kwh:8.2f} kWh  "
                f"eps {log.dqn_epsilon:.3f}  sigma {log.sddpg_sigma:.3f}  "
                f"[{elapsed:6.1f}s]"
            )
            if log.critical_unserved_kwh > 1e-6:
                print(
                    "  WARNING: critical load went unserved during training. "
                    "The safety layer should make this impossible -- investigate "
                    "before trusting this run."
                )

    logs = train(agent, env, train_cfg, on_episode=on_episode)

    save(agent, args.out)
    print(f"\nsaved checkpoint to {args.out}")

    tail = logs[-min(10, len(logs)):]
    print(f"\nlast {len(tail)} episodes:")
    print(f"  mean reward          {sum(l.reward for l in tail) / len(tail):.1f}")
    print(f"  mean fuel            {sum(l.fuel_l for l in tail) / len(tail):.1f} L")
    print(f"  mean load factor     {sum(l.mean_genset_load_frac for l in tail) / len(tail):.3f}")
    print(f"  mean starts/episode  {sum(l.genset_starts for l in tail) / len(tail):.1f}")
    print(f"  max critical unserved {max(l.critical_unserved_kwh for l in tail):.4f} kWh")


if __name__ == "__main__":
    main()
