"""Train DQN, SDDPG, or both jointly, against the safety-projected environment.

    python -m allotrope.train --agent dqn --station maitri
    python -m allotrope.train --agent sddpg --station maitri
    python -m allotrope.train --agent hybrid --station maitri

`--agent dqn` and `--agent sddpg` train one network while the other side of
the joint action is supplied by an *untrained* instance of its counterpart,
so a single-agent run still exercises against a plausible partner rather
than a placeholder. `--agent hybrid` updates both simultaneously, which is
the mode a deployable controller is ultimately trained under.

Every action proposed during training is projected by
`allotrope.safety.projection.SafetyProjection` before it reaches the plant
(`PolarMicrogridEnv(apply_safety=True)`), so exploration -- including a
freshly initialised, effectively random network -- can never damage the
simulated station. This is not a training convenience; it is the same
guarantee `scripts/run_safety_audit.py` audits, applied to the one policy
in this codebase that did not have it built in by construction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from allotrope.agents.dqn import BranchingDQN, DQNConfig
from allotrope.agents.replay_buffer import ReplayBuffer, Transition
from allotrope.agents.sddpg import SDDPG, SDDPGConfig
from allotrope.config import load_station
from allotrope.envs.polar_microgrid import PolarMicrogridEnv
from allotrope.experiment import ExperimentTracker

LOG_EVERY = 500


def _checkpoint_dict(
    agent_kind: str, station: str, obs_dim: int, n_g: int, dispatch_dim: int, dqn: BranchingDQN, sddpg: SDDPG
) -> dict:
    return {
        "agent_kind": agent_kind,
        "station": station,
        "obs_dim": obs_dim,
        "n_gensets": n_g,
        "dispatch_dim": dispatch_dim,
        "dqn": dqn.state_dict(),
        "sddpg": sddpg.state_dict(),
    }


def build_agents(env: PolarMicrogridEnv, seed: int) -> tuple[BranchingDQN, SDDPG]:
    obs_dim = env.observation_space.shape[0]
    n_g = len(env.cfg.gensets)
    dispatch_dim = env.action_space["dispatch"].shape[0]
    dqn = BranchingDQN(obs_dim, n_g, DQNConfig(seed=seed))
    sddpg = SDDPG(obs_dim, dispatch_dim, SDDPGConfig(seed=seed))
    return dqn, sddpg


def train(
    agent_kind: str,
    station: str,
    total_steps: int,
    seed: int,
    episode_steps: int,
    warmup_steps: int,
    buffer_capacity: int,
    runs_dir: Path,
    init_checkpoint: Path | None = None,
    checkpoint_every: int = 0,
) -> Path:
    """Train, optionally warm-started from an existing checkpoint's weights.

    `init_checkpoint` is what `allotrope.federated` uses to start a
    station's local round from the current global model rather than from
    scratch -- a plain `dqn.load_state_dict`/`sddpg.load_state_dict` after
    the usual random init, using the same `weights_only=True` loading
    `allotrope.evaluate.load_checkpoint` uses. `None` (the default)
    reproduces this function's exact prior behaviour: every existing
    caller is unaffected.

    `checkpoint_every` (0 = disabled, the prior behaviour) overwrites the
    same `checkpoint.pt` this function already writes at the end, every
    that many steps, so a training run killed partway through (an
    interrupted machine, not a code failure) loses at most that many
    steps' progress instead of the entire run. `BranchingDQN.state_dict()`
    and `SDDPG.state_dict()` already carry `env_steps`/`train_steps`, so
    resuming via `--init-checkpoint` on this intermediate file preserves
    the epsilon-decay schedule and target-network update cadence exactly
    -- it does not preserve the replay buffer or optimizer momentum, which
    this checkpoint schema was never designed to carry, so a resumed run
    is not bit-identical to an uninterrupted one, only continued from the
    same learned weights.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    cfg = load_station(station)
    env = PolarMicrogridEnv(
        cfg, apply_safety=True, episode_steps=episode_steps, randomise_start=True, seed=seed
    )
    dqn, sddpg = build_agents(env, seed)
    if init_checkpoint is not None:
        state = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        dqn.load_state_dict(state["dqn"])
        sddpg.load_state_dict(state["sddpg"])
    update_dqn = agent_kind in ("dqn", "hybrid")
    update_sddpg = agent_kind in ("sddpg", "hybrid")

    obs_dim = env.observation_space.shape[0]
    n_g = len(cfg.gensets)
    dispatch_dim = env.action_space["dispatch"].shape[0]
    buffer = ReplayBuffer(buffer_capacity, obs_dim, n_g, dispatch_dim)

    tracker = ExperimentTracker(
        agent=agent_kind,
        station=station,
        seed=seed,
        config={
            "total_steps": total_steps,
            "episode_steps": episode_steps,
            "warmup_steps": warmup_steps,
            "buffer_capacity": buffer_capacity,
            "dqn": vars(dqn.cfg),
            "sddpg": vars(sddpg.cfg),
        },
        runs_dir=runs_dir,
    )

    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_return_log: list[float] = []

    for step in range(1, total_steps + 1):
        genset_on = dqn.act(obs, deterministic=False)
        dispatch = sddpg.act(obs, deterministic=False)
        action = {"genset_on": genset_on.astype(np.int8), "dispatch": dispatch}

        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(Transition(obs, genset_on, dispatch, float(reward), next_obs, terminated))
        episode_reward += float(reward)
        obs = next_obs

        if terminated or truncated:
            episode_return_log.append(episode_reward)
            episode_reward = 0.0
            obs, _ = env.reset()

        metrics: dict[str, float | None] = {}
        if len(buffer) >= warmup_steps:
            batch_size = max(dqn.cfg.batch_size, sddpg.cfg.batch_size)
            batch = buffer.sample(batch_size, rng)
            if update_dqn:
                metrics.update(dqn.update(batch))
            if update_sddpg:
                metrics.update(sddpg.update(batch))

        if step % LOG_EVERY == 0:
            # None, not NaN, when no episode has completed yet: `float("nan")`
            # serialises to the bareword `NaN`, which is not valid JSON and
            # breaks any standards-conformant reader of runs/<id>/record.json.
            mean_return = float(np.mean(episode_return_log[-10:])) if episode_return_log else None
            metrics["mean_episode_return"] = mean_return
            metrics["buffer_size"] = len(buffer)
            tracker.log(step, metrics)
            return_str = f"{mean_return:8.3f}" if mean_return is not None else "     n/a"
            print(f"step {step:>7}/{total_steps}  return[-10]={return_str}  {metrics}")

        if checkpoint_every and step % checkpoint_every == 0 and step != total_steps:
            torch.save(
                _checkpoint_dict(agent_kind, station, obs_dim, n_g, dispatch_dim, dqn, sddpg),
                tracker.dir / "checkpoint.pt",
            )

    checkpoint_path = tracker.dir / "checkpoint.pt"
    torch.save(
        _checkpoint_dict(agent_kind, station, obs_dim, n_g, dispatch_dim, dqn, sddpg),
        checkpoint_path,
    )
    final_metrics = {
        "mean_episode_return_last10": (
            float(np.mean(episode_return_log[-10:])) if episode_return_log else None
        ),
        "episodes_completed": len(episode_return_log),
    }
    tracker.finish(final_metrics, checkpoint_path=str(checkpoint_path))
    print(f"\nsaved checkpoint: {checkpoint_path}")
    print(f"run record: {tracker.dir / 'record.json'}")
    return tracker.dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["dqn", "sddpg", "hybrid"], required=True)
    parser.add_argument("--station", default="maitri")
    parser.add_argument("--total-steps", type=int, default=20_000)
    parser.add_argument("--episode-steps", type=int, default=24 * 14)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--buffer-capacity", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--init-checkpoint", default=None, help="warm-start from this checkpoint's weights"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="overwrite checkpoint.pt every N steps (0 = only at the end, the prior default)",
    )
    args = parser.parse_args()

    train(
        agent_kind=args.agent,
        station=args.station,
        total_steps=args.total_steps,
        seed=args.seed,
        episode_steps=args.episode_steps,
        warmup_steps=args.warmup_steps,
        buffer_capacity=args.buffer_capacity,
        runs_dir=Path(args.runs_dir),
        init_checkpoint=Path(args.init_checkpoint) if args.init_checkpoint else None,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
