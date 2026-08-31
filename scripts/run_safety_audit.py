"""Audit the safety layer by attacking it, with the guarantee on and off.

    python scripts/run_safety_audit.py --station maitri --days 30

The claim under audit is that no action -- however random, adversarial or
malformed -- can cause the station to shed life support or freeze. A claim like
that is only worth as much as the attempt made to break it, so this script runs
policies designed to break it and reports what got through.

The unguarded column is the control. If it showed no harm either, the guarded
column would prove nothing.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from allotrope.config import load_station
from allotrope.envs.polar_microgrid import PolarMicrogridEnv


def random_policy(env, rng):
    return env.action_space.sample()


def all_off_policy(env, rng):
    n_g, n_s = len(env.cfg.gensets), len(env.cfg.storage)
    return {
        "genset_on": np.zeros(n_g, dtype=np.int8),
        "dispatch": np.full(n_g + n_s + 1, -1.0, dtype=np.float32),
    }


def max_charge_policy(env, rng):
    n_g, n_s = len(env.cfg.gensets), len(env.cfg.storage)
    dispatch = np.full(n_g + n_s + 1, -1.0, dtype=np.float32)
    dispatch[n_g : n_g + n_s] = -1.0
    return {"genset_on": np.zeros(n_g, dtype=np.int8), "dispatch": dispatch}


def max_melt_policy(env, rng):
    n_g, n_s = len(env.cfg.gensets), len(env.cfg.storage)
    dispatch = np.full(n_g + n_s + 1, -1.0, dtype=np.float32)
    dispatch[-1] = 1.0
    return {"genset_on": np.zeros(n_g, dtype=np.int8), "dispatch": dispatch}


def oscillating_policy(env, rng):
    flip = bool(rng.integers(0, 2))
    n_g, n_s = len(env.cfg.gensets), len(env.cfg.storage)
    return {
        "genset_on": np.full(n_g, int(flip), dtype=np.int8),
        "dispatch": np.full(n_g + n_s + 1, 1.0 if flip else -1.0, dtype=np.float32),
    }


POLICIES = {
    "random": random_policy,
    "shut everything down": all_off_policy,
    "charge flat out": max_charge_policy,
    "melt flat out": max_melt_policy,
    "oscillate commitment": oscillating_policy,
}


def run(station, policy, days, seed, apply_safety):
    env = PolarMicrogridEnv(
        station=station, start="2026-06-01", periods=24 * days, seed=seed,
        apply_safety=apply_safety,
    )
    env.reset(seed=seed)
    env.action_space.seed(seed)
    rng = np.random.default_rng(seed)

    interventions: dict[str, int] = {}
    while True:
        _, _, terminated, truncated, info = env.step(policy(env, rng))
        for name in info.get("safety", {}).get("interventions", []):
            interventions[name] = interventions.get(name, 0) + 1
        if terminated or truncated:
            break
    return env.summary(), interventions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="maitri")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_station(args.station)
    pd.set_option("display.width", 150)

    rows = {}
    all_interventions: dict[str, int] = {}
    for label, policy in POLICIES.items():
        guarded, interventions = run(cfg, policy, args.days, args.seed, apply_safety=True)
        unguarded, _ = run(cfg, policy, args.days, args.seed, apply_safety=False)
        for name, count in interventions.items():
            all_interventions[name] = all_interventions.get(name, 0) + count

        rows[label] = {
            "life support lost, guarded (kWh)": guarded["critical_unserved_kwh"],
            "life support lost, UNGUARDED (kWh)": unguarded["critical_unserved_kwh"],
            "freeze steps, guarded": guarded["freeze_violation_steps"],
            "freeze steps, UNGUARDED": unguarded["freeze_violation_steps"],
            "fuel, guarded (L)": guarded["fuel_l"],
        }

    print(f"\n{cfg.site.name}  |  {args.days} midwinter days  |  seed {args.seed}")
    print("=" * 96)
    print(pd.DataFrame(rows).T.round(2).to_string())

    print("\nsafety interventions across all attacks")
    for name, count in sorted(all_interventions.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<48} {count:>6}")

    worst = max(r["life support lost, guarded (kWh)"] for r in rows.values())
    frozen = max(r["freeze steps, guarded"] for r in rows.values())
    print()
    if worst <= 1e-9 and frozen == 0:
        print("  VERDICT: no attack reached the station through the projection layer.")
    else:
        print(f"  VERDICT: the guarantee was breached -- {worst:.3f} kWh lost, {frozen:.0f} freeze steps.")


if __name__ == "__main__":
    main()
