"""Evaluate a trained agent against both baselines, on seeds it never trained on.

    python scripts/evaluate_agent.py --station maitri --checkpoint checkpoints/maitri.pt

Held-out means held out: training draws its weather from `build_plant(seed=N)`
for `N` in the training range, and this script evaluates on a disjoint range, so
a good score here cannot be explained by the agent having memorised the specific
year it was shown.

The bar is `EfficientRuleBased`, not `LegacyNPlusOne`. A learned policy that
only beats the incumbent has learned what careful, unlearned engineering already
achieves; it has to beat the disciplined rules to justify the added complexity of
training it at all.
"""

from __future__ import annotations

import argparse

import pandas as pd

from allotrope.agents.checkpoint import load
from allotrope.config import load_station
from allotrope.control.baseline import EfficientRuleBased, LegacyNPlusOne
from allotrope.safety.fallback import GuardedController
from allotrope.sim.runner import build_plant, compare, run_episode

HELD_OUT_SEEDS = [100, 101, 102, 103, 104]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="maitri")
    parser.add_argument("--checkpoint", default="checkpoints/agent.pt")
    parser.add_argument("--periods", type=int, default=8760)
    parser.add_argument("--seeds", type=int, nargs="+", default=HELD_OUT_SEEDS)
    args = parser.parse_args()

    cfg = load_station(args.station)
    agent = load(args.checkpoint, cfg)
    guarded_agent = GuardedController(cfg, agent=agent)

    pd.set_option("display.width", 150)
    all_results = {"legacy_n_plus_one": [], "efficient_rule_based": [], "hybrid_dqn_sddpg": []}

    for seed in args.seeds:
        for controller in (LegacyNPlusOne(cfg), EfficientRuleBased(cfg), guarded_agent):
            plant = build_plant(cfg, periods=args.periods, seed=seed)
            result = run_episode(plant, controller)
            label = "hybrid_dqn_sddpg" if controller is guarded_agent else result.controller
            all_results[label].append(result)

    print(f"\n{cfg.site.name}  |  {args.periods} steps  |  held-out seeds {args.seeds}")
    print("=" * 90)

    means = {}
    for label, results in all_results.items():
        keys = results[0].summary.keys()
        means[label] = {k: sum(r.summary[k] for r in results) / len(results) for k in keys}
    summary_df = pd.DataFrame(means).T[
        [
            "fuel_kl",
            "black_carbon_g",
            "mean_genset_load_frac",
            "wet_stacking_fraction",
            "renewable_fraction",
            "genset_starts",
            "critical_unserved_kwh",
            "freeze_violation_steps",
        ]
    ]
    print("mean across held-out seeds:")
    print(summary_df.round(3).to_string())

    fuel_vs_efficient = (
        means["efficient_rule_based"]["fuel_l"] - means["hybrid_dqn_sddpg"]["fuel_l"]
    ) / means["efficient_rule_based"]["fuel_l"]
    starts_vs_efficient = means["hybrid_dqn_sddpg"]["genset_starts"] - means["efficient_rule_based"]["genset_starts"]

    print(f"\nhybrid agent versus the efficient rule-based bar it must clear:")
    print(f"  fuel            {fuel_vs_efficient:+.1%}")
    print(f"  genset starts   {starts_vs_efficient:+.1f} per year")

    max_unserved = max(r.summary["critical_unserved_kwh"] for r in all_results["hybrid_dqn_sddpg"])
    max_freeze = max(r.summary["freeze_violation_steps"] for r in all_results["hybrid_dqn_sddpg"])
    print()
    if max_unserved <= 1e-6 and max_freeze == 0:
        print("  SAFETY: the guarded agent shed no life support and caused no freeze "
              "violation on any held-out seed.")
    else:
        print(f"  SAFETY WARNING: {max_unserved:.3f} kWh unserved, {max_freeze:.0f} freeze steps "
              "-- investigate before reporting this checkpoint.")

    if fuel_vs_efficient <= 0:
        print(
            "\n  This checkpoint does not yet beat the efficient rule-based baseline. "
            "That is a legitimate result to report, not one to hide: say so, and say "
            "by how much, rather than training longer until the number looks better."
        )


if __name__ == "__main__":
    main()
