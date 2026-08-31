"""Run the rule-based controllers over a full year and print the comparison.

    python scripts/run_baseline.py --station maitri --seed 0

This is the calibration and sanity check for the whole plant model. If the
incumbent controller does not reproduce the problem the project exists to solve
-- a fleet loitering below its wet-stacking threshold through the winter -- then
the model is wrong and nothing built on top of it means anything.
"""

from __future__ import annotations

import argparse

import pandas as pd

from allotrope.config import load_station
from allotrope.control.baseline import EfficientRuleBased, LegacyNPlusOne
from allotrope.sim.runner import build_plant, compare, run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="maitri")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--periods", type=int, default=8760)
    parser.add_argument("--freq", default="1h")
    args = parser.parse_args()

    cfg = load_station(args.station)
    pd.set_option("display.width", 140)

    results = []
    for controller_cls in (LegacyNPlusOne, EfficientRuleBased):
        plant = build_plant(cfg, args.start, args.periods, args.freq, seed=args.seed)
        result = run_episode(plant, controller_cls(cfg))
        results.append(result)

    print(f"\n{cfg.site.name}  |  {args.periods} steps at {args.freq}  |  seed {args.seed}")
    print("=" * 78)
    print(compare(results).round(3).to_string())

    legacy, efficient = results
    fuel_saved = legacy.summary["fuel_l"] - efficient.summary["fuel_l"]
    bc_saved = legacy.summary["black_carbon_g"] - efficient.summary["black_carbon_g"]
    print("\nefficient rule-based versus legacy N+1")
    print(f"  fuel          {fuel_saved:9.0f} L   ({fuel_saved / legacy.summary['fuel_l']:6.1%})")
    print(f"  black carbon  {bc_saved:9.0f} g   ({bc_saved / legacy.summary['black_carbon_g']:6.1%})")

    if cfg.fuel_budget:
        budget_kl = cfg.fuel_budget.get("seasonal_jet_a1_kl")
        if budget_kl:
            print(f"\n  published seasonal fuel budget: {budget_kl:.0f} kL")
            for r in results:
                print(f"    {r.controller:<22} {r.summary['fuel_kl']:6.1f} kL")


if __name__ == "__main__":
    main()
