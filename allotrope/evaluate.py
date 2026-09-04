"""Evaluate a trained checkpoint against the rule-based baselines.

    python -m allotrope.evaluate --checkpoint runs/hybrid_maitri_seed0_.../checkpoint.pt \\
        --station maitri --seed 1 --periods 8760

Runs, on a held-out seed the checkpoint was never trained on:

  * LegacyNPlusOne         -- the incumbent practice
  * EfficientRuleBased     -- the best non-learned policy
  * the checkpoint, safety-projected      ("safe" -- what would actually be deployed)
  * the checkpoint, unprojected           (the control column, as in
                                            `scripts/run_safety_audit.py`: shows what the
                                            projection is buying, on this policy specifically)

All four are scored inside `allotrope.sim.runner.run_episode`, so the
comparison uses the same plant, the same clock and the same summary metrics
the rest of the project's benchmarks use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from allotrope.agents.dqn import BranchingDQN, DQNConfig
from allotrope.agents.hybrid import HybridAgent
from allotrope.agents.sddpg import SDDPG, SDDPGConfig
from allotrope.config import load_station
from allotrope.control.baseline import EfficientRuleBased, LegacyNPlusOne
from allotrope.safety.fallback import GuardedController
from allotrope.sim.runner import build_plant, compare, run_episode


def load_checkpoint(path: Path) -> tuple[BranchingDQN, SDDPG, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    dqn = BranchingDQN(state["obs_dim"], state["n_gensets"], DQNConfig())
    sddpg = SDDPG(state["obs_dim"], state["dispatch_dim"], SDDPGConfig())
    dqn.load_state_dict(state["dqn"])
    sddpg.load_state_dict(state["sddpg"])
    return dqn, sddpg, state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--station", default="maitri")
    parser.add_argument("--seed", type=int, default=1, help="held-out seed, distinct from training")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--periods", type=int, default=8760)
    parser.add_argument("--freq", default="1h")
    args = parser.parse_args()

    cfg = load_station(args.station)
    dqn, sddpg, state = load_checkpoint(Path(args.checkpoint))
    pd.set_option("display.width", 150)

    hybrid = HybridAgent(cfg, dqn, sddpg, deterministic=True)
    controllers = {
        "legacy_n_plus_one": LegacyNPlusOne(cfg),
        "efficient_rule_based": EfficientRuleBased(cfg),
        "hybrid_safe": GuardedController(cfg, agent=hybrid),
        "hybrid_unsafe": hybrid,
    }

    results = []
    for name, controller in controllers.items():
        plant = build_plant(cfg, args.start, args.periods, args.freq, seed=args.seed)
        results.append(run_episode(plant, controller))

    print(f"\n{cfg.site.name}  |  checkpoint {args.checkpoint} (trained as {state['agent_kind']})")
    print(f"held-out seed {args.seed}  |  {args.periods} steps at {args.freq}")
    print("=" * 100)
    print(compare(results).round(3).to_string())

    legacy, efficient, safe, unsafe = (r.summary for r in results)
    print("\nhybrid_safe versus baselines")
    print(f"  fuel vs legacy N+1     {legacy['fuel_l'] - safe['fuel_l']:9.0f} L")
    print(f"  fuel vs efficient      {efficient['fuel_l'] - safe['fuel_l']:9.0f} L")
    print("\nwhat the safety projection is buying this specific policy")
    print(f"  critical unserved, safe   {safe['critical_unserved_kwh']:9.2f} kWh")
    print(f"  critical unserved, unsafe {unsafe['critical_unserved_kwh']:9.2f} kWh")
    print(f"  freeze steps, safe        {safe['freeze_violation_steps']:9.0f}")
    print(f"  freeze steps, unsafe      {unsafe['freeze_violation_steps']:9.0f}")


if __name__ == "__main__":
    main()
