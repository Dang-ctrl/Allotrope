"""Scenario-based evaluation: many seeds, not one.

    python -m allotrope.evaluate_scenarios --station maitri --seeds 200
    python -m allotrope.evaluate_scenarios --station maitri --seeds 200 \\
        --checkpoint runs/hybrid_maitri_seed0_.../checkpoint.pt

A single seeded year (`scripts/run_baseline.py`, `allotrope.evaluate`) shows
one weather-and-demand realisation. This module runs a controller against
many independent realisations of the same station's synthetic climate and
demand -- the seed is the only thing that changes -- and reports the
distribution of outcomes, not a single point estimate: mean, median,
standard deviation, min/max, and the 5th/95th percentiles for every headline
metric `allotrope.sim.runner.compare` already tracks.

What this *is*: a real statistical spread over the natural stochasticity the
project's own synthetic climate and demand generators already produce --
cold snaps, blizzards, and the demand noise process all vary by seed, so
"200 seeds" genuinely covers a range from mild to severe winters, not 200
copies of the same year.

What this is *not* (yet): scenario-specific fault injection. There is no
mechanism in `allotrope.sim` today to force a genset, PV string, or wind
turbine offline mid-run, corrupt telemetry, or delay a control decision --
building one is real, separate work, not a relabelling of what exists. The
adjacent claim this project *can* back today is the adversarial-policy audit
(`scripts/run_safety_audit.py`, `tests/test_safety.py`, `tests/test_agents.py`)
covering NaN/invalid/malformed *actions* and agent timeout/exception paths,
which is a different (and already-tested) axis from asset or sensor failure.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from allotrope.config import StationConfig, load_station
from allotrope.control.baseline import EfficientRuleBased, LegacyNPlusOne
from allotrope.sim.runner import build_plant, run_episode

METRIC_KEYS = [
    "fuel_kl",
    "black_carbon_g",
    "specific_fuel_l_per_kwh",
    "mean_genset_load_frac",
    "wet_stacking_fraction",
    "mean_deposit",
    "renewable_fraction",
    "curtailed_kwh",
    "genset_run_hours",
    "genset_starts",
    "critical_unserved_kwh",
    "freeze_violation_steps",
    "unmet_water_kwh",
]


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


@dataclass
class ScenarioSuiteResult:
    """One controller's outcomes across many independent seeds, on one station."""

    station: str
    controller: str
    seeds: list[int]
    per_seed: dict[str, list[float]] = field(default_factory=dict)
    stats: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "n_seeds": self.n_seeds}


def run_scenario_suite(
    station: str | StationConfig,
    controller_factory: Callable[[StationConfig], Any],
    controller_name: str,
    seeds: list[int],
    start: str = "2026-01-01",
    periods: int = 8760,
    freq: str = "1h",
) -> ScenarioSuiteResult:
    """Run one controller across `seeds` independent weather/demand realisations."""
    cfg = station if isinstance(station, StationConfig) else load_station(station)
    per_seed: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    for seed in seeds:
        plant = build_plant(cfg, start, periods, freq, seed=seed)
        controller = controller_factory(cfg)
        result = run_episode(plant, controller)
        for k in METRIC_KEYS:
            per_seed[k].append(float(result.summary.get(k, float("nan"))))
    stats = {k: summarize(v) for k, v in per_seed.items()}
    return ScenarioSuiteResult(
        station=cfg.site.id, controller=controller_name, seeds=list(seeds), per_seed=per_seed, stats=stats
    )


def _rl_controller_factory(checkpoint_path: Path, guarded: bool) -> Callable[[StationConfig], Any]:
    """Build a factory for a trained checkpoint's controller, guarded or not.

    Imports torch-dependent modules lazily, at call time, so this module
    stays importable (and every rule-based scenario runnable) in an
    environment without the `rl` extra installed.
    """

    def factory(cfg: StationConfig) -> Any:
        from allotrope.evaluate import load_checkpoint
        from allotrope.agents.hybrid import HybridAgent
        from allotrope.safety.fallback import GuardedController

        dqn, sddpg, _ = load_checkpoint(checkpoint_path)
        hybrid = HybridAgent(cfg, dqn, sddpg, deterministic=True)
        # enforce_latency_budget=False: an offline scenario replay, not a real
        # control loop -- the evaluation machine's CPU scheduling must not
        # change which seeds a policy is scored as having handled well.
        return GuardedController(cfg, agent=hybrid, enforce_latency_budget=False) if guarded else hybrid

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--station", default="maitri")
    parser.add_argument("--seeds", type=int, default=200, help="number of independent seeds, 0..N-1")
    parser.add_argument("--seed-offset", type=int, default=0, help="first seed (lets held-out ranges avoid training seeds)")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--periods", type=int, default=8760)
    parser.add_argument("--freq", default="1h")
    parser.add_argument("--checkpoint", default=None, help="optional trained checkpoint to include")
    parser.add_argument("--out", default=None, help="write machine-readable results as JSON to this path")
    args = parser.parse_args()

    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    controllers: list[tuple[str, Callable[[StationConfig], Any]]] = [
        ("legacy_n_plus_one", LegacyNPlusOne),
        ("efficient_rule_based", EfficientRuleBased),
    ]
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        controllers.append(("hybrid_safe", _rl_controller_factory(checkpoint_path, guarded=True)))
        controllers.append(("hybrid_unsafe", _rl_controller_factory(checkpoint_path, guarded=False)))

    results: list[ScenarioSuiteResult] = []
    for name, factory in controllers:
        print(f"running {args.seeds} seeds for {name}...")
        results.append(
            run_scenario_suite(args.station, factory, name, seeds, args.start, args.periods, args.freq)
        )

    print(f"\n{args.station}  |  {len(seeds)} seeds ({seeds[0]}..{seeds[-1]})  |  {args.periods} steps at {args.freq}")
    print("=" * 100)
    headline = ["fuel_kl", "genset_starts", "critical_unserved_kwh", "freeze_violation_steps"]
    for r in results:
        print(f"\n{r.controller}")
        for key in headline:
            s = r.stats[key]
            print(
                f"  {key:26s} mean={s['mean']:10.3f}  median={s['median']:10.3f}  std={s['std']:9.3f}"
                f"  p5={s['p5']:10.3f}  p95={s['p95']:10.3f}  worst={s['max']:10.3f}"
            )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "station": args.station,
                    "seeds": seeds,
                    "periods": args.periods,
                    "freq": args.freq,
                    "results": [r.as_dict() for r in results],
                },
                indent=2,
            )
        )
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()


__all__ = ["ScenarioSuiteResult", "run_scenario_suite", "summarize", "METRIC_KEYS"]
