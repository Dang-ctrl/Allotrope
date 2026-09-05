"""One station's local update, and the validation gate a global model must
pass before a federated round is accepted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from allotrope.config import load_station
from allotrope.train import train

DEFAULT_VALIDATION_PERIODS = 24 * 14
"""Two weeks per station, per validation run -- enough to see genset cycling
and at least one cold snap, short enough that validating every round stays
cheap relative to the local training it's gating."""


@dataclass
class LocalUpdateResult:
    """What one station's local round produced."""

    station: str
    steps: int
    seed: int
    checkpoint_path: str
    mean_episode_return_last10: float | None

    def as_dict(self) -> dict:
        return {
            "station": self.station,
            "steps": self.steps,
            "seed": self.seed,
            "checkpoint_path": self.checkpoint_path,
            "mean_episode_return_last10": self.mean_episode_return_last10,
        }


@dataclass
class ValidationResult:
    """Whether an aggregated global checkpoint may become the round's output."""

    accepted: bool
    reason: str
    per_station: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"accepted": self.accepted, "reason": self.reason, "per_station": self.per_station}


def run_local_update(
    station: str,
    steps: int,
    seed: int,
    episode_steps: int,
    warmup_steps: int,
    buffer_capacity: int,
    init_checkpoint: Path | None,
    runs_dir: Path,
) -> LocalUpdateResult:
    """Train one station locally, optionally warm-started from the current global model.

    This *is* the "local dataset" in this project's federation: each
    station's own `PolarMicrogridEnv`, built from its own synthetic weather
    and demand generators (`allotrope.synth`). Nothing about that
    environment or the transitions it produces crosses a station boundary
    -- only the resulting checkpoint does, and only into
    `allotrope.federated.aggregate.fedavg_checkpoint`, which touches
    tensors, never observations.
    """
    run_dir = train(
        agent_kind="hybrid",
        station=station,
        total_steps=steps,
        seed=seed,
        episode_steps=episode_steps,
        warmup_steps=warmup_steps,
        buffer_capacity=buffer_capacity,
        runs_dir=runs_dir,
        init_checkpoint=init_checkpoint,
    )
    record = json.loads((run_dir / "record.json").read_text())
    return LocalUpdateResult(
        station=station,
        steps=steps,
        seed=seed,
        checkpoint_path=str(run_dir / "checkpoint.pt"),
        mean_episode_return_last10=record["metrics"].get("mean_episode_return_last10"),
    )


def default_validator(
    global_checkpoint_path: Path,
    stations: list[str],
    validation_seed: int = 999,
    periods: int = DEFAULT_VALIDATION_PERIODS,
) -> ValidationResult:
    """Run the guarded global model on a held-out seed for every participating station.

    The one hard gate: `critical_unserved_kwh` must come back exactly zero
    on every station. Stated plainly, this is not really testing whether
    *aggregation* is safe -- `allotrope.safety.projection.SafetyProjection`
    has no learned parameters at all, so no amount of federated averaging
    can touch the guarantee it makes (see `allotrope.federated`'s package
    docstring). It's measured here anyway, the same way every other safety
    claim in this project is measured rather than assumed, and because a
    validator that skipped the one check that could never plausibly fail
    would be a validator no one could trust to catch the one that could.

    A weaker, non-fatal check also runs: fuel use per station must not
    exceed the incumbent `LegacyNPlusOne` baseline's, on the theory that a
    federated round that regresses below the very baseline this project
    exists to beat isn't a round worth keeping either, even though it
    can't have compromised safety.
    """
    from allotrope.agents.hybrid import HybridAgent
    from allotrope.control.baseline import LegacyNPlusOne
    from allotrope.evaluate import load_checkpoint
    from allotrope.safety.fallback import GuardedController
    from allotrope.sim.runner import build_plant, run_episode

    dqn, sddpg, _ = load_checkpoint(global_checkpoint_path)
    per_station: dict[str, dict[str, float]] = {}
    reasons: list[str] = []

    for station in stations:
        cfg = load_station(station)
        hybrid = HybridAgent(cfg, dqn, sddpg, deterministic=True)
        guard = GuardedController(cfg, agent=hybrid)
        plant = build_plant(cfg, periods=periods, seed=validation_seed)
        result = run_episode(plant, guard)
        summary = result.summary

        legacy_plant = build_plant(cfg, periods=periods, seed=validation_seed)
        legacy_summary = run_episode(legacy_plant, LegacyNPlusOne(cfg)).summary

        per_station[station] = {
            "critical_unserved_kwh": summary["critical_unserved_kwh"],
            "fuel_kl": summary["fuel_kl"],
            "genset_starts": summary["genset_starts"],
            "legacy_fuel_kl": legacy_summary["fuel_kl"],
        }
        if summary["critical_unserved_kwh"] > 1e-6:
            reasons.append(
                f"{station}: critical_unserved_kwh={summary['critical_unserved_kwh']:.6f} (safety gate)"
            )
        elif summary["fuel_kl"] > legacy_summary["fuel_kl"]:
            reasons.append(
                f"{station}: fuel {summary['fuel_kl']:.1f} kL exceeds legacy baseline "
                f"{legacy_summary['fuel_kl']:.1f} kL (performance gate)"
            )

    accepted = not reasons
    return ValidationResult(accepted=accepted, reason="ok" if accepted else "; ".join(reasons), per_station=per_station)


__all__ = [
    "LocalUpdateResult",
    "ValidationResult",
    "run_local_update",
    "default_validator",
    "DEFAULT_VALIDATION_PERIODS",
]
