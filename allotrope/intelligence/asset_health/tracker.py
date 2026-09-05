"""Asset health tracking, built entirely on state the simulator already computes.

The plant (`allotrope.sim.plant.PolarMicrogrid`) already carries the physical
memory that matters for maintenance: a genset's cumulative starts and run hours
live on `GensetState`, its exhaust fouling lives in `GensetState.deposit`
(`allotrope.sim.assets.Genset._update_deposit`), and a battery's state of
charge and cumulative throughput live on `BatteryState`. This module adds no
parallel physics. It watches the plant's own per-step telemetry (the dict
`PolarMicrogrid.step()` returns) and accumulates it into per-asset health
records, plus one composite "wear score" per genset that is explicitly a
weighted combination of measured quantities -- not a probability of failure.

Every metric this module exposes carries a `MetricLabel` saying exactly how
trustworthy it is:

  MEASURED  -- read directly from simulator state (e.g. `battery_soc`).
  MODELED   -- computed by a physical/engineering model already in this
               codebase (e.g. genset exhaust deposit, from `assets.py`).
  PROXY     -- a stand-in for something the simulator does not model directly
               (e.g. the wear score: higher for more-stressed equipment, but
               not a calibrated measurement of anything physical).
  ESTIMATED -- a derived statistic computed from measured/modeled quantities
               using a named, standard formula (e.g. full-equivalent-cycles).

Nothing here computes a failure probability, a remaining-useful-life estimate,
or any other number that implies knowledge this project has no evidence for.
See `docs/asset-health.md` for the full labeling and what is explicitly not
done in this pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from allotrope.config import StationConfig


class MetricLabel(str, Enum):
    """How much epistemic weight a metric can bear. See module docstring."""

    MEASURED = "measured"
    MODELED = "modeled"
    PROXY = "proxy"
    ESTIMATED = "estimated"


@dataclass(frozen=True)
class Metric:
    """One labeled number. `value` is never presented without `label` attached."""

    value: float
    label: MetricLabel
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "label": self.label.value, "note": self.note}


# Low-load threshold for "harmful low-load operation": reused, not invented.
#
# `GensetSpec.wet_stack_threshold_frac` is already the load fraction below
# which `Genset._update_deposit` (allotrope/sim/assets.py) starts accruing
# exhaust deposit -- the project's own physical definition of harmful
# part-load running. Defining "low-load hours" against any other threshold
# would create a second, competing notion of "harmful" that disagrees with
# the deposit model this tracker also reports. So low-load hours are simply
# the hours a genset spends online below its own wet-stacking threshold.

# Wear-score weights: reused, not invented.
#
# `allotrope.envs.reward.RewardWeights` already prices `genset_start_per_event`
# and `deposit_growth_per_unit` in rupees, as the reward function's own stated
# valuation of a cold start ("maintenance life consumed... amortised") and of
# fouling ("fouling accrued over a full deposit range, as deferred
# maintenance"). Reusing those two weights to combine starts and deposit into
# one number keeps the wear score consistent with the economic weighting the
# rest of this project already trusts, instead of inventing a fresh set of
# weights with no grounding. Low-load hours are reported separately (below)
# rather than folded into the score a second time, because they are already
# the mechanism by which deposit accumulates -- adding them again would
# double-count the same physical effect.
DEFAULT_START_WEIGHT = 1500.0
"""Rupees per start, from `RewardWeights.genset_start_per_event`."""
DEFAULT_DEPOSIT_WEIGHT = 8000.0
"""Rupees per full [0, 1] unit of deposit, from `RewardWeights.deposit_growth_per_unit`."""

# SOC-extreme band: the last 5 percentage points against each bound of a
# battery's own configured envelope (`StorageSpec.soc_min` / `soc_max`).
# Chosen because that envelope is already the chemistry-specific limit this
# project enforces (`Battery.max_charge_kw` / `max_discharge_kw`); operating
# within 5 points of it is the operating band those functions already
# derate hardest, not an arbitrary percentage of full range.
SOC_EXTREME_BAND = 0.05


@dataclass
class GensetHealth:
    """Accumulated health record for one genset, labeled field by field."""

    id: str
    rated_kw: float
    wet_stack_threshold_frac: float
    run_hours: float = 0.0
    starts: int = 0
    low_load_hours: float = 0.0
    deposit: float = 0.0
    _prev_online: bool = False

    def update(self, online: bool, load_frac: float, deposit: float, dt_h: float) -> None:
        """Ingest one step's telemetry for this genset.

        `online` and `load_frac` come straight off the plant's per-step
        `genset_online` / `genset_load_frac` telemetry lists; `deposit` off
        `genset_deposit`. A start is detected as an off-to-on transition,
        which is exactly how `Genset.set_commitment` records one internally
        (anti-cycling means a set cannot re-start without first going
        through `online=False`), so this reproduces the simulator's own
        `total_starts` count without needing a second telemetry field for it.
        """
        if online and not self._prev_online:
            self.starts += 1
        if online:
            self.run_hours += dt_h
            if load_frac < self.wet_stack_threshold_frac:
                self.low_load_hours += dt_h
        self.deposit = deposit
        self._prev_online = online

    def wear_score(
        self, start_weight: float = DEFAULT_START_WEIGHT, deposit_weight: float = DEFAULT_DEPOSIT_WEIGHT
    ) -> Metric:
        """A PROXY maintenance-proximity indicator, not a failure probability.

        `wear = start_weight * cumulative_starts + deposit_weight * deposit`.
        Higher means more stressed, in the same rupee-scaled units the
        project's own reward function already uses to price these two
        effects. It carries no claim about time-to-failure or probability of
        failure -- this project has no evidence to support either, and the
        house rule against fabricated numbers forbids inventing them.
        """
        value = start_weight * self.starts + deposit_weight * self.deposit
        return Metric(
            value=value,
            label=MetricLabel.PROXY,
            note=(
                f"{start_weight:.0f}*starts + {deposit_weight:.0f}*deposit; "
                "higher = more maintenance-relevant stress observed so far, "
                "not a failure probability or remaining-useful-life estimate."
            ),
        )

    def report(self) -> dict[str, Metric]:
        return {
            "run_hours": Metric(self.run_hours, MetricLabel.MEASURED, "sum of dt_h while online"),
            "starts": Metric(
                float(self.starts), MetricLabel.MEASURED, "off-to-on transitions observed"
            ),
            "low_load_hours": Metric(
                self.low_load_hours,
                MetricLabel.MEASURED,
                f"hours online below wet_stack_threshold_frac={self.wet_stack_threshold_frac:.2f}",
            ),
            "deposit": Metric(
                self.deposit,
                MetricLabel.MODELED,
                "exhaust fouling proxy in [0, 1] from Genset._update_deposit",
            ),
            "wear_score": self.wear_score(),
        }


@dataclass
class BatteryHealth:
    """Accumulated health record for one battery, labeled field by field."""

    id: str
    capacity_kwh: float
    soc_min: float
    soc_max: float
    throughput_kwh: float = 0.0
    soc: float = 0.5
    low_soc_hours: float = 0.0
    high_soc_hours: float = 0.0
    cold_charge_blocks: int = 0

    def update(self, power_kw: float, soc: float, dt_h: float, cold_charge_blocked: bool) -> None:
        """Ingest one step's telemetry for this battery.

        `power_kw` and `soc` come off the plant's per-step `battery_kw` /
        `battery_soc` telemetry lists. Throughput accumulates the same
        `abs(delivered) * dt_h` quantity `Battery.step` already tracks on
        `BatteryState.throughput_kwh`, so the full-equivalent-cycle estimate
        below is derived from the identical simulated energy flow.
        """
        self.throughput_kwh += abs(power_kw) * dt_h
        self.soc = soc
        low_bound = self.soc_min + SOC_EXTREME_BAND * (self.soc_max - self.soc_min)
        high_bound = self.soc_max - SOC_EXTREME_BAND * (self.soc_max - self.soc_min)
        if soc <= low_bound:
            self.low_soc_hours += dt_h
        if soc >= high_bound:
            self.high_soc_hours += dt_h
        if cold_charge_blocked:
            self.cold_charge_blocks += 1

    def full_equivalent_cycles(self) -> Metric:
        """FEC = cumulative throughput / (2 * nameplate capacity).

        Standard proxy for cycle count under variable-depth cycling: two full
        capacity-worths of throughput (one charge, one discharge) count as one
        equivalent full cycle. It is an ESTIMATED statistic, not a measured
        cycle count -- the simulator never labels a "cycle" as such.
        """
        value = self.throughput_kwh / max(2.0 * self.capacity_kwh, 1e-9)
        return Metric(
            value,
            MetricLabel.ESTIMATED,
            "throughput_kwh / (2 * capacity_kwh); standard full-equivalent-cycle proxy",
        )

    def report(self) -> dict[str, Metric]:
        return {
            "soc": Metric(self.soc, MetricLabel.MEASURED, "state of charge at end of episode"),
            "throughput_kwh": Metric(
                self.throughput_kwh, MetricLabel.MEASURED, "sum of abs(power_kw) * dt_h"
            ),
            "full_equivalent_cycles": self.full_equivalent_cycles(),
            "low_soc_hours": Metric(
                self.low_soc_hours,
                MetricLabel.MEASURED,
                f"hours at/below soc_min + {SOC_EXTREME_BAND:.0%} of envelope",
            ),
            "high_soc_hours": Metric(
                self.high_soc_hours,
                MetricLabel.MEASURED,
                f"hours at/above soc_max - {SOC_EXTREME_BAND:.0%} of envelope",
            ),
            "cold_charge_blocks": Metric(
                float(self.cold_charge_blocks),
                MetricLabel.MEASURED,
                "steps where a requested charge was refused for being too cold",
            ),
        }


class AssetHealthTracker:
    """Accumulates per-asset health from a plant's own per-step telemetry.

    Usage: construct against the `StationConfig` a `PolarMicrogrid` was built
    from (for nameplate ratings and envelopes), then feed it either the raw
    telemetry dict `PolarMicrogrid.step()` returns, or a `pandas.Series` /
    row from `EpisodeResult.telemetry` (which flattens the same lists into
    `..._0`, `..._1`, ... columns) via `update`. `report()` returns the
    accumulated, labeled record for every genset and battery.

    This tracker does not touch dispatch. It is read-only observation of a
    plant that runs exactly as it would without it.
    """

    def __init__(self, cfg: StationConfig, dt_h: float) -> None:
        self.dt_h = dt_h
        self.gensets: dict[str, GensetHealth] = {
            g.id: GensetHealth(id=g.id, rated_kw=g.rated_kw, wet_stack_threshold_frac=g.wet_stack_threshold_frac)
            for g in cfg.gensets
        }
        self._genset_ids = [g.id for g in cfg.gensets]
        self.batteries: dict[str, BatteryHealth] = {
            s.id: BatteryHealth(
                id=s.id, capacity_kwh=s.capacity_kwh, soc_min=s.soc_min, soc_max=s.soc_max
            )
            for s in cfg.storage
        }
        self._battery_ids = [s.id for s in cfg.storage]
        self._prev_cold_charge_blocks: list[int] | None = None

    def update(self, telemetry: dict[str, Any]) -> None:
        """Ingest one step of raw `PolarMicrogrid.step()` telemetry (list-valued fields)."""
        online = telemetry["genset_online"]
        load_frac = telemetry["genset_load_frac"]
        deposit = telemetry["genset_deposit"]
        for k, gid in enumerate(self._genset_ids):
            self.gensets[gid].update(bool(online[k]), float(load_frac[k]), float(deposit[k]), self.dt_h)

        battery_kw = telemetry["battery_kw"]
        battery_soc = telemetry["battery_soc"]
        for k, bid in enumerate(self._battery_ids):
            # The per-step telemetry does not carry a cold-charge-block flag
            # (that lives only in the plant's cumulative `summary()`), so this
            # tracker does not attribute a per-step block; `update_from_episode`
            # reconciles the cumulative count directly from the plant instead.
            self.batteries[bid].update(float(battery_kw[k]), float(battery_soc[k]), self.dt_h, False)

    def update_from_records(self, records: Iterable[dict[str, Any]]) -> None:
        """Ingest a sequence of raw telemetry dicts, e.g. one per `plant.step()` call."""
        for record in records:
            self.update(record)

    def reconcile_cold_charge_blocks(self, plant: Any) -> None:
        """Pull the plant's own cumulative cold-charge-block count onto each battery.

        `BatteryState.cold_charge_blocks` is incremented inside `Battery.step`
        and is not part of the per-step telemetry dict, so it cannot be
        accumulated from `update()` alone. Call this once after an episode
        (or periodically) against the live `PolarMicrogrid` to pull the
        simulator's own count rather than re-deriving it.
        """
        for battery, bid in zip(plant.batteries, self._battery_ids):
            self.batteries[bid].cold_charge_blocks = battery.state.cold_charge_blocks

    def report(self) -> dict[str, dict[str, Any]]:
        """The full labeled health record: `{"gensets": {...}, "batteries": {...}}`."""
        return {
            "gensets": {
                gid: {k: m.as_dict() for k, m in health.report().items()}
                for gid, health in self.gensets.items()
            },
            "batteries": {
                bid: {k: m.as_dict() for k, m in health.report().items()}
                for bid, health in self.batteries.items()
            },
        }


__all__ = ["AssetHealthTracker", "GensetHealth", "BatteryHealth", "Metric", "MetricLabel"]
