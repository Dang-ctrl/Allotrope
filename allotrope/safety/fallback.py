"""The deterministic fallback, and the guard that decides when to use it.

The projection layer in `projection.py` bounds what an action may be. This module
handles the other failure mode: the agent producing no usable action at all --
raising, hanging, or returning tensors full of NaN after a bad update or a
corrupted checkpoint.

The fallback is deliberately the dullest code in the repository. It has no
learned parameters, no state, no allocation of consequence and no branch that can
loop. It cannot be improved by training and it cannot be broken by it. Its only
job is to keep a station alive in a condition nobody anticipated.

A note on scope, so the claim is not overstated. The station-level fallback here
is *dispatch* logic: which sets turn, how storage is used, what discretionary
load runs. The inverter-level Volt-VAr and Volt-Watt curves that complete the
picture act on voltage, and voltage does not exist in a power-balance model --
they arrive with the OpenDSS network twin, at which point they sit underneath
this layer rather than replacing it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np

from allotrope.config import StationConfig
from allotrope.safety.projection import SafetyProjection, SafetyReport
from allotrope.safety.voltage import InverterVoltageLayer, VoltageReport
from allotrope.sim.plant import DispatchCommand, PolarMicrogrid


class FallbackReason(str, Enum):
    """Why control was taken away from the agent on a given step."""

    AGENT_RAISED = "agent_raised_exception"
    AGENT_TIMED_OUT = "agent_exceeded_latency_budget"
    AGENT_RETURNED_INVALID = "agent_returned_invalid_command"
    AGENT_ABSENT = "no_agent_configured"


@dataclass
class DeterministicFallback:
    """Minimum viable dispatch. No learning, no state, no way to fail.

    The strategy is the one an engineer would write on a whiteboard: commit the
    fewest sets that cover the firm load with reserve, load them evenly, hold
    storage still, and melt at the rate the day requires. It is not efficient.
    It is not supposed to be. It is supposed to be correct on the worst day of
    the year with a corrupted model file.
    """

    cfg: StationConfig
    name: str = "deterministic_fallback"

    def reset(self) -> None:
        return None

    def act(self, observation: dict, plant: PolarMicrogrid) -> DispatchCommand:
        cfg = self.cfg
        firm_kw = observation["electrical_load_kw"] + cfg.criticality.reserve_margin_kw

        # Commit largest-first until the firm load is covered.
        order = sorted(range(len(cfg.gensets)), key=lambda k: -cfg.gensets[k].rated_kw)
        on = [False] * len(cfg.gensets)
        covered_kw = 0.0
        for k in order:
            if covered_kw >= firm_kw:
                break
            on[k] = True
            covered_kw += cfg.gensets[k].rated_kw
        if not any(on):
            on[order[0]] = True

        # Share the load evenly across whatever is committed.
        committed = [k for k, flag in enumerate(on) if flag]
        share_kw = observation["electrical_load_kw"] / max(len(committed), 1)
        setpoints = [
            float(np.clip(share_kw, cfg.gensets[k].min_stable_kw, cfg.gensets[k].rated_kw))
            if on[k]
            else 0.0
            for k in range(len(cfg.gensets))
        ]

        # Storage is held still. A fallback does not speculate with the reserve.
        battery = [0.0] * len(cfg.storage)

        # Melt whatever the day still owes, spread over the hours that remain.
        i = min(plant.state.step_index, plant.n_steps - 1)
        hours_left = max(24.0 - plant.index[i].hour, plant.dt_h)
        melt_kw = plant.state.snow_melt_remaining_kwh / hours_left

        return DispatchCommand(
            genset_on=tuple(on),
            genset_setpoint_kw=tuple(setpoints),
            battery_kw=tuple(battery),
            snow_melt_kw=melt_kw,
        )


class Agent(Protocol):
    name: str

    def act(self, observation: dict, plant: PolarMicrogrid) -> DispatchCommand: ...


@dataclass
class GuardStats:
    """How often the guard had to intervene, and why. Reported to the HMI."""

    steps: int = 0
    fallbacks: int = 0
    projections: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    max_latency_ms: float = 0.0

    def note_fallback(self, reason: FallbackReason) -> None:
        self.fallbacks += 1
        self.reasons[reason.value] = self.reasons.get(reason.value, 0) + 1

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / max(self.steps, 1)

    @property
    def projection_rate(self) -> float:
        return self.projections / max(self.steps, 1)


@dataclass
class GuardedController:
    """Runs an agent behind the fallback and the projection layer.

    This is the object that would actually be deployed. Nothing reaches the
    plant without passing both guards, so the safety argument does not depend on
    which agent is loaded, on how well it was trained, or on whether its weights
    survived the trip over the satellite link intact.
    """

    cfg: StationConfig
    agent: Any | None = None
    latency_budget_ms: float = 10.0
    """A late answer is a wrong answer: the gRPC control path budgets 10 ms."""
    inverter_layer: InverterVoltageLayer | None = None
    """Inverter-level Volt-Watt curtailment (allotrope.safety.voltage), for a
    station with a network model. None (the default) reproduces this class's
    behaviour from before this layer existed exactly -- every existing
    caller that doesn't pass one is unaffected. Build one with
    `allotrope.safety.voltage.build_inverter_layer(cfg)`."""

    name: str = "guarded"
    stats: GuardStats = field(default_factory=GuardStats)
    last_report: SafetyReport | None = None
    last_fallback_reason: FallbackReason | None = None
    last_voltage_report: VoltageReport | None = None

    def __post_init__(self) -> None:
        self.fallback = DeterministicFallback(self.cfg)
        self.projection = SafetyProjection(self.cfg)
        if self.agent is not None:
            self.name = f"guarded_{getattr(self.agent, 'name', type(self.agent).__name__)}"

    def reset(self) -> None:
        self.stats = GuardStats()
        self.last_report = None
        self.last_fallback_reason = None
        self.last_voltage_report = None
        for target in (self.agent, self.fallback):
            if target is not None and hasattr(target, "reset"):
                target.reset()

    def act(self, observation: dict, plant: PolarMicrogrid) -> DispatchCommand:
        self.stats.steps += 1
        command, reason = self._propose(observation, plant)

        if reason is not None:
            self.stats.note_fallback(reason)
            command = self.fallback.act(observation, plant)
        self.last_fallback_reason = reason

        safe, report = self.projection.project(command, observation, plant)
        self.last_report = report
        if report.intervened:
            self.stats.projections += 1

        if self.inverter_layer is not None:
            safe, voltage_report = self.inverter_layer.apply(safe, observation)
            self.last_voltage_report = voltage_report

        return safe

    def _propose(
        self, observation: dict, plant: PolarMicrogrid
    ) -> tuple[DispatchCommand | None, FallbackReason | None]:
        """Ask the agent, and decide whether the answer can be used."""
        if self.agent is None:
            return None, FallbackReason.AGENT_ABSENT

        start = time.perf_counter()
        try:
            command = self.agent.act(observation, plant)
        except Exception:
            # A controller that crashes with the station is no controller at all.
            return None, FallbackReason.AGENT_RAISED

        latency_ms = (time.perf_counter() - start) * 1000.0
        self.stats.max_latency_ms = max(self.stats.max_latency_ms, latency_ms)
        if latency_ms > self.latency_budget_ms:
            return None, FallbackReason.AGENT_TIMED_OUT
        if not self._is_well_formed(command):
            return None, FallbackReason.AGENT_RETURNED_INVALID
        return command, None

    def _is_well_formed(self, command: Any) -> bool:
        """Structural check only; the projection handles the numeric bounds."""
        if not isinstance(command, DispatchCommand):
            return False
        try:
            if len(command.genset_on) != len(self.cfg.gensets):
                return False
            if len(command.genset_setpoint_kw) != len(self.cfg.gensets):
                return False
            if len(command.battery_kw) != len(self.cfg.storage):
                return False
            float(command.snow_melt_kw)
        except (TypeError, ValueError):
            return False
        return True


__all__ = ["DeterministicFallback", "GuardedController", "GuardStats", "FallbackReason"]
