"""The safety projection layer: an analytic bound on every action, learned or not.

This is the component that makes a learned controller acceptable at -40 C. A
policy network is a function whose output is not guaranteed to be anything in
particular -- least of all safe -- and no amount of training reward converts a
statistical tendency into a guarantee. So the guarantee is placed outside the
network, where it can be reasoned about:

    agent proposes  ->  projection bounds  ->  plant executes

The projection is analytic and closed-form. It solves no optimisation problem,
calls no solver, and cannot itself fail to converge, because a safety layer that
can time out is not a safety layer. It runs in microseconds and its behaviour can
be checked by reading it.

What it guarantees, for *any* input including adversarial or malformed ones:

  1. committed generating capacity always covers life support plus reserve;
  2. no set is stopped if stopping it would breach that cover;
  3. battery commands stay inside the envelope the cells can actually accept,
     which in a polar winter is not the nameplate envelope;
  4. discretionary load never displaces critical load;
  5. the station's heat supply is never left short of what the envelope needs.

Every intervention is recorded rather than applied silently. An engineer at the
station sees exactly which bound bit and why, which is the difference between a
controller that can be trusted and one that merely works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from allotrope.config import StationConfig
from allotrope.sim.plant import DispatchCommand, PolarMicrogrid


class Intervention(str, Enum):
    """Why the projection changed an action. Surfaced to the operator HMI."""

    SANITISED_NAN = "sanitised_non_finite_action"
    FORCED_START = "forced_start_for_capacity"
    BLOCKED_STOP = "blocked_stop_that_would_breach_reserve"
    CLIPPED_SETPOINT = "clipped_setpoint_to_machine_limits"
    RAISED_SETPOINT = "raised_setpoint_to_cover_critical_load"
    CLIPPED_BATTERY = "clipped_battery_to_thermal_envelope"
    LIMITED_CHARGE = "limited_charging_to_protect_critical_load"
    CLIPPED_MELT = "clipped_discretionary_load"
    SHED_MELT_FOR_CRITICAL = "shed_discretionary_load_for_critical"
    FORCED_START_FOR_HEAT = "forced_start_to_protect_heating"
    RAISED_SETPOINT_FOR_HEAT = "raised_setpoint_to_cover_heat_shortfall"


@dataclass
class SafetyReport:
    """What the projection did to one action, and why."""

    interventions: list[Intervention] = field(default_factory=list)
    required_capacity_kw: float = 0.0
    committed_capacity_kw: float = 0.0
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def intervened(self) -> bool:
        return bool(self.interventions)

    def record(self, intervention: Intervention) -> None:
        if intervention not in self.interventions:
            self.interventions.append(intervention)

    def as_dict(self) -> dict[str, object]:
        return {
            "intervened": self.intervened,
            "interventions": [i.value for i in self.interventions],
            "required_capacity_kw": self.required_capacity_kw,
            "committed_capacity_kw": self.committed_capacity_kw,
            **self.detail,
        }


class SafetyProjection:
    """Projects a proposed dispatch onto the set of actions that cannot harm the station.

    The projection is deliberately conservative in one direction only. It will
    start machines the agent did not ask for and refuse to stop machines the
    agent wanted stopped; it will never stop a machine on its own initiative,
    because every failure mode worth protecting against at a polar station is a
    failure of supply rather than an excess of it.
    """

    def __init__(self, cfg: StationConfig, melt_ceiling_multiple: float = 4.0) -> None:
        self.cfg = cfg
        self.melt_ceiling_multiple = melt_ceiling_multiple

    # -- public -----------------------------------------------------------

    def project(
        self, command: DispatchCommand, observation: dict, plant: PolarMicrogrid
    ) -> tuple[DispatchCommand, SafetyReport]:
        """Return a safe command and a record of every bound that bit."""
        report = SafetyReport()

        genset_on = self._sanitise_bools(command.genset_on, report)
        setpoints = self._sanitise_floats(
            command.genset_setpoint_kw, len(self.cfg.gensets), report
        )
        battery = self._sanitise_floats(command.battery_kw, len(self.cfg.storage), report)
        melt = self._sanitise_scalar(command.snow_melt_kw, report)

        required_kw = self._required_capacity_kw(observation)
        report.required_capacity_kw = required_kw

        genset_on = self._enforce_capacity(genset_on, observation, required_kw, report)
        genset_on = self._enforce_heat(genset_on, observation, report)
        setpoints = self._bound_setpoints(setpoints, genset_on, required_kw, observation, report)
        setpoints = self._raise_setpoints_for_heat(setpoints, genset_on, observation, report)
        battery = self._bound_battery(battery, genset_on, setpoints, observation, report)
        melt = self._bound_melt(melt, genset_on, setpoints, battery, observation, report)

        report.committed_capacity_kw = self._capacity_kw(genset_on, observation)
        recovered_heat_kw = sum(
            setpoints[k] * self.cfg.gensets[k].chp_heat_ratio for k in range(len(self.cfg.gensets))
        )
        heat_shortfall_kw = observation["firm_thermal_kw"] - self.cfg.thermal.boiler_rated_kw
        report.detail.update(
            {
                "critical_load_kw": observation["critical_load_kw"],
                "reserve_margin_kw": self.cfg.criticality.reserve_margin_kw,
                "indoor_temp_c": observation["indoor_temp_c"],
                # >0 means even the fleet at its raised setpoints, fully committed,
                # cannot recover enough heat -- a physical CHP-capacity shortfall
                # this layer cannot manufacture away, surfaced rather than hidden.
                "unmet_heat_shortfall_kw": max(heat_shortfall_kw - recovered_heat_kw, 0.0),
            }
        )

        return (
            DispatchCommand(
                genset_on=tuple(genset_on),
                genset_setpoint_kw=tuple(setpoints),
                battery_kw=tuple(battery),
                snow_melt_kw=melt,
            ),
            report,
        )

    def melt_ceiling_kw(self) -> float:
        """The physical limit of the melters, independent of what is asked of them."""
        therm = self.cfg.thermal
        peak_daily_kwh = (
            self.cfg.occupancy.summer_crew
            * therm.water_l_per_person_day
            * therm.snow_melt_kwh_per_l
        )
        return self.melt_ceiling_multiple * peak_daily_kwh / 24.0

    # -- sanitising -------------------------------------------------------

    def _sanitise_bools(self, values, report: SafetyReport) -> list[bool]:
        """A policy network can emit anything. Treat non-finite as 'off' and move on."""
        out = []
        for v in values:
            try:
                out.append(bool(v) if np.isfinite(float(v)) else self._flag_nan(report))
            except (TypeError, ValueError):
                out.append(self._flag_nan(report))
        while len(out) < len(self.cfg.gensets):
            out.append(self._flag_nan(report))
        return out[: len(self.cfg.gensets)]

    def _sanitise_floats(self, values, width: int, report: SafetyReport) -> list[float]:
        out = []
        for v in values:
            out.append(self._sanitise_scalar(v, report))
        while len(out) < width:
            out.append(self._sanitise_scalar(float("nan"), report))
        return out[:width]

    def _sanitise_scalar(self, value, report: SafetyReport) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            report.record(Intervention.SANITISED_NAN)
            return 0.0
        if not np.isfinite(v):
            report.record(Intervention.SANITISED_NAN)
            return 0.0
        return v

    @staticmethod
    def _flag_nan(report: SafetyReport) -> bool:
        report.record(Intervention.SANITISED_NAN)
        return False

    # -- capacity ---------------------------------------------------------

    def _required_capacity_kw(self, observation: dict) -> float:
        """Firm capacity the station must have turning, whatever the agent believes.

        Renewables are deliberately excluded. Committing against the wind is
        exactly the mistake that leaves a station dark when the wind drops, and
        no forecast available in Antarctica is good enough to justify it.
        """
        crit = self.cfg.criticality
        return observation["critical_load_kw"] + crit.reserve_margin_kw

    def _capacity_kw(self, genset_on: list[bool], observation: dict) -> float:
        """Capacity that will actually be turning under a proposed commitment.

        This is evaluated against the whole commitment at once, never one set at
        a time. Checking stops individually is unsound: with two sets online and
        both commanded off, each stop looks safe because the other set is still
        running, and the plant ends up with nothing turning at all.
        """
        return sum(
            g.rated_kw
            for k, g in enumerate(self.cfg.gensets)
            if self._effective_online(genset_on, observation)[k]
        )

    def _enforce_capacity(
        self, genset_on: list[bool], observation: dict, required_kw: float, report: SafetyReport
    ) -> list[bool]:
        """Keep enough plant turning to cover life support and its reserve.

        Cover is restored in the order that costs the station least: first by
        cancelling stops, since a set that is already turning costs nothing to
        keep, and only then by starting cold machines.
        """
        if self._capacity_kw(genset_on, observation) >= required_kw:
            return genset_on

        # Cancel stops, largest set first.
        for k in sorted(range(len(self.cfg.gensets)), key=lambda k: -self.cfg.gensets[k].rated_kw):
            if self._capacity_kw(genset_on, observation) >= required_kw:
                return genset_on
            if observation["genset_online"][k] and not genset_on[k]:
                genset_on[k] = True
                report.record(Intervention.BLOCKED_STOP)

        # Then start cold sets, largest first.
        for k in sorted(range(len(self.cfg.gensets)), key=lambda k: -self.cfg.gensets[k].rated_kw):
            if self._capacity_kw(genset_on, observation) >= required_kw:
                break
            if not genset_on[k]:
                genset_on[k] = True
                report.record(Intervention.FORCED_START)
        return genset_on

    def _enforce_heat(
        self, genset_on: list[bool], observation: dict, report: SafetyReport
    ) -> list[bool]:
        """Guarantee the heat supply, not merely the electricity.

        The boilers carry most of the heating, so this bound rarely binds. When
        it does -- a deep cold snap with firm heat demand beyond boiler rating --
        recovered heat is the only thing standing between the station and a
        falling indoor temperature, and a set must be turning to recover any.
        """
        therm = self.cfg.thermal
        firm_thermal_kw = observation["firm_thermal_kw"]
        shortfall_kw = firm_thermal_kw - therm.boiler_rated_kw
        if shortfall_kw <= 0.0:
            return genset_on

        order = sorted(range(len(self.cfg.gensets)), key=lambda k: -self.cfg.gensets[k].rated_kw)
        for k in order:
            recovered = sum(
                self.cfg.gensets[j].rated_kw * self.cfg.gensets[j].chp_heat_ratio
                for j in range(len(self.cfg.gensets))
                if genset_on[j]
            )
            if recovered >= shortfall_kw:
                break
            if not genset_on[k]:
                genset_on[k] = True
                report.record(Intervention.FORCED_START_FOR_HEAT)
        return genset_on

    def _raise_setpoints_for_heat(
        self, setpoints: list[float], genset_on: list[bool], observation: dict, report: SafetyReport
    ) -> list[float]:
        """Close the gap `_enforce_heat` cannot see on its own.

        `_enforce_heat` decides which sets must be *committed* using each
        set's rated CHP output, because at that point in the pipeline no
        setpoint has been chosen yet. But `_bound_setpoints` only raises a
        committed set's ceiling far enough to cover the *electrical* load --
        a set sitting at `min_stable_kw` recovers a fraction of what
        `_enforce_heat` assumed available. Left uncorrected, the projection
        reports the heat guarantee satisfied while the station is still
        short: committing a set is not the same as letting it produce, and
        that gap is exactly what this step closes, the same way
        `_bound_setpoints` already closes it for the electrical requirement.
        """
        therm = self.cfg.thermal
        shortfall_kw = observation["firm_thermal_kw"] - therm.boiler_rated_kw
        if shortfall_kw <= 0.0:
            return setpoints

        out = list(setpoints)
        committed = [k for k, on in enumerate(genset_on) if on]
        recovered = sum(out[k] * self.cfg.gensets[k].chp_heat_ratio for k in committed)
        if recovered >= shortfall_kw - 1e-9:
            return out

        # Raise the largest sets first -- fewer machines pushed harder costs
        # less wear than spreading the raise thin across the whole fleet.
        for k in sorted(committed, key=lambda k: -self.cfg.gensets[k].rated_kw):
            if recovered >= shortfall_kw - 1e-9:
                break
            g = self.cfg.gensets[k]
            headroom_kw = g.rated_kw - out[k]
            if headroom_kw <= 1e-9 or g.chp_heat_ratio <= 0.0:
                continue
            heat_needed_kw = shortfall_kw - recovered
            raise_kw = min(headroom_kw, heat_needed_kw / g.chp_heat_ratio)
            out[k] += raise_kw
            recovered += raise_kw * g.chp_heat_ratio
            report.record(Intervention.RAISED_SETPOINT_FOR_HEAT)
        return out

    # -- bounds -----------------------------------------------------------

    def _effective_online(self, genset_on: list[bool], observation: dict) -> list[bool]:
        """Which sets will actually be turning this step, not merely commanded on.

        A set inside its minimum down time is going to produce nothing however
        firmly it is commanded, so every bound that depends on available power
        has to be computed against this mask rather than against the command.
        Confusing the two is how a projection layer comes to believe it has cover
        that does not exist.
        """
        out = []
        for k in range(len(self.cfg.gensets)):
            online = bool(observation["genset_online"][k])
            if genset_on[k]:
                # Commanded on: it runs if it is already turning or may start now.
                out.append(online or bool(observation["genset_can_start"][k]))
            else:
                # Commanded off: it still runs if it is not yet allowed to stop.
                out.append(online and not bool(observation["genset_can_stop"][k]))
        return out

    def _bound_setpoints(
        self,
        setpoints: list[float],
        genset_on: list[bool],
        required_kw: float,
        observation: dict,
        report: SafetyReport,
    ) -> list[float]:
        """Clip to each machine's band, then raise the fleet to cover the firm load.

        Committing a set is not the same as letting it produce. Because the sets
        load-follow only up to their setpoints, a setpoint is a ceiling, and a
        fleet committed at minimum stable load will happily leave life support
        unserved while three machines idle. Capacity without a ceiling to match
        it is not cover, so the ceiling is raised here.
        """
        out = []
        for k, g in enumerate(self.cfg.gensets):
            if not genset_on[k]:
                out.append(0.0)
                continue
            bounded = float(np.clip(setpoints[k], g.min_stable_kw, g.rated_kw))
            if abs(bounded - setpoints[k]) > 1e-9:
                report.record(Intervention.CLIPPED_SETPOINT)
            out.append(bounded)

        committed = [k for k, flag in enumerate(self._effective_online(genset_on, observation)) if flag]
        headroom = sum(self.cfg.gensets[k].rated_kw - out[k] for k in committed)
        deficit = required_kw - sum(out[k] for k in committed)
        if deficit > 1e-9 and headroom > 1e-9:
            report.record(Intervention.RAISED_SETPOINT)
            share = min(deficit / headroom, 1.0)
            for k in committed:
                g = self.cfg.gensets[k]
                out[k] += (g.rated_kw - out[k]) * share
        return out

    def _bound_battery(
        self,
        battery: list[float],
        genset_on: list[bool],
        setpoints: list[float],
        observation: dict,
        report: SafetyReport,
    ) -> list[float]:
        """Clip to the cells' envelope, then to what the bus can spare.

        Two separate bounds, for two separate reasons. The first is the pack: a
        LiFePO4 pack below freezing reports a charge limit of zero, and the
        agent's request is refused rather than silently failing. The second is
        the station: charging is *demand*, and demand that outruns generation
        comes out of somebody's load. Never life support's.
        """
        out = []
        for k in range(len(self.cfg.storage)):
            lo = -float(observation["battery_max_charge_kw"][k])
            hi = float(observation["battery_max_discharge_kw"][k])
            bounded = float(np.clip(battery[k], lo, hi))
            if abs(bounded - battery[k]) > 1e-9:
                report.record(Intervention.CLIPPED_BATTERY)
            out.append(bounded)

        # Renewables are excluded from the charging budget for the same reason
        # they are excluded from the capacity requirement: the wind can stop.
        effective = self._effective_online(genset_on, observation)
        firm_supply_kw = sum(setpoints[k] for k in range(len(setpoints)) if effective[k])
        firm_supply_kw += sum(max(p, 0.0) for p in out)
        charge_budget_kw = max(firm_supply_kw - observation["critical_load_kw"], 0.0)
        requested_charge_kw = sum(max(-p, 0.0) for p in out)

        if requested_charge_kw > charge_budget_kw + 1e-9:
            report.record(Intervention.LIMITED_CHARGE)
            scale = charge_budget_kw / requested_charge_kw if requested_charge_kw > 0 else 0.0
            out = [p * scale if p < 0.0 else p for p in out]
        return out

    def _bound_melt(
        self,
        melt: float,
        genset_on: list[bool],
        setpoints: list[float],
        battery: list[float],
        observation: dict,
        report: SafetyReport,
    ) -> float:
        """Discretionary load must never compete with critical load.

        Melting is the dump load and the deferrable sink, which makes it the
        obvious thing for a poorly trained policy to over-commit. It is bounded
        by the melters' physical rate and then, if supply is tight, by whatever
        is left after life support has been served.
        """
        ceiling = self.melt_ceiling_kw()
        bounded = float(np.clip(melt, 0.0, ceiling))
        if abs(bounded - melt) > 1e-9:
            report.record(Intervention.CLIPPED_MELT)

        effective = self._effective_online(genset_on, observation)
        available_kw = (
            sum(setpoints[k] for k in range(len(setpoints)) if effective[k])
            + observation["pv_available_kw"]
            + observation["wind_available_kw"]
            + sum(max(p, 0.0) for p in battery)
        )
        headroom_kw = available_kw - observation["critical_load_kw"] - sum(
            max(-p, 0.0) for p in battery
        )
        if bounded > headroom_kw:
            bounded = max(headroom_kw, 0.0)
            report.record(Intervention.SHED_MELT_FOR_CRITICAL)
        return bounded


__all__ = ["SafetyProjection", "SafetyReport", "Intervention"]
