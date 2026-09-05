"""Plant assets: conversion models and the stateful machines the controller acts on.

The renewable converters are pure functions of weather. The generating sets and
batteries are state machines, because the quantities that matter to this project
are precisely the ones with memory: exhaust deposits accumulated over weeks of
part-load running, minimum up and down times that forbid chattering, and a state
of charge that a cold battery cannot always accept.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from allotrope.config import GensetSpec, PVSpec, StorageSpec, WindSpec
from allotrope.synth.climate import STANDARD_AIR_DENSITY

FREEZING_C = 0.0


def pv_power_kw(
    spec: PVSpec,
    poa_w_m2: np.ndarray | float,
    air_temp_c: np.ndarray | float,
    snow_cover: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """Available PV power from plane-of-array irradiance.

    Two polar effects pull in opposite directions and both are modelled. Cold
    cells are efficient cells, so a panel at -25 C outperforms its nameplate by
    several percent; but a panel under snow or rime produces almost nothing, and
    in the polar night nothing sheds it.
    """
    poa = np.asarray(poa_w_m2, dtype=float)
    cell_temp = np.asarray(air_temp_c, dtype=float) + (spec.noct_c - 20.0) / 800.0 * poa
    temp_factor = 1.0 + spec.temp_coeff_per_c * (cell_temp - 25.0)
    snow_factor = 1.0 - spec.snow_cover_loss_max * np.clip(snow_cover, 0.0, 1.0)
    power = spec.rated_kwp * (poa / 1000.0) * temp_factor * spec.system_derate * snow_factor
    return np.maximum(power, 0.0)


def wind_power_kw(
    spec: WindSpec,
    hub_speed_ms: np.ndarray | float,
    air_density_kg_m3: np.ndarray | float = STANDARD_AIR_DENSITY,
) -> np.ndarray | float:
    """Available wind power from a cubic-ramp power curve, density corrected.

    Cold air is dense air. At -30 C a turbine sees roughly a fifth more mass
    flow than the 1.225 kg/m3 its nameplate assumes, so the density correction
    is a real gain in polar service rather than a rounding term.
    """
    v = np.asarray(hub_speed_ms, dtype=float)
    rho = np.asarray(air_density_kg_m3, dtype=float)

    ramp = (v**3 - spec.cut_in_ms**3) / (spec.rated_ms**3 - spec.cut_in_ms**3)
    fraction = np.where(
        v < spec.cut_in_ms,
        0.0,
        np.where(v < spec.rated_ms, np.clip(ramp, 0.0, 1.0), 1.0),
    )
    fraction = np.where(v >= spec.cut_out_ms, 0.0, fraction)

    power = spec.rated_kw_total * fraction * (rho / STANDARD_AIR_DENSITY)
    return np.clip(power, 0.0, spec.rated_kw_total)


@dataclass
class GensetState:
    online: bool = False
    power_kw: float = 0.0
    deposit: float = 0.0
    """Exhaust fouling in [0, 1]. Zero is a clean stack, one is fully wet-stacked."""
    minutes_in_state: float = 1e6
    total_fuel_l: float = 0.0
    total_kwh: float = 0.0
    total_starts: int = 0
    total_black_carbon_mg: float = 0.0
    run_hours: float = 0.0


@dataclass
class GensetStepResult:
    power_kw: float
    fuel_l: float
    heat_kw: float
    black_carbon_mg: float
    started: bool
    stopped: bool
    wet_stacking: bool


@dataclass
class Genset:
    """A single generating set with fuel, fouling and anti-cycling dynamics."""

    spec: GensetSpec
    state: GensetState = field(default_factory=GensetState)

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def can_start(self) -> bool:
        return not self.state.online and self.state.minutes_in_state >= self.spec.min_down_time_min

    @property
    def can_stop(self) -> bool:
        return self.state.online and self.state.minutes_in_state >= self.spec.min_up_time_min

    def available_kw(self) -> float:
        return self.spec.rated_kw if self.state.online else 0.0

    def fuel_rate_l_per_h(self, power_kw: float) -> float:
        """Willans-line fuel flow, penalised by accumulated fouling."""
        if power_kw <= 0.0 and not self.state.online:
            return 0.0
        fouling_penalty = 1.0 + 0.05 * self.state.deposit
        return (
            self.spec.willans_intercept_l_per_h
            + self.spec.willans_slope_l_per_kwh * max(power_kw, 0.0)
        ) * fouling_penalty

    def black_carbon_mg_per_kwh(self, load_frac: float) -> float:
        """Emission factor, driven by both current load and accumulated fouling.

        Running dirty is self-reinforcing: light load raises the emission factor
        directly, and it also lays down the deposits that raise it further.
        """
        s = self.spec
        by_deposit = s.bc_ef_clean_mg_per_kwh + (
            s.bc_ef_fouled_mg_per_kwh - s.bc_ef_clean_mg_per_kwh
        ) * self.state.deposit
        shortfall = max(s.wet_stack_threshold_frac - load_frac, 0.0) / s.wet_stack_threshold_frac
        return by_deposit * (1.0 + 1.5 * shortfall)

    def set_commitment(self, command_on: bool) -> tuple[bool, bool]:
        """Start or stop the set, honouring minimum up and down times.

        Anti-cycling is enforced here rather than trusted to the controller: a
        command that would violate a minimum up or down time is simply ignored,
        so no policy, learned or otherwise, can wear the machine out.

        Commitment is separated from power because that is how a real plant
        works. An operator decides which sets are turning; the bus decides how
        hard each one is worked.
        """
        st = self.state
        started = stopped = False

        if command_on and not st.online and self.can_start:
            st.online, started = True, True
            st.minutes_in_state = 0.0
            st.total_starts += 1
            st.total_fuel_l += self.spec.start_fuel_l
        elif not command_on and st.online and self.can_stop:
            st.online, stopped = False, True
            st.minutes_in_state = 0.0

        return started, stopped

    def apply_power(
        self, power_kw: float, dt_h: float, started: bool = False, stopped: bool = False
    ) -> GensetStepResult:
        """Advance the set by dt_h at the power the bus actually drew from it."""
        st, spec = self.state, self.spec

        if st.online:
            power = float(np.clip(power_kw, spec.min_stable_kw, spec.rated_kw))
        else:
            power = 0.0

        load_frac = power / spec.rated_kw if st.online else 0.0
        fuel_l = self.fuel_rate_l_per_h(power) * dt_h if st.online else 0.0
        energy_kwh = power * dt_h
        heat_kw = power * spec.chp_heat_ratio if st.online else 0.0

        bc_mg = self.black_carbon_mg_per_kwh(load_frac) * energy_kwh if st.online else 0.0
        wet = st.online and load_frac < spec.wet_stack_threshold_frac
        self._update_deposit(st.online, load_frac, dt_h)

        st.power_kw = power
        st.minutes_in_state += dt_h * 60.0
        st.total_fuel_l += fuel_l
        st.total_kwh += energy_kwh
        st.total_black_carbon_mg += bc_mg
        if st.online:
            st.run_hours += dt_h

        return GensetStepResult(power, fuel_l, heat_kw, bc_mg, started, stopped, wet)

    def step(self, command_on: bool, setpoint_kw: float, dt_h: float) -> GensetStepResult:
        """Commit and load in one call, for callers that dispatch a single set."""
        started, stopped = self.set_commitment(command_on)
        return self.apply_power(setpoint_kw, dt_h, started, stopped)

    def _update_deposit(self, online: bool, load_frac: float, dt_h: float) -> None:
        """Deposits build below the wet-stacking threshold and burn off above it.

        This is the state variable the whole burn-off strategy exists to manage.
        A set held at 25 percent load fouls steadily; the same set pushed to 75
        percent for a few hours clears itself.
        """
        spec, st = self.spec, self.state
        if not online:
            return
        if load_frac < spec.wet_stack_threshold_frac:
            severity = 1.0 - load_frac / spec.wet_stack_threshold_frac
            st.deposit += spec.deposit_accum_per_h * severity * dt_h
        elif load_frac > spec.burn_off_threshold_frac:
            intensity = (load_frac - spec.burn_off_threshold_frac) / (
                1.0 - spec.burn_off_threshold_frac
            )
            st.deposit -= spec.deposit_burn_per_h * intensity * dt_h
        st.deposit = float(np.clip(st.deposit, 0.0, 1.0))


@dataclass
class BatteryState:
    soc: float = 0.5
    power_kw: float = 0.0
    """Positive when discharging to the bus, negative when charging."""
    throughput_kwh: float = 0.0
    temperature_c: float = 20.0
    cold_charge_blocks: int = 0


@dataclass
class Battery:
    """A storage unit whose usable envelope depends on its own temperature.

    The dual-chemistry premise of this project lives in this class. A LiFePO4
    pack in the heated core keeps its full envelope all winter; an LTO pack on
    the exterior keeps most of its own down to -30 C, and neither can be charged
    below the floor its chemistry sets. Ignoring that is how a nominally
    well-sized polar battery turns out to be unavailable in August.
    """

    spec: StorageSpec
    state: BatteryState = field(default_factory=BatteryState)

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def energy_kwh(self) -> float:
        return self.state.soc * self.spec.capacity_kwh

    def set_temperature(self, ambient_c: float, indoor_c: float) -> None:
        self.state.temperature_c = ambient_c if self.spec.is_exterior else indoor_c

    def cold_derate(self) -> float:
        """Power capability multiplier from cell temperature.

        Full capability at 15 C and above, tapering to a third of it at the
        chemistry's operating floor.
        """
        t = self.state.temperature_c
        floor = self.spec.min_operating_temp_c
        if t >= 15.0:
            return 1.0
        span = max(15.0 - floor, 1e-6)
        return float(np.clip(0.35 + 0.65 * (t - floor) / span, 0.0, 1.0))

    def max_charge_kw(self, dt_h: float = 0.25) -> float:
        """Charge power available now, zero when the pack is too cold to accept.

        Bounded by whichever is tighter: the nameplate/thermal envelope, or
        the energy the pack can actually absorb over one control step of
        length `dt_h`. The second bound matters as much as the first: a pack
        with little headroom left can accept a large *instantaneous* power
        but not that power sustained for a full hourly step, and a caller
        that only checked the nameplate envelope would ask for more energy
        than physically fits. `dt_h` defaults to a quarter hour for callers,
        such as plain unit tests, that have no step length in scope; every
        call from the plant itself passes its own `dt_h` explicitly.
        """
        if self.state.temperature_c < self.spec.min_operating_temp_c:
            return 0.0
        headroom_kwh = (self.spec.soc_max - self.state.soc) * self.spec.capacity_kwh
        return max(min(self.spec.max_charge_kw * self.cold_derate(), headroom_kwh / dt_h), 0.0)

    def max_discharge_kw(self, dt_h: float = 0.25) -> float:
        """Discharge is permitted colder than charge, as the chemistry allows.

        See `max_charge_kw` for why `dt_h` matters: the energy-based bound
        must reflect the length of the step the plant is actually about to
        take, or a projection computed from this bound can authorise a
        discharge the pack cannot sustain for the step's full duration --
        which is exactly what leaves demand, including critical demand,
        unmet mid-step despite having passed the safety projection.
        """
        available_kwh = (self.state.soc - self.spec.soc_min) * self.spec.capacity_kwh
        return max(min(self.spec.max_discharge_kw * self.cold_derate(), available_kwh / dt_h), 0.0)

    def step(self, power_kw: float, dt_h: float) -> float:
        """Apply a power request, positive to discharge, and return what was met."""
        spec, st = self.spec, self.state
        eta = spec.one_way_efficiency

        if power_kw >= 0.0:
            delivered = min(power_kw, self.max_discharge_kw(dt_h))
            drawn_kwh = delivered * dt_h / eta
            st.soc -= drawn_kwh / spec.capacity_kwh
        else:
            requested = -power_kw
            if requested > 0.0 and self.max_charge_kw(dt_h) == 0.0:
                st.cold_charge_blocks += 1
            accepted = min(requested, self.max_charge_kw(dt_h))
            stored_kwh = accepted * dt_h * eta
            st.soc += stored_kwh / spec.capacity_kwh
            delivered = -accepted

        st.soc = float(np.clip(st.soc, spec.soc_min, spec.soc_max))
        st.power_kw = delivered
        st.throughput_kwh += abs(delivered) * dt_h
        return delivered


__all__ = [
    "pv_power_kw",
    "wind_power_kw",
    "Genset",
    "GensetState",
    "GensetStepResult",
    "Battery",
    "BatteryState",
]
