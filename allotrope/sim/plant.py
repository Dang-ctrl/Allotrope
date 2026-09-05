"""The microgrid plant: one steppable simulation of a polar station's energy system.

This is the object a controller acts on and, later, the object a hardware-in-the-loop
rig replaces. Everything the controller is allowed to decide arrives as a
DispatchCommand; everything it must live with -- weather, crew, the laws of the
energy balance -- happens here.

Two coupled buses are simulated, because a polar station is not an electrical
system with some heating attached. It is a combined-heat-and-power system in
which the heat is as critical as the electricity, and in which the interaction
between them is where the savings are:

  electrical:  PV + wind + gensets + battery = demand + melting + charging + curtailment
  thermal:     recovered CHP heat + boilers = space heat + hot water + snow melt

Space heating is deliberately not a controller decision. It is served
automatically -- by recovered heat first, by the auxiliary boilers after that --
because no learned policy should ever be in a position to let a station freeze.
The controller's influence on heating is therefore indirect and honest: run the
sets well and there is recovered heat to spare, which is boiler fuel not burnt.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from allotrope.config import StationConfig
from allotrope.sim.assets import Battery, Genset, pv_power_kw, wind_power_kw
from allotrope.synth.climate import ClimateSeries
from allotrope.synth.loads import LoadSeries


@dataclass(frozen=True)
class DispatchCommand:
    """Everything a controller is permitted to decide in one timestep.

    Setpoints are advisory: the plant clips them to what each machine can
    actually do, and the difference between what was asked and what happened is
    reported back in the telemetry so a controller can be held to account for it.
    """

    genset_on: tuple[bool, ...]
    genset_setpoint_kw: tuple[float, ...]
    battery_kw: tuple[float, ...]
    """Per unit, positive to discharge to the bus and negative to charge."""
    snow_melt_kw: float
    """Requested melting rate. The deferrable sink, and the burn-off dump load."""
    renewable_limit_kw: float | None = None
    """A ceiling on combined PV+wind injection this step, or None for no ceiling.

    This is the inverter-level Volt-Watt curtailment lever
    (`allotrope.safety.voltage.VoltWattCurve`): the point-of-interconnection
    real-power limit an inverter would enforce to hold bus voltage inside its
    ride-through band. No other part of the plant sets this field -- it
    exists so a layer downstream of the analytic safety projection can act on
    a network voltage solve the projection itself has no model of.
    """

    @classmethod
    def all_off(cls, cfg: StationConfig) -> DispatchCommand:
        return cls(
            genset_on=tuple(False for _ in cfg.gensets),
            genset_setpoint_kw=tuple(0.0 for _ in cfg.gensets),
            battery_kw=tuple(0.0 for _ in cfg.storage),
            snow_melt_kw=0.0,
        )

    def with_gensets(
        self, on: tuple[bool, ...], setpoints: tuple[float, ...]
    ) -> DispatchCommand:
        return replace(self, genset_on=on, genset_setpoint_kw=setpoints)


@dataclass
class PlantState:
    """The plant's memory between timesteps."""

    step_index: int = 0
    indoor_temp_c: float = 20.0
    snow_melt_remaining_kwh: float = 0.0
    """Melting energy still owed today. Deferral is allowed; default is not."""
    current_doy: int = -1

    # Cumulative counters, for episode-level accounting.
    total_fuel_l: float = 0.0
    total_black_carbon_mg: float = 0.0
    total_load_kwh: float = 0.0
    total_renewable_kwh: float = 0.0
    total_curtailed_kwh: float = 0.0
    total_unserved_kwh: float = 0.0
    total_critical_unserved_kwh: float = 0.0
    wet_stacking_steps: int = 0
    freeze_violation_steps: int = 0
    unmet_water_kwh: float = 0.0


class PolarMicrogrid:
    """A polar station microgrid, steppable one dispatch interval at a time."""

    def __init__(
        self,
        cfg: StationConfig,
        climate: ClimateSeries,
        loads: LoadSeries,
        dt_h: float | None = None,
    ) -> None:
        if len(climate) != len(loads):
            raise ValueError("climate and load series must share a time index")
        self.cfg = cfg
        self.climate = climate
        self.loads = loads
        self.index: pd.DatetimeIndex = climate.index
        self.dt_h = dt_h if dt_h is not None else self._infer_dt_h(self.index)

        self.gensets = [Genset(spec) for spec in cfg.gensets]
        self.batteries = [Battery(spec) for spec in cfg.storage]
        self.state = PlantState(indoor_temp_c=cfg.thermal.indoor_setpoint_c)

        # Renewable availability is a property of the weather, so it is computed
        # once for the whole series rather than per step.
        self.pv_available_kw = np.asarray(
            pv_power_kw(cfg.pv, climate.poa_w_m2, climate.air_temp_c, climate.snow_cover)
        )
        self.wind_available_kw = np.asarray(
            wind_power_kw(cfg.wind, climate.hub_wind_speed_ms, climate.air_density_kg_m3)
        )

    @staticmethod
    def _infer_dt_h(index: pd.DatetimeIndex) -> float:
        if len(index) < 2:
            raise ValueError("cannot infer a timestep from fewer than two samples")
        return float((index[1] - index[0]).total_seconds()) / 3600.0

    @property
    def n_steps(self) -> int:
        return len(self.index)

    @property
    def done(self) -> bool:
        return self.state.step_index >= self.n_steps

    def reset(self, start_index: int = 0, initial_soc: float | None = None) -> None:
        self.gensets = [Genset(spec) for spec in self.cfg.gensets]
        self.batteries = [Battery(spec) for spec in self.cfg.storage]
        for battery in self.batteries:
            if initial_soc is not None:
                battery.state.soc = float(
                    np.clip(initial_soc, battery.spec.soc_min, battery.spec.soc_max)
                )
        self.state = PlantState(
            step_index=start_index,
            indoor_temp_c=self.cfg.thermal.indoor_setpoint_c,
        )
        self._roll_over_day(start_index)

    # -- observation -------------------------------------------------------

    def observe(self) -> dict[str, Any]:
        """The plant's externally visible condition, before this step is dispatched.

        Battery temperature is re-derated against *this* step's ambient/indoor
        reading before the envelope is reported, exactly as `step()` will do
        before it executes -- not the reading left over from the previous
        step. Reporting the stale figure understates how far a cold snap has
        already cut a pack's capability by the time a command computed from
        this observation is executed, which is exactly the gap a safety
        projection using every last watt of reported headroom can fall
        through.
        """
        i = min(self.state.step_index, self.n_steps - 1)
        for battery in self.batteries:
            battery.set_temperature(float(self.climate.air_temp_c[i]), self.state.indoor_temp_c)
        return {
            "timestamp": self.index[i],
            "electrical_load_kw": float(self.loads.electrical_kw[i]),
            "critical_load_kw": float(self.loads.critical_kw[i]),
            "firm_thermal_kw": float(self.loads.firm_thermal_kw[i]),
            "pv_available_kw": float(self.pv_available_kw[i]),
            "wind_available_kw": float(self.wind_available_kw[i]),
            "air_temp_c": float(self.climate.air_temp_c[i]),
            "wind_speed_ms": float(self.climate.wind_speed_ms[i]),
            "indoor_temp_c": self.state.indoor_temp_c,
            "snow_melt_remaining_kwh": self.state.snow_melt_remaining_kwh,
            "genset_online": [g.state.online for g in self.gensets],
            "genset_power_kw": [g.state.power_kw for g in self.gensets],
            "genset_deposit": [g.state.deposit for g in self.gensets],
            "genset_can_start": [g.can_start for g in self.gensets],
            "genset_can_stop": [g.can_stop for g in self.gensets],
            "battery_soc": [b.state.soc for b in self.batteries],
            "battery_max_charge_kw": [b.max_charge_kw(self.dt_h) for b in self.batteries],
            "battery_max_discharge_kw": [b.max_discharge_kw(self.dt_h) for b in self.batteries],
        }

    # -- dispatch ----------------------------------------------------------

    def step(self, command: DispatchCommand) -> dict[str, Any]:
        """Advance the plant one dispatch interval under the given command."""
        if self.done:
            raise RuntimeError("simulation has run past the end of its weather series")

        i = self.state.step_index
        dt = self.dt_h
        cfg = self.cfg
        self._roll_over_day(i)

        for battery in self.batteries:
            battery.set_temperature(float(self.climate.air_temp_c[i]), self.state.indoor_temp_c)

        # 1. Commitment. Which sets are turning is the controller's decision, and
        #    anti-cycling is enforced inside each machine, so the command is a
        #    request rather than an order.
        commitment = [
            g.set_commitment(bool(on)) for g, on in zip(self.gensets, command.genset_on)
        ]
        online = [g.state.online for g in self.gensets]

        # 2. Batteries, then the electrical balance that decides how hard the
        #    committed sets are actually worked.
        electrical_load_kw = float(self.loads.electrical_kw[i])
        renewable_available_kw = float(self.pv_available_kw[i] + self.wind_available_kw[i])
        voltage_curtailed_kw = 0.0
        if command.renewable_limit_kw is not None:
            limited_kw = min(renewable_available_kw, max(command.renewable_limit_kw, 0.0))
            voltage_curtailed_kw = renewable_available_kw - limited_kw
            renewable_available_kw = limited_kw
        battery_kw = [
            b.step(float(p), dt) for b, p in zip(self.batteries, command.battery_kw)
        ]
        battery_net_kw = sum(battery_kw)
        charging_kw = max(-battery_net_kw, 0.0)
        discharging_kw = max(battery_net_kw, 0.0)

        firm_thermal_kw = float(self.loads.firm_thermal_kw[i])
        space_heat_demand_kw = float(self.loads.space_heat_kw[i])
        melt_request_kw = max(command.snow_melt_kw, 0.0)

        # Melting can be served by spare recovered heat, which depends on how
        # hard the sets run, which depends in turn on how much melting is served
        # electrically. Two passes settle it: the first assumes all melting is
        # electric, the second uses the recovered heat that resulted.
        melt_electric_kw = melt_request_kw
        for _ in range(2):
            demand_kw = electrical_load_kw + melt_electric_kw + charging_kw
            genset_kw = self._load_follow(
                demand_kw - renewable_available_kw - discharging_kw,
                online,
                command.genset_setpoint_kw,
            )
            chp_heat_kw = sum(
                genset_kw[k] * g.spec.chp_heat_ratio for k, g in enumerate(self.gensets)
            )
            spare_chp_heat_kw = max(chp_heat_kw - firm_thermal_kw, 0.0)
            melt_from_heat_kw = min(spare_chp_heat_kw, melt_request_kw)
            melt_electric_kw = melt_request_kw - melt_from_heat_kw

        genset_results = [
            g.apply_power(genset_kw[k], dt, *commitment[k]) for k, g in enumerate(self.gensets)
        ]
        genset_total_kw = sum(r.power_kw for r in genset_results)
        chp_heat_kw = sum(r.heat_kw for r in genset_results)
        fuel_l = sum(r.fuel_l for r in genset_results)
        black_carbon_mg = sum(r.black_carbon_mg for r in genset_results)
        wet_stacking = any(r.wet_stacking for r in genset_results)

        # 3. Thermal balance. Recovered heat first, boilers for the rest.
        heat_to_firm_kw = min(chp_heat_kw, firm_thermal_kw)
        spare_chp_heat_kw = chp_heat_kw - heat_to_firm_kw
        melt_from_heat_kw = min(spare_chp_heat_kw, melt_request_kw)
        melt_electric_kw = melt_request_kw - melt_from_heat_kw

        # Whatever heat the sets did not recover is covered by the auxiliary
        # boilers, burning the same Jet A-1 directly. This is what makes
        # recovered heat valuable: it is not a bonus, it is boiler fuel avoided.
        # Only if the boilers themselves cannot keep up does heating fall back
        # on electricity, and that is a failure condition, not a strategy.
        firm_heat_shortfall_kw = max(firm_thermal_kw - heat_to_firm_kw, 0.0)
        boiler_heat_kw = min(firm_heat_shortfall_kw, cfg.thermal.boiler_rated_kw)
        boiler_fuel_l = self._boiler_fuel_l(boiler_heat_kw, dt)
        electric_heat_kw = firm_heat_shortfall_kw - boiler_heat_kw
        fuel_l += boiler_fuel_l

        demand_kw = electrical_load_kw + electric_heat_kw + melt_electric_kw + charging_kw
        dispatchable_kw = genset_total_kw + discharging_kw

        # Renewables are free, so they displace fuel wherever the sets have left
        # room. What they cannot displace is curtailed: a committed set cannot
        # go below its minimum stable load to make way for the wind.
        renewable_used_kw = min(renewable_available_kw, max(demand_kw - dispatchable_kw, 0.0))
        curtailed_kw = renewable_available_kw - renewable_used_kw
        supply_kw = dispatchable_kw + renewable_used_kw
        unserved_kw = max(demand_kw - supply_kw, 0.0)

        # Shedding order under a shortfall: melting first, then comfort load,
        # and life support only if there is genuinely nothing left.
        shed_kw = unserved_kw
        shed_melt_kw = min(shed_kw, melt_electric_kw)
        shed_kw -= shed_melt_kw
        melt_electric_kw -= shed_melt_kw
        sheddable_comfort_kw = max(electrical_load_kw - float(self.loads.critical_kw[i]), 0.0)
        shed_comfort_kw = min(shed_kw, sheddable_comfort_kw)
        shed_kw -= shed_comfort_kw
        critical_unserved_kw = shed_kw

        # 4. Heat actually delivered, and the indoor temperature that follows.
        heat_delivered_kw = heat_to_firm_kw + boiler_heat_kw + max(
            electric_heat_kw - critical_unserved_kw, 0.0
        )
        space_heat_delivered_kw = min(heat_delivered_kw, space_heat_demand_kw)
        self._update_indoor_temperature(space_heat_delivered_kw, float(self.climate.air_temp_c[i]))

        # 5. Water accounting. Melting is deferrable within the day, not beyond it.
        melt_delivered_kw = melt_from_heat_kw + melt_electric_kw
        self.state.snow_melt_remaining_kwh = max(
            self.state.snow_melt_remaining_kwh - melt_delivered_kw * dt, 0.0
        )

        self._accumulate(
            fuel_l=fuel_l,
            black_carbon_mg=black_carbon_mg,
            load_kwh=electrical_load_kw * dt,
            renewable_kwh=renewable_used_kw * dt,
            curtailed_kwh=curtailed_kw * dt,
            unserved_kwh=(shed_melt_kw + shed_comfort_kw + critical_unserved_kw) * dt,
            critical_unserved_kwh=critical_unserved_kw * dt,
            wet_stacking=wet_stacking,
        )

        telemetry = {
            "timestamp": self.index[i],
            "genset_kw": genset_total_kw,
            "genset_power_kw": [r.power_kw for r in genset_results],
            "genset_online": [g.state.online for g in self.gensets],
            "genset_load_frac": [
                r.power_kw / g.spec.rated_kw for r, g in zip(genset_results, self.gensets)
            ],
            "genset_deposit": [g.state.deposit for g in self.gensets],
            "genset_starts": sum(r.started for r in genset_results),
            "wet_stacking": wet_stacking,
            "fuel_l": fuel_l,
            "black_carbon_mg": black_carbon_mg,
            "chp_heat_kw": chp_heat_kw,
            "chp_heat_used_kw": heat_to_firm_kw + melt_from_heat_kw,
            "boiler_heat_kw": boiler_heat_kw,
            "boiler_fuel_l": boiler_fuel_l,
            "electric_heat_kw": electric_heat_kw,
            "pv_available_kw": float(self.pv_available_kw[i]),
            "wind_available_kw": float(self.wind_available_kw[i]),
            "renewable_used_kw": renewable_used_kw,
            "curtailed_kw": curtailed_kw,
            "voltage_curtailed_kw": voltage_curtailed_kw,
            "battery_kw": battery_kw,
            "battery_soc": [b.state.soc for b in self.batteries],
            "electrical_load_kw": electrical_load_kw,
            "firm_thermal_kw": firm_thermal_kw,
            "melt_kw": melt_delivered_kw,
            "melt_from_heat_kw": melt_from_heat_kw,
            "melt_electric_kw": melt_electric_kw,
            "snow_melt_remaining_kwh": self.state.snow_melt_remaining_kwh,
            "unserved_kw": shed_melt_kw + shed_comfort_kw + critical_unserved_kw,
            "critical_unserved_kw": critical_unserved_kw,
            "indoor_temp_c": self.state.indoor_temp_c,
            "air_temp_c": float(self.climate.air_temp_c[i]),
            "renewable_fraction": (
                renewable_used_kw / max(electrical_load_kw, 1e-6)
            ),
        }

        self.state.step_index += 1
        return telemetry

    # -- internals ---------------------------------------------------------

    def _load_follow(
        self, required_kw: float, online: list[bool], setpoints: tuple[float, ...]
    ) -> list[float]:
        """Share the bus deficit across the committed sets, within their limits.

        A generating set does not hold an output because it was told to; it holds
        whatever the bus draws from it, floored at its minimum stable load and
        capped by its rating. The controller's setpoints act as ceilings and as
        the sharing key between sets, not as fixed outputs. This is also where
        curtailment comes from: several sets committed against a light load
        cannot collectively go below their minimum stable loads, so surplus wind
        has nowhere to go.
        """
        committed = [k for k, flag in enumerate(online) if flag]
        power = [0.0] * len(self.gensets)
        if not committed:
            return power

        floor = sum(self.gensets[k].spec.min_stable_kw for k in committed)
        ceilings = {
            k: float(np.clip(setpoints[k], self.gensets[k].spec.min_stable_kw, self.gensets[k].spec.rated_kw))
            for k in committed
        }
        ceiling = sum(ceilings.values())
        target = float(np.clip(required_kw, floor, ceiling))

        # Share above the common floor in proportion to each set's headroom, so
        # that machines with more room take more of the swing.
        headroom = {k: ceilings[k] - self.gensets[k].spec.min_stable_kw for k in committed}
        total_headroom = sum(headroom.values())
        surplus = target - floor
        for k in committed:
            share = headroom[k] / total_headroom if total_headroom > 1e-9 else 0.0
            power[k] = self.gensets[k].spec.min_stable_kw + surplus * share
        return power

    def _boiler_fuel_l(self, heat_kw: float, dt_h: float) -> float:
        """Fuel burnt by the auxiliary boilers to deliver a given heat output."""
        lhv_kwh_per_l = self.cfg.gensets[0].fuel_lhv_mj_per_l / 3.6
        return heat_kw * dt_h / (lhv_kwh_per_l * self.cfg.thermal.boiler_efficiency)

    def _roll_over_day(self, i: int) -> None:
        """Reset the day's melting obligation when the station clock passes midnight."""
        doy = int(self.index[min(i, self.n_steps - 1)].dayofyear)
        if doy == self.state.current_doy:
            return
        if self.state.current_doy != -1:
            # Whatever melting was still owed at midnight was simply not done.
            self.state.unmet_water_kwh += self.state.snow_melt_remaining_kwh
        self.state.current_doy = doy
        self.state.snow_melt_remaining_kwh = float(self.loads.snow_melt_daily_kwh[i])

    def _update_indoor_temperature(self, heat_delivered_kw: float, outdoor_c: float) -> None:
        """First-order envelope: capacitance against conduction to the outside."""
        therm = self.cfg.thermal
        loss_kw = therm.ua_kw_per_c * (self.state.indoor_temp_c - outdoor_c)
        net_kw = heat_delivered_kw - loss_kw
        self.state.indoor_temp_c += net_kw * self.dt_h / therm.thermal_capacitance_kwh_per_c
        if self.state.indoor_temp_c < self.cfg.criticality.min_indoor_temp_c:
            self.state.freeze_violation_steps += 1

    def _accumulate(self, *, wet_stacking: bool, **kwh: float) -> None:
        st = self.state
        st.total_fuel_l += kwh["fuel_l"]
        st.total_black_carbon_mg += kwh["black_carbon_mg"]
        st.total_load_kwh += kwh["load_kwh"]
        st.total_renewable_kwh += kwh["renewable_kwh"]
        st.total_curtailed_kwh += kwh["curtailed_kwh"]
        st.total_unserved_kwh += kwh["unserved_kwh"]
        st.total_critical_unserved_kwh += kwh["critical_unserved_kwh"]
        st.wet_stacking_steps += int(wet_stacking)

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, float]:
        """Episode-level outcomes, in the units the project is actually judged on."""
        st = self.state
        steps = max(st.step_index, 1)
        run_hours = sum(g.state.run_hours for g in self.gensets)
        genset_kwh = sum(g.state.total_kwh for g in self.gensets)
        return {
            "fuel_l": st.total_fuel_l,
            "fuel_kl": st.total_fuel_l / 1000.0,
            "black_carbon_g": st.total_black_carbon_mg / 1000.0,
            "load_kwh": st.total_load_kwh,
            "genset_kwh": genset_kwh,
            "renewable_kwh": st.total_renewable_kwh,
            "curtailed_kwh": st.total_curtailed_kwh,
            "renewable_fraction": st.total_renewable_kwh / max(st.total_load_kwh, 1e-9),
            "specific_fuel_l_per_kwh": st.total_fuel_l / max(genset_kwh, 1e-9),
            "mean_genset_load_frac": (
                genset_kwh / max(sum(g.spec.rated_kw * g.state.run_hours for g in self.gensets), 1e-9)
            ),
            "wet_stacking_fraction": st.wet_stacking_steps / steps,
            "mean_deposit": float(np.mean([g.state.deposit for g in self.gensets])),
            "genset_run_hours": run_hours,
            "genset_starts": sum(g.state.total_starts for g in self.gensets),
            "unserved_kwh": st.total_unserved_kwh,
            "critical_unserved_kwh": st.total_critical_unserved_kwh,
            "freeze_violation_steps": float(st.freeze_violation_steps),
            "unmet_water_kwh": st.unmet_water_kwh,
            "cold_charge_blocks": float(sum(b.state.cold_charge_blocks for b in self.batteries)),
        }


__all__ = ["PolarMicrogrid", "DispatchCommand", "PlantState"]
