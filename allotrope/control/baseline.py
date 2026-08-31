"""Rule-based controllers: the incumbent practice, and the best non-learned policy.

LegacyNPlusOne is written to be a fair representation of how a station is
actually run, not a strawman. Its inefficiency is not carelessness; it is the
rational response to having no forecast, no storage worth dispatching and no
tolerance for a blackout in August. It keeps a spare set spinning because a set
that is already turning cannot fail to start. The cost of that caution -- a fleet
averaging well under its wet-stacking threshold all winter -- is exactly the cost
this project sets out to remove.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from allotrope.config import StationConfig
from allotrope.sim.plant import DispatchCommand, PolarMicrogrid


@dataclass
class LegacyNPlusOne:
    """Keep one more set online than the load needs, and share load equally.

    No storage dispatch, no load shifting, no burn-off. Melting runs at its flat
    nominal rate around the clock because nothing tells it to do otherwise.
    """

    cfg: StationConfig
    name: str = "legacy_n_plus_one"
    min_online: int = 2

    def reset(self) -> None:
        return None

    def act(self, observation: dict, plant: PolarMicrogrid) -> DispatchCommand:
        cfg = self.cfg
        nominal_melt_kw = self._nominal_melt_kw(plant)
        demand_kw = observation["electrical_load_kw"] + nominal_melt_kw

        # Sizing ignores the renewables entirely: an operator who cannot forecast
        # them cannot commit a generating set against them.
        rated = np.array([g.rated_kw for g in cfg.gensets])
        needed = self._units_for(demand_kw + cfg.criticality.reserve_margin_kw, rated)
        online_count = min(max(needed + 1, self.min_online), len(cfg.gensets))

        on = tuple(i < online_count for i in range(len(cfg.gensets)))
        share = demand_kw / max(online_count, 1)
        setpoints = tuple(
            float(np.clip(share, 0.0, g.rated_kw)) if on[i] else 0.0
            for i, g in enumerate(cfg.gensets)
        )
        return DispatchCommand(
            genset_on=on,
            genset_setpoint_kw=setpoints,
            battery_kw=tuple(0.0 for _ in cfg.storage),
            snow_melt_kw=nominal_melt_kw,
        )

    @staticmethod
    def _units_for(demand_kw: float, rated: np.ndarray) -> int:
        if demand_kw <= 0:
            return 0
        return max(1, ceil(demand_kw / float(rated.max())))

    @staticmethod
    def _nominal_melt_kw(plant: PolarMicrogrid) -> float:
        i = min(plant.state.step_index, plant.n_steps - 1)
        return float(plant.loads.snow_melt_kw[i])


@dataclass
class EfficientRuleBased:
    """Commit the fewest sets that can carry the load, and load them hard.

    The three moves that separate this from the incumbent are all things an
    operator could do with a forecast and a battery, and none of them require
    learning:

      * count the renewables against the commitment, holding storage back as the
        reserve that the spare spinning set used to provide;
      * push melting into surplus hours so that surplus is absorbed rather than
        curtailed, and drop it in deficit hours since water can wait;
      * when a set is fouled, deliberately raise its load and dump the excess
        into the melters until the deposits burn off.
    """

    cfg: StationConfig
    name: str = "efficient_rule_based"
    target_load_frac: float = 0.80
    burn_off_deposit_threshold: float = 0.35

    def reset(self) -> None:
        return None

    def act(self, observation: dict, plant: PolarMicrogrid) -> DispatchCommand:
        cfg = self.cfg
        dt_h = plant.dt_h
        i = min(plant.state.step_index, plant.n_steps - 1)

        electrical_kw = observation["electrical_load_kw"]
        renewable_kw = observation["pv_available_kw"] + observation["wind_available_kw"]

        # Melting is owed by the end of the day; spread what remains over the
        # hours left, then let surplus and deficit push it around that baseline.
        melt_kw = self._melt_baseline_kw(plant, i, dt_h)

        net_kw = electrical_kw + melt_kw - renewable_kw
        # Storage substitutes for the spinning reserve, which is the spare set
        # the incumbent keeps turning. It does not substitute for the load
        # itself: discharging against base load in August is how a station
        # arrives at midwinter with an empty battery and nothing committed.
        reserve_kw = min(
            self._storage_reserve_kw(observation), cfg.criticality.reserve_margin_kw
        )
        commit_kw = max(net_kw + cfg.criticality.reserve_margin_kw - reserve_kw, net_kw, 0.0)

        on, setpoints = self._commit(commit_kw, net_kw, observation)

        # Burn-off: a fouled set gets pushed into its clean band on purpose, and
        # the extra output is absorbed by the melters rather than wasted.
        extra_kw = 0.0
        deposits = observation["genset_deposit"]
        for idx, g in enumerate(cfg.gensets):
            if on[idx] and deposits[idx] > self.burn_off_deposit_threshold:
                target = g.rated_kw * (g.burn_off_threshold_frac + 0.15)
                extra_kw += max(target - setpoints[idx], 0.0)
                setpoints[idx] = min(target, g.rated_kw)

        supply_kw = sum(setpoints) + renewable_kw
        surplus_kw = max(supply_kw - electrical_kw - melt_kw, 0.0) + extra_kw
        melt_kw, battery_kw = self._absorb(surplus_kw, melt_kw, net_kw, observation)

        return DispatchCommand(
            genset_on=tuple(on),
            genset_setpoint_kw=tuple(setpoints),
            battery_kw=tuple(battery_kw),
            snow_melt_kw=melt_kw,
        )

    def _melt_baseline_kw(self, plant: PolarMicrogrid, i: int, dt_h: float) -> float:
        """Spread the day's remaining melting obligation over the hours that are left."""
        remaining_kwh = plant.state.snow_melt_remaining_kwh
        hours_left = max(24.0 - plant.index[i].hour, dt_h)
        return remaining_kwh / hours_left

    def _storage_reserve_kw(self, observation: dict) -> float:
        """Discharge capability available right now, which can stand in for a spare set."""
        return float(sum(observation["battery_max_discharge_kw"]))

    def _commit(
        self, commit_kw: float, net_kw: float, observation: dict
    ) -> tuple[list[bool], list[float]]:
        """Choose the fewest sets that can carry the load, loaded near their sweet spot."""
        cfg = self.cfg
        order = sorted(range(len(cfg.gensets)), key=lambda k: -cfg.gensets[k].rated_kw)
        on = [False] * len(cfg.gensets)
        setpoints = [0.0] * len(cfg.gensets)

        if commit_kw <= 0.0:
            # Respect minimum up times: a set that cannot legally stop stays on,
            # and is given real load rather than being left to idle dirty.
            for idx, g in enumerate(cfg.gensets):
                if observation["genset_online"][idx] and not observation["genset_can_stop"][idx]:
                    on[idx] = True
                    setpoints[idx] = g.rated_kw * self.target_load_frac
            return on, setpoints

        remaining_kw = commit_kw
        for idx in order:
            if remaining_kw <= 0.0:
                break
            g = cfg.gensets[idx]
            on[idx] = True
            take = min(remaining_kw, g.rated_kw)
            setpoints[idx] = max(take, g.min_stable_kw)
            remaining_kw -= take

        # Keep any set that is legally stuck online usefully loaded.
        for idx, g in enumerate(cfg.gensets):
            if observation["genset_online"][idx] and not observation["genset_can_stop"][idx]:
                on[idx] = True
                setpoints[idx] = max(setpoints[idx], g.min_stable_kw)

        # Rebalance across the committed sets so none sits in the fouling band.
        committed = [idx for idx, flag in enumerate(on) if flag]
        if committed:
            total_target = max(net_kw, sum(cfg.gensets[k].min_stable_kw for k in committed))
            capacity = sum(cfg.gensets[k].rated_kw for k in committed)
            frac = float(np.clip(total_target / capacity, 0.0, 1.0))
            for k in committed:
                g = cfg.gensets[k]
                setpoints[k] = float(np.clip(g.rated_kw * frac, g.min_stable_kw, g.rated_kw))
        return on, setpoints

    def _absorb(
        self, surplus_kw: float, melt_kw: float, net_kw: float, observation: dict
    ) -> tuple[float, list[float]]:
        """Route surplus into melting first, then storage; cover deficit the other way.

        Melting comes before charging because it has no round-trip loss and no
        cycle cost: energy put into a water tank comes back as water, not as
        ninety-four percent of itself.
        """
        cfg = self.cfg
        battery_kw = [0.0] * len(cfg.storage)

        if surplus_kw > 0.0:
            melt_headroom_kw = max(self._melt_ceiling_kw() - melt_kw, 0.0)
            to_melt = min(surplus_kw, melt_headroom_kw)
            melt_kw += to_melt
            left = surplus_kw - to_melt
            for idx, limit in enumerate(observation["battery_max_charge_kw"]):
                take = min(left, float(limit))
                battery_kw[idx] = -take
                left -= take
                if left <= 0.0:
                    break
        elif net_kw < 0.0:
            # More renewable than the station can use, and nothing committed.
            deficit_kw = 0.0
            for idx, limit in enumerate(observation["battery_max_discharge_kw"]):
                take = min(deficit_kw, float(limit))
                battery_kw[idx] = take
                deficit_kw -= take

        return melt_kw, battery_kw

    def _melt_ceiling_kw(self) -> float:
        """How fast the melters can physically run, as a multiple of the daily average."""
        therm = self.cfg.thermal
        daily_kwh = self.cfg.occupancy.summer_crew * therm.water_l_per_person_day * therm.snow_melt_kwh_per_l
        return 4.0 * daily_kwh / 24.0


__all__ = ["LegacyNPlusOne", "EfficientRuleBased"]
