"""Plant-level invariants, and the behaviour the project's claims rest on.

The energy-balance test is the important one. Every headline figure this project
will quote is a difference between two runs of this simulator, so if the
simulator can create or destroy energy, none of those figures mean anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from allotrope.config import load_station
from allotrope.control.baseline import EfficientRuleBased, LegacyNPlusOne
from allotrope.sim.plant import DispatchCommand
from allotrope.sim.runner import build_plant, run_episode

WEEK = 24 * 7


@pytest.fixture(scope="module")
def cfg():
    return load_station("maitri")


@pytest.fixture
def plant(cfg):
    p = build_plant(cfg, start="2026-06-01", periods=WEEK, seed=1)
    p.reset()
    return p


def test_electrical_balance_closes_every_step(plant, cfg):
    """Supply equals demand, or the difference is accounted for as unserved."""
    controller = EfficientRuleBased(cfg)
    for _ in range(plant.n_steps):
        t = plant.step(controller.act(plant.observe(), plant))

        supply = (
            t["genset_kw"]
            + t["renewable_used_kw"]
            + sum(max(b, 0.0) for b in t["battery_kw"])
        )
        demand = (
            t["electrical_load_kw"]
            + t["electric_heat_kw"]
            + t["melt_electric_kw"]
            + sum(max(-b, 0.0) for b in t["battery_kw"])
        )
        assert supply == pytest.approx(demand - t["unserved_kw"], abs=1e-6), (
            f"energy balance broke at {t['timestamp']}"
        )


def test_renewable_generation_is_either_used_or_curtailed(plant, cfg):
    controller = EfficientRuleBased(cfg)
    for _ in range(plant.n_steps):
        t = plant.step(controller.act(plant.observe(), plant))
        available = t["pv_available_kw"] + t["wind_available_kw"]
        assert t["renewable_used_kw"] + t["curtailed_kw"] == pytest.approx(available, abs=1e-9)
        assert t["renewable_used_kw"] >= -1e-12
        assert t["curtailed_kw"] >= -1e-12


def test_thermal_demand_is_met_by_recovered_heat_or_boilers(plant, cfg):
    controller = EfficientRuleBased(cfg)
    for _ in range(plant.n_steps):
        t = plant.step(controller.act(plant.observe(), plant))
        recovered = min(t["chp_heat_kw"], t["firm_thermal_kw"])
        assert recovered + t["boiler_heat_kw"] + t["electric_heat_kw"] == pytest.approx(
            t["firm_thermal_kw"], abs=1e-6
        )
        assert t["boiler_heat_kw"] <= cfg.thermal.boiler_rated_kw + 1e-9


def test_an_idle_plant_serves_nothing_and_burns_only_boiler_fuel(plant, cfg):
    """With every set stopped, only the boilers can run, and load goes unserved."""
    command = DispatchCommand.all_off(cfg)
    total_unserved = 0.0
    for _ in range(24):
        t = plant.step(command)
        assert t["genset_kw"] == 0.0
        assert t["fuel_l"] == pytest.approx(t["boiler_fuel_l"])
        total_unserved += t["unserved_kw"]
    assert total_unserved > 0.0


def test_the_envelope_loses_heat_when_it_is_not_supplied(plant):
    """Without heat the station must actually cool, or freezing is unreachable."""
    outdoor_c = float(plant.climate.air_temp_c[0])
    start_temp = plant.state.indoor_temp_c
    for _ in range(48):
        plant._update_indoor_temperature(heat_delivered_kw=0.0, outdoor_c=outdoor_c)

    assert plant.state.indoor_temp_c < start_temp - 5.0
    assert plant.state.indoor_temp_c > outdoor_c, "the envelope cooled past ambient"
    assert plant.state.freeze_violation_steps > 0


def test_committed_sets_cannot_go_below_minimum_stable_load(plant, cfg):
    """This is where curtailment comes from, so it must hold exactly."""
    on = tuple(True for _ in cfg.gensets)
    setpoints = tuple(g.rated_kw for g in cfg.gensets)
    command = DispatchCommand(
        genset_on=on,
        genset_setpoint_kw=setpoints,
        battery_kw=tuple(0.0 for _ in cfg.storage),
        snow_melt_kw=0.0,
    )
    for _ in range(48):
        t = plant.step(command)
        for k, g in enumerate(cfg.gensets):
            if t["genset_online"][k]:
                assert t["genset_power_kw"][k] >= g.min_stable_kw - 1e-9


def test_state_of_charge_never_leaves_its_envelope(plant, cfg):
    controller = EfficientRuleBased(cfg)
    for _ in range(plant.n_steps):
        t = plant.step(controller.act(plant.observe(), plant))
        for k, s in enumerate(cfg.storage):
            assert s.soc_min - 1e-9 <= t["battery_soc"][k] <= s.soc_max + 1e-9


def test_cumulative_fuel_matches_the_step_by_step_trace(cfg):
    result = run_episode(build_plant(cfg, periods=WEEK, seed=2), EfficientRuleBased(cfg))
    assert result.summary["fuel_l"] == pytest.approx(result.telemetry["fuel_l"].sum(), rel=1e-9)


def test_running_past_the_end_of_the_weather_is_an_error(plant, cfg):
    command = DispatchCommand.all_off(cfg)
    for _ in range(plant.n_steps):
        plant.step(command)
    with pytest.raises(RuntimeError, match="past the end"):
        plant.step(command)


def test_reset_clears_every_accumulator(plant, cfg):
    controller = EfficientRuleBased(cfg)
    for _ in range(24):
        plant.step(controller.act(plant.observe(), plant))
    assert plant.state.total_fuel_l > 0.0

    plant.reset()
    assert plant.state.total_fuel_l == 0.0
    assert plant.state.step_index == 0
    assert all(not g.state.online for g in plant.gensets)
    assert all(g.state.deposit == 0.0 for g in plant.gensets)


# -- the claims the project actually makes ------------------------------------


@pytest.fixture(scope="module")
def winter_comparison(cfg):
    """A midwinter month under both rule-based controllers."""
    results = []
    for controller_cls in (LegacyNPlusOne, EfficientRuleBased):
        plant = build_plant(cfg, start="2026-06-01", periods=24 * 30, seed=4)
        results.append(run_episode(plant, controller_cls(cfg)))
    return results


def test_legacy_practice_reproduces_the_wet_stacking_problem(winter_comparison):
    """If the incumbent does not foul its machines, there is nothing to fix."""
    legacy = winter_comparison[0].summary
    assert legacy["mean_genset_load_frac"] < 0.30
    assert legacy["wet_stacking_fraction"] > 0.5
    assert legacy["mean_deposit"] > 0.5


def test_disciplined_dispatch_holds_the_sets_in_their_efficient_band(winter_comparison):
    efficient = winter_comparison[1].summary
    assert efficient["mean_genset_load_frac"] > 0.45
    assert efficient["wet_stacking_fraction"] < 0.15
    assert efficient["mean_deposit"] < 0.2


def test_better_dispatch_saves_fuel_and_black_carbon(winter_comparison):
    legacy, efficient = (r.summary for r in winter_comparison)
    assert efficient["fuel_l"] < legacy["fuel_l"]
    assert efficient["black_carbon_g"] < legacy["black_carbon_g"] * 0.5


def test_neither_controller_is_allowed_to_endanger_the_station(winter_comparison):
    """A fuel saving bought with unserved life support is not a saving."""
    for result in winter_comparison:
        assert result.summary["critical_unserved_kwh"] == pytest.approx(0.0)
        assert result.summary["freeze_violation_steps"] == 0.0
        assert result.telemetry["indoor_temp_c"].min() > 16.0
