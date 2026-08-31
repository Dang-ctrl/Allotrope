"""Asset dynamics: fuel, fouling, anti-cycling and cold-limited storage."""

from __future__ import annotations

import numpy as np
import pytest

from allotrope.config import load_station
from allotrope.sim.assets import Battery, Genset, pv_power_kw, wind_power_kw


@pytest.fixture
def cfg():
    return load_station("maitri")


@pytest.fixture
def genset(cfg):
    return Genset(cfg.gensets[0])


def _run_online(genset, load_frac, hours, dt_h=1.0):
    genset.set_commitment(True)
    genset.state.minutes_in_state = 1e6
    for _ in range(int(hours / dt_h)):
        genset.apply_power(genset.spec.rated_kw * load_frac, dt_h)


# -- generating sets ----------------------------------------------------------


def test_light_load_costs_more_fuel_per_kwh(genset):
    spec = genset.spec
    genset.set_commitment(True)
    light = genset.fuel_rate_l_per_h(spec.rated_kw * 0.2) / (spec.rated_kw * 0.2)
    heavy = genset.fuel_rate_l_per_h(spec.rated_kw * 0.9) / (spec.rated_kw * 0.9)
    assert light > heavy * 1.3


def test_running_below_the_threshold_accumulates_deposits(genset):
    _run_online(genset, load_frac=0.20, hours=48)
    assert genset.state.deposit > 0.3


def test_running_hard_burns_deposits_off(genset):
    _run_online(genset, load_frac=0.20, hours=72)
    fouled = genset.state.deposit
    assert fouled > 0.5
    _run_online(genset, load_frac=0.85, hours=12)
    assert genset.state.deposit < fouled * 0.5


def test_deposits_stay_within_bounds(genset):
    _run_online(genset, load_frac=0.0, hours=1000)
    assert genset.state.deposit == pytest.approx(1.0)
    _run_online(genset, load_frac=1.0, hours=1000)
    assert genset.state.deposit == pytest.approx(0.0)


def test_fouling_raises_the_black_carbon_factor(genset):
    clean = genset.black_carbon_mg_per_kwh(0.8)
    _run_online(genset, load_frac=0.15, hours=100)
    fouled = genset.black_carbon_mg_per_kwh(0.8)
    assert fouled > clean * 4


def test_light_load_raises_emissions_even_when_clean(genset):
    assert genset.black_carbon_mg_per_kwh(0.10) > genset.black_carbon_mg_per_kwh(0.80)


def test_a_stopped_set_burns_nothing(genset):
    result = genset.step(command_on=False, setpoint_kw=100.0, dt_h=1.0)
    assert result.power_kw == 0.0
    assert result.fuel_l == 0.0
    assert result.black_carbon_mg == 0.0


def test_minimum_up_time_prevents_immediate_stopping(genset):
    genset.step(command_on=True, setpoint_kw=100.0, dt_h=0.1)
    assert genset.state.online
    genset.step(command_on=False, setpoint_kw=0.0, dt_h=0.1)
    assert genset.state.online, "a set stopped inside its minimum up time"


def test_minimum_down_time_prevents_immediate_restart(genset):
    _run_online(genset, load_frac=0.5, hours=2)
    genset.state.minutes_in_state = 1e6
    genset.set_commitment(False)
    assert not genset.state.online
    genset.step(command_on=True, setpoint_kw=100.0, dt_h=0.1)
    assert not genset.state.online


def test_output_is_floored_at_minimum_stable_load(genset):
    genset.set_commitment(True)
    result = genset.apply_power(1.0, dt_h=1.0)
    assert result.power_kw == pytest.approx(genset.spec.min_stable_kw)


def test_output_is_capped_at_the_rating(genset):
    genset.set_commitment(True)
    result = genset.apply_power(10_000.0, dt_h=1.0)
    assert result.power_kw == pytest.approx(genset.spec.rated_kw)


def test_starting_costs_fuel(genset):
    genset.step(command_on=True, setpoint_kw=0.0, dt_h=0.0)
    assert genset.state.total_fuel_l >= genset.spec.start_fuel_l


# -- renewables ---------------------------------------------------------------


def test_pv_yields_nothing_without_irradiance(cfg):
    assert pv_power_kw(cfg.pv, 0.0, -20.0) == pytest.approx(0.0)


def test_cold_cells_outperform_the_nameplate(cfg):
    cold = pv_power_kw(cfg.pv, 800.0, -25.0)
    warm = pv_power_kw(cfg.pv, 800.0, 25.0)
    assert cold > warm


def test_snow_cover_all_but_stops_the_array(cfg):
    clear = pv_power_kw(cfg.pv, 800.0, -20.0, snow_cover=0.0)
    buried = pv_power_kw(cfg.pv, 800.0, -20.0, snow_cover=1.0)
    assert buried < clear * 0.1


def test_wind_power_curve_respects_its_breakpoints(cfg):
    w = cfg.wind
    assert wind_power_kw(w, w.cut_in_ms - 0.5) == pytest.approx(0.0)
    assert wind_power_kw(w, w.cut_out_ms + 0.5) == pytest.approx(0.0)
    rated = wind_power_kw(w, w.rated_ms, 1.225)
    assert rated == pytest.approx(w.rated_kw_total, rel=1e-6)


def test_wind_power_never_exceeds_the_rating_even_in_dense_air(cfg):
    speeds = np.linspace(0.0, 30.0, 200)
    power = wind_power_kw(cfg.wind, speeds, 1.55)
    assert power.max() <= cfg.wind.rated_kw_total + 1e-9


def test_dense_air_produces_more_below_rated(cfg):
    mid = (cfg.wind.cut_in_ms + cfg.wind.rated_ms) / 2
    assert wind_power_kw(cfg.wind, mid, 1.45) > wind_power_kw(cfg.wind, mid, 1.225)


# -- storage ------------------------------------------------------------------


def _battery(cfg, chemistry):
    spec = next(s for s in cfg.storage if s.chemistry == chemistry)
    return Battery(spec)


def test_a_freezing_lfp_pack_refuses_charge(cfg):
    lfp = _battery(cfg, "lifepo4")
    lfp.set_temperature(ambient_c=-30.0, indoor_c=-5.0)
    assert lfp.max_charge_kw() == 0.0
    assert lfp.state.temperature_c == -5.0, "an interior pack follows the indoor temperature"


def test_the_heated_core_keeps_the_lfp_pack_available(cfg):
    lfp = _battery(cfg, "lifepo4")
    lfp.set_temperature(ambient_c=-40.0, indoor_c=20.0)
    assert lfp.max_charge_kw() > 0.0
    assert lfp.cold_derate() == pytest.approx(1.0)


def test_the_lto_pack_still_works_outdoors_in_deep_cold(cfg):
    """This is the entire reason for the dual-chemistry design."""
    lto = _battery(cfg, "lto")
    lfp = _battery(cfg, "lifepo4")
    lto.set_temperature(ambient_c=-25.0, indoor_c=20.0)
    lfp_outdoors = Battery(lfp.spec.__class__(**{**lfp.spec.__dict__, "location": "exterior"}))
    lfp_outdoors.set_temperature(ambient_c=-25.0, indoor_c=20.0)

    assert lto.max_charge_kw() > 0.0
    assert lfp_outdoors.max_charge_kw() == 0.0


def test_charging_below_the_floor_is_recorded_as_a_block(cfg):
    lfp = _battery(cfg, "lifepo4")
    lfp.set_temperature(ambient_c=-30.0, indoor_c=-10.0)
    lfp.step(-50.0, dt_h=1.0)
    assert lfp.state.cold_charge_blocks == 1


def test_round_trip_efficiency_is_actually_lost(cfg):
    lfp = _battery(cfg, "lifepo4")
    lfp.set_temperature(-20.0, 20.0)
    lfp.state.soc = 0.5
    start_energy = lfp.energy_kwh

    charged = -lfp.step(-40.0, dt_h=1.0)
    gained = lfp.energy_kwh - start_energy
    assert gained < charged, "charging stored more than it drew"
    assert gained == pytest.approx(charged * lfp.spec.one_way_efficiency, rel=1e-6)


def test_state_of_charge_stays_inside_its_limits(cfg):
    lfp = _battery(cfg, "lifepo4")
    lfp.set_temperature(-20.0, 20.0)
    for _ in range(200):
        lfp.step(500.0, dt_h=1.0)
    assert lfp.state.soc >= lfp.spec.soc_min - 1e-9
    for _ in range(200):
        lfp.step(-500.0, dt_h=1.0)
    assert lfp.state.soc <= lfp.spec.soc_max + 1e-9


def test_an_empty_pack_delivers_nothing(cfg):
    lfp = _battery(cfg, "lifepo4")
    lfp.set_temperature(-20.0, 20.0)
    lfp.state.soc = lfp.spec.soc_min
    assert lfp.max_discharge_kw() == pytest.approx(0.0)
