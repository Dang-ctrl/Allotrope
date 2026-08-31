"""Station configurations must be complete, coherent and physically sane."""

from __future__ import annotations

import copy

import pytest

from allotrope.config import ConfigError, available_stations, load_station
from allotrope.config import _build as build_config


@pytest.fixture(params=available_stations())
def station(request):
    return load_station(request.param)


def test_shipped_stations_load_and_validate(station):
    station.validate()
    assert station.gensets
    assert station.storage
    assert station.total_genset_kw > station.criticality.life_support_kw


def test_every_station_is_actually_polar(station):
    assert station.site.is_polar, "a polar EMS validated at a non-polar site proves nothing"


def test_dual_chemistry_covers_both_thermal_regimes(station):
    """The project premise: one pack outdoors, one in the heated core."""
    locations = {s.location for s in station.storage}
    assert {"exterior", "heated_core"} <= locations
    exterior = [s for s in station.storage if s.is_exterior]
    interior = [s for s in station.storage if not s.is_exterior]
    assert all(s.min_operating_temp_c <= -20.0 for s in exterior), (
        "an exterior pack must accept charge well below freezing"
    )
    assert all(s.min_operating_temp_c >= -5.0 for s in interior)


def test_part_load_is_penalised(station):
    """The Willans line must make light load genuinely more expensive per kWh."""
    for g in station.gensets:
        sfc_full = g.willans_intercept_l_per_h / g.rated_kw + g.willans_slope_l_per_kwh
        quarter = g.rated_kw * 0.25
        sfc_quarter = g.willans_intercept_l_per_h / quarter + g.willans_slope_l_per_kwh
        assert sfc_quarter > sfc_full * 1.15
        assert sfc_full == pytest.approx(g.sfc_rated_l_per_kwh, rel=1e-9)


def test_boiler_can_cover_design_heat_load(station):
    """A station whose boilers cannot meet peak heat demand would simply freeze."""
    delta_t = station.thermal.indoor_setpoint_c - station.climate.temp_winter_mean_c
    peak_space_heat_kw = station.thermal.ua_kw_per_c * (delta_t + station.climate.cold_snap_depth_c)
    assert station.thermal.boiler_rated_kw > peak_space_heat_kw


def test_recovered_heat_covers_most_of_design_heat_load(station):
    """CHP recovery must be worth dispatching for, or the thermal coupling is fiction."""
    delta_t = station.thermal.indoor_setpoint_c - station.climate.temp_winter_mean_c
    space_heat_kw = station.thermal.ua_kw_per_c * delta_t
    recovered_at_half_load_kw = 0.5 * station.total_genset_kw * station.gensets[0].chp_heat_ratio
    assert recovered_at_half_load_kw > space_heat_kw


def test_unknown_station_names_are_rejected():
    with pytest.raises(ConfigError, match="no station config"):
        load_station("mcmurdo")


def _mutated(station, path, value):
    raw = copy.deepcopy(station.raw)
    node = raw
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return raw


def test_incoherent_wet_stacking_threshold_is_rejected(station):
    raw = _mutated(station, ["generation", "genset_common", "wet_stack_threshold_frac"], 0.10)
    with pytest.raises(ConfigError, match="wet-stacking threshold"):
        build_config(raw)


def test_inverted_soc_limits_are_rejected(station):
    raw = copy.deepcopy(station.raw)
    raw["storage"][0]["soc_min"] = 0.99
    with pytest.raises(ConfigError, match="soc_min"):
        build_config(raw)


def test_freezing_setpoint_is_rejected(station):
    raw = _mutated(station, ["loads", "thermal", "indoor_setpoint_c"], 5.0)
    with pytest.raises(ConfigError, match="indoor floor"):
        build_config(raw)


def test_missing_section_names_itself(station):
    raw = copy.deepcopy(station.raw)
    del raw["criticality"]
    with pytest.raises(ConfigError, match="criticality"):
        build_config(raw)
