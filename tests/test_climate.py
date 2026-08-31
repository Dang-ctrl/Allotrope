"""The synthetic climate must reproduce the polar features that drive the design.

These are not tests of a fitted model against data; there is no data. They are
tests that the generator obeys the physics and geometry it claims to, which is
the only thing that makes a synthetic training environment defensible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from allotrope.config import load_station
from allotrope.synth.climate import (
    ClimateGenerator,
    air_density,
    ar1_process,
    clear_sky_ghi,
    solar_position,
)

STANDARD_DENSITY = 1.225


@pytest.fixture(scope="module")
def maitri():
    return load_station("maitri")


@pytest.fixture(scope="module")
def year(maitri):
    return ClimateGenerator(maitri, seed=3).generate("2026-01-01", periods=8760, freq="1h")


def test_polar_night_exists_and_falls_in_midwinter(year):
    """At 70.8 S the sun stays down for weeks around midwinter."""
    nights = year.is_polar_night
    days = nights.sum() / 24.0
    assert 50 < days < 90, f"expected a polar night of weeks, got {days:.0f} days"

    months = year.index[nights].month.unique().tolist()
    assert set(months) <= {5, 6, 7, 8}


def test_no_irradiance_during_polar_night(year):
    """A PV array in the polar night yields exactly nothing, not merely little."""
    assert year.ghi_w_m2[year.is_polar_night].max() == 0.0
    assert year.poa_w_m2[year.is_polar_night].max() == 0.0


def test_midsummer_sun_never_sets_far_enough_to_stop_generation(year):
    december = year.index.month == 12
    assert year.ghi_w_m2[december].max() > 400.0


def test_snow_albedo_lifts_plane_of_array_above_horizontal(year):
    """The ground-reflected term is why polar arrays are mounted near-vertical."""
    ghi = year.ghi_w_m2.sum()
    poa = year.poa_w_m2.sum()
    assert poa > ghi * 1.1


def test_seasonal_temperature_has_the_right_sign_and_range(year, maitri):
    monthly = pd.Series(year.air_temp_c, index=year.index).resample("1ME").mean()
    assert monthly.idxmin().month in (6, 7, 8)
    assert monthly.idxmax().month in (12, 1, 2)
    assert monthly.min() < maitri.climate.temp_winter_mean_c + 4.0
    assert year.air_temp_c.min() < -35.0


def test_cold_air_is_dense_air(year):
    """Density correction is a real gain for a turbine in polar service."""
    assert year.air_density_kg_m3.mean() > STANDARD_DENSITY
    coldest = np.argmin(year.air_temp_c)
    assert year.air_density_kg_m3[coldest] > STANDARD_DENSITY * 1.15


def test_wind_is_persistent_not_white_noise(year):
    """A turbine must not ride out a blizzard in a single timestep."""
    w = year.wind_speed_ms
    lag1 = np.corrcoef(w[:-1], w[1:])[0, 1]
    assert lag1 > 0.8


def test_wind_speeds_are_physically_plausible(year):
    assert 4.0 < year.wind_speed_ms.mean() < 14.0
    assert year.wind_speed_ms.max() > 20.0
    assert year.wind_speed_ms.min() >= 0.0


def test_hub_height_extrapolation_raises_speed(year):
    assert (year.hub_wind_speed_ms >= year.wind_speed_ms).all()


def test_generation_is_reproducible(maitri):
    a = ClimateGenerator(maitri, seed=11).generate(periods=720)
    b = ClimateGenerator(maitri, seed=11).generate(periods=720)
    c = ClimateGenerator(maitri, seed=12).generate(periods=720)
    assert np.array_equal(a.air_temp_c, b.air_temp_c)
    assert not np.array_equal(a.air_temp_c, c.air_temp_c)


def test_statistics_survive_a_change_of_timestep(maitri):
    """AR(1) parameters are defined by a correlation time, not by the timestep."""
    hourly = ClimateGenerator(maitri, seed=5).generate("2026-03-01", periods=24 * 60, freq="1h")
    quarterly = ClimateGenerator(maitri, seed=5).generate(
        "2026-03-01", periods=24 * 60 * 4, freq="15min"
    )
    assert quarterly.air_temp_c.std() == pytest.approx(hourly.air_temp_c.std(), rel=0.25)
    assert quarterly.wind_speed_ms.mean() == pytest.approx(hourly.wind_speed_ms.mean(), rel=0.25)


def test_ar1_process_has_the_requested_spread():
    rng = np.random.default_rng(0)
    x = ar1_process(200_000, dt_h=1.0, tau_h=6.0, rng=rng, std=2.5)
    assert x.std() == pytest.approx(2.5, rel=0.05)
    assert x.mean() == pytest.approx(0.0, abs=0.1)


def test_sun_is_north_of_a_southern_station_at_local_noon(maitri):
    """A sanity check on the azimuth convention, which is easy to get backwards."""
    index = pd.date_range("2026-12-21", periods=24, freq="1h", tz="UTC")
    pos = solar_position(index, maitri.site.latitude_deg, maitri.site.longitude_deg)
    highest = int(np.argmax(pos.elevation_deg))
    azimuth = pos.azimuth_deg[highest]
    assert min(azimuth, 360.0 - azimuth) < 20.0


def test_clear_sky_is_zero_below_the_horizon():
    assert clear_sky_ghi(np.array([-0.5, -0.01, 0.0])).max() == 0.0
    assert clear_sky_ghi(np.array([1.0]))[0] > 900.0


def test_air_density_matches_the_ideal_gas_law_at_sea_level():
    assert air_density(np.array([15.0]), 0.0)[0] == pytest.approx(1.225, rel=0.01)
