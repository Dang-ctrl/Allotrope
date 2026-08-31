"""Synthetic station demand: electrical, thermal, critical and deferrable.

The split this module draws matters more than the absolute numbers. A polar
station's demand is not one quantity to be met but three with very different
standing:

  * life support, which must be served every second of every winter-over;
  * comfort and science load, which can flex a little; and
  * snow melting for potable water, which is a large thermal demand that can be
    moved hours around the clock at no cost to anyone.

That third category is what makes the whole strategy work. It is the sink for
surplus wind, and it is the load that lets a generating set be pushed up into
its efficient band instead of idling dirty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from allotrope.config import StationConfig
from allotrope.synth.climate import ClimateSeries, ar1_process


def occupancy_profile(index: pd.DatetimeIndex, cfg: StationConfig) -> np.ndarray:
    """Crew count over time, ramping between winter-over and summer campaign.

    The summer window wraps the new year in the southern hemisphere, so the
    membership test is done on a circular day-of-year array and the shoulder
    ramp is applied as a circular moving average.
    """
    occ = cfg.occupancy
    days = np.arange(1, 367)
    start, end = occ.summer_start_doy, occ.summer_end_doy
    in_summer = (
        ((days >= start) | (days <= end)) if start > end else ((days >= start) & (days <= end))
    )

    width = max(int(occ.shoulder_days), 1)
    kernel = np.ones(width) / width
    padded = np.concatenate([in_summer[-width:], in_summer, in_summer[:width]]).astype(float)
    smoothed = np.convolve(padded, kernel, mode="same")[width : width + len(days)]

    crew_by_doy = occ.winter_crew + smoothed * (occ.summer_crew - occ.winter_crew)
    return crew_by_doy[index.dayofyear.to_numpy() - 1]


def _station_local_hour(index: pd.DatetimeIndex, longitude_deg: float) -> np.ndarray:
    """Hour on the station clock, which is set near local solar time."""
    offset = round(longitude_deg / 15.0)
    hours = index.hour.to_numpy(dtype=float) + index.minute.to_numpy(dtype=float) / 60.0
    return (hours + offset) % 24.0


@dataclass(frozen=True)
class LoadSeries:
    """Station demand, decomposed by service and by deferrability."""

    index: pd.DatetimeIndex
    crew: np.ndarray
    electrical_kw: np.ndarray
    critical_kw: np.ndarray
    space_heat_kw: np.ndarray
    service_hot_water_kw: np.ndarray
    snow_melt_kw: np.ndarray
    snow_melt_daily_kwh: np.ndarray

    def __len__(self) -> int:
        return len(self.index)

    @property
    def thermal_kw(self) -> np.ndarray:
        """Total heat demand including the nominal snow-melt schedule."""
        return self.space_heat_kw + self.service_hot_water_kw + self.snow_melt_kw

    @property
    def firm_thermal_kw(self) -> np.ndarray:
        """Heat demand that cannot be moved in time."""
        return self.space_heat_kw + self.service_hot_water_kw

    @property
    def deferrable_kw(self) -> np.ndarray:
        """The dispatchable thermal sink available to absorb surplus power."""
        return self.snow_melt_kw

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "crew": self.crew,
                "electrical_kw": self.electrical_kw,
                "critical_kw": self.critical_kw,
                "space_heat_kw": self.space_heat_kw,
                "service_hot_water_kw": self.service_hot_water_kw,
                "snow_melt_kw": self.snow_melt_kw,
                "thermal_kw": self.thermal_kw,
            },
            index=self.index,
        )


class LoadGenerator:
    """Generates station demand consistent with a given weather realisation."""

    def __init__(self, cfg: StationConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def generate(self, climate: ClimateSeries) -> LoadSeries:
        index = climate.index
        n = len(index)
        dt_h = float(pd.Timedelta(index.freq or (index[1] - index[0])).total_seconds()) / 3600.0
        elec = self.cfg.electrical
        therm = self.cfg.thermal

        crew = occupancy_profile(index, self.cfg)
        local_hour = _station_local_hour(index, self.cfg.site.longitude_deg)
        summer_crew_frac = np.clip(
            (crew - self.cfg.occupancy.winter_crew)
            / max(self.cfg.occupancy.summer_crew - self.cfg.occupancy.winter_crew, 1e-9),
            0.0,
            1.0,
        )

        # Activity peaks mid-afternoon on the station clock and troughs overnight.
        activity = 1.0 + elec.diurnal_amp_frac * np.sin(
            2.0 * np.pi * (local_hour - 9.0) / 24.0
        )
        noise = 1.0 + ar1_process(n, dt_h, elec.noise_tau_h, self.rng, elec.noise_std_frac)

        variable_kw = crew * elec.base_kw_per_person + summer_crew_frac * elec.science_summer_kw
        electrical = (elec.fixed_kw + variable_kw * activity) * noise
        electrical = np.maximum(electrical, elec.fixed_kw * 0.9)

        # Life support is the floor the safety layer must never breach.
        critical = np.full(n, self.cfg.criticality.life_support_kw)

        space_heat = np.maximum(therm.ua_kw_per_c * (therm.indoor_setpoint_c - climate.air_temp_c), 0.0)
        # Wind strips heat from the envelope faster than still air of the same
        # temperature, which is why a blizzard is a heating event as much as an
        # electrical one.
        wind_chill_factor = 1.0 + 0.02 * np.maximum(climate.wind_speed_ms - 5.0, 0.0)
        space_heat = space_heat * np.clip(wind_chill_factor, 1.0, 1.6)

        service_hw = np.full(n, therm.service_hot_water_kw) * (crew / self.cfg.occupancy.winter_crew)

        daily_water_l = crew * therm.water_l_per_person_day
        snow_melt_daily_kwh = daily_water_l * therm.snow_melt_kwh_per_l
        # The nominal schedule spreads melting evenly; the controller is free to
        # move it, and the daily energy is what it must ultimately deliver.
        snow_melt_kw = snow_melt_daily_kwh / 24.0

        return LoadSeries(
            index=index,
            crew=crew,
            electrical_kw=electrical,
            critical_kw=critical,
            space_heat_kw=space_heat,
            service_hot_water_kw=service_hw,
            snow_melt_kw=snow_melt_kw,
            snow_melt_daily_kwh=snow_melt_daily_kwh,
        )


__all__ = ["LoadGenerator", "LoadSeries", "occupancy_profile"]
