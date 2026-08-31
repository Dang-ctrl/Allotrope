"""Synthetic polar climate: solar geometry, irradiance, wind and air temperature.

There is no public telemetry from Maitri or Bharati, so the environment a policy
trains against has to be generated. This module generates it from physics and
station latitude rather than from resampled mid-latitude data, because the two
features that dominate a polar energy system do not survive resampling:

  * the polar night, during which PV yield is exactly zero for weeks, and
  * the snow albedo, which at a steep tilt returns a quarter of global
    irradiance back onto the panel.

Stochastic components are generated as Ornstein-Uhlenbeck style AR(1) processes
parameterised by a correlation time in hours, so that a series generated at a
1-minute step has the same statistics as one generated at 15 minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from allotrope.config import StationConfig

SOLAR_CONSTANT_W_M2 = 1361.0
STANDARD_AIR_DENSITY = 1.225  # kg/m3 at 15 C, sea level
GAS_CONSTANT_DRY_AIR = 287.05  # J/(kg K)


def ar1_process(
    n: int,
    dt_h: float,
    tau_h: float,
    rng: np.random.Generator,
    std: float = 1.0,
) -> np.ndarray:
    """A zero-mean AR(1) series with correlation time tau_h and stationary std.

    The coefficient is derived from the timestep so that the statistics of the
    series do not change when the simulation resolution changes.
    """
    if tau_h <= 0:
        return rng.normal(0.0, std, size=n)
    phi = float(np.exp(-dt_h / tau_h))
    innovation = std * np.sqrt(max(1.0 - phi * phi, 0.0))
    noise = rng.normal(0.0, 1.0, size=n)
    out = np.empty(n)
    out[0] = std * noise[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + innovation * noise[i]
    return out


def _poisson_events(
    n: int, dt_h: float, prob_per_day: float, rng: np.random.Generator
) -> np.ndarray:
    """Boolean mask of event onsets, at the given per-day probability."""
    p_step = 1.0 - np.exp(-prob_per_day * dt_h / 24.0)
    return rng.random(n) < p_step


def _decay_kernel(onsets: np.ndarray, dt_h: float, duration_h: float) -> np.ndarray:
    """Turn onsets into unit-amplitude pulses that decay over duration_h."""
    decay = float(np.exp(-dt_h / max(duration_h, 1e-9)))
    out = np.zeros(len(onsets))
    level = 0.0
    for i, onset in enumerate(onsets):
        level = max(level * decay, 1.0 if onset else 0.0)
        out[i] = level
    return out


@dataclass(frozen=True)
class SolarPosition:
    """Sun position and the extraterrestrial irradiance normal to the beam."""

    elevation_deg: np.ndarray
    azimuth_deg: np.ndarray
    cos_zenith: np.ndarray
    extraterrestrial_w_m2: np.ndarray

    @property
    def is_daylight(self) -> np.ndarray:
        return self.elevation_deg > 0.0


def solar_position(
    index: pd.DatetimeIndex, latitude_deg: float, longitude_deg: float
) -> SolarPosition:
    """NOAA solar position, evaluated on a UTC index.

    Accurate to well under a degree, which is far finer than the irradiance
    model that consumes it.
    """
    doy = index.dayofyear.to_numpy(dtype=float)
    hour = (
        index.hour.to_numpy(dtype=float)
        + index.minute.to_numpy(dtype=float) / 60.0
        + index.second.to_numpy(dtype=float) / 3600.0
    )

    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hour - 12.0) / 24.0)
    eq_time_min = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.001480 * np.sin(3 * gamma)
    )

    true_solar_time_min = (hour * 60.0 + eq_time_min + 4.0 * longitude_deg) % 1440.0
    hour_angle = np.deg2rad(true_solar_time_min / 4.0 - 180.0)

    lat = np.deg2rad(latitude_deg)
    # Sun direction in the local East-North-Up frame. The vector form avoids the
    # quadrant ambiguities that plague the closed-form azimuth expressions.
    east = -np.cos(decl) * np.sin(hour_angle)
    north = np.cos(lat) * np.sin(decl) - np.sin(lat) * np.cos(decl) * np.cos(hour_angle)
    up = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)

    elevation = np.rad2deg(np.arcsin(np.clip(up, -1.0, 1.0)))
    azimuth = np.rad2deg(np.arctan2(east, north)) % 360.0
    e0 = SOLAR_CONSTANT_W_M2 * (1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0))

    return SolarPosition(
        elevation_deg=elevation,
        azimuth_deg=azimuth,
        cos_zenith=np.clip(up, 0.0, 1.0),
        extraterrestrial_w_m2=e0,
    )


def clear_sky_ghi(cos_zenith: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky global horizontal irradiance."""
    cz = np.clip(cos_zenith, 0.0, 1.0)
    with np.errstate(divide="ignore", over="ignore"):
        ghi = 1098.0 * cz * np.exp(-0.059 / np.where(cz > 1e-3, cz, 1e-3))
    return np.where(cz > 1e-3, ghi, 0.0)


def _erbs_diffuse_fraction(kt: np.ndarray) -> np.ndarray:
    """Erbs correlation: diffuse fraction of global irradiance from clearness."""
    kt = np.clip(kt, 0.0, 1.0)
    return np.piecewise(
        kt,
        [kt <= 0.22, (kt > 0.22) & (kt <= 0.80), kt > 0.80],
        [
            lambda k: 1.0 - 0.09 * k,
            lambda k: 0.9511
            - 0.1604 * k
            + 4.388 * k**2
            - 16.638 * k**3
            + 12.336 * k**4,
            lambda k: np.full_like(k, 0.165),
        ],
    )


def plane_of_array(
    solar: SolarPosition,
    ghi: np.ndarray,
    tilt_deg: float,
    azimuth_deg: float,
    albedo: float,
) -> np.ndarray:
    """Irradiance on a tilted plane, isotropic sky with ground reflection.

    The ground-reflected term is not a rounding detail here. At a 70 degree tilt
    over snow of albedo 0.8 it contributes roughly a quarter of GHI, which is
    why polar arrays are mounted near-vertical.
    """
    beta = np.deg2rad(tilt_deg)
    gamma = np.deg2rad(azimuth_deg)
    normal = np.array([np.sin(beta) * np.sin(gamma), np.sin(beta) * np.cos(gamma), np.cos(beta)])

    elev = np.deg2rad(solar.elevation_deg)
    az = np.deg2rad(solar.azimuth_deg)
    sun = np.stack([np.cos(elev) * np.sin(az), np.cos(elev) * np.cos(az), np.sin(elev)])
    cos_aoi = np.clip(normal @ sun, 0.0, 1.0)

    e0_horizontal = solar.extraterrestrial_w_m2 * solar.cos_zenith
    kt = np.divide(ghi, e0_horizontal, out=np.zeros_like(ghi), where=e0_horizontal > 1.0)
    diffuse = ghi * _erbs_diffuse_fraction(kt)
    beam_horizontal = np.maximum(ghi - diffuse, 0.0)
    dni = np.divide(
        beam_horizontal,
        solar.cos_zenith,
        out=np.zeros_like(ghi),
        where=solar.cos_zenith > 0.02,
    )

    poa_beam = dni * cos_aoi
    poa_sky = diffuse * (1.0 + np.cos(beta)) / 2.0
    poa_ground = ghi * albedo * (1.0 - np.cos(beta)) / 2.0
    return poa_beam + poa_sky + poa_ground


def air_density(temp_c: np.ndarray, elevation_m: float) -> np.ndarray:
    """Air density from temperature and a barometric pressure at elevation.

    Cold air is dense air: at -30 C a turbine sees about 20 percent more mass
    flow than the 1.225 kg/m3 the nameplate assumes, and produces accordingly.
    """
    pressure_pa = 101325.0 * np.exp(-elevation_m / 8434.5)
    return pressure_pa / (GAS_CONSTANT_DRY_AIR * (temp_c + 273.15))


@dataclass(frozen=True)
class ClimateSeries:
    """One realisation of station weather, indexed in UTC."""

    index: pd.DatetimeIndex
    air_temp_c: np.ndarray
    wind_speed_ms: np.ndarray
    hub_wind_speed_ms: np.ndarray
    air_density_kg_m3: np.ndarray
    cloud_fraction: np.ndarray
    solar_elevation_deg: np.ndarray
    ghi_w_m2: np.ndarray
    poa_w_m2: np.ndarray
    snow_cover: np.ndarray
    blizzard: np.ndarray

    def __len__(self) -> int:
        return len(self.index)

    @property
    def is_polar_night(self) -> np.ndarray:
        """True where the sun does not clear the horizon at any point that day."""
        daily_max = (
            pd.Series(self.solar_elevation_deg, index=self.index).resample("1D").max()
        )
        return (daily_max.reindex(self.index, method="ffill") <= 0.0).to_numpy()

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "air_temp_c": self.air_temp_c,
                "wind_speed_ms": self.wind_speed_ms,
                "hub_wind_speed_ms": self.hub_wind_speed_ms,
                "air_density_kg_m3": self.air_density_kg_m3,
                "cloud_fraction": self.cloud_fraction,
                "solar_elevation_deg": self.solar_elevation_deg,
                "ghi_w_m2": self.ghi_w_m2,
                "poa_w_m2": self.poa_w_m2,
                "snow_cover": self.snow_cover,
                "blizzard": self.blizzard,
            },
            index=self.index,
        )


class ClimateGenerator:
    """Generates station weather realisations from a station configuration."""

    def __init__(self, cfg: StationConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        start: str | pd.Timestamp = "2026-01-01",
        periods: int = 8760,
        freq: str = "1h",
    ) -> ClimateSeries:
        index = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
        dt_h = float(pd.Timedelta(freq).total_seconds()) / 3600.0
        n = len(index)
        clim = self.cfg.climate

        summer_phase = self._summer_phase(index)
        temp = self._temperature(index, dt_h, n, summer_phase)
        cloud = self._cloud(dt_h, n)
        wind = self._wind(dt_h, n, summer_phase)
        blizzard_pulse = _decay_kernel(
            _poisson_events(n, dt_h, clim.blizzard_prob_per_day, self.rng), dt_h, 8.0
        )
        wind = wind + blizzard_pulse * (clim.blizzard_gust_ms - wind) * 0.8
        wind = np.maximum(wind, 0.0)

        solar = solar_position(
            index, self.cfg.site.latitude_deg, self.cfg.site.longitude_deg
        )
        clear = clear_sky_ghi(solar.cos_zenith)
        # Kasten-Czeplak cloud attenuation of global irradiance.
        ghi = clear * (1.0 - 0.75 * np.clip(cloud, 0.0, 1.0) ** 3.4)

        snow_cover = self._snow_cover(dt_h, n, blizzard_pulse, wind, temp, ghi)
        poa = plane_of_array(
            solar, ghi, self.cfg.pv.tilt_deg, self.cfg.pv.azimuth_deg, clim.snow_albedo
        )

        wind_hub = self._to_hub_height(wind)
        return ClimateSeries(
            index=index,
            air_temp_c=temp,
            wind_speed_ms=wind,
            hub_wind_speed_ms=wind_hub,
            air_density_kg_m3=air_density(temp, self.cfg.site.elevation_m),
            cloud_fraction=cloud,
            solar_elevation_deg=solar.elevation_deg,
            ghi_w_m2=ghi,
            poa_w_m2=poa,
            snow_cover=snow_cover,
            blizzard=blizzard_pulse > 0.5,
        )

    def _summer_phase(self, index: pd.DatetimeIndex) -> np.ndarray:
        """1 at midsummer (austral, ~doy 355), 0 at midwinter."""
        doy = index.dayofyear.to_numpy(dtype=float)
        return 0.5 * (1.0 + np.cos(2.0 * np.pi * (doy - 355.0) / 365.25))

    def _temperature(
        self, index: pd.DatetimeIndex, dt_h: float, n: int, summer_phase: np.ndarray
    ) -> np.ndarray:
        clim = self.cfg.climate
        seasonal = clim.temp_winter_mean_c + summer_phase * (
            clim.temp_summer_mean_c - clim.temp_winter_mean_c
        )
        hour = index.hour.to_numpy(dtype=float) + index.minute.to_numpy(dtype=float) / 60.0
        # The diurnal swing itself fades in the polar night, when there is no
        # daily forcing to drive it.
        diurnal = (
            -clim.temp_diurnal_amp_c
            * summer_phase
            * np.cos(2.0 * np.pi * (hour - 14.0) / 24.0)
        )
        noise = ar1_process(n, dt_h, clim.temp_noise_tau_h, self.rng, clim.temp_noise_std_c)
        snaps = _decay_kernel(
            _poisson_events(n, dt_h, clim.cold_snap_prob_per_day, self.rng), dt_h, 36.0
        )
        return seasonal + diurnal + noise - snaps * clim.cold_snap_depth_c

    def _cloud(self, dt_h: float, n: int) -> np.ndarray:
        clim = self.cfg.climate
        latent = ar1_process(n, dt_h, clim.cloud_tau_h, self.rng, 1.0)
        # Squash a Gaussian latent into [0, 1] with the requested mean.
        return np.clip(clim.cloud_mean + 0.35 * latent, 0.0, 1.0)

    def _wind(self, dt_h: float, n: int, summer_phase: np.ndarray) -> np.ndarray:
        """Weibull-distributed wind with realistic temporal persistence.

        A Gaussian AR(1) latent is mapped through its own CDF into a Weibull
        quantile, which preserves both the marginal distribution and the
        autocorrelation. Independent draws would let a turbine ride out a
        blizzard in a single timestep.
        """
        clim = self.cfg.climate
        latent = ar1_process(n, dt_h, clim.wind_tau_h, self.rng, 1.0)
        from scipy.stats import norm

        uniform = np.clip(norm.cdf(latent), 1e-9, 1.0 - 1e-9)
        scale = clim.wind_scale_winter_ms + summer_phase * (
            clim.wind_scale_summer_ms - clim.wind_scale_winter_ms
        )
        return scale * (-np.log(1.0 - uniform)) ** (1.0 / clim.wind_weibull_k)

    def _to_hub_height(self, wind: np.ndarray) -> np.ndarray:
        w = self.cfg.wind
        log_ratio = np.log(w.hub_height_m / w.roughness_length_m) / np.log(
            w.ref_height_m / w.roughness_length_m
        )
        return wind * log_ratio

    def _snow_cover(
        self,
        dt_h: float,
        n: int,
        blizzard: np.ndarray,
        wind: np.ndarray,
        temp: np.ndarray,
        ghi: np.ndarray,
    ) -> np.ndarray:
        """Fractional snow and rime cover on the PV array.

        Cover accumulates during blizzards and is shed by wind scouring and by
        solar gain on the panel. A steeply tilted array sheds well, which is the
        design intent behind the tilt in the station configuration; but during
        the polar night there is no solar term at all, so an array that fouls in
        June stays fouled until the sun returns.
        """
        steepness = np.sin(np.deg2rad(self.cfg.pv.tilt_deg))
        cover = np.zeros(n)
        level = 0.0
        for i in range(n):
            level += blizzard[i] * 0.6 * dt_h
            shed_wind = 0.04 * steepness * max(wind[i] - 8.0, 0.0) * dt_h
            shed_sun = 0.0
            if ghi[i] > 120.0 and temp[i] > -15.0:
                shed_sun = 0.25 * steepness * dt_h
            level = float(np.clip(level - shed_wind - shed_sun, 0.0, 1.0))
            cover[i] = level
        return cover


__all__ = [
    "ClimateGenerator",
    "ClimateSeries",
    "SolarPosition",
    "solar_position",
    "clear_sky_ghi",
    "plane_of_array",
    "air_density",
    "ar1_process",
]
