"""Station configuration: typed, validated views over the YAML files in stations/.

Every physical parameter of a station lives in YAML, never in code. The
dataclasses here are a thin, validated projection of that YAML so the rest of
the package can rely on attribute access and on the invariants checked in
StationConfig.validate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

STATIONS_DIR = Path(__file__).parent / "stations"


class ConfigError(ValueError):
    """A station configuration is missing a field or violates an invariant."""


def _req(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"{where}: missing required key {key!r}")
    return d[key]


@dataclass(frozen=True)
class Site:
    id: str
    name: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float

    @property
    def is_polar(self) -> bool:
        """True when the site sees a true polar night, i.e. beyond a polar circle."""
        return abs(self.latitude_deg) >= 66.5


@dataclass(frozen=True)
class Occupancy:
    winter_crew: int
    summer_crew: int
    summer_start_doy: int
    summer_end_doy: int
    shoulder_days: int


@dataclass(frozen=True)
class Climate:
    temp_summer_mean_c: float
    temp_winter_mean_c: float
    temp_diurnal_amp_c: float
    temp_noise_std_c: float
    temp_noise_tau_h: float
    cold_snap_prob_per_day: float
    cold_snap_depth_c: float
    wind_weibull_k: float
    wind_scale_summer_ms: float
    wind_scale_winter_ms: float
    wind_tau_h: float
    blizzard_prob_per_day: float
    blizzard_gust_ms: float
    cloud_mean: float
    cloud_tau_h: float
    snow_albedo: float


@dataclass(frozen=True)
class GensetSpec:
    """One generating set. Fuel and fouling parameters are shared across sets."""

    id: str
    rated_kw: float
    chp_heat_ratio: float
    fuel: str
    fuel_lhv_mj_per_l: float
    sfc_rated_l_per_kwh: float
    idle_fuel_frac: float
    min_stable_load_frac: float
    wet_stack_threshold_frac: float
    burn_off_threshold_frac: float
    deposit_accum_per_h: float
    deposit_burn_per_h: float
    min_up_time_min: float
    min_down_time_min: float
    start_fuel_l: float
    bc_ef_clean_mg_per_kwh: float
    bc_ef_fouled_mg_per_kwh: float

    @property
    def min_stable_kw(self) -> float:
        return self.rated_kw * self.min_stable_load_frac

    @property
    def fuel_rate_rated_l_per_h(self) -> float:
        return self.sfc_rated_l_per_kwh * self.rated_kw

    @property
    def willans_intercept_l_per_h(self) -> float:
        """No-load fuel flow: the intercept a in the Willans line a + b * P.

        This intercept is the whole reason part-load operation is expensive. It
        is paid in full whether the set carries 10 kW or its full rating.
        """
        return self.idle_fuel_frac * self.fuel_rate_rated_l_per_h

    @property
    def willans_slope_l_per_kwh(self) -> float:
        return (1.0 - self.idle_fuel_frac) * self.sfc_rated_l_per_kwh


@dataclass(frozen=True)
class PVSpec:
    rated_kwp: float
    tilt_deg: float
    azimuth_deg: float
    temp_coeff_per_c: float
    noct_c: float
    system_derate: float
    snow_cover_loss_max: float


@dataclass(frozen=True)
class WindSpec:
    turbines: int
    rated_kw_each: float
    hub_height_m: float
    ref_height_m: float
    roughness_length_m: float
    cut_in_ms: float
    rated_ms: float
    cut_out_ms: float

    @property
    def rated_kw_total(self) -> float:
        return self.turbines * self.rated_kw_each


@dataclass(frozen=True)
class StorageSpec:
    id: str
    chemistry: str
    location: str
    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    round_trip_efficiency: float
    soc_min: float
    soc_max: float
    min_operating_temp_c: float

    @property
    def one_way_efficiency(self) -> float:
        """Charge and discharge each carry the square root of the round trip."""
        return self.round_trip_efficiency**0.5

    @property
    def is_exterior(self) -> bool:
        return self.location == "exterior"


@dataclass(frozen=True)
class ElectricalLoadSpec:
    base_kw_per_person: float
    fixed_kw: float
    science_summer_kw: float
    diurnal_amp_frac: float
    noise_std_frac: float
    noise_tau_h: float


@dataclass(frozen=True)
class ThermalLoadSpec:
    ua_kw_per_c: float
    indoor_setpoint_c: float
    thermal_capacitance_kwh_per_c: float
    boiler_rated_kw: float
    boiler_efficiency: float
    water_l_per_person_day: float
    snow_melt_kwh_per_l: float
    service_hot_water_kw: float
    deferrable_fraction: float


@dataclass(frozen=True)
class Criticality:
    life_support_kw: float
    min_indoor_temp_c: float
    reserve_margin_kw: float


@dataclass(frozen=True)
class NetworkBus:
    id: str


@dataclass(frozen=True)
class NetworkLine:
    from_bus: str
    to_bus: str
    r_ohm: float
    """Positive-sequence resistance for the whole line, not per unit length."""
    x_ohm: float
    """Positive-sequence reactance for the whole line, not per unit length."""


@dataclass(frozen=True)
class NetworkConfig:
    """A single-feeder radial LV network, for `allotrope.sim.network.DistributionNetwork`.

    Optional: a station with no `network:` section in its YAML has no
    electrical-network model, and every consumer of `StationConfig.network`
    must treat `None` as "not configured" rather than assuming a network
    exists. This is deliberately synthetic like every other station
    parameter -- see the station YAML's own `src` tags -- calibrated to be
    a plausible small LV feeder, not a measured one.
    """

    base_kv: float
    """Nominal three-phase line-to-line voltage, kV."""
    source_bus: str
    """The bus the plant's aggregate generation and storage connect to; held at 1.0 pu."""
    buses: tuple[NetworkBus, ...]
    lines: tuple[NetworkLine, ...]
    v_min_pu: float
    v_max_pu: float

    def bus_ids(self) -> tuple[str, ...]:
        return tuple(b.id for b in self.buses)


@dataclass(frozen=True)
class StationConfig:
    site: Site
    occupancy: Occupancy
    climate: Climate
    gensets: tuple[GensetSpec, ...]
    pv: PVSpec
    wind: WindSpec
    storage: tuple[StorageSpec, ...]
    electrical: ElectricalLoadSpec
    thermal: ThermalLoadSpec
    criticality: Criticality
    network: NetworkConfig | None = None
    fuel_budget: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def total_genset_kw(self) -> float:
        return sum(g.rated_kw for g in self.gensets)

    @property
    def total_storage_kwh(self) -> float:
        return sum(s.capacity_kwh for s in self.storage)

    def storage_by_id(self, sid: str) -> StorageSpec:
        for s in self.storage:
            if s.id == sid:
                return s
        raise KeyError(sid)

    def validate(self) -> None:
        """Check the invariants that make a configuration physically coherent."""
        if not self.gensets:
            raise ConfigError("station has no gensets")
        for g in self.gensets:
            if g.rated_kw <= 0:
                raise ConfigError(f"{g.id}: rated_kw must be positive")
            if not 0.0 <= g.idle_fuel_frac < 1.0:
                raise ConfigError(f"{g.id}: idle_fuel_frac must lie in [0, 1)")
            if not g.min_stable_load_frac < g.wet_stack_threshold_frac:
                raise ConfigError(
                    f"{g.id}: wet-stacking threshold must sit above the minimum stable "
                    "load, otherwise the set can never run cleanly"
                )
            if not g.wet_stack_threshold_frac < g.burn_off_threshold_frac <= 1.0:
                raise ConfigError(
                    f"{g.id}: burn-off threshold must exceed the wet-stacking one"
                )
        for s in self.storage:
            if not 0.0 <= s.soc_min < s.soc_max <= 1.0:
                raise ConfigError(f"{s.id}: require 0 <= soc_min < soc_max <= 1")
            if not 0.0 < s.round_trip_efficiency <= 1.0:
                raise ConfigError(f"{s.id}: round-trip efficiency must lie in (0, 1]")
        if self.criticality.min_indoor_temp_c > self.thermal.indoor_setpoint_c:
            raise ConfigError("the hard indoor floor cannot exceed the comfort setpoint")
        if self.criticality.life_support_kw > self.total_genset_kw:
            raise ConfigError("life-support demand exceeds total installed generation")
        if not self.wind.cut_in_ms < self.wind.rated_ms < self.wind.cut_out_ms:
            raise ConfigError("wind turbine speeds must satisfy cut_in < rated < cut_out")
        if self.network is not None:
            net = self.network
            bus_ids = net.bus_ids()
            if len(bus_ids) != len(set(bus_ids)):
                raise ConfigError("network: bus ids must be unique")
            if net.source_bus not in bus_ids:
                raise ConfigError(f"network: source_bus {net.source_bus!r} is not a declared bus")
            for line in net.lines:
                if line.from_bus not in bus_ids or line.to_bus not in bus_ids:
                    raise ConfigError(
                        f"network: line {line.from_bus}->{line.to_bus} references an undeclared bus"
                    )
                if line.r_ohm <= 0.0 and line.x_ohm <= 0.0:
                    raise ConfigError(
                        f"network: line {line.from_bus}->{line.to_bus} has zero impedance"
                    )
            if not 0.0 < net.v_min_pu < 1.0 < net.v_max_pu:
                raise ConfigError("network: require 0 < v_min_pu < 1.0 < v_max_pu")
            if net.base_kv <= 0.0:
                raise ConfigError("network: base_kv must be positive")


def _build_network(data: dict[str, Any]) -> NetworkConfig:
    return NetworkConfig(
        base_kv=float(_req(data, "base_kv", "network")),
        source_bus=_req(data, "source_bus", "network"),
        buses=tuple(NetworkBus(id=b["id"]) for b in _req(data, "buses", "network")),
        lines=tuple(
            NetworkLine(
                from_bus=_req(line, "from", "network line"),
                to_bus=_req(line, "to", "network line"),
                r_ohm=float(_req(line, "r_ohm", "network line")),
                x_ohm=float(_req(line, "x_ohm", "network line")),
            )
            for line in _req(data, "lines", "network")
        ),
        v_min_pu=float(data.get("v_min_pu", 0.94)),
        v_max_pu=float(data.get("v_max_pu", 1.06)),
    )


def _build(data: dict[str, Any]) -> StationConfig:
    gen = _req(data, "generation", "station")
    common = dict(_req(gen, "genset_common", "generation"))
    gensets = tuple(
        GensetSpec(
            id=_req(g, "id", "genset"),
            rated_kw=float(_req(g, "rated_kw", "genset")),
            chp_heat_ratio=float(g.get("chp_heat_ratio", 1.0)),
            **{k: (v if k == "fuel" else float(v)) for k, v in common.items()},
        )
        for g in _req(gen, "gensets", "generation")
    )
    cfg = StationConfig(
        site=Site(**_req(data, "station", "root")),
        occupancy=Occupancy(**_req(data, "occupancy", "root")),
        climate=Climate(**_req(data, "climate", "root")),
        gensets=gensets,
        pv=PVSpec(**_req(gen, "pv", "generation")),
        wind=WindSpec(**_req(gen, "wind", "generation")),
        storage=tuple(StorageSpec(**s) for s in _req(data, "storage", "root")),
        electrical=ElectricalLoadSpec(
            **_req(_req(data, "loads", "root"), "electrical", "loads")
        ),
        thermal=ThermalLoadSpec(**_req(_req(data, "loads", "root"), "thermal", "loads")),
        criticality=Criticality(**_req(data, "criticality", "root")),
        network=_build_network(data["network"]) if data.get("network") else None,
        fuel_budget=data.get("fuel_budget", {}),
        raw=data,
    )
    cfg.validate()
    return cfg


def load_station(name_or_path: str | Path) -> StationConfig:
    """Load a station by short name ("maitri") or by path to a YAML file."""
    path = Path(name_or_path)
    if not path.suffix:
        path = STATIONS_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise ConfigError(f"no station config at {path} (available: {available_stations()})")
    with open(path, encoding="utf-8") as fh:
        return _build(yaml.safe_load(fh))


def available_stations() -> list[str]:
    return sorted(p.stem for p in STATIONS_DIR.glob("*.yaml"))


__all__ = [
    "ConfigError",
    "StationConfig",
    "GensetSpec",
    "PVSpec",
    "WindSpec",
    "StorageSpec",
    "NetworkBus",
    "NetworkLine",
    "NetworkConfig",
    "load_station",
    "available_stations",
]
