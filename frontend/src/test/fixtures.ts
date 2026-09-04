/**
 * Fixtures shaped exactly like real allotrope/api/app.py responses -- the
 * values below were taken from an actual running `uvicorn
 * allotrope.api.app:app` process during development (see docs/api.md), not
 * invented. Component tests mock `fetch` with these so they exercise the
 * real rendering logic against the real response contract, without needing
 * a live backend process for every test run.
 */

import type {
  HealthStatus,
  Metrics,
  SafetyStatus,
  StationDetail,
  StationState,
  StationSummary,
} from "../api/types";

export const stationSummaries: StationSummary[] = [
  { id: "bharati", name: "Bharati", running: false, step: 0, n_steps: 8760 },
  { id: "maitri", name: "Maitri", running: false, step: 1, n_steps: 8760 },
];

export const health: HealthStatus = {
  status: "ok",
  uptime_s: 12.4,
  stations: ["bharati", "maitri"],
};

export const maitriDetail: StationDetail = {
  id: "maitri",
  name: "Maitri",
  latitude_deg: -70.766,
  longitude_deg: 11.731,
  gensets: [
    { id: "G1", rated_kw: 125 },
    { id: "G2", rated_kw: 125 },
    { id: "G3", rated_kw: 125 },
  ],
  storage: [
    { id: "BESS_LFP", capacity_kwh: 200 },
    { id: "BESS_LTO", capacity_kwh: 60 },
  ],
  controller: { name: "guarded_efficient_rule_based", type: "GuardedController", wrapped_agent: "EfficientRuleBased" },
};

export const maitriState: StationState = {
  station_id: "maitri",
  mode: "simulation",
  running: false,
  step: 1,
  n_steps: 8760,
  done: false,
  timestamp: "2026-01-01T01:00:00+00:00",
  observation: {
    timestamp: "2026-01-01T01:00:00+00:00",
    electrical_load_kw: 121.37,
    critical_load_kw: 45.0,
    firm_thermal_kw: 63.44,
    pv_available_kw: 2.14,
    wind_available_kw: 4.02,
    air_temp_c: -2.51,
    wind_speed_ms: 6.71,
    indoor_temp_c: 20.04,
    snow_melt_remaining_kwh: 363.69,
    genset_online: [false, true, false],
    genset_power_kw: [0, 57.61, 0],
    genset_deposit: [0.0, 0.0, 0.0],
    genset_can_start: [true, false, true],
    genset_can_stop: [false, true, false],
    battery_soc: [0.5, 0.5],
    battery_max_charge_kw: [60, 27],
    battery_max_discharge_kw: [70, 24],
  },
};

export const maitriSafety: SafetyStatus = {
  last_report: {
    intervened: false,
    interventions: [],
    required_capacity_kw: 65.0,
    committed_capacity_kw: 250.0,
    detail: { critical_load_kw: 45.0, reserve_margin_kw: 20.0, indoor_temp_c: 20.0 },
  },
  last_fallback_reason: null,
  steps: 1,
  fallbacks: 0,
  projections: 0,
  fallback_rate: 0,
  projection_rate: 0,
  fallback_reasons: {},
  max_latency_ms: 0.092,
  voltage: {
    bus_voltage_pu: { plant: 1.0, renewables: 1.001, load: 0.942 },
    converged: true,
    curtailed: false,
    renewable_available_kw: 6.15,
    renewable_limit_kw: null,
  },
};

export const maitriMetrics: Metrics = {
  fuel_l: 34.38,
  fuel_kl: 0.034,
  black_carbon_g: 1.38,
  load_kwh: 121.37,
  genset_kwh: 57.61,
  renewable_kwh: 6.15,
  curtailed_kwh: 0,
  renewable_fraction: 0.0507,
  specific_fuel_l_per_kwh: 0.283,
  mean_genset_load_frac: 0.461,
  wet_stacking_fraction: 0,
  mean_deposit: 0,
  genset_run_hours: 1,
  genset_starts: 1,
  unserved_kwh: 0,
  critical_unserved_kwh: 0,
  freeze_violation_steps: 0,
  unmet_water_kwh: 0,
  cold_charge_blocks: 0,
};
