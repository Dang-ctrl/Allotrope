/**
 * Types mirroring allotrope/api/app.py's actual JSON responses exactly.
 *
 * Every field here has a real backing value in the Python API — see
 * allotrope/api/simulation.py's StationSimulation methods. Nothing in this
 * file is a placeholder shape for data the backend doesn't produce yet.
 */

export interface StationSummary {
  id: string;
  name: string;
  running: boolean;
  step: number;
  n_steps: number;
}

export interface GensetSpec {
  id: string;
  rated_kw: number;
}

export interface StorageSpec {
  id: string;
  capacity_kwh: number;
}

export interface ControllerStatus {
  name: string;
  type: string;
  wrapped_agent: string | null;
}

export interface StationDetail {
  id: string;
  name: string;
  latitude_deg: number;
  longitude_deg: number;
  gensets: GensetSpec[];
  storage: StorageSpec[];
  controller: ControllerStatus;
}

export interface Observation {
  timestamp: string;
  electrical_load_kw: number;
  critical_load_kw: number;
  firm_thermal_kw: number;
  pv_available_kw: number;
  wind_available_kw: number;
  air_temp_c: number;
  wind_speed_ms: number;
  indoor_temp_c: number;
  snow_melt_remaining_kwh: number;
  genset_online: boolean[];
  genset_power_kw: number[];
  genset_deposit: number[];
  genset_can_start: boolean[];
  genset_can_stop: boolean[];
  battery_soc: number[];
  battery_max_charge_kw: number[];
  battery_max_discharge_kw: number[];
}

export interface StationState {
  station_id: string;
  mode: "simulation";
  running: boolean;
  step: number;
  n_steps: number;
  done: boolean;
  timestamp: string;
  observation: Observation;
}

export interface SafetyReport {
  intervened: boolean;
  interventions: string[];
  required_capacity_kw: number;
  committed_capacity_kw: number;
  // allotrope/safety/projection.py's Report.as_dict() spreads `detail`'s
  // keys directly into this object (**self.detail) rather than nesting
  // them under a "detail" key, so there is no such field on the wire --
  // an earlier version of this type claimed one existed. Extra keys land
  // here as an untyped bag: nothing in this app currently reads them.
  [extra: string]: unknown;
}

export interface VoltageStatus {
  bus_voltage_pu: Record<string, number>;
  converged: boolean;
  curtailed: boolean;
  renewable_available_kw: number;
  renewable_limit_kw: number | null;
}

export interface SafetyStatus {
  last_report: SafetyReport | null;
  last_fallback_reason: string | null;
  steps: number;
  fallbacks: number;
  projections: number;
  fallback_rate: number;
  projection_rate: number;
  fallback_reasons: Record<string, number>;
  max_latency_ms: number;
  /** null for a station with no network model -- see docs/network-safety.md. */
  voltage: VoltageStatus | null;
}

export interface Metrics {
  fuel_l: number;
  fuel_kl: number;
  black_carbon_g: number;
  load_kwh: number;
  genset_kwh: number;
  renewable_kwh: number;
  curtailed_kwh: number;
  renewable_fraction: number;
  specific_fuel_l_per_kwh: number;
  mean_genset_load_frac: number;
  wet_stacking_fraction: number;
  mean_deposit: number;
  genset_run_hours: number;
  genset_starts: number;
  unserved_kwh: number;
  critical_unserved_kwh: number;
  freeze_violation_steps: number;
  unmet_water_kwh: number;
  cold_charge_blocks: number;
}

export interface TelemetryRecord {
  timestamp: string;
  genset_kw: number;
  electrical_load_kw: number;
  critical_load_kw: number;
  renewable_used_kw: number;
  curtailed_kw: number;
  unserved_kw: number;
  critical_unserved_kw: number;
  fuel_l: number;
  black_carbon_mg: number;
  safety: SafetyReport | null;
  fallback_reason: string | null;
  [key: string]: unknown;
}

export interface HealthStatus {
  status: "ok";
  uptime_s: number;
  stations: string[];
}
