export interface GensetConfig {
  id: string
  rated_kw: number
  chp_heat_ratio: number
  wet_stack_threshold_frac: number
  burn_off_threshold_frac: number
  min_stable_load_frac: number
}

export interface StorageConfig {
  id: string
  chemistry: string
  location: string
  capacity_kwh: number
  soc_min: number
  soc_max: number
}

export interface StationConfig {
  id: string
  name: string
  lat: number
  lon: number
  elevation_m: number
  is_polar: boolean
  occupancy: { winter_crew: number; summer_crew: number }
  gensets: GensetConfig[]
  storage: StorageConfig[]
  criticality: {
    life_support_kw: number
    min_indoor_temp_c: number
    reserve_margin_kw: number
  }
  total_genset_kw: number
  total_storage_kwh: number
}

export interface TelemetryRow {
  time: string
  station_id: string
  genset_kw: number | null
  fuel_l: number | null
  black_carbon_mg: number | null
  renewable_used_kw: number | null
  curtailed_kw: number | null
  electrical_load_kw: number | null
  melt_kw: number | null
  unserved_kw: number | null
  critical_unserved_kw: number | null
  indoor_temp_c: number | null
  air_temp_c: number | null
  battery_soc_mean: number | null
}

export interface Telemetry {
  genset_kw: number
  fuel_l: number
  black_carbon_mg: number
  renewable_used_kw: number
  curtailed_kw: number
  electrical_load_kw: number
  melt_kw: number
  unserved_kw: number
  critical_unserved_kw: number
  indoor_temp_c: number
  air_temp_c: number
  battery_soc: number[]
  dispatch_latency_ms: number
}

export interface Observation {
  electrical_load_kw: number
  critical_load_kw: number
  firm_thermal_kw: number
  pv_available_kw: number
  wind_available_kw: number
  air_temp_c: number
  wind_speed_ms: number
  indoor_temp_c: number
  genset_online: boolean[]
  genset_power_kw: number[]
  genset_deposit: number[]
  battery_soc: number[]
}

export interface SafetyReport {
  intervened: boolean
  interventions: string[]
}

export interface SafetyEvent extends SafetyReport {
  ts: number
}

export type LiveMessage =
  | { type: 'snapshot'; telemetry: Telemetry | null; observation: Observation | null; safety_events: SafetyEvent[] }
  | { type: 'telemetry'; data: Telemetry; ts: number }
  | { type: 'observation'; data: Observation; ts: number }
  | { type: 'safety'; data: SafetyReport; ts: number }
