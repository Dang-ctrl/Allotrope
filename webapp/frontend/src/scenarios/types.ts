// Mirrors the exact shape scripts/generate_scenarios.py writes to
// scenarios.json (copied verbatim into webapp/frontend/public/). This is
// static, played-back data -- not the live feed -- so it is deliberately a
// separate module from ../types.ts.

export interface StormRow {
  t: string
  date: string
  temp: number
  wind: number
  blizzard: boolean
  aiFailed: boolean
  gensetG: number
  gensetU: number
  critG: number
  critU: number
  cumCritG: number
  cumCritU: number
  cumShedG: number
}

export interface WetstackRow {
  day: number
  date: string
  depositLegacy: number
  depositEfficient: number
  wetStackLegacy: number
  wetStackEfficient: number
  fuelLegacy: number
  fuelEfficient: number
  bcLegacy: number
  bcEfficient: number
  loadLegacy: number
  loadEfficient: number
}

export interface Wetstack {
  station: string
  rows: WetstackRow[]
  totalFuelLegacy: number
  totalFuelEfficient: number
  totalBcLegacy: number
  totalBcEfficient: number
}

export interface FreeEnergyRow {
  t: string
  renewAvail: number
  renewUsedLegacy: number
  renewUsedEfficient: number
  curtailedLegacy: number
  curtailedEfficient: number
  gensetLegacy: number
  gensetEfficient: number
}

export interface FreeEnergy {
  station: string
  date: string
  rows: FreeEnergyRow[]
  totalCurtailedLegacy: number
  totalCurtailedEfficient: number
  gensetOffHoursEfficient: number
}

export interface GridStressRow {
  mult: number
  installedPv: number
  installedWind: number
  vPvRaw: number
  vWindRaw: number
  vPvFallback: number
  vWindFallback: number
  curtailPv: number
  curtailWind: number
  intervened: string[]
}

export interface GridStress {
  station: string
  rows: GridStressRow[]
  realPv: number
  realWind: number
}

export interface Scenarios {
  storm: StormRow[]
  wetstack: Wetstack
  freeenergy: FreeEnergy
  gridstress: GridStress
}
