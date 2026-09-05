import type { StationConfig, TelemetryRow } from '../types'

export async function fetchStations(): Promise<StationConfig[]> {
  const res = await fetch('/api/stations')
  if (!res.ok) throw new Error(`GET /api/stations -> ${res.status}`)
  return res.json()
}

export async function fetchHistory(
  stationId: string,
  minutes = 60,
): Promise<{ station_id: string; rows: TelemetryRow[] }> {
  const res = await fetch(`/api/stations/${stationId}/telemetry/history?minutes=${minutes}`)
  if (!res.ok) throw new Error(`GET telemetry/history -> ${res.status}`)
  return res.json()
}
