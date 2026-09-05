import { useEffect, useState } from 'react'
import { Section } from '@astryxdesign/core/Section'
import { Grid } from '@astryxdesign/core/Grid'
import { fetchHistory } from '../api/client'
import { FuelAndBlackCarbonChart } from './FuelAndBlackCarbonChart'
import { LoadVsGensetChart } from './LoadVsGensetChart'
import type { TelemetryRow } from '../types'

const REFRESH_MS = 15_000

export function TelemetryCharts({ stationId }: { stationId: string }) {
  const [rows, setRows] = useState<TelemetryRow[]>([])

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const { rows } = await fetchHistory(stationId, 60)
        if (!cancelled) setRows(rows)
      } catch {
        // A transient fetch failure just means the charts keep their last
        // known data until the next tick.
      }
    }

    load()
    const interval = setInterval(load, REFRESH_MS)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [stationId])

  return (
    <Section>
      <Grid columns={2}>
        <FuelAndBlackCarbonChart rows={rows} />
        <LoadVsGensetChart rows={rows} />
      </Grid>
    </Section>
  )
}
