import { Grid } from '@astryxdesign/core/Grid'
import { KpiCard } from './KpiCard'
import type { Telemetry } from '../types'

export function KpiRow({ telemetry }: { telemetry: Telemetry | null }) {
  const criticalUnserved = telemetry?.critical_unserved_kw ?? 0
  const isSafe = criticalUnserved <= 0.001

  return (
    <Grid columns={4}>
      <KpiCard
        label="Fuel burn"
        value={telemetry ? telemetry.fuel_l.toFixed(1) : '—'}
        unit="L / sample"
      />
      <KpiCard
        label="Black carbon"
        value={telemetry ? telemetry.black_carbon_mg.toFixed(0) : '—'}
        unit="mg / sample"
      />
      <KpiCard
        label="Genset output"
        value={telemetry ? telemetry.genset_kw.toFixed(0) : '—'}
        unit="kW"
      />
      <KpiCard
        label="Critical unserved"
        value={criticalUnserved.toFixed(2)}
        unit="kW — should be zero"
        statusVariant={isSafe ? 'success' : 'error'}
        statusLabel={isSafe ? 'life support met' : 'life support at risk'}
      />
    </Grid>
  )
}
