import { Card } from '@astryxdesign/core/Card'
import { Heading } from '@astryxdesign/core/Heading'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { TelemetryRow } from '../types'

export function FuelAndBlackCarbonChart({ rows }: { rows: TelemetryRow[] }) {
  const data = rows.map((r) => ({
    time: new Date(r.time).toLocaleTimeString(),
    fuel_l: r.fuel_l,
    black_carbon_mg: r.black_carbon_mg,
  }))

  return (
    <Card variant="default" elevation="low">
      <Heading level={3}>Fuel &amp; black carbon</Heading>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" tick={{ fontSize: 11 }} minTickGap={40} />
          <YAxis yAxisId="fuel" tick={{ fontSize: 11 }} />
          <YAxis yAxisId="bc" orientation="right" tick={{ fontSize: 11 }} />
          <Tooltip />
          <Line yAxisId="fuel" type="monotone" dataKey="fuel_l" name="Fuel (L)" stroke="#3b82f6" dot={false} />
          <Line
            yAxisId="bc"
            type="monotone"
            dataKey="black_carbon_mg"
            name="Black carbon (mg)"
            stroke="#ef4444"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  )
}
