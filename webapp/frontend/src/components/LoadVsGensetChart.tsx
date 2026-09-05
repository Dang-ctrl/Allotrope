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

export function LoadVsGensetChart({ rows }: { rows: TelemetryRow[] }) {
  const data = rows.map((r) => ({
    time: new Date(r.time).toLocaleTimeString(),
    electrical_load_kw: r.electrical_load_kw,
    genset_kw: r.genset_kw,
    renewable_used_kw: r.renewable_used_kw,
  }))

  return (
    <Card variant="default" elevation="low">
      <Heading level={3}>Load vs. generation (kW)</Heading>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" tick={{ fontSize: 11 }} minTickGap={40} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip />
          <Line type="monotone" dataKey="electrical_load_kw" name="Load" stroke="#f59e0b" dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="genset_kw" name="Genset" stroke="#22c55e" dot={false} isAnimationActive={false} />
          <Line
            type="monotone"
            dataKey="renewable_used_kw"
            name="Renewable"
            stroke="#06b6d4"
            dot={false} isAnimationActive={false}
            />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  )
}
