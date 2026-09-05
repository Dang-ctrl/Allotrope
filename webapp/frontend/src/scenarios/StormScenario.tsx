import { Card } from '@astryxdesign/core/Card'
import { Heading } from '@astryxdesign/core/Heading'
import { Text } from '@astryxdesign/core/Text'
import { Grid } from '@astryxdesign/core/Grid'
import { Badge } from '@astryxdesign/core/Badge'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { KpiCard } from '../components/KpiCard'
import type { StormRow } from './types'

export function StormScenario({ rows }: { rows: StormRow[] }) {
  const last = rows[rows.length - 1]
  const failureRows = rows.filter((r) => r.aiFailed)
  const failureWindow =
    failureRows.length > 0
      ? `${failureRows[0].date} ${failureRows[0].t}–${failureRows[failureRows.length - 1].t}`
      : 'n/a'

  return (
    <Grid columns={1}>
      <Card variant="default" elevation="low">
        <Heading level={2}>Blizzard, record cold, and an injected AI failure</Heading>
        <Text type="body">
          A learned controller is simulated crashing mid-inference for the duration of the
          blizzard — the worst possible moment. Replayed twice against identical weather: once
          behind the safety projection (guarded), once without it (unguarded, the control).
        </Text>
        <Badge variant="warning" label={`AI failure window: ${failureWindow}`} />
      </Card>

      <Grid columns={3}>
        <KpiCard
          label="Critical unserved, guarded"
          value={last.cumCritG.toFixed(0)}
          unit="kWh cumulative"
          statusVariant={last.cumCritG <= 0.001 ? 'success' : 'error'}
          statusLabel={last.cumCritG <= 0.001 ? 'life support held' : 'life support at risk'}
        />
        <KpiCard
          label="Critical unserved, unguarded"
          value={last.cumCritU.toFixed(0)}
          unit="kWh cumulative — the control"
          statusVariant={last.cumCritU > 0.001 ? 'error' : 'success'}
          statusLabel={last.cumCritU > 0.001 ? 'would have failed' : 'held anyway'}
        />
        <KpiCard
          label="Discretionary load shed, guarded"
          value={last.cumShedG.toFixed(0)}
          unit="kWh cumulative"
        />
      </Grid>

      <Card variant="default" elevation="low">
        <Heading level={3}>Cumulative life-support energy unserved (kWh)</Heading>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" tick={{ fontSize: 11 }} minTickGap={30} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="cumCritG"
              name="Guarded"
              stroke="#22c55e"
              dot={false}
              strokeWidth={2} isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="cumCritU"
              name="Unguarded (control)"
              stroke="#ef4444"
              dot={false}
              strokeWidth={2} isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card variant="default" elevation="low">
        <Heading level={3}>Weather during the window</Heading>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" tick={{ fontSize: 11 }} minTickGap={30} />
            <YAxis yAxisId="temp" tick={{ fontSize: 11 }} />
            <YAxis yAxisId="wind" orientation="right" tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            <Line
              yAxisId="temp"
              type="monotone"
              dataKey="temp"
              name="Air temp (°C)"
              stroke="#38bdf8"
              dot={false} isAnimationActive={false}
            />
            <Line
              yAxisId="wind"
              type="monotone"
              dataKey="wind"
              name="Wind (m/s)"
              stroke="#a78bfa"
              dot={false} isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </Grid>
  )
}
