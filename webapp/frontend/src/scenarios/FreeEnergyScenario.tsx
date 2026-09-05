import { Card } from '@astryxdesign/core/Card'
import { Heading } from '@astryxdesign/core/Heading'
import { Text } from '@astryxdesign/core/Text'
import { Grid } from '@astryxdesign/core/Grid'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { KpiCard } from '../components/KpiCard'
import type { FreeEnergy } from './types'

export function FreeEnergyScenario({ data }: { data: FreeEnergy }) {
  const curtailedAvoided = data.totalCurtailedLegacy - data.totalCurtailedEfficient

  return (
    <Grid columns={1}>
      <Card variant="default" elevation="low">
        <Heading level={2}>A windy day, and what disciplined dispatch does with it</Heading>
        <Text type="body">
          {data.station}, {data.date} — the larger of the two fleets. Same available wind and
          solar, two dispatch policies: one curtails it away rather than ramping gensets down,
          the other captures nearly all of it.
        </Text>
      </Card>

      <Grid columns={3}>
        <KpiCard
          label="Curtailed, legacy"
          value={data.totalCurtailedLegacy.toFixed(0)}
          unit="kWh wasted"
        />
        <KpiCard
          label="Curtailed, efficient"
          value={data.totalCurtailedEfficient.toFixed(0)}
          unit={`kWh — ${curtailedAvoided.toFixed(0)} kWh recovered`}
          statusVariant="success"
          statusLabel="renewable energy captured"
        />
        <KpiCard
          label="Hours off gensets"
          value={data.gensetOffHoursEfficient.toString()}
          unit="of 24, under efficient dispatch"
        />
      </Grid>

      <Card variant="default" elevation="low">
        <Heading level={3}>Renewable available vs. used (kW)</Heading>
        <ResponsiveContainer width="100%" height={260}>
          <ComposedChart data={data.rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" tick={{ fontSize: 11 }} minTickGap={30} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            <Area
              type="monotone"
              dataKey="renewAvail"
              name="Available"
              stroke="#94a3b8"
              fill="#94a3b8"
              fillOpacity={0.15} isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="renewUsedLegacy"
              name="Used, legacy"
              stroke="#ef4444"
              dot={false}
              strokeWidth={2} isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="renewUsedEfficient"
              name="Used, efficient"
              stroke="#22c55e"
              dot={false}
              strokeWidth={2} isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </Card>
    </Grid>
  )
}
