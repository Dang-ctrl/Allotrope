import { Card } from '@astryxdesign/core/Card'
import { Heading } from '@astryxdesign/core/Heading'
import { Text } from '@astryxdesign/core/Text'
import { Grid } from '@astryxdesign/core/Grid'
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
import type { Wetstack } from './types'

export function WetstackScenario({ data }: { data: Wetstack }) {
  const fuelSavedPct = (100 * (1 - data.totalFuelEfficient / data.totalFuelLegacy)).toFixed(1)
  const bcSavedPct = (100 * (1 - data.totalBcEfficient / data.totalBcLegacy)).toFixed(1)

  return (
    <Grid columns={1}>
      <Card variant="default" elevation="low">
        <Heading level={2}>The founding problem: one generator, two weeks</Heading>
        <Text type="body">
          Same station ({data.station}), same weather, two dispatch policies over the same
          fortnight. `LegacyNPlusOne` idles a spare set at a bare fraction of its rating;
          `EfficientRuleBased` runs sets in their efficient band instead.
        </Text>
      </Card>

      <Grid columns={4}>
        <KpiCard label="Fuel, legacy" value={data.totalFuelLegacy.toFixed(0)} unit="L / 2 weeks" />
        <KpiCard
          label="Fuel, efficient"
          value={data.totalFuelEfficient.toFixed(0)}
          unit={`L — ${fuelSavedPct}% less`}
          statusVariant="success"
          statusLabel="fuel saved"
        />
        <KpiCard label="Black carbon, legacy" value={data.totalBcLegacy.toFixed(0)} unit="g / 2 weeks" />
        <KpiCard
          label="Black carbon, efficient"
          value={data.totalBcEfficient.toFixed(0)}
          unit={`g — ${bcSavedPct}% less`}
          statusVariant="success"
          statusLabel="black carbon avoided"
        />
      </Grid>

      <Grid columns={2}>
        <Card variant="default" elevation="low">
          <Heading level={3}>Wet-stack deposit (mean, %)</Heading>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={data.rows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="depositLegacy"
                name="Legacy"
                stroke="#ef4444"
                dot={false}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="depositEfficient"
                name="Efficient"
                stroke="#22c55e"
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card variant="default" elevation="low">
          <Heading level={3}>Wet-stacking fraction of steps (%)</Heading>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={data.rows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="wetStackLegacy"
                name="Legacy"
                stroke="#ef4444"
                dot={false}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="wetStackEfficient"
                name="Efficient"
                stroke="#22c55e"
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </Grid>
    </Grid>
  )
}
