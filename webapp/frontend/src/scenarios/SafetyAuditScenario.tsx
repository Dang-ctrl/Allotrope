import { Card } from '@astryxdesign/core/Card'
import { Heading } from '@astryxdesign/core/Heading'
import { Text } from '@astryxdesign/core/Text'
import { Grid } from '@astryxdesign/core/Grid'
import { HStack } from '@astryxdesign/core/HStack'
import { Badge } from '@astryxdesign/core/Badge'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { KpiCard } from '../components/KpiCard'
import type { SafetyAudit } from './types'

export function SafetyAuditScenario({ data }: { data: SafetyAudit }) {
  // Guarded is exactly 0 on every attack -- a bar of height 0 shows nothing,
  // so it's a reference line ("always 0") rather than a second, invisible
  // series. Only "Unguarded" (the control) needs a bar at all.
  const chartData = data.rows.map((r) => ({
    attack: r.attack,
    Unguarded: r.critLostUnguarded,
  }))

  return (
    <Grid columns={1}>
      <Card variant="default" elevation="low">
        <Heading level={2}>Attacking the safety layer, on purpose</Heading>
        <Text type="body">
          {data.station}, {data.days} midwinter days, five adversarial policies — random
          actions, shutting every machine down, charging storage flat out, melting flat out,
          and oscillating commitment every step — each replayed guarded and unguarded. The
          unguarded column is the control: without it, the guarded column proves nothing.
        </Text>
      </Card>

      <Grid columns={2}>
        <KpiCard
          label="Life support lost, guarded"
          value="0"
          unit={`kWh, every attack, ${data.days} days`}
          statusVariant="success"
          statusLabel="no attack got through"
        />
        <KpiCard
          label="Worst unguarded loss"
          value={data.rows.reduce((m, r) => Math.max(m, r.critLostUnguarded), 0).toFixed(0)}
          unit="kWh — the control"
          statusVariant="error"
          statusLabel="what the guard is for"
        />
      </Grid>

      <Card variant="default" elevation="low">
        <Heading level={3}>Life support lost per attack, unguarded (kWh)</Heading>
        <Text type="supporting">
          Guarded is not a second bar here because it is exactly zero on every attack —
          the dashed line below marks it.
        </Text>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="attack" tick={{ fontSize: 10 }} interval={0} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <ReferenceLine y={0} stroke="#1097ba" strokeWidth={2} label={{ value: 'Guarded: 0 kWh, always', position: 'insideTopLeft', fontSize: 11, fill: '#1097ba' }} />
            <Bar dataKey="Unguarded" name="Unguarded (control)" fill="#bc2a49" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card variant="default" elevation="low">
        <Heading level={3}>What the projection actually did, across all five attacks</Heading>
        <Grid columns={1}>
          {data.interventionCounts.map((i) => (
            <HStack key={i.name} gap={3}>
              <Badge variant="warning" label={i.name.replaceAll('_', ' ')} />
              <Text type="supporting">{i.count} times</Text>
            </HStack>
          ))}
        </Grid>
      </Card>
    </Grid>
  )
}
