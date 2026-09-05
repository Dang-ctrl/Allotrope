import { Card } from '@astryxdesign/core/Card'
import { Heading } from '@astryxdesign/core/Heading'
import { Text } from '@astryxdesign/core/Text'
import { Grid } from '@astryxdesign/core/Grid'
import { Badge } from '@astryxdesign/core/Badge'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { KpiCard } from '../components/KpiCard'
import type { Federated } from './types'

export function FederatedScenario({ data }: { data: Federated }) {
  const fuelData = [
    {
      station: 'Maitri',
      Efficient: data.maitri.meanFuelEfficient,
      Federated: data.maitri.meanFuelFederated,
      'Own checkpoint': data.maitri.meanFuelOwnCheckpoint,
    },
    {
      station: 'Bharati',
      Efficient: data.bharati.meanFuelEfficient,
      Federated: data.bharati.meanFuelFederated,
      'Own checkpoint': data.bharati.meanFuelOwnCheckpoint,
    },
  ]

  const bothSafe =
    data.maitri.maxCriticalUnservedFederated <= 0.001 &&
    data.maitri.maxFreezeFederated === 0 &&
    data.bharati.maxCriticalUnservedFederated <= 0.001 &&
    data.bharati.maxFreezeFederated === 0

  const beatsEfficient =
    data.maitri.meanFuelFederated < data.maitri.meanFuelEfficient &&
    data.bharati.meanFuelFederated < data.bharati.meanFuelEfficient

  return (
    <Grid columns={1}>
      <Card variant="default" elevation="low">
        <Heading level={2}>Federated training: a negative result, reported as one</Heading>
        <Text type="body">
          FedAvg across Maitri and Bharati simultaneously — only network parameters cross the
          station's satellite link, never weather or telemetry. Evaluated the same way as
          every other checkpoint, against both stations' held-out seeds.
        </Text>
        {!beatsEfficient && (
          <Badge
            variant="warning"
            label="The federated policy does not beat EfficientRuleBased at either station"
          />
        )}
      </Card>

      <Card variant="default" elevation="low">
        <Heading level={3}>Fuel by station (kL/year)</Heading>
        <Text type="supporting">
          Federated vs. the rule-based bar vs. each station's own dedicated single-station
          checkpoint — the comparison that actually matters.
        </Text>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={fuelData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="station" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="Efficient" fill="#1097ba" isAnimationActive={false} />
            <Bar dataKey="Federated" fill="#bc2a49" isAnimationActive={false} />
            <Bar dataKey="Own checkpoint" fill="#c9793a" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Grid columns={2}>
        <KpiCard
          label="Life support, both stations"
          value={bothSafe ? '0' : 'unsafe'}
          unit="kWh unserved — every held-out seed"
          statusVariant={bothSafe ? 'success' : 'error'}
          statusLabel={bothSafe ? 'held regardless of policy quality' : 'investigate'}
        />
        <KpiCard
          label="What this shows"
          value="Safety ≠ policy quality"
          unit="the guarantee holds even when the training doesn't"
        />
      </Grid>
    </Grid>
  )
}
