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
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { ColdBattery } from './types'

export function ColdBatteryScenario({ data }: { data: ColdBattery }) {
  const packs = Object.entries(data.packs)
  const temps = packs[0][1].rows.map((r) => r.tempC)
  const chartData = temps.map((t, i) => {
    const row: Record<string, number> = { tempC: t }
    packs.forEach(([id, pack]) => {
      row[`${id}_charge`] = pack.rows[i].maxChargeKw
      row[`${id}_discharge`] = pack.rows[i].maxDischargeKw
    })
    return row
  })

  const colors: Record<string, string> = { BESS_LFP: '#1097ba', BESS_LTO: '#c9793a' }

  return (
    <Grid columns={1}>
      <Card variant="default" elevation="low">
        <Heading level={2}>Dual chemistry, and why it isn't redundant</Heading>
        <Text type="body">
          {data.station}'s battery packs, swept across temperature using their own asset
          model (`allotrope.sim.assets.Battery`) — not an illustration. The heated-core LFP
          pack and the exterior LTO pack are different chemistries for a reason: each has a
          different floor below which it can no longer be charged at all.
        </Text>
        {packs.map(([id, pack]) => (
          <Badge
            key={id}
            variant="neutral"
            label={`${id} (${pack.chemistry}, ${pack.location}) — charge floor ${pack.minOperatingTempC}°C`}
          />
        ))}
      </Card>

      <Grid columns={2}>
        <Card variant="default" elevation="low">
          <Heading level={3}>Max charge power vs. temperature (kW)</Heading>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="tempC" tick={{ fontSize: 11 }} label={{ value: '°C', position: 'insideBottom', offset: -5, fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              {packs.map(([id, pack]) => (
                <ReferenceLine
                  key={id}
                  x={pack.minOperatingTempC}
                  stroke={colors[id] ?? '#8b94a3'}
                  strokeDasharray="3 3"
                />
              ))}
              {packs.map(([id]) => (
                <Line
                  key={id}
                  type="monotone"
                  dataKey={`${id}_charge`}
                  name={id}
                  stroke={colors[id] ?? '#8b94a3'}
                  dot={false}
                  strokeWidth={2} isAnimationActive={false}
            />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card variant="default" elevation="low">
          <Heading level={3}>Max discharge power vs. temperature (kW)</Heading>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="tempC" tick={{ fontSize: 11 }} label={{ value: '°C', position: 'insideBottom', offset: -5, fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              {packs.map(([id]) => (
                <Line
                  key={id}
                  type="monotone"
                  dataKey={`${id}_discharge`}
                  name={id}
                  stroke={colors[id] ?? '#8b94a3'}
                  dot={false}
                  strokeWidth={2} isAnimationActive={false}
            />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </Grid>
      <Text type="supporting">
        Dashed vertical lines mark each pack's charge floor — below it, that pack accepts no
        charge at all, though (per the model) it may still discharge somewhat colder than
        that, since real chemistry permits discharge below the charge floor.
      </Text>
    </Grid>
  )
}
