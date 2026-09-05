import { Card } from '@astryxdesign/core/Card'
import { Heading } from '@astryxdesign/core/Heading'
import { Text } from '@astryxdesign/core/Text'
import { Grid } from '@astryxdesign/core/Grid'
import { Badge } from '@astryxdesign/core/Badge'
import { HStack } from '@astryxdesign/core/HStack'
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
import type { GridStress } from './types'

const VOLTAGE_CEILING_PU = 1.10

export function GridStressScenario({ data }: { data: GridStress }) {
  const firstIntervened = data.rows.find((r) => r.intervened.length > 0)

  return (
    <Grid columns={1}>
      <Card variant="default" elevation="low">
        <Heading level={2}>Forward stress test: scaling renewables 1×–6×</Heading>
        <Text type="body">
          Not a claim about today's grid — {data.station}'s installed capacity today never
          approaches this limit ({data.realPv.toFixed(0)} kW PV, {data.realWind.toFixed(0)} kW
          wind available at this snapshot). This scales the fleet the way a "Maitri II"-scale
          expansion might, to see where the Volt-VAr/Volt-Watt fallback actually has to act.
        </Text>
        {firstIntervened && (
          <Badge
            variant="warning"
            label={`Fallback first intervenes at ${firstIntervened.mult}× installed capacity`}
          />
        )}
      </Card>

      <Card variant="default" elevation="low">
        <Heading level={3}>Bus voltage vs. installed capacity multiplier (pu)</Heading>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data.rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="mult" tick={{ fontSize: 11 }} label={{ value: '× installed', position: 'insideBottom', offset: -5, fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} domain={[0.95, 1.15]} />
            <Tooltip />
            <Legend />
            <ReferenceLine
              y={VOLTAGE_CEILING_PU}
              stroke="#ef4444"
              strokeDasharray="4 4"
              label={{ value: '1.10 pu ceiling', position: 'insideTopRight', fontSize: 11, fill: '#ef4444' }}
            />
            <Line type="monotone" dataKey="vPvRaw" name="PV bus, uncorrected" stroke="#f59e0b" strokeDasharray="4 2" dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="vPvFallback" name="PV bus, with fallback" stroke="#f59e0b" dot={false} strokeWidth={2} isAnimationActive={false} />
            <Line type="monotone" dataKey="vWindRaw" name="Wind bus, uncorrected" stroke="#38bdf8" strokeDasharray="4 2" dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="vWindFallback" name="Wind bus, with fallback" stroke="#38bdf8" dot={false} strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card variant="default" elevation="low">
        <Heading level={3}>Curtailment and interventions, per multiplier</Heading>
        <Text type="supporting">
          `curtailPv`/`curtailWind` are the fraction of available power still
          allowed through (1.0 = none curtailed, 0.0 = fully curtailed) — shown
          below as the curtailed percentage, its complement.
        </Text>
        <Grid columns={1}>
          {data.rows.map((r) => (
            <HStack key={r.mult} gap={3}>
              <Text type="body">{r.mult}×</Text>
              <Text type="supporting">
                PV curtailed {((1 - r.curtailPv) * 100).toFixed(0)}% · Wind curtailed{' '}
                {((1 - r.curtailWind) * 100).toFixed(0)}%
              </Text>
              {r.intervened.length > 0 ? (
                r.intervened.map((bus) => <Badge key={bus} variant="warning" label={bus} />)
              ) : (
                <Badge variant="success" label="no intervention needed" />
              )}
            </HStack>
          ))}
        </Grid>
      </Card>
    </Grid>
  )
}
