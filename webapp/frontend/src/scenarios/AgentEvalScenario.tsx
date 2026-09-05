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
import type { AgentEval } from './types'

export function AgentEvalScenario({ data }: { data: AgentEval }) {
  const clearsBar = data.fuelVsEfficientPct > 0
  const startsWorse = data.startsVsEfficientDelta > 0
  const isSafe = data.maxCriticalUnservedAgent <= 0.001 && data.maxFreezeAgent === 0

  const fuelData = data.rows.map((r) => ({
    seed: `#${r.seed}`,
    Legacy: r.fuelLegacy,
    Efficient: r.fuelEfficient,
    'Hybrid agent': r.fuelAgent,
  }))
  const startsData = data.rows.map((r) => ({
    seed: `#${r.seed}`,
    Legacy: r.startsLegacy,
    Efficient: r.startsEfficient,
    'Hybrid agent': r.startsAgent,
  }))

  return (
    <Grid columns={1}>
      <Card variant="default" elevation="low">
        <Heading level={2}>
          The trained agent, on weather it never saw — {data.station}
        </Heading>
        <Text type="body">
          `HybridAgent` (DQN + SDDPG), evaluated on 5 held-out seeds disjoint from every
          training seed, full year each. The bar is `EfficientRuleBased`, not the incumbent —
          beating disciplined rules is what justifies training an agent at all.
        </Text>
        {!clearsBar && (
          <Badge
            variant="warning"
            label="This checkpoint does not clear the efficient rule-based bar on fuel — reported as measured, not re-trained until it did"
          />
        )}
      </Card>

      <Grid columns={4}>
        <KpiCard
          label="Fuel vs. efficient rules"
          value={(clearsBar ? '−' : '+') + Math.abs(data.fuelVsEfficientPct).toFixed(1)}
          unit="% (negative = less fuel)"
          statusVariant={clearsBar ? 'success' : 'warning'}
          statusLabel={clearsBar ? 'clears the bar' : 'does not clear the bar'}
        />
        <KpiCard
          label="Genset starts vs. efficient rules"
          value={(startsWorse ? '+' : '') + data.startsVsEfficientDelta.toFixed(0)}
          unit="per year"
          statusVariant={startsWorse ? 'warning' : 'success'}
          statusLabel={startsWorse ? 'more cycling' : 'less cycling'}
        />
        <KpiCard label="Mean fuel, agent" value={data.meanFuelAgent.toFixed(1)} unit="kL / year" />
        <KpiCard
          label="Life support, every held-out seed"
          value={isSafe ? '0' : data.maxCriticalUnservedAgent.toFixed(2)}
          unit="kWh max unserved"
          statusVariant={isSafe ? 'success' : 'error'}
          statusLabel={isSafe ? 'held on every seed' : 'investigate'}
        />
      </Grid>

      <Card variant="default" elevation="low">
        <Heading level={3}>Fuel per held-out seed (kL/year)</Heading>
        <Text type="supporting">Five independent seeds — not one favorable run.</Text>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={fuelData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="seed" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="Legacy" fill="#8b94a3" isAnimationActive={false} />
            <Bar dataKey="Efficient" fill="#1097ba" isAnimationActive={false} />
            <Bar dataKey="Hybrid agent" fill="#c9793a" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card variant="default" elevation="low">
        <Heading level={3}>Genset starts per held-out seed</Heading>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={startsData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="seed" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="Legacy" fill="#8b94a3" isAnimationActive={false} />
            <Bar dataKey="Efficient" fill="#1097ba" isAnimationActive={false} />
            <Bar dataKey="Hybrid agent" fill="#c9793a" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </Grid>
  )
}
