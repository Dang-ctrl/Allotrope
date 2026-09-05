import { useState } from 'react'
import { TabList, Tab } from '@astryxdesign/core/TabList'
import { Spinner } from '@astryxdesign/core/Spinner'
import { Banner } from '@astryxdesign/core/Banner'
import { Grid } from '@astryxdesign/core/Grid'
import { useScenarios } from './useScenarios'
import { StormScenario } from './StormScenario'
import { WetstackScenario } from './WetstackScenario'
import { FreeEnergyScenario } from './FreeEnergyScenario'
import { GridStressScenario } from './GridStressScenario'

const TABS = [
  { value: 'storm', label: 'Storm + AI failure' },
  { value: 'wetstack', label: 'Wet-stacking' },
  { value: 'freeenergy', label: 'Free energy' },
  { value: 'gridstress', label: 'Grid stress test' },
] as const

export function ScenarioExplorer() {
  const [tab, setTab] = useState<string>('storm')
  const state = useScenarios()

  if (state.status === 'loading') {
    return <Spinner label="Loading scenario data" />
  }
  if (state.status === 'error') {
    return (
      <Banner
        status="error"
        title="Could not load scenarios.json"
        description={`${state.message} — run \`python scripts/generate_scenarios.py\` and copy the output into webapp/frontend/public/.`}
        isDismissable={false}
      />
    )
  }

  const { data } = state

  return (
    <Grid columns={1}>
      <TabList value={tab} onChange={setTab} aria-label="Scenario">
        {TABS.map((t) => (
          <Tab key={t.value} value={t.value} label={t.label} />
        ))}
      </TabList>

      {tab === 'storm' && <StormScenario rows={data.storm} />}
      {tab === 'wetstack' && <WetstackScenario data={data.wetstack} />}
      {tab === 'freeenergy' && <FreeEnergyScenario data={data.freeenergy} />}
      {tab === 'gridstress' && <GridStressScenario data={data.gridstress} />}
    </Grid>
  )
}
