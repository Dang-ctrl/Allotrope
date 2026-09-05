import { useEffect, useState } from 'react'
import { AppShell } from '@astryxdesign/core/AppShell'
import { TopNav, TopNavHeading } from '@astryxdesign/core/TopNav'
import { SegmentedControl, SegmentedControlItem } from '@astryxdesign/core/SegmentedControl'
import { Grid } from '@astryxdesign/core/Grid'
import { Spinner } from '@astryxdesign/core/Spinner'
import { fetchStations } from './api/client'
import { useLiveStation } from './api/useLiveStation'
import { StationSwitcher } from './components/StationSwitcher'
import { ConnectionBanner } from './components/ConnectionBanner'
import { StationHeader } from './components/StationHeader'
import { KpiRow } from './components/KpiRow'
import { GensetGrid } from './components/GensetGrid'
import { BatteryGauges } from './components/BatteryGauges'
import { TelemetryCharts } from './components/TelemetryCharts'
import { SafetyFeed } from './components/SafetyFeed'
import { ScenarioExplorer } from './scenarios/ScenarioExplorer'
import type { StationConfig } from './types'

type View = 'live' | 'scenarios'

function Dashboard({ station }: { station: StationConfig }) {
  const live = useLiveStation(station.id)

  return (
    <Grid columns={1}>
      <ConnectionBanner connected={live.connected} />
      <StationHeader station={station} />
      <KpiRow telemetry={live.telemetry} />
      <GensetGrid gensets={station.gensets} observation={live.observation} />
      <BatteryGauges storage={station.storage} observation={live.observation} />
      <TelemetryCharts stationId={station.id} />
      <SafetyFeed events={live.safetyEvents} />
    </Grid>
  )
}

export default function App() {
  const [stations, setStations] = useState<StationConfig[] | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [view, setView] = useState<View>('live')

  useEffect(() => {
    fetchStations().then((data) => {
      setStations(data)
      setSelected((current) => current ?? data[0]?.id ?? null)
    })
  }, [])

  const station = stations?.find((s) => s.id === selected) ?? null

  return (
    <AppShell
      contentPadding={4}
      topNav={
        <TopNav
          label="Allotrope navigation"
          heading={<TopNavHeading heading="Allotrope Mission Control" />}
          startContent={
            <SegmentedControl value={view} onChange={(v) => setView(v as View)} label="View">
              <SegmentedControlItem value="live" label="Live" />
              <SegmentedControlItem value="scenarios" label="Scenarios" />
            </SegmentedControl>
          }
          endContent={
            view === 'live' &&
            stations && <StationSwitcher stations={stations} value={selected!} onChange={setSelected} />
          }
        />
      }
    >
      {view === 'scenarios' ? (
        <ScenarioExplorer />
      ) : station ? (
        <Dashboard station={station} />
      ) : (
        <Spinner label="Loading station data" />
      )}
    </AppShell>
  )
}
