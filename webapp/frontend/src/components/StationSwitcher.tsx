import { SegmentedControl, SegmentedControlItem } from '@astryxdesign/core/SegmentedControl'
import type { StationConfig } from '../types'

export function StationSwitcher({
  stations,
  value,
  onChange,
}: {
  stations: StationConfig[]
  value: string
  onChange: (id: string) => void
}) {
  return (
    <SegmentedControl value={value} onChange={onChange} label="Station">
      {stations.map((s) => (
        <SegmentedControlItem key={s.id} value={s.id} label={s.name} />
      ))}
    </SegmentedControl>
  )
}
