import { Card } from '@astryxdesign/core/Card'
import { Badge } from '@astryxdesign/core/Badge'
import { ProgressBar } from '@astryxdesign/core/ProgressBar'
import type { StorageConfig } from '../types'

export function BatteryGaugeCard({ storage, soc }: { storage: StorageConfig; soc: number | undefined }) {
  const socPct = (soc ?? 0) * 100
  return (
    <Card variant="default" elevation="low">
      <Badge variant="neutral" label={storage.chemistry.toUpperCase()} />
      <ProgressBar
        label={`${storage.id} state of charge`}
        value={socPct}
        hasValueLabel
        marks={[
          { value: storage.soc_min * 100, label: 'min' },
          { value: storage.soc_max * 100, label: 'max' },
        ]}
      />
    </Card>
  )
}
