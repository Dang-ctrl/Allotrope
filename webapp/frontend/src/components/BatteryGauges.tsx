import { HStack } from '@astryxdesign/core/HStack'
import { BatteryGaugeCard } from './BatteryGaugeCard'
import type { Observation, StorageConfig } from '../types'

export function BatteryGauges({
  storage,
  observation,
}: {
  storage: StorageConfig[]
  observation: Observation | null
}) {
  return (
    <HStack gap={4}>
      {storage.map((s, i) => (
        <BatteryGaugeCard key={s.id} storage={s} soc={observation?.battery_soc[i]} />
      ))}
    </HStack>
  )
}
