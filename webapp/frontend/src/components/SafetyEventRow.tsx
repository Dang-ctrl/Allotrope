import { HStack } from '@astryxdesign/core/HStack'
import { Badge } from '@astryxdesign/core/Badge'
import { Timestamp } from '@astryxdesign/core/Timestamp'
import type { SafetyEvent } from '../types'

export function SafetyEventRow({ event }: { event: SafetyEvent }) {
  return (
    <HStack gap={2}>
      <Timestamp value={event.ts} format="relative_short" />
      <HStack gap={1}>
        {event.interventions.map((i) => (
          <Badge key={i} variant="warning" label={i.replaceAll('_', ' ')} />
        ))}
      </HStack>
    </HStack>
  )
}
