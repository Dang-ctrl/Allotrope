import { Card } from '@astryxdesign/core/Card'
import { Section } from '@astryxdesign/core/Section'
import { Heading } from '@astryxdesign/core/Heading'
import { VStack } from '@astryxdesign/core/VStack'
import { EmptyState } from '@astryxdesign/core/EmptyState'
import { SafetyEventRow } from './SafetyEventRow'
import type { SafetyEvent } from '../types'

export function SafetyFeed({ events }: { events: SafetyEvent[] }) {
  return (
    <Card variant="default" elevation="low">
      <Section>
        <Heading level={2}>Safety interventions</Heading>
        {events.length === 0 ? (
          <EmptyState
            title="No interventions"
            description="The projection layer has not had to act."
          />
        ) : (
          <VStack gap={2}>
            {events
              .slice()
              .reverse()
              .map((event, i) => (
                <SafetyEventRow key={`${event.ts}-${i}`} event={event} />
              ))}
          </VStack>
        )}
      </Section>
    </Card>
  )
}
