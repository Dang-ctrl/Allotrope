import { Card } from '@astryxdesign/core/Card'
import { Heading } from '@astryxdesign/core/Heading'
import { Text } from '@astryxdesign/core/Text'
import { StatusDot } from '@astryxdesign/core/StatusDot'
import type { StatusDotVariant } from '@astryxdesign/core/StatusDot'

export function KpiCard({
  label,
  value,
  unit,
  statusVariant,
  statusLabel,
}: {
  label: string
  value: string
  unit?: string
  statusVariant?: StatusDotVariant
  statusLabel?: string
}) {
  return (
    <Card variant="default" elevation="low">
      <Heading level={3}>{label}</Heading>
      <Text size="xl">
        {value}
        {unit ? ` ${unit}` : ''}
      </Text>
      {statusVariant && statusLabel && (
        <StatusDot variant={statusVariant} label={statusLabel} isPulsing={statusVariant === 'error'} />
      )}
    </Card>
  )
}
