import { Card } from '@astryxdesign/core/Card'
import { Text } from '@astryxdesign/core/Text'
import { StatusDot } from '@astryxdesign/core/StatusDot'
import { ProgressBar } from '@astryxdesign/core/ProgressBar'
import type { GensetConfig } from '../types'

export function GensetCard({
  genset,
  online,
  powerKw,
  deposit,
}: {
  genset: GensetConfig
  online: boolean | undefined
  powerKw: number | undefined
  deposit: number | undefined
}) {
  const depositPct = (deposit ?? 0) * 100
  const variant =
    depositPct >= genset.burn_off_threshold_frac * 100
      ? 'error'
      : depositPct >= genset.wet_stack_threshold_frac * 100
        ? 'warning'
        : 'accent'

  return (
    <Card variant="default" elevation="low">
      <StatusDot
        variant={online ? 'success' : 'neutral'}
        label={online ? 'running' : 'stopped'}
        isPulsing={online === true}
      />
      <Text type="body">
        {genset.id} — {(powerKw ?? 0).toFixed(1)} / {genset.rated_kw} kW
      </Text>
      <ProgressBar
        label="wet-stack deposit"
        value={depositPct}
        hasValueLabel
        variant={variant}
        marks={[
          { value: genset.wet_stack_threshold_frac * 100, label: 'wet-stack onset' },
          { value: genset.burn_off_threshold_frac * 100, label: 'burn-off' },
        ]}
      />
    </Card>
  )
}
