import { Grid } from '@astryxdesign/core/Grid'
import { GensetCard } from './GensetCard'
import type { GensetConfig, Observation } from '../types'

export function GensetGrid({
  gensets,
  observation,
}: {
  gensets: GensetConfig[]
  observation: Observation | null
}) {
  return (
    <Grid columns={3}>
      {gensets.map((g, i) => (
        <GensetCard
          key={g.id}
          genset={g}
          online={observation?.genset_online[i]}
          powerKw={observation?.genset_power_kw[i]}
          deposit={observation?.genset_deposit[i]}
        />
      ))}
    </Grid>
  )
}
