import { Section } from '@astryxdesign/core/Section'
import { Heading } from '@astryxdesign/core/Heading'
import { MetadataList, MetadataListItem } from '@astryxdesign/core/MetadataList'
import type { StationConfig } from '../types'

export function StationHeader({ station }: { station: StationConfig }) {
  return (
    <Section>
      <Heading level={1}>{station.name}</Heading>
      <MetadataList columns={4}>
        <MetadataListItem label="Position">
          {station.lat.toFixed(2)}°, {station.lon.toFixed(2)}°
        </MetadataListItem>
        <MetadataListItem label="Elevation">{station.elevation_m} m</MetadataListItem>
        <MetadataListItem label="Winter crew">{station.occupancy.winter_crew}</MetadataListItem>
        <MetadataListItem label="Summer crew">{station.occupancy.summer_crew}</MetadataListItem>
      </MetadataList>
    </Section>
  )
}
