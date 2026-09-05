import { Banner } from '@astryxdesign/core/Banner'

export function ConnectionBanner({ connected }: { connected: boolean }) {
  if (connected) return null
  return (
    <Banner
      status="warning"
      title="Live feed disconnected"
      description="Reconnecting to the station's telemetry feed…"
      isDismissable={false}
    />
  )
}
