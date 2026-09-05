"""A web-facing API over the station telemetry/config/safety surfaces.

Nothing here calls `Dispatch` -- the station service already drives its own
control loop via that RPC, and a second caller would double-step the plant.
This package only ever reads: `Observe` over gRPC, the telemetry/safety topics
over MQTT, the `telemetry` table in TimescaleDB, and the station YAML.
"""
