# Deployment stack

The infrastructure the deck describes: MQTT for telemetry, TimescaleDB for
storage, Grafana for the HMI, gRPC for actuation, one container per station.

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Then:
- Grafana at http://localhost:3000 (anonymous viewer access enabled; admin/allotrope)
- MQTT broker at localhost:1883
- TimescaleDB at localhost:5432 (allotrope/allotrope)

## What is proven where

This compose stack was written and reasoned through in a sandboxed development
environment with the Docker **CLI** present but no running daemon --
`docker compose up` has not been executed against it end to end. That is stated
plainly rather than glossed over, and it is also why the project's test suite
does not depend on this file at all:

| Piece | How it is actually verified |
|---|---|
| MQTT publish/subscribe, malformed-payload handling | `tests/test_mqtt.py`, against a real embedded MQTT broker |
| gRPC actuation, safety projection over the wire | `tests/test_rpc.py`, client and server in-process |
| TimescaleDB bridge SQL and error handling | `tests/test_timescale_bridge.py`, against a fake connection |
| The full loop: plant to gRPC to MQTT to subscriber | smoke-tested manually in this environment, not in CI |
| Grafana rendering the dashboard against real data | **not verified here** -- needs a running Postgres and Grafana |
| The compose file's service wiring itself | **not verified here** -- needs a running Docker daemon |

Everything in the left column is real, tested Python. The compose file wires
that already-correct code to real infrastructure; running it is the remaining
step, not a rewrite.

## Services

- `mosquitto` -- the telemetry and federated-update broker.
- `timescaledb` -- schema applied from `init-timescaledb.sql` on first start.
- `grafana` -- provisioned datasource and dashboard from `grafana/provisioning/`.
- `bridge` -- `scripts/run_timescale_bridge.py`, MQTT to TimescaleDB.
- `station-maitri`, `station-bharati` -- `scripts/run_station_service.py`,
  each running its own plant, actuation server, and controller loop.

## Running a station without Docker

Every piece also runs as a plain Python process, which is how it was actually
exercised during development:

```bash
# broker (needs mosquitto installed, or point --mqtt-host at any broker)
mosquitto -c deploy/mosquitto.conf

# a station, with a trained checkpoint if one exists
python scripts/run_station_service.py --station maitri --checkpoint checkpoints/maitri.pt

# the bridge (needs a running Postgres/TimescaleDB)
python scripts/run_timescale_bridge.py --stations maitri bharati
```
