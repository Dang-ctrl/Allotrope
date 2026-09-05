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

The full stack has been run end to end, not just written: all six containers
started, stayed up, and moved real telemetry from a simulated plant through
gRPC actuation, the safety projection, MQTT, and a TimescaleDB insert, queryable
by the exact SQL the provisioned Grafana dashboard uses.

| Piece | How it is actually verified |
|---|---|
| MQTT publish/subscribe, malformed-payload handling | `tests/test_mqtt.py`, against a real embedded MQTT broker |
| gRPC actuation, safety projection over the wire | `tests/test_rpc.py`, client and server in-process |
| TimescaleDB bridge SQL and error handling | `tests/test_timescale_bridge.py`, against a fake connection |
| The full loop: plant to gRPC to MQTT to subscriber to TimescaleDB | **run for real**: `docker compose up`, both stations publishing, rows landing in `telemetry` with `critical_unserved_kw = 0` on every row |
| Grafana's datasource and provisioned dashboard | **run for real**: `/api/datasources` and `/api/search` confirm the TimescaleDB datasource and "Allotrope Station Overview" dashboard both loaded; the dashboard's own panel queries (genset/load/renewable, the critical-unserved stat) return real rows when run directly against the database |
| The compose file's service wiring | **run for real**: `docker compose ps` after several minutes shows all six containers still `Up`, no crash loop |

Two real bugs were caught only by actually starting the containers, not by
`docker compose config` or by anything in the test suite: `protobuf` (needed by
the generated gRPC stubs) and `psycopg` (needed by the TimescaleDB bridge) were
both installed by hand in the development venv at some point and never added
to `pyproject.toml`'s actual dependency list. Locally this was invisible --
`grpcio-tools` pulls in `protobuf` as a side effect, and `psycopg` had simply
been `pip install`ed directly into the venv and forgotten. In a fresh container
building only from `pyproject.toml`, both services crashed on import within a
second of starting. Fixed by declaring both as real dependencies; this is
exactly the class of gap containerizing is supposed to surface, and it did.

**What is not yet verified**: the dashboard's panels have not been visually
inspected rendering in a browser, only confirmed to be wired to a working
datasource with queries that return correct data against it. That is a much
smaller gap than "the stack has never been started" was.

## Services

- `mosquitto` -- the telemetry and federated-update broker.
- `timescaledb` -- schema applied from `init-timescaledb.sql` on first start.
- `grafana` -- provisioned datasource and dashboard from `grafana/provisioning/`.
- `bridge` -- `scripts/run_timescale_bridge.py`, MQTT to TimescaleDB.
- `station-maitri`, `station-bharati` -- `scripts/run_station_service.py`,
  each running its own plant, actuation server, and controller loop.

## Running a station without Docker

Every piece also runs as a plain Python process:

```bash
# broker (needs mosquitto installed, or point --mqtt-host at any broker)
mosquitto -c deploy/mosquitto.conf

# a station, with a trained checkpoint if one exists
python scripts/run_station_service.py --station maitri --checkpoint checkpoints/maitri.pt

# the bridge (needs a running Postgres/TimescaleDB)
python scripts/run_timescale_bridge.py --stations maitri bharati
```
