# Deployment stack

The infrastructure the deck describes: MQTT for telemetry, TimescaleDB for
storage, Grafana for the HMI, gRPC for actuation, one container per station.

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Then:
- Grafana at http://localhost:3000 (anonymous viewer access enabled; admin/allotrope)
- The web API at http://localhost:8000 (`/api/health` is the quickest check)
- MQTT broker at localhost:1883
- TimescaleDB at localhost:5432 (allotrope/allotrope)

The operator UI is **not** in this stack on purpose — it runs from its own
toolchain so a demo never waits on an image build:

```bash
cd webapp/frontend && npm install && npm run dev   # then http://localhost:5173
```

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
| The compose file's service wiring | **run for real**: `docker compose ps` after several minutes shows all seven containers still `Up`, no crash loop |
| The web API (`allotrope/api`) | **run for real**: `/api/stations`, `/api/health` (mqtt and grpc true for both stations), `/api/stations/{id}/telemetry/history` and the `/ws/stations/{id}` feed all serving live data from the running stack |
| The operator UI (`webapp/frontend`) | **run for real** in a browser against the live stack: both stations, live KPIs, per-genset wet-stack deposit, per-pack battery SoC, history charts, and real safety interventions in the feed |
| Recovery from a broker restart | **run for real**, twice: a fast `docker compose restart mosquitto` resumes rows in ~45 s; a 5-minute `stop`/`start` outage took ~2.5 min to resume (paho's reconnect backoff doubles to a 120 s ceiling). Unattended in both cases, and the UI's disconnect banner appears during the outage and clears on its own after |

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

A third bug was caught the same way, by taking a running service away rather
than by any test: `TelemetrySubscriber` subscribed to its topics once in
`__init__` instead of in an `on_connect` callback. paho reconnects to a
restarted broker on its own, but a reconnect is a fresh MQTT session carrying
no subscriptions, and paho does not restore them -- so after
`docker compose restart mosquitto` the bridge reconnected, looked healthy, and
never wrote another row. Both subscribers now subscribe in `on_connect`, with
tests that restart a real broker mid-test.

**What is not yet verified**: Grafana's panels have since been confirmed
rendering real moving data in a browser. What remains unverified is running a
*trained checkpoint* inside a container -- every container run so far has used
the rule-based controller -- and anything about behaviour under a constrained
satellite-like link or multiple concurrent UI viewers.

## Services

- `mosquitto` -- the telemetry and federated-update broker.
- `timescaledb` -- schema applied from `init-timescaledb.sql` on first start.
- `grafana` -- provisioned datasource and dashboard from `grafana/provisioning/`.
- `bridge` -- `scripts/run_timescale_bridge.py`, MQTT to TimescaleDB.
- `station-maitri`, `station-bharati` -- `scripts/run_station_service.py`,
  each running its own plant, actuation server, and controller loop.
- `api` -- `scripts/run_api.py`, the read-only web API the operator UI runs on.
  It polls each station's gRPC `Observe` and subscribes to both MQTT topics; it
  never calls `Dispatch`, which would double-step a plant the station service
  is already stepping.

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
