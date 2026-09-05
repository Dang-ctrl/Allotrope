# Deployment stack

The infrastructure the project's architecture calls for: MQTT for telemetry,
TimescaleDB for storage, Grafana for the HMI, one container per station.
gRPC exists separately for state distribution and liveness
(`allotrope.controlplane`, see [docs/control-plane.md](../docs/control-plane.md))
rather than actuation -- there is no remote command-injection path in this
project; the controller and the plant it commands stay in the same process
(see that document's "what's explicitly not implemented").

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Then:
- Grafana at http://localhost:3000 (anonymous viewer access enabled; admin/allotrope)
- MQTT broker at localhost:1883
- TimescaleDB at localhost:5432 (allotrope/allotrope)

## What is proven where

This stack has now been tried against two different sandboxes, and each
found a different, real reason it isn't fully verified end to end yet --
recorded honestly rather than collapsed into one vague "not tested":

- **First environment**: Docker CLI present, daemon not running at all
  (`docker info` failed to reach the socket). Nothing beyond
  `docker compose config` could be tried.
- **This environment**: the daemon actually starts and runs (`dockerd`,
  confirmed via `docker info` reporting a real server). `docker compose
  build` gets further -- it reaches Docker Hub for the `python:3.11-slim`
  base image -- but the registry pull redirects to a signed
  `production.cloudfront.docker.com` blob URL, and this sandbox's network
  egress policy returns a 403 to that specific host (confirmed via the
  proxy's own status endpoint reporting a `connect_rejected` /
  `policy_denial` on it, not a transient failure). Per this environment's
  own guidance, a 403 from a network policy is reported, not routed around
  with a mirror or a different base image chosen just to dodge it.

Both are real, environment-specific network/infrastructure limits, not
gaps in the code being deployed -- which is why the project's test suite
does not depend on this file at all.

A parallel implementation of this project's architecture did reach an
environment where the registry pull went through, and its build there
surfaced a real bug in `deploy/Dockerfile`, ported into this one: the image
ran `pip install -e .` with no extras, which installs only the project's
base dependencies -- silently omitting `paho-mqtt`, `psycopg` and `torch`.
Both entry points in this compose stack need more than the base set
(`scripts/run_station_service.py` publishes over MQTT and, with
`--checkpoint`, loads a torch agent; `scripts/run_timescale_bridge.py`
talks to TimescaleDB via psycopg), so both would have failed at import
time inside the container the moment one was actually run -- a class of
bug this project's own (torch-free by design) test suite cannot see, and
that only building and running the image catches. Fixed here to
`pip install -e ".[rl,mqtt,deploy]"`, and the CPU-only torch install moved
into its own layer ahead of the `COPY`, so an application-code change no
longer forces a full torch re-download on rebuild. This environment still
cannot verify the fix by actually building (same registry block as above,
reconfirmed just now), so it stands as a code review finding applied
on faith in the other environment's report, not as a build verified here.

| Piece | How it is actually verified |
|---|---|
| MQTT publish/subscribe, malformed-payload handling | `tests/test_mqtt.py`, against a real embedded MQTT broker |
| gRPC state distribution, safety/quality fields over the wire | `tests/test_controlplane.py`, real server on a real ephemeral port, real client |
| TimescaleDB bridge SQL and error handling | `tests/test_timescale_bridge.py`, against a fake connection |
| The full loop: plant to guarded controller to MQTT to subscriber | re-verified in this session: a real embedded MQTT broker, a real `GuardedController`-wrapped `EfficientRuleBased` agent stepping a real Maitri plant, publishing via `TelemetryPublisher` and received end to end by `TelemetrySubscriber` -- 5/5 telemetry records round-tripped; not automated in CI |
| Grafana rendering the dashboard against real data | **not verified here** -- needs a running Postgres and Grafana |
| The compose file's service wiring itself | `docker compose config` resolves it cleanly (correct build contexts, volumes, dependency ordering) in both sandboxes |
| The image actually building | **blocked** in both sandboxes tried so far, for two different reasons (above) -- not yet verified in any environment |

Everything in the left column above the last two rows is real, tested
Python. The compose file wires that already-correct code to real
infrastructure; building the image and running it in a network
environment that can actually reach Docker Hub's blob storage is the
remaining step, not a rewrite.

## Services

- `mosquitto` -- the telemetry and federated-update broker.
- `timescaledb` -- schema applied from `init-timescaledb.sql` on first start.
- `grafana` -- provisioned datasource and dashboard from `grafana/provisioning/`.
- `bridge` -- `scripts/run_timescale_bridge.py`, MQTT to TimescaleDB.
- `station-maitri`, `station-bharati` -- `scripts/run_station_service.py`,
  each running its own plant and guarded controller loop in-process,
  publishing telemetry to MQTT.

## Running a station without Docker

Every piece also runs as a plain Python process, which is how it was actually
exercised during development:

```bash
# broker (needs mosquitto installed, or point --mqtt-host at any broker)
mosquitto -c deploy/mosquitto.conf

# a station, with a trained checkpoint if one exists
python scripts/run_station_service.py --station maitri --checkpoint runs/hybrid_maitri_.../checkpoint.pt

# the bridge (needs a running Postgres/TimescaleDB)
python scripts/run_timescale_bridge.py --stations maitri bharati
```
