# Control plane

`allotrope/controlplane/` is a gRPC service distributing station state and
liveness — the wire-protocol half of README.md's architecture diagram
("safety projection -> actuation, gRPC < 10 ms"). It wraps the same
`allotrope.api.simulation.StationSimulation` objects the REST API serves:
one process, one set of simulations, two transports, never two sources of
truth.

## Run it

```bash
pip install -e ".[controlplane]"
python -m allotrope.controlplane.server --port 50051
```

The schema lives in `allotrope/controlplane/allotrope.proto` — read it
first; its own comments state what each field and RPC means and, in two
places, explicitly what it does *not* cover. Regenerate the Python stubs
after editing it:

```bash
python -m grpc_tools.protoc -I allotrope/controlplane \
  --python_out=allotrope/controlplane --grpc_python_out=allotrope/controlplane \
  allotrope/controlplane/allotrope.proto
```

(`allotrope_pb2_grpc.py`'s generated `import allotrope_pb2` needs
hand-editing to `from . import allotrope_pb2` afterward — protoc doesn't
know it's generating into a package. Both generated files are committed,
not built at install time, same as any other checked-in generated code.)

## What's implemented

- **`GetState(station_id) -> StationState`** — one snapshot: load, critical
  load, renewable availability, genset/battery state, and a nested
  `SafetyStatus` (interventions, fallback reason, guard rates) and
  `ControllerStatus` (name, type, the git commit the server was built
  from). `NOT_FOUND` for an unknown station — never a default or empty
  state standing in for one.
- **`StreamState(station_id) -> stream StationState`** — the same, pushed
  once a second until the client disconnects. Reconnecting is just calling
  it again: there's no server-side session to resume, and
  `sequence_number` (monotonic per station, never reset) lets a
  reconnecting client detect any gap itself.
- **`Heartbeat() -> HeartbeatResponse`** — liveness, uptime, model version,
  and the station list, independent of any one station. What a client
  should poll to tell "the controller process is gone" apart from "this
  one station has a problem."
- **A `Quality` flag on every state**: `GOOD` while the simulation is
  actively advancing, `STALE` when it isn't (a real, distinct signal from
  "the number is wrong" — it just isn't live), `INVALID` once the run is
  out of data. All three states are exercised in
  `tests/test_controlplane.py`, not just declared in the schema and left
  untested.
- **Real robustness behaviour, tested against a live server on a real
  ephemeral port with a real client** (not an in-memory stub):
  unknown-station requests get `NOT_FOUND` on both the unary and streaming
  RPCs; an already-expired client deadline gets `DEADLINE_EXCEEDED`, not a
  hang; cancelling a stream and opening a new one works with no leaked
  server-side state; loopback `GetState` calls average under the project's
  own 10 ms control-path figure (stated in the test as a loopback
  measurement only — this makes no claim about real network conditions).

## What's explicitly not implemented

- **No command-injection RPC.** The schema distributes *state*; it does not
  accept a command from a remote caller. This is a deliberate consequence
  of how the rest of the project is built, not an oversight: computing a
  command needs the full `PolarMicrogrid` object
  (`GuardedController.act(observation, plant)`), which a remote caller
  cannot supply without either serialising the entire plant on every call
  (defeating a low-latency path) or having the server accept a command
  blind, bypassing the safety projection this project's central guarantee
  depends on. The controller and the plant it commands stay in the same
  process; gRPC here is for a remote HMI or monitor to observe that
  process, not to drive it. See the `.proto` file's own module comment.
- **No MQTT.** The project's own architecture notes list "gRPC / MQTT as
  appropriate." MQTT is a pub/sub broker pattern suited to many lightweight
  sensor publishers and subscribers on a shared topic tree — a real fit for
  ingesting field telemetry from physical sensors, none of which exist for
  this project yet (`README.md`, "On the data"). Standing up a broker and a
  duplicate transport for the same simulation state gRPC already serves
  would be technology added for its own sake, which the project's own
  rules warn against. If and when this project ingests real per-sensor
  telemetry rather than a simulator's own state, that's the point MQTT
  becomes the right tool, not before.
- **No authentication or TLS** — `add_insecure_port`, matching the REST
  API having none either. Both need addressing before either is reachable
  from anywhere but localhost.
- **No persistence.** Sequence numbers and simulation state live only as
  long as the server process does, same as the REST API.
