# Backend API

`allotrope/api/` serves one thing honestly: the same in-process `PolarMicrogrid`
+ `GuardedController` state every CLI script and test in this repository
already exercises. There is no physical link to Maitri or Bharati (see
[docs/calibration.md](calibration.md) and README's "On the data") -- every
response is `"mode": "simulation"`, and every consumer, including the
frontend this exists for, should surface that rather than imply otherwise.

## Run it

```bash
pip install -e ".[api]"
uvicorn allotrope.api.app:app --reload
```

Interactive docs (from FastAPI's own OpenAPI generation, not written by
hand) at `http://localhost:8000/docs`.

## What's here

One `StationSimulation` per configured station (`maitri`, `bharati`) lives
for the life of the process: a real `PolarMicrogrid` stepping through a
synthetic year, wrapped in a real `GuardedController`. The default
controller is `EfficientRuleBased` -- the best policy this project can
currently stand behind by default, since no learned checkpoint is
production-ready yet (`docs/reinforcement-learning.md`, "Honest status").
Swapping in a trained checkpoint is a change to
`allotrope.api.simulation.default_controller`, not to the API surface.

| Endpoint | Backed by |
|---|---|
| `GET /stations` | `available_stations()` + each simulation's live step count |
| `GET /stations/{id}` | the station's `StationConfig` (gensets, storage, site) |
| `GET /stations/{id}/state` | `plant.observe()` on the current step, `mode: "simulation"` |
| `GET /stations/{id}/telemetry` | the last `HISTORY_LEN` (500) steps of real `plant.step()` telemetry |
| `GET /stations/{id}/metrics` | `plant.summary()` -- cumulative fuel, black carbon, starts, etc. |
| `GET /stations/{id}/safety` | `GuardedController.stats` and `.last_report` -- real intervention counts, not a mock. Also carries a `voltage` field from the inverter-level Volt-Watt layer (see [docs/network-safety.md](network-safety.md)) for a station with a network model; `null` for one without (Bharati, currently). |
| `GET /stations/{id}/controller` | which controller class is wired in |
| `POST /stations/{id}/simulation/start` | starts a background asyncio loop stepping the plant every `interval_s` |
| `POST /stations/{id}/simulation/stop` | stops it, holding the current state |
| `POST /stations/{id}/simulation/reset` | stops, then resets the plant and clears telemetry history |
| `POST /stations/{id}/simulation/step` | advances exactly one step (409 if the auto-loop is running) |

Tested end to end in `tests/test_api.py` via FastAPI's `TestClient`, driven
against the real simulation objects -- not a second, divergent mock of them.

## What's deliberately not here yet

This is the honest half of this document, per the project's own rule against
implying more exists than does:

- **No model/evaluation endpoints** (`GET /models`, `GET /evaluations`).
  There is no model registry or evaluation-run store yet; `allotrope/train.py`
  and `allotrope/evaluate.py` write their records to `runs/<id>/record.json`
  today (see `docs/reinforcement-learning.md`), and exposing those over HTTP
  is a real but separate piece of work, not a relabelling of what exists.
- **No forecast endpoint.** The synthetic climate generator in `allotrope/synth`
  produces the *actual* weather a run will see, which is not the same thing as
  a forecast a controller would have to act under uncertainty from -- serving
  the generator's own output as a "forecast" would be exactly the kind of
  invented state this project's rules forbid.
- **No auth, no rate limiting.** `CORSMiddleware` is wide open
  (`allow_origins=["*"]`) for local frontend development. This is a
  simulation-only API with no secrets and no write access to anything
  outside its own in-memory state; before this API is reachable from
  anywhere but localhost, both need addressing.
- **No trained RL checkpoint wired in by default.** `default_controller`
  documents exactly where one would be loaded once training produces a
  checkpoint worth deploying.
- **One process, one simulation per station, no persistence.** Restarting
  the process loses all history and resets every station to step 0. There is
  no database yet (`allotrope`'s own roadmap lists one as a later,
  deliberately deferred phase) -- state here is exactly as durable as the
  process, and no more.
