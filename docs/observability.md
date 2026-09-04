# Observability

`allotrope/observability.py` is deliberately small: one JSON object per log
line to stdout, the same reasoning `allotrope/experiment.py` already gives
for local-file experiment tracking applied to logs -- no observability
platform (an OTel collector, Grafana) is reachable from this environment or
guaranteed to exist at edge deployment, so this gives the thing every such
platform actually ingests instead of pretending one is wired up.

## What's instrumented today

`allotrope.api.simulation.StationSimulation.step()` emits structured events
on the shared `allotrope.events` logger:

| Event | Level | When |
|---|---|---|
| `simulation.done` | INFO | the plant's weather/demand series is exhausted |
| `controller.fallback` | WARNING | `GuardedController` fell back to `DeterministicFallback` this step, with the reason |
| `safety.intervened` | INFO | the projection modified the proposed action this step, with which interventions fired |
| `simulation.step_latency_high` | WARNING | the API's own observe+act+step wall-clock time exceeded 10 ms (see the docstring in `simulation.py`: this is a separate measurement from `GuardedController`'s own enforced agent-latency budget in `allotrope/safety/fallback.py`, not a duplicate of it) |

`GET /health` reports the API process's own uptime and configured stations --
real values (`time.monotonic()` since startup), not placeholders.

## What's not instrumented yet

- **RL training and evaluation** (`allotrope/train.py`, `allotrope/evaluate.py`,
  `allotrope/evaluate_scenarios.py`) still log via plain `print()` to their
  own CLI output, not through this structured logger. Their real
  per-run record already exists as machine-readable JSON via
  `allotrope/experiment.py` and `evaluate_scenarios.py --out`, which serves
  the same purpose for that kind of long-running batch job; wiring them
  through `log_event` as well is straightforward follow-up, not done here.
- **No metrics/tracing backend.** This module produces logs, not the
  counters/histograms a Prometheus-style `/metrics` endpoint or an OTel
  trace would need. `GuardedController.stats` already tracks the counters
  that would feed one (fallback/projection rates, max latency) and is
  exposed today via `GET /stations/{id}/safety` -- turning that into a
  scrape-able metrics endpoint is real, separate work.
- **No log shipping.** Lines go to stdout only; forwarding them anywhere
  (a file, a collector) is a deployment-time concern, not this module's.
