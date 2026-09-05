"""Federated learning across stations: local training, weight-only
aggregation, validation, and rollback.

What crosses a station boundary is a model checkpoint's tensors
(`allotrope.federated.aggregate.fedavg_checkpoint`), never a station's
weather, demand, or telemetry -- each station trains against its own
`PolarMicrogridEnv`, built from its own synthetic generators
(`allotrope.federated.round.run_local_update`).

The safety guarantee is untouched by any of this, structurally rather than
by policy: `allotrope.safety.projection.SafetyProjection` and
`allotrope.safety.fallback.GuardedController` have no learned parameters at
all, so there is nothing in them for federated averaging to reach. A
federated round can only ever change how well `HybridAgent` proposes, never
whether the guard behind it holds -- `allotrope.federated.round.
default_validator`'s safety gate measures exactly this rather than assuming
it.

See `docs/federated-learning.md` for the honest status: this is a real,
tested mechanism, run for a handful of rounds with modest local step
counts, not a claim that it has been run to convergence or that it beats
per-station training.
"""
