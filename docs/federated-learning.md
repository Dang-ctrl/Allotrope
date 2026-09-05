# Federated learning across stations

`allotrope/federated/` implements the architecture the project's own
roadmap named: local training per station, aggregation into a global
model, and station-specific adaptation from there — without a station's
weather, demand, or telemetry ever leaving the process that generated it.

```
local training (Maitri)     local training (Bharati)
        |                            |
   local checkpoint            local checkpoint
        \___________      ___________/
                    \    /
              fedavg_checkpoint          <- allotrope/federated/aggregate.py
                     |
          candidate global checkpoint
                     |
              default_validator          <- allotrope/federated/round.py
                /         \
          accepted      rejected
          (promoted,    (kept for
           carried      provenance,
           forward)      never promoted)
```

## Run it

```bash
python -m allotrope.federated.coordinator --rounds 2 --local-steps 5000
```

Every round writes `runs/federated/round_<n>/round_record.json` — the full
provenance: which stations participated, each one's local steps/seed/
checkpoint path, the aggregation weights, the validator's per-station
result, accept/reject, the git commit, and start/finish timestamps. A
rejected round's candidate checkpoint stays on disk (as
`global_checkpoint_candidate.pt`, never renamed to `global_checkpoint.pt`)
rather than being deleted — the point of provenance is being able to look
at what a rejected round actually produced, not just that it was rejected.

## What crosses a station boundary, precisely

**Only tensors.** `allotrope.federated.round.run_local_update` trains one
station against its own `PolarMicrogridEnv`, built from that station's own
`allotrope.synth` climate and demand generators (`allotrope.federated`'s
package docstring). `allotrope.federated.aggregate.fedavg_checkpoint` reads
`allotrope.train.train`-format checkpoints — `obs_dim`/`n_gensets`/
`dispatch_dim` plus the DQN and SDDPG network state_dicts — and has no
interface through which an observation, a reward, or a weather value could
pass. This is checkable by reading the module, not just a design intent:
`aggregate.py` never imports anything from `allotrope.synth`,
`allotrope.sim`, or `allotrope.envs`.

FedAvg requires every participant's network to have the same shape.
`fedavg_checkpoint` checks `obs_dim`, `n_gensets`, and `dispatch_dim`
match across all local checkpoints and raises rather than silently
averaging mismatched tensors. This holds for Maitri and Bharati today
(three gensets, two storage packs, each) — see
`test_fedavg_checkpoint_rejects_architecture_mismatch` in
`tests/test_federated.py` for what happens when it doesn't.

## Validation and rollback

`allotrope.federated.round.default_validator` runs the candidate global
model, guarded, against a held-out seed on every participating station,
and gates on two things:

1. **Safety — hard gate.** `critical_unserved_kwh` must be exactly zero.
   Stated honestly: this cannot fail from anything federated averaging did,
   because `SafetyProjection` and `GuardedController` carry no learned
   parameters at all (`allotrope/federated/__init__.py`'s docstring) — no
   amount of weight averaging can touch a guard that isn't made of
   weights. It's still measured every round, the same way every other
   safety claim in this project is measured rather than assumed.
2. **Performance — soft gate.** Fuel use must not exceed the incumbent
   `LegacyNPlusOne` baseline's, on the same held-out run. This one *can*
   fail, and is the gate that actually exercises rollback:
   `test_a_rejected_round_is_never_promoted_and_is_kept_for_provenance`
   and `test_run_rounds_carries_forward_only_the_last_accepted_checkpoint`
   in `tests/test_federated.py` force it to fail (via a validator override
   — `run_round`/`run_rounds` accept one) and confirm the rejected
   checkpoint is never promoted and never warm-starts the next round.

A validator is a plain function of `(checkpoint_path, stations) ->
ValidationResult` passed into `run_round`/`run_rounds`; the default one
above is what `coordinator.main()` uses, but a deployment could substitute
a stricter one without touching the aggregation or rollback logic.

## Honest status

**Two real rounds were run** (`python -m allotrope.federated.coordinator
--rounds 2 --local-steps 5000`, one round per Maitri+Bharati local update
of 5,000 steps each) as this feature's own smoke test, on top of
`tests/test_federated.py`'s 13 tests (which use far smaller local step
counts — 50-100 — chosen for test speed, not to demonstrate competitive
performance). See the table below for what that produced.

This is a **real, tested mechanism** — aggregation is exact FedAvg over
real network tensors (spot-checked against the literal weighted mean in
`test_fedavg_checkpoint_averages_real_network_weights`), warm-starting
from a global checkpoint is verified bit-exact when no gradient step
occurs (`test_warm_started_training_with_no_update_reproduces_the_
initial_weights`), and validation/rollback is exercised end to end, not
merely code that exists. It is **not** a claim that federated training at
this scale beats per-station training, or that it has been run to
convergence — 5,000-10,000 local steps per round is far below the 500,000
steps `docs/reinforcement-learning.md` reports it takes to make meaningful
progress against `EfficientRuleBased` in the single-station case, and no
number of rounds this pass ran was chosen to make a competitive claim.
Whether more rounds, more local steps, or a different aggregation cadence
converges to something competitive with (or better than) per-station
training is open, and is next work, not a claim made here.
