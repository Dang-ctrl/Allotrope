# Reinforcement learning

Phase 3. This document describes the learned controllers in `allotrope/agents/`,
how they are trained and evaluated, and — as with every other claim in this
repository — what they are not yet entitled to say.

## Why two algorithms, not one

The control problem `PolarMicrogridEnv` presents is genuinely hybrid (see the
module docstring in `allotrope/envs/polar_microgrid.py`). Which gensets are
committed is a discrete decision with minimum up/down times attached; how hard
committed plant is worked, and what storage and melting do, is continuous.
Flattening that into one space would either explode combinatorially (`2^n`
joint commitments times a continuous product) or hide that these are different
decisions with different natural algorithms:

- **`BranchingDQN`** (`allotrope/agents/dqn.py`) — one binary Q-head per
  genset over a shared trunk (action-branching architecture), trained as
  Double DQN with a joint, branch-coordinated target. Answers "which sets are
  on."
- **`SDDPG`** (`allotrope/agents/sddpg.py`) — a squashed-Gaussian stochastic
  actor with twin critics and soft-updated target networks (DDPG/TD3-style
  off-policy actor-critic; the "S" is the stochastic policy, not a specific
  published algorithm of that name). Answers "how hard, and what does storage
  and melting do."
- **`HybridAgent`** (`allotrope/agents/hybrid.py`) — composes the two into one
  `.act(observation, plant) -> DispatchCommand` policy, so it can be scored by
  `allotrope.sim.runner.run_episode` exactly like `LegacyNPlusOne` and
  `EfficientRuleBased`.

## The proposal, not the guarantee

None of the three classes above touch `allotrope.safety`. Like the rule-based
baselines, they only ever propose. `allotrope.safety.fallback.GuardedController`
is the single place — already implemented, already tested in
`tests/test_safety.py` — that wraps an agent with the deterministic fallback
and the safety projection for deployment or evaluation:

```python
guard = GuardedController(cfg, agent=HybridAgent(cfg, dqn, sddpg))
```

Giving the learned agent its own parallel safety path was considered and
rejected during this work: it would duplicate logic the project already has
and had already audited, and it would let this one controller silently diverge
from how every other controller is guarded. `tests/test_agents.py` runs the
same class of attack `tests/test_safety.py` runs against random and
adversarial policies — an untrained network, and a network whose weights have
been overwritten with NaN — through `GuardedController`, with Hypothesis
driving 25 random seeds rather than a handful chosen by hand. Both must, and
do, leave `critical_unserved_kwh == 0` and `freeze_violation_steps == 0`.

## Training against a plant the agent cannot damage

`PolarMicrogridEnv.step` applies the safety projection *inside* the
environment (`apply_safety=True`, the training default), before the plant
ever sees the action. This is deliberate, and is the environment's own design
choice, not something added for the agents: exploration is safe from the
first random action, so the replay buffer stores the agent's *raw proposal*
alongside the reward the plant actually produced once that proposal was
projected. What the agent learns is how to propose well inside constraints it
never had to discover, not how to avoid harming a plant it was never able to
harm in the first place.

```
python -m allotrope.train --agent dqn --station maitri     # only the Q-network updates
python -m allotrope.train --agent sddpg --station maitri   # only the actor/critic update
python -m allotrope.train --agent hybrid --station maitri  # both update jointly (deployable mode)
```

`--agent dqn` and `--agent sddpg` still act through an instance of the other
network (untrained, frozen) so the environment always receives a complete
joint action; only the selected network's parameters are updated. `hybrid` is
the mode a controller intended for evaluation or deployment is actually
trained under.

## Evaluation

```
python -m allotrope.evaluate --checkpoint runs/<run>/checkpoint.pt --station maitri --seed 1
```

Runs, on a seed the checkpoint never trained on: `LegacyNPlusOne`,
`EfficientRuleBased`, the checkpoint under `GuardedController` (what would
actually be deployed), and the bare checkpoint unguarded (the control column —
what the projection is buying this specific policy, the same methodology
`scripts/run_safety_audit.py` uses for adversarial policies).

## Experiment tracking

No MLflow (or other) tracking server is reachable from this environment, and
an edge-first project's controller cannot depend on one existing at
deployment time either. `allotrope/experiment.py` records what a hosted
tracker would, to `runs/<run_id>/record.json`: git commit and dirty flag,
agent kind, station, seed, full hyperparameter config, a training-metric
history, final metrics, and the checkpoint path. A run is reproducible from
that file alone. Swapping this for a hosted tracker later is a change to
`ExperimentTracker`, not to every call site that logs a metric.

## Honest status

**Two real runs, reported in full, same seed and evaluation protocol.**
`hybrid`, Maitri, seed 0, evaluated on held-out seed 1 over a full synthetic
year (`python -m allotrope.evaluate --checkpoint <checkpoint> --station
maitri --seed 1 --periods 8760`):

| | Legacy N+1 | Efficient rules | Hybrid, guarded (60k steps) | Hybrid, guarded (500k steps) | Hybrid, unguarded (500k steps) |
|---|---|---|---|---|---|
| Fuel | 254.6 kL | 214.6 kL | 237.2 kL | **223.3 kL** | 155.3 kL |
| Black carbon | 71 826 g | 11 008 g | 59 730 g | **38 269 g** | 18 399 g |
| Wet-stacking fraction | 0.794 | 0.023 | 0.368 | **0.167** | 0.148 |
| Genset starts | 21 | 286 | 947 | **495** | 3 749 |
| Unmet water | 3 863 kWh | 1 262 kWh | -- | **66 328 kWh** | 79 030 kWh |
| Critical unserved | 0 kWh | 0 kWh | **0 kWh** | **0 kWh** | 197 146 kWh |
| Freeze violation steps | 0 | 0 | 0 | 0 | 0 |

(`--`: not reported for the 60k run's writeup. `python -m allotrope.train
--agent hybrid --station maitri --total-steps 500000 --episode-steps 336
--warmup-steps 1000 --buffer-capacity 100000` reproduces the 500k run.
The 500k column was re-measured after fixing a reproducibility bug --
see "A correction" below -- so it differs slightly, in the fourth digit,
from an earlier version of this table.)

**A correction.** An earlier version of this table reported 497 genset
starts for the 500k run. That number was never wrong exactly, but it was
never exactly reproducible either: this project's own adversarial audit
found that `GuardedController`'s real-time latency budget -- a correct
safety property for actual control, where a late answer really is a wrong
answer -- was also being enforced during *offline* evaluation, where it
has no business being. Because the budget is checked against a wall-clock
measurement of each forward pass, re-running the identical evaluation
(same checkpoint, same seed) on a machine under different load silently
substituted the deterministic fallback a different number of times,
giving genset_starts anywhere from 489 to 527 across six otherwise
identical runs. `GuardedController` now takes an `enforce_latency_budget`
flag (default `True`, unchanged for real deployment); `allotrope.evaluate`,
`allotrope.evaluate_scenarios`, and the federated round validator all set
it `False`, and every number in this table above is now a pure function of
(checkpoint, seed) -- reconfirmed by running the 500k evaluation four times
and getting 495 starts every time.

Read this for what it actually shows, not more:

- **More training closed roughly 60 % of the gap to the best baseline, on
  the metric the 60k run named as the problem.** Genset starts fell from 947
  to 495 -- still well above `EfficientRuleBased`'s 286, but the fuel gap to
  that baseline shrank from 22.6 kL to 8.6 kL, and wet-stacking fraction
  dropped by more than half (0.368 to 0.167). This is exactly the
  `genset_start_per_event` penalty finally being learned, as the 60k
  writeup predicted it would need more training to do -- not a different
  mechanism, just more of the same one working.
- **The safety claim holds at 500k steps exactly as it did at 60k, on a
  policy that has become more dangerous unguarded, not less.** The unguarded
  checkpoint's fuel dropped further (155.3 kL, below even the guarded
  figure) because it is now aggressive enough to starve life support to get
  there: 197 146 kWh of critical load lost, worse than the 60k run's
  181 717 kWh. The guarded column still loses zero. This is the point of
  putting the guarantee outside the network rather than in the reward: a
  policy optimising harder for fuel got *more* willing to cut corners the
  reward doesn't itself forbid, and the projection caught every instance of
  it regardless.
- **A cost the reward under-weights showed up once training pushed hard
  enough on the ones it doesn't: unmet water.** 66 328 kWh of deferred
  melting against the efficient baseline's 1 262 kWh is a real regression
  this run surfaces for the first time, not noise -- `unmet_water_per_kwh`
  in `RewardWeights` (₹60/kWh) is small next to `fuel_per_l` (₹250/L) and
  `genset_start_per_event` (₹1 500), so a policy under pressure to cut starts
  and fuel is trading melting away first. This is worth reward-weight
  attention before the next training run, not a claim that the current
  weights are wrong -- they simply have not been tested this hard before.
- **Still not competitive with `EfficientRuleBased` on fuel or starts.**
  495 starts and 223.3 kL beat the incumbent (`LegacyNPlusOne`) by a wide
  margin but remain behind the best rule-based baseline. Whether closing
  the rest of this gap needs more steps, reward reshaping around the
  unmet-water finding above, or architecture changes (Section "RL
  architecture" above) is open; this document states the current numbers
  rather than predicting which lever closes it.
- **Not yet built:** ONNX export for edge inference, and federated training
  across Maitri and Bharati. A scenario-based benchmark across many seeds
  now exists (`allotrope/evaluate_scenarios.py`, see
  [docs/evaluation.md](evaluation.md)) for the rule-based baselines; it
  accepts a trained checkpoint via `--checkpoint` but has not yet been run
  against one at scale, since a single held-out-seed comparison was the more
  urgent question this pass answered.
