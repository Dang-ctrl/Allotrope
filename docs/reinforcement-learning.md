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

- **Trained, not tuned.** The hyperparameters in `DQNConfig` and `SDDPGConfig`
  are reasonable defaults, not the result of a sweep. A short run (a few
  thousand steps) already shows the episode return improving monotonically,
  which is the correctness signal this phase set out to establish; a
  policy competitive with `EfficientRuleBased` needs materially more training
  than fits in a single interactive session, and the honest comparison run is
  recorded under `runs/`, not asserted here.
- **The safety guarantee is unchanged.** `HybridAgent` adds a third proposer
  to a safety architecture that already existed and was already proven
  against random and adversarial policies. Nothing in this phase weakened,
  bypassed, or added a special case to `allotrope/safety/`.
- **Not yet built:** a scenario-based benchmark across hundreds/thousands of
  seeds with confidence intervals (Section 6 of the project's own roadmap),
  ONNX export for edge inference, and federated training across Maitri and
  Bharati. `allotrope/train.py` and `allotrope/evaluate.py` are the
  foundation those build on, not a replacement for them.
