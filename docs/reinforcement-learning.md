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

**One real run, reported in full.** `hybrid`, Maitri, seed 0, 60 000 env
steps (`python -m allotrope.train --agent hybrid --station maitri
--total-steps 60000 --episode-steps 336`), evaluated on held-out seed 1 over
a full synthetic year (`python -m allotrope.evaluate --checkpoint
runs/hybrid_maitri_seed0_1788552940/checkpoint.pt --station maitri --seed 1
--periods 8760`):

| | Legacy N+1 | Efficient rules | Hybrid, **guarded** | Hybrid, unguarded |
|---|---|---|---|---|
| Fuel | 254.6 kL | 214.6 kL | 237.2 kL | 172.2 kL |
| Black carbon | 71 826 g | 10 845 g | 59 730 g | 29 803 g |
| Wet-stacking fraction | 0.794 | 0.024 | 0.368 | 0.252 |
| Genset starts | 21 | 286 | 947 | 1 214 |
| Critical unserved | 0 kWh | 0 kWh | **0 kWh** | **181 717 kWh** |
| Freeze violation steps | 0 | 0 | 0 | 0 |

Read this for what it actually shows, not more:

- **The safety claim, demonstrated on this agent specifically.** The
  unguarded checkpoint -- the same weights, no projection -- loses 181 717 kWh
  of life-support power over the year. The guarded one loses none. This is
  the same result `scripts/run_safety_audit.py` reports for hand-designed
  adversarial policies, now shown for an actual trained network rather than
  a policy built to attack the projection on purpose. Training did not, and
  structurally could not, touch this number: the guard sits entirely outside
  the network.
- **Beats the incumbent it exists to replace, not yet the best baseline.**
  237.2 kL versus the legacy fleet's 254.6 kL is real progress on exactly the
  problem this project states (a fleet loitering below its wet-stacking
  threshold); `EfficientRuleBased`'s 214.6 kL is still ahead. The gap is
  explained by genset starts, not commitment strategy: 947 cold starts
  against the efficient baseline's 286 says the policy has not yet learned
  what `genset_start_per_event` in `RewardWeights` is pricing at ₹1 500 each.
  Unguarded starts (1 214) are even higher, which is expected -- an
  unprojected policy is also thrashing commitment in ways the guard
  currently absorbs rather than the reward correcting.
- **60 000 steps is a correctness run, not a competitive one.** The episode
  return improved monotonically and substantially over the run (roughly
  -3 900 to -600 on the last-ten-episode mean), which is the signal this
  phase needed to establish: the agents learn, the pipeline is sound end to
  end, and the safety guarantee holds throughout. Matching or beating
  `EfficientRuleBased` on fuel and starts needs materially more training
  (and likely per-station hyperparameter attention) than one interactive
  session provides; that is future work, not a claim made here.
- **Not yet built:** a scenario-based benchmark across hundreds/thousands of
  seeds with confidence intervals (Section 6 of the project's own roadmap),
  ONNX export for edge inference, and federated training across Maitri and
  Bharati. `allotrope/train.py` and `allotrope/evaluate.py` are the
  foundation those build on, not a replacement for them.
