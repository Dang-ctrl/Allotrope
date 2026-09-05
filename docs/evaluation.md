# Scenario-based evaluation

`allotrope/evaluate_scenarios.py` runs a controller against many independent
seeds of a station's synthetic climate and demand, and reports the
*distribution* of outcomes rather than one number from one year:

```bash
python -m allotrope.evaluate_scenarios --station maitri --seeds 200 \
    --out runs/scenario_suite_maitri_200seeds.json

# once a trained checkpoint exists:
python -m allotrope.evaluate_scenarios --station maitri --seeds 200 \
    --checkpoint runs/hybrid_maitri_seed0_.../checkpoint.pt \
    --out runs/scenario_suite_maitri_200seeds_with_rl.json
```

For every metric `allotrope.sim.runner.compare` already tracks (fuel,
black carbon, genset starts, critical/comfort unserved energy, freeze
violations, ...), it reports mean, median, standard deviation, min, max, and
the 5th/95th percentiles across the seed range -- machine-readable, as
`runs/<name>.json`.

## What "many seeds" actually varies

Each seed drives an independent draw from `allotrope.synth.climate` and
`allotrope.synth.loads`: the cold-snap and blizzard processes, the wind and
irradiance realisations, and the demand noise are all reseeded per run. That
is a real spread from mild to severe winters -- `test_different_seeds_give_
different_weather_and_therefore_different_fuel` in `tests/test_evaluate_
scenarios.py` checks it isn't running the same year 200 times by accident --
and it is the honest scope of what this module claims: **statistical
variation in weather and demand**, run identically for every controller
under comparison.

## What it does not (yet) claim

The project's own rules forbid implying more exists than does, so this is
stated plainly rather than left to be discovered:

- **No fault injection.** There is no mechanism in `allotrope.sim` today to
  force a genset, PV string, or wind turbine offline mid-run, corrupt a
  sensor reading, or delay a control decision. "Genset failure," "PV
  failure," "telemetry dropout," and "stale sensor data" as *scenario suite*
  categories are not implemented -- adding them means adding real fault
  models to `allotrope/sim/assets.py` and `allotrope/sim/plant.py`, which is
  separate, real work, not a relabelling of the seed sweep above.
- **Invalid/NaN/malformed *actions*, agent timeout, and agent exceptions are
  already covered -- just not by this module.** That is the adversarial-policy
  audit's job: `scripts/run_safety_audit.py` (five attack policies, a full
  midwinter month, guarded vs. unguarded) and the Hypothesis-driven property
  tests in `tests/test_safety.py` and `tests/test_agents.py` (300+ examples
  run manually against the untrained-agent property; 25 in CI). Those already
  demonstrate zero critical-load loss under NaN, infinity, wrong-length, and
  adversarial commands. This module is about *environmental* variation
  (weather, demand); that one is about *action-space* variation.
- **Comparisons currently cover the two rule-based baselines by default.**
  A trained RL checkpoint is included via `--checkpoint` (both the guarded
  and unguarded columns, matching `allotrope.evaluate`'s methodology) once
  one exists worth reporting on scenario scale -- see
  `docs/reinforcement-learning.md`'s "Honest status" for where training
  currently stands.
