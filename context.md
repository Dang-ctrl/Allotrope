# Working context

**Read this first when resuming work on Allotrope.** It is the operational state
of the project: where things stand, how to run them, what has been decided, and
what is open. For *what the project is* and why it is built the way it is, read
[docs/PROJECT_BIBLE.md](docs/PROJECT_BIBLE.md).

Last updated: 2026-08-31, end of Phase 2.

---

## Where we are

| | |
|---|---|
| Repo | https://github.com/Dang-ctrl/Allotrope (public, `main`) |
| Local | `E:\CODE\Allotrope` |
| Phase | 2 of 5 complete — the plant, and the guarantee |
| Tests | 127 passing |
| Source | ~4 650 lines across `allotrope/`, `scripts/`, `tests/` |
| Commits | `db4b9ab` Phase 1: the plant · `6d42c9b` Phase 2: the guarantee |

## Environment

Python **3.11** in a venv at `.venv`. This is not the machine default — `python`
resolves to 3.13, which has **no pip installed**; 3.11 does. Create with
`py -3.11 -m venv .venv`.

```bash
.venv/Scripts/python.exe -m pytest -q
```

```bash
.venv/Scripts/python.exe scripts/run_baseline.py --station maitri --seed 0
```

```bash
.venv/Scripts/python.exe scripts/run_safety_audit.py --station maitri --days 30
```

Installed: `numpy` 2.4.6, `pandas` 3.0.5, `scipy` 1.17.1, `pyyaml` 6.0.3,
`gymnasium` 1.3.0, `pypsa` 1.3.0, `opendssdirect.py`, `pytest` 9.1.1. The package
is installed editable (`pip install -e .`).

**Not yet installed: `torch`.** Needed for Phase 3. Decision taken but not
executed: install **CPU-only** (~200 MB) rather than the CUDA build (~2.5 GB),
since the problem size does not need a GPU. Confirm with the user before pulling
the CUDA wheel.

### Shell gotchas on this machine

- Writing Python via bash heredoc **fails when the content contains backticks**
  (they appear in docstrings). Use the Write tool for Python files; heredocs are
  fine for YAML and Markdown.
- `pypsa` and `opendssdirect` are installed but **not yet used** — they arrive
  with the network twin in Phase 4. The current plant is a power-balance model.

## What exists

```
allotrope/
  config/       station YAML (maitri, bharati) + typed validated loader
  synth/        climate.py  polar weather from solar geometry
                loads.py    electrical, thermal, deferrable demand
  sim/          assets.py   gensets, batteries, PV, wind
                plant.py    the two-bus microgrid, steppable
                runner.py   run a controller, collect telemetry
  control/      baseline.py LegacyNPlusOne, EfficientRuleBased
  safety/       projection.py  the analytic bound on every action
                fallback.py    DeterministicFallback, GuardedController
  envs/         polar_microgrid.py  Gymnasium env
                reward.py           priced in physical units
docs/           calibration.md, PROJECT_BIBLE.md
scripts/        run_baseline.py, run_safety_audit.py
tests/          127 tests
```

## Next: Phase 3, the agents

The two-layer design the deck commits to: **DQN** for discrete commitment,
**SDDPG** for continuous dispatch. The environment already presents these as
separate spaces (`Dict{genset_on: MultiBinary(3), dispatch: Box(-1,1,(6,))}`), so
no environment change should be needed.

Order of work:

1. Install torch (CPU).
2. Replay buffer and training loop scaffolding in `allotrope/agents/`.
3. SDDPG first, with commitment held by the rule-based logic — isolates the
   continuous problem and gives an early signal that learning helps at all.
4. DQN for commitment on top.
5. Evaluation harness: held-out seeds, both stations, versus both baselines,
   under the same reward via `env.encode_command`.

**The bar to clear:** `EfficientRuleBased` at 213.8 kL/yr and 52.2 % load factor.
A learned policy that does not beat that is not worth deploying, and saying so
plainly is better than shipping a weak result.

**Two things to get right.** Train *with* the safety layer on, so exploration is
safe from the first random action. And keep held-out evaluation seeds genuinely
held out — `build_plant(seed=N)` is the only thing that distinguishes one weather
realisation from another.

## Open questions and known gaps

- **Genset starts.** `EfficientRuleBased` makes 307 starts/year against the
  incumbent's 22. All legal under minimum up and down times, but roughly one a
  day is a real wear cost. It is now priced in the reward; the agent should beat
  it, and if it does not, that is a finding worth reporting.
- **Freeze guarantee untested.** The safety audit shows zero freeze violations in
  *both* guarded and unguarded conditions, because the boilers cover heat
  independently of the controller. The guarantee is real but the audit does not
  currently demonstrate it. To test it properly, an attack scenario needs boiler
  capacity constrained below peak firm thermal demand.
- **Unmet water** is small but non-zero for both baselines (2 772 and 996 kWh/yr
  out of ~58 000). Cause is crew count changing across a day, so the melting
  obligation set at midnight does not exactly match what the flat rate delivers.
  Benign, but it should not be quoted as a controller failing.
- **Volt-VAr / Volt-Watt** curves cannot be built yet — they act on voltage, and
  the plant is a power-balance model. They arrive with the OpenDSS twin. The
  README says so; do not let this quietly become an unbacked claim.
- **Federated learning** is a deck commitment with no code yet. It needs a second
  station training concurrently, which `bharati.yaml` already supports.

## Conventions

- Every physical parameter lives in station YAML, never in code, and carries a
  `[public]` / `[derived]` / `[assumed]` tag.
- Claims in the README are reproducible by a script in `scripts/`, and the
  invariants behind them are asserted in `tests/`.
- No personal data in the repo. The SIH deck's team slide carries registration
  numbers, personal emails and mobile numbers; the `.pptx` is gitignored and
  those details are deliberately absent from all documentation.
- Commit messages explain *why*, including bugs found and claims deliberately not
  made.

## Maintenance

**Update this file at the end of any session that changes the state of the
project** — phase, test count, commits, decisions taken, questions opened or
closed. Update [docs/PROJECT_BIBLE.md](docs/PROJECT_BIBLE.md) whenever the
architecture, parameters, results or roadmap change.
