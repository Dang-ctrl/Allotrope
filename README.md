# Allotrope

**AI-driven safe energy management for polar research station microgrids.**

Smart India Hackathon 2026 · Problem Statement **SIH26061** · Clean and Green
Technology · Software

---

## The problem

Through the Antarctic winter-over, the generating sets at Maitri and Bharati are
held far below 30 % of their rating. Unburnt fuel coats the exhaust manifolds —
*wet stacking* — and the black carbon that follows settles on the surrounding
ice, darkening it and lowering its albedo. Every litre burnt was carried in by
ice-class vessel or DROMLAN flight.

The cause is not carelessness. It is the rational response to having no
forecast, no dispatchable storage and no tolerance whatsoever for a blackout in
August: a spare set that is already turning cannot fail to start. Rule-based
controllers and model-predictive control cannot adapt to an environment this
stochastic, this multi-energy, and this hardware-degrading.

## What this is

An edge-hosted controller for the station microgrid, trained against a digital
twin of it. The architecture, in the order the signal flows:

```
sensors and DERs  ->  digital twin  ->  safe DRL agent  ->  safety projection  ->  actuation
CHP, PV, wind,        PyPSA +           DQN (discrete)      hard limits on         gRPC < 10 ms
BESS, thermal         OpenDSS           + SDDPG             heating and            to inverters
                      state est.        (continuous)        life support           and gensets
```

Two properties are not negotiable and are built in rather than trained in:

- **The safety projection layer** bounds every action analytically. The agent
  cannot breach life-support power or heating limits, whatever it has learned.
- **A deterministic fallback** takes over instantly if the networks time out,
  raise, or return invalid tensors. At the dispatch level it is implemented and
  audited today; the inverter-level Volt-VAr / Volt-Watt curves that complete it
  arrive with the OpenDSS network twin, since they act on voltage.

Only model gradients cross the station's 4 MHz satellite link; all inference runs
on station.

## Status

Phase 1 (**the plant**) and Phase 2 (**the guarantee**) are complete. Phase 3
(**the agents**) is implemented, safety-integrated and tested. A 500k-step
training run cut genset starts by 48% and closed 58% of the fuel gap to the
best rule-based baseline versus an earlier 60k-step run — real progress, not
yet a win: still behind `EfficientRuleBased` on fuel and starts. See
[docs/reinforcement-learning.md](docs/reinforcement-learning.md)'s "Honest
status" for the full numbers, including a real regression (unmet water) this
run surfaced.

| Component | State |
|---|---|
| Station configuration (Maitri, Bharati) | done |
| Synthetic polar climate — solar geometry, irradiance, wind, temperature | done |
| Demand model — electrical, thermal, deferrable | done |
| Asset models — gensets, dual-chemistry storage, PV, wind | done |
| Two-bus plant simulator with CHP and boiler coupling | done |
| Rule-based baselines — legacy N+1 and efficient | done |
| Safety projection layer and deterministic fallback | done |
| Gymnasium environment and reward | done |
| Branching DQN (commitment) + SDDPG (dispatch), safety-integrated | implemented and tested; see [docs/reinforcement-learning.md](docs/reinforcement-learning.md) |
| Local experiment tracking (`allotrope/experiment.py`, `runs/`) | done |
| Backend API over the live simulation — see [docs/api.md](docs/api.md) | implemented and tested; no trained checkpoint or frontend wired in yet |
| Scenario benchmark across many seeds — see [docs/evaluation.md](docs/evaluation.md) | implemented for weather/demand variation; asset-failure and sensor-fault scenarios not yet built |
| Structured logging — see [docs/observability.md](docs/observability.md) | implemented for the API/simulation loop; training/evaluation CLIs and a metrics endpoint not yet wired |
| Frontend Command Center — see [frontend/README.md](frontend/README.md) | one real screen against live API data; no browser/E2E test tool was available to verify it visually, only build/type-check/component tests and manual curl checks against a live server pair |
| Federated learning across stations | planned |
| MQTT / gRPC control plane, Grafana HMI, containers | planned |

## Results so far

A synthetic year at Maitri, hourly, seed 0 — reproduce with
`python scripts/run_baseline.py`:

| | Legacy N+1 | Efficient rules |
|---|---|---|
| Fuel | 254.1 kL | 213.8 kL |
| Black carbon | 72 324 g | 11 190 g |
| Mean genset load factor | **26.5 %** | 52.2 % |
| Steps wet-stacking | **80.6 %** | 2.3 % |
| Mean deposit level | **1.00** | 0.00 |
| Renewable fraction | 15.8 % | 16.1 % |
| Life-support energy unserved | 0 | 0 |
| Freeze violations | 0 | 0 |

The incumbent reproduces the problem the project exists to solve: a fleet
loitering at a quarter of its rating, wet-stacking four steps in five, deposits
saturated. Disciplined rules alone recover 15.9 % of the fuel and 84.5 % of the
black carbon — deliberately leaving headroom, because a baseline that already
captured everything would leave the learned agent nothing to demonstrate.

## The safety guarantee

A policy network's output is not guaranteed to be anything in particular, and no
amount of training reward converts a statistical tendency into a guarantee. So
the guarantee lives outside the network, in an analytic projection that bounds
every action before it reaches a machine. It solves no optimisation problem and
calls no solver, because a safety layer that can fail to converge is not a safety
layer.

The claim is that **no action — random, adversarial or malformed — can make the
station shed life support or freeze.** A claim like that is worth only as much as
the attempt made to break it, so the audit attacks it and reports what got
through. Thirty midwinter days at Maitri, via `python scripts/run_safety_audit.py`:

| Attack policy | Life support lost, guarded | Unguarded |
|---|---|---|
| Random actions | **0 kWh** | 4 075 kWh |
| Shut every machine down | **0 kWh** | 26 901 kWh |
| Charge storage flat out | **0 kWh** | 26 901 kWh |
| Melt flat out | **0 kWh** | 26 901 kWh |
| Oscillate commitment every step | **0 kWh** | 33 309 kWh |

The unguarded column is the control: without it, the guarded column would prove
nothing. The projection also survives NaN and infinity in every field, commands
of the wrong length, agents that raise, and agents that exceed the 10 ms control
budget — a late answer being treated as a wrong answer.

Two honest caveats. The freeze column is zero in *both* conditions, because the
auxiliary boilers protect the heat supply independently of the controller; the
freeze guarantee is therefore real but currently untested by these attacks.
And the deterministic fallback here is dispatch logic — the inverter-level
Volt-VAr and Volt-Watt curves act on voltage, which does not exist in a
power-balance model, and arrive with the OpenDSS network twin.

One bug found and fixed during this work is worth recording, because it is the
kind that survives casual review: the projection originally checked each machine
stop in isolation. With two sets online and both commanded off, each stop looked
safe because the other was still running — and the plant went to zero. Cover is
now evaluated against the whole commitment at once, and
`test_capacity_cover_is_evaluated_jointly_not_per_machine` keeps it that way.

## Install

Requires Python 3.11 or newer.

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -e ".[dev]"
```

```bash
python scripts/run_baseline.py --station maitri --seed 0
```

```bash
python scripts/run_safety_audit.py --station maitri --days 30
```

```bash
python -m pytest
```

## Layout

```
allotrope/
  config/        station YAML and its typed, validated loader
  synth/         synthetic climate and demand generation
  sim/           asset models, the plant, the episode runner
  control/       rule-based baselines
  safety/        the projection layer, the deterministic fallback, GuardedController
  envs/          the Gymnasium environment and the reward
  agents/        BranchingDQN, SDDPG, HybridAgent, the replay buffer
  train.py       training CLI: python -m allotrope.train --agent {dqn,sddpg,hybrid}
  evaluate.py    evaluation CLI: python -m allotrope.evaluate --checkpoint ...
  evaluate_scenarios.py  many-seed statistical evaluation, see docs/evaluation.md
  experiment.py  local, file-based experiment tracking (runs/<run_id>/record.json)
  observability.py  structured JSON logging, see docs/observability.md
  api/           FastAPI backend over the live simulation, see docs/api.md
frontend/        React/TypeScript Command Center UI over the API, see frontend/README.md
docs/            calibration, reinforcement learning, and other design notes
scripts/         entry points
tests/           invariants, including the ones the project's claims rest on
```

## On the data

There is no public telemetry from Maitri or Bharati, so the training environment
is synthetic — generated from physics and station latitude, calibrated against
the one hard public figure available (Bharati's 296 kL seasonal fuel budget) and
against published climate for the sites. Every parameter is tagged in the station
YAML as published, derived or assumed.

**[docs/calibration.md](docs/calibration.md) states where each number came from,
which assumptions the results are most sensitive to, and which claims this model
is not entitled to make.** It is the first thing to read before quoting any
figure from this repository.

## Team

Team **Allotrope**, SRM Institute of Science and Technology.
