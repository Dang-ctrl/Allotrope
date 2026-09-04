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
  raise, or return invalid tensors — implemented and audited at the dispatch
  level (`allotrope.safety.fallback`). The inverter-level Volt-VAr / Volt-Watt
  curves that act on voltage now also exist (`allotrope.network.twin`), run
  against the OpenDSS network twin; the two are not yet merged into one
  combined runtime path.

Only model parameters cross the station's 4 MHz satellite link during
federated training (`allotrope.agents.federated`); all inference runs on
station.

## Status

All five phases are code-complete. Phases 1–4 are backed by a full test suite
run in development; Phase 5's infrastructure (the container stack, Grafana
against real data) has not been started end to end in that environment — see
[docs/PROJECT_BIBLE.md §11](docs/PROJECT_BIBLE.md) for exactly which claims
that covers and which it doesn't.

| Component | State |
|---|---|
| Station configuration (Maitri, Bharati) | done |
| Synthetic polar climate, demand model, asset models | done |
| Two-bus plant simulator, rule-based baselines | done |
| Safety projection layer, deterministic fallback, Gymnasium env | done |
| DQN + SDDPG agents, training, checkpointing | done — see results below |
| OpenDSS network twin, Volt-VAr / Volt-Watt fallback | done |
| Federated learning across stations (FedAvg) | mechanism done and tested; a real 30-round run **did not beat** the single-station agents — see below |
| gRPC actuation interface | done |
| MQTT telemetry link, TimescaleDB bridge | done |
| Containers, Grafana HMI | code written, **not run end to end** in this environment |

## Results so far

A synthetic year, hourly, seed 0 — reproduce with `python scripts/run_baseline.py --station <maitri|bharati>`:

**Maitri**

| | Legacy N+1 | Efficient rules |
|---|---|---|
| Fuel | 254.1 kL | 213.8 kL |
| Black carbon | 72 324 g | 11 190 g |
| Mean genset load factor | **26.5 %** | 52.2 % |
| Steps wet-stacking | **80.6 %** | 2.3 % |
| Renewable fraction | 15.8 % | 16.1 % |
| Life-support energy unserved | 0 | 0 |

**Bharati** — also the calibration check, since 264.2 kL against a published
296 kL seasonal budget is the strongest validation available without station
telemetry:

| | Legacy N+1 | Efficient rules |
|---|---|---|
| Fuel | 264.2 kL | 204.4 kL |
| Black carbon | 95 251 g | 39 889 g |

The incumbent reproduces the problem the project exists to solve: a fleet
loitering at a quarter of its rating, wet-stacking four steps in five, deposits
saturated. Disciplined rules alone recover 15.9–22.6 % of the fuel and
58–84.5 % of the black carbon depending on station — deliberately leaving
headroom, because a baseline that already captured everything would leave the
learned agent nothing to demonstrate.

### The learned agent

`HybridAgent` (DQN + SDDPG), evaluated on **held-out seeds** (100–104,
disjoint from every training seed) the policy never trained on —
`python scripts/evaluate_agent.py --station <maitri|bharati> --checkpoint checkpoints/<station>.pt`:

| | Efficient rules | **Hybrid DQN + SDDPG** |
|---|---|---|
| **Maitri** fuel | 213.4 kL | **209.6 kL** (−1.8 %) |
| Maitri black carbon | **10 617 g** | 15 931 g |
| Maitri genset starts/year | 272.2 | **210.0** (−22.9 %) |
| **Bharati** fuel | 205.4 kL | **193.8 kL** (−5.6 %) |
| Bharati black carbon | 40 654 g | 40 722 g (flat) |
| Bharati genset starts/year | 140.0 | **14.8** (−89 %) |
| Life support unserved, every held-out seed, both stations | 0 | **0** |

The agent beats the rule-based bar it was built to clear at both stations —
less fuel in each case — but finds a *different* trade-off at each one. At
Maitri it cuts starts by 23 % and trades away some black-carbon performance;
at Bharati it nearly eliminates cycling (14.8 starts/year, close to the
incumbent's own habits) while holding black carbon flat. Neither trade is a
bug: `RewardWeights` prices fuel and starts more heavily than black carbon in
absolute terms, so both policies are optimising the same stated objective,
just landing at different points its trade-off surface allows. Report both
numbers, not only the more flattering one.

### Federated training — a negative result, reported as one

`scripts/run_federated.py`, 30 rounds × 15 local episodes per station, FedAvg
across Maitri and Bharati simultaneously. Evaluated the same way as above:

| | Efficient rules | **Federated** |
|---|---|---|
| Maitri fuel | 213.4 kL | 225.3 kL (**+5.6 %**, worse) |
| Bharati fuel | 205.4 kL | 210.1 kL (**+2.3 %**, worse) |
| Life support unserved, both stations, every seed | 0 | **0** |

The federated policy does not beat `EfficientRuleBased` at either station, and
underperforms each station's own dedicated single-agent checkpoint above by a
wide margin. The training log shows why: reward fluctuates across all 30
rounds with no visible convergence trend, consistent with FedAvg client drift
— each site gets only 15 local episodes to adapt before its weights are
averaged back toward the other site's differently-adapted ones. The safety
guarantee holds exactly as well as everywhere else in this project regardless
of training quality, which is the one property that has to be true here no
matter what. The mechanism itself is real and tested end to end
(`tests/test_federated.py`); this particular training configuration simply
does not yet produce a policy worth deploying, and that is reported plainly
rather than tuned until the number looked better.

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

One honest caveat remains: the freeze column is zero in *both* conditions,
because the auxiliary boilers protect the heat supply independently of the
controller, so the freeze guarantee is real but still untested by these
attacks. The other caveat this section used to carry — that Volt-VAr/Volt-Watt
couldn't exist yet because the plant had no voltage in it — is resolved: an
OpenDSS network twin now exists (`allotrope/network/twin.py`), with a tested,
working two-stage Volt-VAr/Volt-Watt fallback. It runs alongside, not yet
merged into, `DeterministicFallback`'s dispatch-level logic — see
[docs/PROJECT_BIBLE.md §9](docs/PROJECT_BIBLE.md) for the detail.

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
python scripts/train_agent.py --station maitri --episodes 500 --out checkpoints/maitri.pt
python scripts/evaluate_agent.py --station maitri --checkpoint checkpoints/maitri.pt
```

```bash
python -m pytest
```

`torch` installs CPU-only by default (`pip install -e ".[dev]"` above does not
pull CUDA). The networks are small — 128-unit MLPs over a 25-dimensional
observation — and gain nothing from a GPU at this problem size.

To regenerate the gRPC stubs after editing `allotrope/rpc/allotrope.proto`:

```bash
python scripts/gen_proto.py
```

To try the full stack (plant, gRPC actuation, MQTT, TimescaleDB, Grafana) —
written and unit-tested as described in
[docs/PROJECT_BIBLE.md §11](docs/PROJECT_BIBLE.md), but not run end to end as
a container stack in this repository's own development environment:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

## Layout

```
allotrope/
  config/        station YAML and its typed, validated loader
  synth/         synthetic climate and demand generation
  sim/           asset models, the plant, the episode runner
  control/       rule-based baselines
  safety/        the projection layer and the deterministic fallback
  envs/          the Gymnasium environment and the reward
  agents/        DQN + SDDPG, training, checkpointing, federated averaging
  network/       the OpenDSS twin and the Volt-VAr / Volt-Watt fallback
  rpc/           the gRPC actuation interface (proto + server + client)
  mqtt/          telemetry pub/sub and the TimescaleDB bridge
deploy/          Dockerfile, docker-compose, Grafana provisioning, DB schema
docs/            calibration, design notes, and the project bible
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
