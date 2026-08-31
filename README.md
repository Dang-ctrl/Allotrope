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
- **A deterministic fallback** — hard-coded Volt-VAr / Volt-Watt curves — takes
  over instantly if the networks time out or return invalid tensors.

Only model gradients cross the station's 4 MHz satellite link; all inference runs
on station.

## Status

Phase 1 of 5 is complete: **the plant**. The simulator, the synthetic polar
environment, and the rule-based controllers that everything later must beat.

| Component | State |
|---|---|
| Station configuration (Maitri, Bharati) | done |
| Synthetic polar climate — solar geometry, irradiance, wind, temperature | done |
| Demand model — electrical, thermal, deferrable | done |
| Asset models — gensets, dual-chemistry storage, PV, wind | done |
| Two-bus plant simulator with CHP and boiler coupling | done |
| Rule-based baselines — legacy N+1 and efficient | done |
| Gymnasium environment and reward | next |
| Safety projection layer and deterministic fallback | next |
| DQN + SDDPG agents | planned |
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

## Install

Requires Python 3.11 or newer.

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -e ".[dev]"
```

```bash
python scripts/run_baseline.py --station maitri --seed 0
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
  control/       rule-based baselines; learned agents to follow
docs/            calibration and design notes
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
