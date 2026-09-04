# Working context

**Read this first when resuming work on Allotrope.** It is the operational state
of the project: where things stand, how to run them, what has been decided, and
what is open. For *what the project is* and why it is built the way it is, read
[docs/PROJECT_BIBLE.md](docs/PROJECT_BIBLE.md).

Last updated: 2026-09-05, end of Phase 5 (autonomous overnight session).

---

## Where we are

| | |
|---|---|
| Repo | https://github.com/Dang-ctrl/Allotrope (public, `main`) |
| Local | `E:\CODE\Allotrope` |
| Phase | **5 of 5 code-complete** — see caveats below; not everything is tested to the same depth |
| Tests | 194 passing |
| Commits | `db4b9ab` Plant · `6d42c9b` Guarantee · `d0f9ca9` Docs · `14e4303` Agents · `6f11585` Twin · `ca028cf` System |

**This was an autonomous overnight session** (user asked to "complete the project" while asleep, using judgment for decisions without stopping to ask). Everything below was built, tested, and committed without further confirmation. Nothing has been pushed to the remote yet as of this writing — confirm with the user before pushing, since the repo is public and this is a large, unreviewed batch of work.

## Environment

Python **3.11** in a venv at `.venv`. The machine default `python` is 3.13, which has **no pip**; 3.11 does.

```bash
.venv/Scripts/python.exe -m pytest -q
```

Installed beyond Phase 1–2: `torch` 2.14 (CPU-only — deliberate, the networks are
small 128-unit MLPs and gain nothing from a GPU), `grpcio` + `grpcio-tools`,
`paho-mqtt`, `psycopg[binary]`, `amqtt` (dev-only, an embedded pure-Python MQTT
broker used by the test suite so `tests/test_mqtt.py` runs against a real
broker rather than a mock).

### Shell gotchas (still true)

- Writing Python via a bash heredoc fails when the content has backticks. Use
  the Write tool for Python; heredocs are fine for YAML/Markdown/SQL.
- `pytest` runs print a Python stack trace to stderr on the *first* test module
  that imports `opendssdirect` (a cosmetic fault-handler registration in that
  library) even when every test passes with exit code 0. **This is benign** —
  check the actual pass/fail summary line, not the presence of a traceback.
- Docker CLI is present on this machine but **the daemon is not running**
  (Docker Desktop not started). `deploy/docker-compose.yml` has not been
  exercised end to end for that reason — see `deploy/README.md` for exactly
  which pieces are and are not verified.
- No `mosquitto` binary and no live Postgres/TimescaleDB in this environment.
  MQTT is tested against an embedded `amqtt` broker (real protocol, no
  external service); the TimescaleDB bridge is tested against a fake
  connection object (real SQL-building logic, no real database).

## What exists (additions since Phase 2)

```
allotrope/
  agents/       dqn.py, sddpg.py, hybrid.py, networks.py, replay.py, train.py,
                checkpoint.py, federated.py — the two-layer learned controller
  network/      twin.py — OpenDSS radial feeder + Volt-VAr/Volt-Watt fallback
  rpc/          allotrope.proto + generated stubs, server.py, client.py,
                convert.py — the gRPC actuation interface
  mqtt/         codec.py, topics.py, publisher.py, subscriber.py,
                timescale_bridge.py — the telemetry link
deploy/         Dockerfile, docker-compose.yml, Grafana provisioning,
                TimescaleDB schema, mosquitto.conf, README.md (tested-vs-not table)
scripts/        train_agent.py, evaluate_agent.py, gen_proto.py,
                run_station_service.py, run_timescale_bridge.py
```

## Phase 3 results — the trained agent

`checkpoints/maitri.pt`, 500 episodes, evaluated on **held-out seeds 100–104**
(disjoint from training), full year (8760 steps) each, via
`scripts/evaluate_agent.py`:

| | Legacy N+1 | Efficient rules | **Hybrid DQN+SDDPG** |
|---|---|---|---|
| Fuel | 253.8 kL | 213.4 kL | **209.6 kL** |
| Black carbon | 72 141 g | **10 617 g** | 15 931 g |
| Mean load factor | 0.264 | 0.524 | 0.514 |
| Wet-stacking fraction | 0.813 | **0.018** | 0.051 |
| Genset starts/year | 23.8 | 272.2 | **210.0** |
| Life support unserved | 0 | 0 | **0** |
| Freeze violations | 0 | 0 | **0** |

**The agent clears its bar**: 1.8% less fuel than `EfficientRuleBased` and 22.9%
fewer genset starts, on seeds it never trained on. **It does not clear every
bar** — black carbon is higher than the rule-based baseline's, because the
reward weighs fuel and starts more heavily in absolute terms than black carbon
(see `RewardWeights`), so the learned policy correctly optimises a slightly
different point on the trade-off surface, not a worse one by its own
objective. State this honestly if asked: the agent beats the target metric it
was built to beat, and trades a secondary one to do it.

**A real bug was found and fixed getting to this number**: the first training
run (300 episodes, discarded) used exploration-decay defaults tuned for
~1000 episodes, so it finished with the DQN still 40% random. Every number
from that run was noise, not policy. `scripts/train_agent.py` now derives the
decay schedule from the actual episode count requested. A second bug, in
`scripts/evaluate_agent.py` itself, transposed the summary DataFrame
backwards (`pd.DataFrame(means)` needed `.T`) and was caught only because the
script crashed outright rather than silently printing wrong numbers — no test
covers this CLI script directly, which is a real gap (see Open questions).

## Phase 4 results — the twin

`allotrope/network/twin.py`: an OpenDSS radial LV feeder per station, one bus
per asset group, and the IEEE 1547-2018 default Volt-VAr/Volt-Watt curves as
the deterministic inverter fallback. Verified under a stress case (600 kW PV
export against a 50 kWp rating): raw voltage reaches 1.11 pu; VAr support alone
partially corrects it; full Volt-Watt curtailment is what actually returns the
bus under the 1.10 pu ceiling. This is real, working code, not a stub — 17
tests in `tests/test_network.py`.

## Phase 5 — what's real and what's infrastructure-only

Read `deploy/README.md`'s table before quoting anything about the deployment
stack. Short version: gRPC actuation, MQTT pub/sub (including malformed-payload
handling), and the TimescaleDB bridge's SQL logic are **genuinely tested** —
39 tests across `test_rpc.py`, `test_mqtt.py`, `test_timescale_bridge.py`.
The `docker-compose.yml` stack itself has **not been run** (no Docker daemon
here) and Grafana rendering real data has **not been verified** (no Postgres
here). Both are real, reasonable infrastructure code; neither is proven to
work end to end by anything in this session.

## Open questions and known gaps

- **Bharati agent + federated training**: were queued to run in the background
  after Maitri's evaluation. Check `checkpoints/bharati.pt` and this file's
  git history / commit log for whether they landed and what they showed. If
  this section still says "pending" and no later commit addresses it, the
  session ended before they finished — pick them up before claiming the
  federated learning result as validated.
- **`evaluate_agent.py` and `run_baseline.py` have no dedicated tests.** The
  library code they call is thoroughly tested; the scripts' own
  DataFrame-construction and printing logic is not, and one real bug there
  (see above) was only caught by a crash. Worth a minimal smoke test.
- **The freeze guarantee is still not exercised by the safety audit** — same
  gap noted at the end of Phase 2. Boilers cover heat independently of the
  controller in every attack scenario tried. Unchanged by Phases 3–5.
- **Genset starts, while much improved, are still non-trivial** (210/year for
  the trained agent, versus the incumbent's 24). This is priced in the reward
  and could be pushed further with a higher `genset_start_per_event` weight or
  more training; not pursued further in this session for diminishing returns.
- **Nothing has been pushed.** All six commits above are local to `main`.
  Confirm with the user before pushing — this is a large, unreviewed batch of
  autonomous work landing on a public repo with four other collaborators.

## Conventions (unchanged from Phase 1–2)

- Every physical parameter lives in station YAML, tagged `[public]` /
  `[derived]` / `[assumed]`.
- Claims are reproducible by a script in `scripts/`, invariants asserted in
  `tests/`.
- No personal data in the repo.
- Commit messages explain *why*, including bugs found and claims deliberately
  not made.

## Maintenance

**Update this file at the end of any session that changes the state of the
project.** Update [docs/PROJECT_BIBLE.md](docs/PROJECT_BIBLE.md) whenever the
architecture, parameters, results, roadmap or claims change — and re-check its
section 8 ("not entitled to claim") every time, since that list decays fastest.
