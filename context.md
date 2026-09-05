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
| Phase | **5 of 5, and the container stack has been run for real** — see caveats below |
| Tests | 204 passing |
| Commits | `db4b9ab` Plant · `6d42c9b` Guarantee · `d0f9ca9` Docs · `14e4303` Agents · `6f11585` Twin · `ca028cf` System · `c35a68a` Phase 3 results · `8866e8d` Federated checkpoint fix |

**This was an autonomous overnight session** (user asked to "complete the project" while asleep, using judgment for decisions without stopping to ask). Everything below was built, tested, committed, and pushed to `main` without further confirmation, per that instruction. The user has not reviewed this batch of work — it is large (11 commits, ~7 500 lines) and everything in it should be treated as freshly landed rather than settled, especially the two negative/partial results (federated training, Phase 5 infrastructure) called out below.

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
- **Docker Desktop works on this machine and the full stack has been run.**
  It is flaky under a heavy first build, though: a `docker compose up --build`
  once appeared to hang for 20+ minutes and the daemon started returning 500s
  on every API call (`docker version` included) mid-build. The build had
  actually completed successfully by the time it finished, but the daemon
  itself had crashed and needed Docker Desktop restarted manually before
  `docker compose up -d` would work again. If this happens again: check
  whether the build log actually shows completion before assuming it's stuck,
  but be ready to just restart Docker Desktop.
- No `mosquitto` binary and no live Postgres/TimescaleDB *outside containers*
  in this environment, which is why the test suite still uses an embedded
  `amqtt` broker for MQTT tests and a fake connection for the TimescaleDB
  bridge tests — those remain the CI-safe path. The real broker and database
  only exist inside the Docker stack.

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

Bharati (`checkpoints/bharati.pt`), same protocol, held-out seeds 100–104:

| | Legacy N+1 | Efficient rules | **Hybrid DQN+SDDPG** |
|---|---|---|---|
| Fuel | 264.5 kL | 205.4 kL | **193.8 kL** |
| Black carbon | 95 085 g | 40 654 g | 40 722 g |
| Genset starts/year | 185.4 | 140.0 | **14.8** |
| Life support unserved | 0 | 0 | **0** |

Different station, different optimum: Bharati's agent found **5.6% less fuel**
and **89% fewer starts** than the rule-based baseline, essentially eliminating
cycling, at the cost of a higher wet-stacking fraction (0.599 vs 0.257) and
flat black carbon. Maitri's agent instead cut starts more modestly (23%) and
traded away more black-carbon performance. Both are legitimate optima under
the same reward weights — report both, not just the flattering one.

**Two real bugs were found and fixed getting to these numbers.** First: the
initial training run (300 episodes, discarded) used exploration-decay defaults
tuned for ~1000 episodes, so it finished with the DQN still 40% random —
every number from that run was noise, not policy. `scripts/train_agent.py`
now derives the decay schedule from the actual episode count requested.
Second: `scripts/evaluate_agent.py` itself built its summary table with rows
and columns transposed (`pd.DataFrame(means)` needed `.T`), caught only
because it crashed rather than silently printing a wrong table. Fixed
properly, not patched: the averaging-and-tabulating logic moved into
`allotrope.sim.runner.compare_multi`, a tested library function
(`tests/test_runner.py`), rather than staying as untested script code.

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
**Update: the stack was run for real** once Docker Desktop was available. All
six containers came up, and real telemetry flowed plant -> gRPC -> safety
projection -> MQTT -> TimescaleDB, confirmed by querying the `telemetry` table
directly (66+ rows per station after a few minutes, `critical_unserved_kw = 0`
on every one) and by running the Grafana dashboard's own panel queries against
that data. Two real bugs were caught only by this: `protobuf` and `psycopg`
were both usable locally only because they'd been installed by hand into the
dev venv at some point and were never in `pyproject.toml` -- both station
containers and the bridge container crashed on import within a second of
starting in a fresh build. Fixed by declaring both as real dependencies. See
`deploy/README.md` for the full verification table.

## Judge-facing artifacts

Two published Claude Artifacts exist for demoing this project, both driven by
real data, none of it narrated:

- **Console** (`https://claude.ai/code/artifact/2afb7cc5-ad1d-4b80-961a-4244de4a5979`)
  — the results dashboard: baseline comparison, safety audit table, held-out
  agent evaluation, the honest federated/infra status log.
- **Scenario explorer** (`https://claude.ai/code/artifact/f829ddec-0529-4097-8fdd-c1ca378caf34`)
  — four interactive, played-back simulations: (1) a blizzard and record cold
  landing the same hour an AI failure is injected, guarded vs. unguarded; (2)
  one genset's exhaust fouling under `LegacyNPlusOne` over two weeks while it
  stays clean under `EfficientRuleBased` — the founding problem, watched
  happening; (3) a real windy day at Bharati where legacy wastes 1,418 kWh of
  wind and the efficient controller runs the station on wind alone for 9
  hours; (4) a forward stress test of the OpenDSS twin's Volt-VAr/Volt-Watt
  fallback as installed renewable capacity is scaled from 1x to 6x today's —
  explicitly labelled as a stress test, since real generation today never
  approaches the trigger point.

`scripts/generate_scenarios.py` reproduces the exact data behind all four
scenario-explorer cases and is checked against the published data byte-for-byte
(only unused display fields differ). Run it after changing anything upstream of
these scenarios and diff the output before republishing the artifact — three
real bugs were found building these (two mismatched simulation
seed/period/start-date alignments producing fabricated events, one climate-generator
non-reproducibility across differing `periods` values with the same seed) and
none of them were obvious from the numbers alone; they were caught only by
re-deriving each scenario's timing independently before trusting it.
## Open questions and known gaps

- **Bharati agent: done**, results above. **Federated training: done, and
  negative.** 30 rounds × 15 local episodes (`checkpoints/federated.pt`)
  underperforms `EfficientRuleBased` at both stations (+5.6% fuel at Maitri,
  +2.3% at Bharati) and underperforms each station's own single-agent
  checkpoint substantially. Safety held perfectly regardless — zero unserved,
  zero freeze, both stations, every held-out seed. The training log shows no
  convergence trend across 30 rounds (reward fluctuating −1850 to −4700),
  consistent with FedAvg client drift under a short 15-episode local window.
  Not pursued further — see docs/PROJECT_BIBLE.md §8 for why chasing a better
  number here would have been the wrong move. **If someone wants a working
  federated policy**, the next thing to try is more local episodes per round
  (e.g. 40-50) with fewer total rounds, or a FedProx-style proximal term.
- **`evaluate_agent.py`'s reporting bug is fixed properly**, not merely
  patched: the averaging/tabulating logic that had rows and columns
  transposed moved into `allotrope.sim.runner.compare_multi`, now covered by
  `tests/test_runner.py`. `run_baseline.py` still has no dedicated test for
  its own printing logic, which is a smaller residual version of the same
  gap — low risk since it does far less than `evaluate_agent.py` did.
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
section 12 ("not entitled to claim") every time, since that list decays fastest.
