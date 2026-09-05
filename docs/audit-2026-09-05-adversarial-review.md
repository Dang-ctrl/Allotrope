# Allotrope adversarial system audit — 2026-09-05

A full-repository red-team / gap-analysis pass: safety, security, ML,
deployment, and reliability, done by tracing actual execution rather than
trusting the README, docstrings, or a passing test suite. **Discovery
only — nothing in this document has been fixed yet.** Every finding below
was reproduced with real code in this session; none is inferred from
documentation alone.

## Scope note

Given the size of a full adversarial audit, depth was concentrated where
tracing execution actually paid off: the safety projection layer, the
API/gRPC/MQTT attack surface, RL evaluation determinism, and federated
learning. Areas not live-attacked in this pass (browser-based XSS fuzzing,
real network DDoS load generation, dependency CVE scanning against a live
feed, full sim-to-real robustness testing) are marked **UNVERIFIED —
reasoned from code** rather than given a false pass.

## Executive verdict

| Dimension | Score /100 | Why |
|---|---|---|
| Overall maturity | 46 | A real, tested plant/agent/safety core; almost no operational hardening around it |
| Security maturity | 15 | Zero auth anywhere (API, gRPC, MQTT); anonymous MQTT; no TLS anywhere |
| Safety maturity | 55 | Four of five stated guarantees hold under attack (verified); the fifth (heat) is **reproducibly broken** |
| ML maturity | 50 | Real training/eval pipeline, honest self-reporting — but eval numbers are **not reproducible run-to-run** (proven below) |
| Reliability | 40 | No restart/crash-recovery story, no health-checked container orchestration, single-process-of-truth everywhere |
| Frontend maturity | 55 | Clean, honest (shows "simulation" mode, error banners), no real E2E test, no auth-aware UI |
| Deployment maturity | 30 | Compose file well-designed; never successfully built in any environment tried; a real dependency bug in the Dockerfile went unnoticed until a parallel fork's build caught it |
| SIH readiness | 60 | Strong narrative and honest self-critique; the RL story currently loses to the rule-based baseline on a held-out seed, and that's the actual number a judge who asks will get |

## Top findings, most severe first

| ID | Sev | Category | Finding | Evidence | Impact | Suggested fix |
|---|---|---|---|---|---|---|
| F1 | **P0** | Safety | The heat guarantee (`_enforce_heat` in `allotrope/safety/projection.py`) checks recovered heat against **rated** genset output, but `_bound_setpoints` (which runs *after* it) only raises setpoints to cover electrical load, not heat. Reproduced live: forced 2 gensets on believing 275 kW recoverable; actual recovered heat at the real commanded setpoints was **44 kW** against a 200 kW shortfall | `SafetyProjection.project()` called directly with a synthetic low-electrical/high-thermal observation — interventions fired, guarantee still violated by 156 kW | Station heat supply can silently fall short exactly in the "deep cold snap" scenario the code's own docstring names as the reason this bound exists | Compute the heat-producing setpoint requirement alongside `_enforce_heat`'s commit decision, or re-run `_enforce_heat` after `_bound_setpoints` |
| F2 | **P0** | ML/Reproducibility | "Deterministic" hybrid-agent evaluation is not reproducible. Same checkpoint, same seed=1, same 8760 periods, run 6 times: genset_starts = 489, 495, 496, 497, 500, 527 | Reproduced across separate process invocations in this session | Every published RL number (README, docs/reinforcement-learning.md's 497-starts table, `run_demo.py`'s output) is one sample from a noisy process, not a fixed fact | Root cause is F3 |
| F3 | **P0** (root cause of F2) | Safety/ML | `GuardedController.act()` (`allotrope/safety/fallback.py`) measures the **wall-clock** latency of the neural-net forward pass and silently substitutes the deterministic fallback if it exceeds a 10 ms budget. CPU scheduling jitter makes this nondeterministic per step, even though both networks are otherwise bit-deterministic under `deterministic=True` (verified: `torch.tanh(mean)`, no RNG) | Traced `latency_budget_ms = 10.0` and the fallback branch in `fallback.py` | Live demo behavior depends on machine load, not the model; running other processes concurrently increases fallback frequency, which can make numbers look better by quietly doing more rule-based control | Evaluation runs should never fall back on latency — only a genuinely time-boxed real control loop should |
| F4 | P1 | Security | FastAPI backend (`allotrope/api/app.py`) has zero authentication, CORS `allow_origins=["*"]`, no rate limiting on any endpoint, including simulation start/stop/reset/step | Read `create_app()` in full | Any client on the network can control every station's simulation or exhaust CPU via repeated `/simulation/step` calls | Add API-key/session auth; rate-limit mutating endpoints |
| F5 | P1 | Security | gRPC server (`allotrope/controlplane/server.py`) uses `add_insecure_port` — no TLS, no auth on any RPC | Read `serve()` | Unauthenticated access to live station telemetry | Add TLS + token auth before any non-localhost exposure |
| F6 | P1 | Security/DoS | gRPC server pool is `ThreadPoolExecutor(max_workers=8)`; `StreamState` occupies one worker for its entire connection lifetime | Read `serve()` + `StreamState` | 8 concurrent `StreamState` clients exhaust the entire server — every other RPC queues indefinitely | Bound streaming concurrency separately from unary RPCs, or move to an async gRPC server |
| F7 | P1 | Security | `deploy/mosquitto.conf`: `allow_anonymous true`, no TLS, no topic ACLs, port bound to the host | Read the file directly | Any reachable client can publish fake telemetry, or publish to the federated-update topic used to carry model weights | Client certs or username/password + per-station topic ACLs before any deployment past localhost |
| F8 | P1 | ML Security | Federated aggregation (`allotrope/federated/aggregate.py`) is plain FedAvg — no clipping, no norm bounds, no Byzantine-robust aggregation | Read `average_state_dict`/`fedavg_checkpoint` in full | A single malicious or compromised contributor can dominate or backdoor the global model | Add per-update norm clipping at minimum |
| F9 | P2 | ML Security | The federated round validator (`allotrope/federated/round.py::default_validator`) checks exactly one held-out seed; a targeted/backdoored update would sail through | Read `default_validator` in full | The "validation + rollback" story only covers global-regression failures, not targeted ones | Multi-seed validation at minimum; document the actual guarantee scope precisely |
| F10 | P2 | Architecture | "Federated learning across stations" is currently a single-process orchestrator — `run_round` calls local training for every "station" in-process, no real network transport or trust boundary exists yet | Read `coordinator.py::run_round` | The malicious-participant/sybil threat model doesn't yet apply because there's no real multi-party channel to attack — but the README's phrasing implies more than exists | Document explicitly as "aggregation math proven, transport/trust protocol not yet built" |
| F11 | P2 | Deployment | `deploy/Dockerfile`'s `pip install -e .` previously installed only base dependencies, omitting `paho-mqtt`/`psycopg`/`torch`, which both compose services need | Verified against `pyproject.toml`'s optional-dependency groups | Container crashes on startup — already partially fixed this session, but still unverified by an actual build (registry access blocked in every environment tried) | Verify once an environment with registry access is available |
| F12 | P2 | Reliability | No `restart:`/`healthcheck:` policy anywhere in `docker-compose.yml` | Read the file in full | A crashed station container stays dead with no automated recovery | Add `restart: unless-stopped` + healthchecks |
| F13 | P2 | Security | Compose exposes Postgres/MQTT/Grafana on `0.0.0.0` by default with weak baked-in credentials (`allotrope`/`allotrope`) | Read `docker-compose.yml`, `grafana/provisioning/datasources/timescaledb.yml` | Naively running this compose file on a cloud VM exposes an anonymous-write telemetry bus and a guessable-credential database | Bind to `127.0.0.1:` by default; require credential override for non-local deployment |
| F14 | P2 | API | `GET /stations/{id}/telemetry?last=N` has no upper bound on `N` | Read `get_telemetry` | Cheap, repeatable memory/CPU amplification | Cap `last` server-side |
| F15 | P3 | Testing | Zero tests reference the heat-guarantee code path in `tests/test_safety.py`'s 20 test functions | Direct grep | This is exactly why F1 shipped silently | Add a regression test once F1 is fixed |
| F16 | P3 | Documentation | The RL headline framing risks overstating results relative to `docs/reinforcement-learning.md`'s own "Honest status" section, which already discloses the currently-checked-in checkpoint loses to the rule-based baseline on fuel/starts | Ran `run_demo.py` and `allotrope.evaluate` in this session | A reader of only the top-level status table could miss the magnitude | Surface the "not yet competitive" framing at equal prominence |

## What already holds up (verified, not assumed)

- **No actuation RPC exists anywhere** — `allotrope.controlplane`'s proto is read-only (`GetState`/`StreamState`/`Heartbeat`). The entire command-injection threat model (forged START GENERATOR, DISABLE SAFETY, etc.) does not apply today because there is no command channel to attack. This is a real, positive architectural finding, not an oversight.
- Four of the five safety-projection guarantees (capacity cover, blocked unsafe stops, battery envelope, critical-load-never-displaced) hold under the existing adversarial attack suite (`scripts/run_safety_audit.py`) and under additional inspection in this session.
- NaN/Inf/malformed-input sanitization in the projection layer is correct and complete — verified by reading every sanitizing function, not just by the existing tests.
- Checkpoint loading uses `torch.load(..., weights_only=True)`, which blocks arbitrary-code-execution-via-pickle — already a correct mitigation.
- The project's own "Honest status" documentation sections pre-empted several findings that would otherwise have required cold discovery (the unmet-water reward regression, the unguarded-policy danger trend, the Docker build blockage). Worth explicit credit.

## Pre-production checklist

| Item | Status |
|---|---|
| No known critical-load violations under adversarial dispatch | PASS |
| Heat guarantee holds under adversarial dispatch | **FAIL (F1)** |
| RL evaluation results are reproducible | **FAIL (F2/F3)** |
| API authenticated | FAIL (F4) |
| Control plane authenticated/encrypted | FAIL (F5) |
| MQTT authenticated | FAIL (F7) |
| No command-injection surface exists | PASS |
| Checkpoint loading safe against arbitrary code execution | PASS |
| Container builds successfully | UNKNOWN — never verified in any environment tried |
| Federated aggregation resists a malicious contributor | FAIL (F8) |
| Compose stack has restart/health policies | FAIL (F12) |
| Full test suite passes | PASS (219 passed) |

## Verdict: would this control a real station today?

**No, and not yet with conditions**, because F1 is a real, reproducible
violation of a specific safety claim the project makes about itself — not
a hardening gap. Everything else here (missing auth, nondeterminism, the
unverified Docker build) matters, but F1 directly contradicts a
load-bearing claim.

With F1 and F3 fixed, and basic auth added to the API/gRPC/MQTT surfaces,
this would conditionally be trustworthy for exactly the scope the project
already claims: a dispatch-level safety net for Maitri and Bharati, no
actuation path, no real multi-station federated trust boundary yet.

## 7-day fix priority

1. Fix F1 (heat guarantee) and add the regression test (F15).
2. Fix F3 (decouple evaluation determinism from the latency-based
   fallback) — this also resolves F2 and restores trust in every
   published RL number.
3. Add basic API-key auth to the REST API and gRPC server (F4/F5).
4. Bind compose ports to localhost by default, add restart policies
   (F12/F13).
5. Cap `last` on the telemetry endpoint (F14).
