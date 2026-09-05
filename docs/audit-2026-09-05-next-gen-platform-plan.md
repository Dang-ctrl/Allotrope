# Next-generation Allotrope: audit and implementation plan

**Discovery and planning only. No production code changed in this document's
commit.** Every "implemented" claim below was checked against the actual
source at commit `2fc28ed` (this repo's `main`, 2026-09-05) — file paths and
line-level behavior, not the README's description of itself. Every proposal
in Parts 5 onward is labeled `MUST HAVE` / `SHOULD HAVE` / `NICE TO HAVE` /
`DO NOT IMPLEMENT` and is a recommendation to evaluate, not a claim about
what exists.

Status legend used throughout:

| Status | Meaning |
|---|---|
| ✅ **Implemented** | Real code, on the main execution path, tested |
| 🟡 **Partial** | Real code, but incomplete, untested, or off the main path |
| 🔵 **Simulated/mock** | Exists but stands in for something not modeled |
| 📄 **Documented, not integrated** | Described somewhere; no wiring found |
| 📋 **Planned** | Explicitly future work in this repo's own docs |
| ❌ **Missing** | No trace in the codebase |

---

## Part 1 — Repository audit

### 1.1 Architecture map (verified)

```
                         ┌─────────────────────────┐
synthetic climate/loads →│  PolarMicrogrid (plant)  │← StationConfig (YAML)
allotrope/synth/         │  allotrope/sim/plant.py  │  allotrope/config/
                         └────────────┬─────────────┘
                                      │ observe() / step(DispatchCommand)
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                  ▼
        rule-based baselines   PolarMicrogridEnv    OpenDSS network twin
        allotrope/control/     (Gymnasium)          allotrope/sim/network.py
        (LegacyNPlusOne,       allotrope/envs/      (Maitri only, single
         EfficientRuleBased)   polar_microgrid.py    feeder, Volt-Watt only)
                    │                 │                  │
                    │        BranchingDQN + SDDPG         │
                    │        allotrope/agents/            │
                    │        → HybridAgent (proposer)      │
                    └────────┬────────┘                  │
                             ▼                            ▼
                  SafetyProjection ──────────────► VoltWattCurve
                  allotrope/safety/projection.py   allotrope/safety/voltage.py
                             │
                             ▼
                  GuardedController (deterministic fallback)
                  allotrope/safety/fallback.py
                             │
              ┌──────────────┼───────────────────┐
              ▼              ▼                    ▼
     FastAPI backend   gRPC control plane    MQTT + TimescaleDB
     allotrope/api/    allotrope/controlplane/  allotrope/mqtt/
              │
              ▼
     React frontend (frontend/) — Command Center dashboard

  Orthogonal, not on the control path:
  - allotrope/federated/  (cross-station FedAvg, offline/manual)
  - allotrope/intelligence/asset_health/  (read-only telemetry observer)
  - allotrope/intelligence/forecasting/   (standalone, not fed to agents)
  - allotrope/evidence/                   (provenance records for metrics)
```

### 1.2 Component-by-component

**Station configuration** — ✅ `allotrope/config/stations/{maitri,bharati}.yaml` + `allotrope/config/__init__.py` (`StationConfig`, `load_station`, `available_stations`). Typed, Pydantic-validated. Every numeric field is tagged published/derived/assumed in `docs/calibration.md`. Two stations exist; adding a third is a YAML file, not code.

**Synthetic climate/demand** — ✅ `allotrope/synth/climate.py`, `allotrope/synth/loads.py`. Deterministic given a seed; solar geometry from latitude, calibrated irradiance/wind/temperature, electrical + thermal + deferrable (snow-melt) demand with a noise process. Real, tested (`tests/test_climate.py`). Not measured station telemetry — there is none public — and `docs/calibration.md` says so.

**Plant / digital twin core** — ✅ `allotrope/sim/plant.py` (`PolarMicrogrid`, 482 lines). A **power-balance model**: kW in, kW out, no bus/line/voltage concept except where the network twin (below) is bolted on. Handles genset commitment + anti-cycling, dual-chemistry battery dispatch with thermal derating, CHP/boiler heat coupling, snow-melt deferrable load, wet-stacking deposit accumulation. This is the actual "digital twin" — real, tested (`tests/test_plant.py`, `tests/test_assets.py`), and the one thing every controller (rule-based, RL, safety audit) runs against.

**PyPSA** — ❌ **Missing.** `grep -r pypsa allotrope/` returns nothing. The README's own architecture diagram (`digital twin -> PyPSA + OpenDSS state est.`) describes something that was never built; the plant is a hand-written power-balance simulator, not a PyPSA network. This is a **documentation/reality mismatch**, not a partial implementation — flagging it here because it would mislead anyone reading the README before the code.

**OpenDSS** — 🟡 **Partial, Maitri only.** `allotrope/sim/network.py` + `allotrope/safety/voltage.py`, via `opendssdirect`. Solves one bus-voltage snapshot per step at unity power factor (no VAr modeled anywhere in the plant, stated explicitly in the module docstring). Feeds `VoltWattCurve` for inverter-level curtailment. Bharati has no network config. Volt-VAr is explicitly not implemented — the plant has no reactive-power balance to act on. Real and tested (`tests/test_network.py`, `tests/test_voltage_safety.py`), just narrow in scope.

**Gymnasium/RL environment** — ✅ `allotrope/envs/polar_microgrid.py` (`PolarMicrogridEnv`, 326 lines). `Dict` action space (`MultiBinary(n_gensets)` + `Box` continuous dispatch), `Box` observation of width `12 + 5*n_gensets + 2*n_storage`. Applies the safety projection **inside** `step()` when `apply_safety=True` (the training default) — replay buffer stores the raw proposal + the reward the *safe* action actually produced. See Part 2 for the full math.

**Reward function** — ✅ `allotrope/envs/reward.py` (150 lines). Every term is a real physical quantity (litres, grams, kWh, starts) times a stated rupee price (`RewardWeights`), summed and negated, scaled by a fixed constant. No arbitrary unitless weights. Safety terms (`critical_unserved`, `freeze`) are priced at 500,000/unit specifically so they dominate any fuel term but are explicitly **not** the safety mechanism — the projection layer is.

**BranchingDQN** — ✅ `allotrope/agents/dqn.py` (134 lines). One binary Q-head per genset over a shared trunk (`DuelingBranchingQNetwork`), Double-DQN target (greedy action from online net, valued by target net), branches coordinate via a **shared mean-branch target**, not `n` independent Bellman backups. Real, trained, tested.

**SDDPG** — ✅ `allotrope/agents/sddpg.py` (130 lines). Squashed-Gaussian stochastic actor + twin critics (TD3-style min-of-two target) + soft-updated targets + delayed policy updates. Named "SDDPG" by this project, not a published algorithm of that name (the module docstring says so explicitly). Proposes loading fraction per genset, power fraction per storage pack, melt rate, all in `[-1, 1]`.

**HybridAgent** — ✅ `allotrope/agents/hybrid.py` (77 lines). Composes DQN + SDDPG into one `.act(observation, plant) -> DispatchCommand`. Carries **no** safety logic itself — same interface contract as the rule-based baselines. `deterministic=True` for eval/deployment.

**Replay buffer** — ✅ `allotrope/agents/replay_buffer.py` (73 lines). Standard uniform-sampling transition buffer, fixed capacity, numpy-backed.

**Training / evaluation / checkpoints** — ✅ `allotrope/train.py`, `allotrope/evaluate.py`, `allotrope/evaluate_scenarios.py`, `allotrope/experiment.py`. Checkpoints are `.pt` files under `runs/`; `ExperimentTracker` records `runs/<run_id>/record.json`. `evaluate_scenarios.py` runs many independent seeds and reports mean/median/std/min/max/percentiles — **this is real stochastic weather/demand variation, not fault injection** (its own module docstring says exactly that; see Part 6).

**Safety projection** — ✅ `allotrope/safety/projection.py` (571 lines, the largest single module in the repo). Deterministic, analytic, no solver. Sanitizes both proposed **commands** and (as of this session's PR #4) **observations** before applying capacity/heat/battery/melt bounds. Evaluates capacity cover jointly across the whole fleet (a real bug this project's own audit found and fixed: per-machine-only cover checks let two sets both look individually safe to stop while zeroing the plant). Audited twice this session (V1, V2) with fixes for every finding.

**Deterministic fallback** — ✅ `allotrope/safety/fallback.py` (256 lines, `GuardedController`). Wraps any agent; takes over on agent exception, NaN/invalid output, or (configurably) a real-time latency budget breach. `enforce_latency_budget` must be `False` for reproducible offline evaluation — a real reproducibility bug this project found and fixed itself (see Part 3).

**Fault handling (adversarial actions)** — ✅ `scripts/run_safety_audit.py` runs 5 adversarial **policies** (random, all-off, max-charge, max-melt, oscillating commitment) against the guarded and unguarded plant. **This is action-space fault injection, not physical/sensor fault injection** — no mechanism exists to force a genset offline, corrupt a sensor mid-run, or drop a communication link (see Part 6/9 gap).

**Federated learning** — 🟡 **Partial, honest about its own limits.** `allotrope/federated/{aggregate,coordinator,round}.py`. Real FedAvg over actual network tensors, real validation-and-rollback gate (round 1 of the project's own 2-round smoke test was rejected for real, unprompted, because the aggregated model used more fuel than Bharati's own baseline). **Not** Byzantine-robust (outlier-norm clipping only). **Not** shown to beat local training — 5,000–10,000 local steps per round, vs. 500,000 for the single-station RL runs. Single-process orchestrator, no real multi-party transport.

**Edge inference** — ❌ **Missing as a deployment concern.** Inference itself is just "load a `.pt` checkpoint, call `.act()`" (works CPU-only today, since nothing in `allotrope/agents/` requires a GPU) — but there is no watchdog, model versioning/rollback, inference-timeout-to-fallback wiring beyond the existing `GuardedController` latency budget, or offline-operation story beyond "the code doesn't require network access." Not designed for it; it happens to not need much.

**gRPC control plane** — ✅ `allotrope/controlplane/` (state distribution + heartbeat, real generated protobuf/gRPC code, `docs/control-plane.md`). Token-authenticated (found unauthenticated in this project's own audit, fixed), `StreamState` concurrency-capped against a worker-pool DoS the same audit reproduced. No TLS. No actuation RPC exists at all — by design, so there is no command-injection surface yet, honestly noted rather than silently absent.

**MQTT** — ✅ `allotrope/mqtt/` (embedded-broker pub/sub, TimescaleDB bridge). Auth required by default (found anonymous, fixed). Docker Compose stack (`deploy/docker-compose.yml`) written with mosquitto/TimescaleDB/Grafana but not run end-to-end in this environment — `deploy/README.md` states exactly which piece is proven where.

**FastAPI backend** — ✅ `allotrope/api/app.py`, `allotrope/api/simulation.py`. Real endpoints over a live `StationSimulation` (not a mock). API-key-gated control endpoints, per-client rate limiting (both found missing by this project's own audit, both fixed). No WebSockets — the frontend **polls** REST endpoints at 1s/5s intervals (`frontend/src/hooks/usePolling.ts`), which is real and tested but not a push architecture.

**WebSockets** — ❌ **Missing.** Not used anywhere; polling is the actual mechanism.

**React frontend** — ✅ `frontend/src/`. One real screen (Command Center) against live API data: power balance, genset fleet, storage, safety projection (with a human-readable intervention-code table), cumulative metrics, and (added this session, PR #10) a recent-trend chart. No E2E/browser test tooling was available in past sessions to verify visually beyond manual Playwright screenshots (this session actually ran it headless and screenshotted it — see PRs #9/#10 for what that caught).

**Grafana** — 📄 **Documented, not integrated.** Provisioning files exist in `deploy/grafana/`; not demonstrated running against real data in this environment.

**Docker** — 🟡 **Partial.** `deploy/Dockerfile`, `docker-compose.yml` exist and were written carefully (documented per-service in `deploy/README.md`), but the full stack has not been run end-to-end in any session so far.

**Configuration** — ✅ Pydantic-validated YAML (`allotrope/config/`), no untyped dict config anywhere in the control path.

**Tests** — ✅ 22 test files (`tests/`), 273 passing as of this session (verified this session, not asserted from memory). Covers plant physics, RL agents (including adversarial/Hypothesis-driven attacks), safety projection, API, gRPC, MQTT, federated learning, and the three modules added this session (asset health, evidence, forecasting).

**Benchmarks / experiment scripts** — ✅ `scripts/run_baseline.py` (single-seed comparison), `allotrope.evaluate_scenarios` (many-seed statistical spread), `scripts/run_safety_audit.py` (adversarial-action audit). **No MILP or MPC baseline exists anywhere in the repository** — confirmed via `grep -rli "milp\|pulp\|cvxpy\|pyomo\|mpc\b"` returning nothing.

**Documentation** — ✅ Extensive and, on inspection, unusually honest by convention (every doc has a "what this is not" section) — with the one real exception found this audit: the README's architecture diagram naming PyPSA, which doesn't exist.

**New this session, not in prior audits** — `allotrope/intelligence/asset_health/` (genset wear-score + battery full-equivalent-cycles, **read-only, not wired into dispatch or RL** — this is the closest thing to a degradation model that exists, and it's observational only), `allotrope/intelligence/forecasting/` (persistence/seasonal-naive/EWMA forecasters, **standalone, not fed to any controller**), `allotrope/evidence/` (provenance records tying a headline metric to its git commit/seed/baseline).

---

## Part 2 — The actual algorithm

### State (observation) representation

`PolarMicrogridEnv._observation_width() = 12 + 5·n_gensets + 2·n_storage`. For Maitri (3 gensets, 2 storage packs): 12 + 15 + 4 = **31 dimensions**. The 12 station-level features are electrical/critical/firm-thermal load, PV/wind available, air/wind/indoor temperature, snow-melt remaining, plus (added this session, PR #5) — no, those are per-genset. Per-genset (×5): online, power_kw, deposit, **can_start, can_stop** (the last two added this session specifically to fix a flapping bug — see below). Per-storage (×2): SOC, and one more envelope field. All continuous, clipped into `[-5, 5]` by the `Box` space (not internally normalized beyond each raw unit's natural scale).

### Discrete action space

`MultiBinary(n_gensets)` — one bit per genset, "should this set be committed." `SafetyProjection._enforce_capacity` and `_enforce_heat` can force additional starts the policy didn't request; they never force a stop the policy didn't request, and never override a stop into a start.

### Continuous action space

`Box(-1, 1, shape=(n_gensets + n_storage + 1,))` = loading fraction per genset (mapped onto each machine's own stable band, not `[0, rated]`), power fraction per storage pack, one scalar for snow-melt rate.

### BranchingDQN — Double DQN with branch-coordinated targets

For genset $i$ at state $s$: $Q_i(s, a_i; \theta)$, two-valued ($a_i \in \{0,1\}$), sharing trunk weights across branches. The target is:

$$y = r + \gamma (1-d) \cdot \frac{1}{n}\sum_{i=1}^{n} Q_i\big(s', \arg\max_{a_i} Q_i(s', a_i; \theta); \theta^-\big)$$

— i.e. **one scalar target, shared across every branch**, using the online network to select the greedy next action per branch (Double-DQN) and the target network to value it, then averaged across branches. This is a real design choice with a real consequence: branches don't get independent value estimates, they get a joint one, which is the mechanism by which "start G2" and "start G3" can be judged as parts of one decision rather than two unrelated ones — and also means a single branch's Bellman error is diluted by however many other branches exist. Loss: smooth-L1 (Huber) between `Q_i(s,a_i)` for the taken action and this shared target, per branch. $\epsilon$-greedy exploration, linearly decayed per-branch-independently (each branch's random/greedy choice is drawn independently even though the target is shared).

### SDDPG — stochastic-actor DDPG/TD3 hybrid

Actor $\pi_\phi(s)$ outputs a squashed-Gaussian (tanh-of-Normal) distribution over the continuous action; `act(deterministic=True)` returns the mean, `deterministic=False` samples. Twin critics $Q_1, Q_2$ (TD3-style), target computed as:

$$y = r + \gamma(1-d)\min(Q_1^-(s', a'), Q_2^-(s', a'))$$

with $a' \sim \pi^-_\phi(s')$ sampled from the **target** actor (not the online one). Critic loss: MSE on both heads against the shared target. Actor updated every `policy_delay=2` critic steps (TD3's delayed-policy-update trick), maximizing $Q_1(s, \pi_\phi(s))$ only (not the min of both critics — a real, if minor, asymmetry from vanilla TD3, which typically also uses $Q_1$ alone for the actor loss, so this matches convention). Soft target updates ($\tau = 0.005$) on both actor and critic targets, gated to the same delayed cadence as the actor update.

### How the two agents interact

They do **not** interact during training — DQN and SDDPG are trained as two entirely separate off-policy learners against the same environment, each drawing its own experience from what appears to be (needs confirming in `train.py`, not fully traced this pass) either a shared or separate replay buffer keyed by `genset_on` / `dispatch` respectively. At **inference** they compose purely functionally: `HybridAgent.act()` calls `dqn.act()` then `sddpg.act()` independently and concatenates the results into one `DispatchCommand` — there is no joint policy network, no shared trunk between the two learned models, and no coordination signal beyond both seeing the same observation vector.

### Reward, episode structure, temporal resolution

Reward is computed **after** the safety projection has already modified the action (training reward reflects the *safe* outcome, not the raw proposal) — see `PolarMicrogridEnv.step`. Episode length is configurable (`episode_steps`, e.g. 336 steps ≈ 14 days at 1-hour resolution in the reported runs); `randomise_start` lets training sample different points in the synthetic year rather than always starting Jan 1. Temporal resolution is `plant.dt_h`, hourly in every reported run.

### Safety projection and fallback (inference path)

`GuardedController.act()`: call the wrapped agent under a latency budget (real-deployment default `True`, must be `False` for reproducible offline eval — a bug this project found in itself) → on exception/timeout/invalid output, substitute the deterministic fallback → otherwise pass the proposal through `SafetyProjection.project()`, which sanitizes then bounds it → execute. This is the **only** path any controller in this project uses to reach the plant; `HybridAgent` has zero special-cased logic.

### Assumptions and documented algorithm/implementation mismatches

- The README calls the actor-critic component "SDDPG" without qualification in prose; the module docstring is the only place that clarifies it's not a published algorithm by that name. A reader who trusts the README literally would misidentify the method.
- The README's architecture diagram names PyPSA; it isn't in the codebase. This is the one outright documentation/reality mismatch found in this audit.
- "500k-step training run" in the README is accurate and reproducible (re-confirmed this session's evaluation four times, 495 starts every time, once the latency-budget-during-eval bug was fixed) — no mismatch there.

---

## Part 3 — Current performance baseline (verified, not invented)

**Reproduced from `docs/reinforcement-learning.md`**, Maitri, seed 0 training / seed 1 held-out evaluation, full synthetic year (8760 hourly steps):

| | Legacy N+1 | Efficient rules | Hybrid, guarded (60k) | Hybrid, guarded (500k) | Hybrid, **un**guarded (500k) |
|---|---|---|---|---|---|
| Fuel | 254.6 kL | 214.6 kL | 237.2 kL | **223.3 kL** | 155.3 kL |
| Black carbon | 71 826 g | 11 008 g | 59 730 g | **38 269 g** | 18 399 g |
| Wet-stacking fraction | 0.794 | 0.023 | 0.368 | **0.167** | 0.148 |
| Genset starts | 21 | 286 | 947 | **495** | 3 749 |
| Critical unserved | 0 kWh | 0 kWh | 0 kWh | **0 kWh** | **197 146 kWh** |
| Freeze violation steps | 0 | 0 | 0 | 0 | 0 |

**Reading it honestly:** the guarded hybrid agent beats the legacy incumbent on every metric, and closes ~60% of the fuel gap to the best rule-based baseline between the 60k and 500k runs — but does **not yet beat `EfficientRuleBased` on fuel or genset starts** (223.3 kL vs 214.6 kL; 495 starts vs 286). The safety claim is unconditional in both runs: the guarded column never loses critical load, while the *same trained weights*, unguarded, lose 197 MWh — direct evidence the projection layer is load-bearing, not decorative.

**A retrain is in progress in this session** (rl/observation-genset-lockout: exposing `genset_can_start`/`genset_can_stop` to the observation, diagnosed as the root cause of excessive starts — the safety layer blocked 6,599 stop requests across the prior year that the agent had no way to know were infeasible). Not complete as of this document; its result will be reported separately once it finishes, not estimated here.

**Federated learning**, 2-round smoke test (`docs/federated-learning.md`): round 1's aggregated model was correctly **rejected** (used more fuel than Bharati's own baseline after averaging two undertrained per-station networks); round 2's aggregated model passed. This demonstrates the rollback gate works, not that federated training beats local training — 5,000–10,000 local steps is far below the 500,000 used for the single-station numbers above, so the two are not comparable yet.

**Cannot currently be reproduced / not measured:**
- **MILP/MPC baselines** — don't exist, so "how close does RL get to optimal" is currently unanswerable.
- **Battery degradation, generator RUL** — no model exists beyond the new (unwired) asset-health wear-score proxy, so there is no "battery degradation under policy X" number to report.
- **Resilience under physical faults** (generator failure, sensor corruption, comms loss) — no fault-injection framework exists; only adversarial-*action* attacks have been measured (Part 1).
- **Inference latency distribution** — `GuardedController` measures and enforces a latency budget per step, but no aggregate latency benchmark report exists in `runs/` or `docs/`.
- **Training wall-clock time** — not recorded by `ExperimentTracker` in a way surfaced in any doc; would need to be pulled from a specific `record.json`, not asserted here from memory.

---

## Part 4 — Real research/engineering gap

RL for microgrids, DQN, DDPG/TD3, hybrid discrete/continuous RL, MPC, MILP, safe RL via action projection, digital twins, FedAvg, and battery-aware optimization are all well-established individually — none of that is this project's claim to novelty, and Part 5 onward does not pretend otherwise.

**What Allotrope can defensibly claim today, verified in this audit:**
1. A safety projection that is *load-bearing and proven so by ablation* — the unguarded-vs-guarded comparison above is a real, reproducible experiment showing the guarantee changes outcomes, not a design that merely exists on paper.
2. An environment where the safety layer runs **inside training**, not bolted on after — so the replay buffer never contains an unsafe-transition artifact, and exploration is safe from the first random action.
3. Extreme-honesty documentation as an engineering practice — most modules explicitly state their own limits, which is unusual and made this audit fast and trustworthy rather than adversarial.
4. A synthetic-but-calibrated Antarctic environment (wet-stacking, black-carbon-on-ice, polar-night solar geometry, dual-chemistry battery thermal derating) that is genuinely domain-specific, not a generic microgrid re-skinned.

**What is currently a gap, not yet a differentiator:** everything in Part 5 (degradation, thermal-electric coupling, uncertainty, controller switching, MILP/MPC baselines, fault-injection resilience benchmark, federated personalization) is *proposed*, not built. The target positioning — "a safety-critical autonomous energy-management and resilience platform for extreme isolated Antarctic microgrids" — is earned by the *integration* of these pieces against a real safety guarantee, not by any one of them being individually new. That integration is exactly what Parts 5–14 below plan.

---

## Part 5 — Proposed target architecture (evaluated per sub-part)

### A. Multi-objective energy management
**Recommendation:** keep the current single-scalar, priced-in-real-units reward (`RewardWeights`) as the RL training signal — it is already principled, not arbitrary, and rewriting it risks losing a scheme this project's own docs argue for carefully. **Add**, not replace: a Pareto/normalized-objective *evaluation* layer (report each objective's raw value alongside the scalar) so a reader can see the trade-off, without asking RL to solve a multi-objective problem it currently doesn't need to. **MUST HAVE**: per-objective breakdown already exists (`RewardBreakdown`) — extend evaluation reporting to surface it consistently. **SHOULD HAVE**: a normalized (0–1 per objective, min-max over a fixed reference range) reporting view for cross-run comparison. **DO NOT IMPLEMENT**: full Pareto-frontier multi-objective RL (e.g. MORL) — high implementation risk, no evidence the single-scalar reward is actually the bottleneck yet (the 500k-step run hasn't converged to beat the rule-based baseline; that's likely a training-budget and observation-design problem, not a reward-shape problem).

### B. Generator health / degradation
The asset-health module (`allotrope/intelligence/asset_health/`) already computes a defensible wear-score proxy (`1500·starts + 8000·deposit`, reusing the reward function's own prices) and full-equivalent-cycles for batteries — **but it is read-only and not wired into RL state, reward, or dispatch.** **MUST HAVE**: feed the existing wear score into the observation vector (cheap — it's already computed) so the agent can see accumulated stress, not just the current step's deposit. **SHOULD HAVE**: an explicit degradation-cost term in the reward, reusing the same prices already in `RewardWeights` (no new arbitrary weight). **DO NOT IMPLEMENT**: a learned or physics-based RUL estimator — no failure-mode dataset exists to validate one against; this project's own house rule against fabricated precision applies directly here.

### C. Battery degradation
`BatteryState.throughput_kwh` and thermal derating already exist in the plant; `asset_health.BatteryHealth` already derives full-equivalent-cycles and SOC-extreme-hours from them. **SHOULD HAVE**: fold a cycling-cost term (rupees per FEC, calibrated the same "priced in real units" way as everything else in `RewardWeights`) into the reward and observation. **NICE TO HAVE**: a capacity-fade curve (FEC → % capacity lost) driving `BatteryState.capacity_kwh` down over an episode — genuinely useful for long-horizon planning (Part E) but adds real complexity to the plant's dynamics; only worth it once the reward/observation wiring above is proven to matter.

### D. Coupled electric + thermal control
Already exists in the plant (CHP-heat coupling, boiler, `firm_thermal_kw`, `_raise_setpoints_for_heat` in the safety layer) — this is **not a gap in modeling**, it's a gap in **whether the RL reward and observation treat it as a first-class objective** the way electrical dispatch is. **MUST HAVE**: verify (this audit did not fully trace it) that the observation vector's thermal fields are sufficient for the agent to anticipate a heat shortfall before the safety layer has to force a start (`forced_start_to_protect_heating` is a real, logged intervention — its frequency is a direct measure of whether the agent is currently blind to this). **SHOULD HAVE**: report `forced_start_to_protect_heating` rate as its own evaluation metric.

### E. Antarctic long-horizon planning
**Recommendation: do not force RL to plan the whole polar night.** A hierarchical split is the right shape: a **simple, deterministic, auditable long-horizon reserve policy** (e.g. "if remaining fuel ÷ historical burn rate < N days of polar night remaining, raise the minimum reserve margin") feeding a constraint into the *existing* safety projection (which already has a reserve-margin concept), with RL/short-horizon control operating underneath that constraint. **MUST HAVE**: this reserve-escalation rule, because it's cheap, deterministic, and testable exactly like the rest of the safety layer. **DO NOT IMPLEMENT**: a learned long-horizon planner (e.g. a seasonal RL policy) — no training signal exists that spans multiple simulated years, and the existing single-year episodes give no basis for one.

### F. Uncertainty-aware control
**Currently fully missing** (Part 1 confirms zero ensemble/confidence/OOD code). **SHOULD HAVE**, in order of implementation cost: (1) a cheap OOD signal — flag when current observation features fall outside the min/max range seen in the last N training episodes (near-zero cost, no retraining); (2) an ensemble of 3–5 independently-seeded checkpoints (already trainable with existing code, just run `train.py` multiple times) with disagreement (e.g. variance in DQN Q-values or SDDPG action) as a confidence proxy. **NICE TO HAVE**: calibrated prediction intervals via quantile critics. **DO NOT IMPLEMENT**: a bespoke uncertainty architecture before the ensemble baseline is measured — there's no evidence yet that OOD conditions are even a practical problem in this environment (the synthetic climate/demand generators are used for both training and eval).

### G. Adaptive controller switching
**Feasibility check first, as the brief requires:** the only alternative controllers that currently exist are the two rule-based baselines (`LegacyNPlusOne`, `EfficientRuleBased`) and `GuardedController`'s deterministic fallback — there is no MPC or MILP controller yet (Parts J/K below), so "RL → MPC → MILP → deterministic → hard safety" cannot be built end-to-end until those exist. **MUST HAVE** (buildable today): RL → rule-based-baseline → deterministic-fallback, gated on the OOD signal from Part F — this is a small addition to `GuardedController`, which already has the "substitute something else" mechanism for exceptions/timeouts. **SHOULD HAVE** (after Part K exists): insert MPC as the middle tier once it's built and benchmarked for edge feasibility. **DO NOT IMPLEMENT**: the full five-tier hierarchy as a first pass — build and prove each tier's transition before adding the next.

### H. Safety architecture
Already strong (Part 1/3): AI-proposal → sanitize → project → execute is exactly the existing pipeline, and it already covers generator limits, battery bounds, reserve margin, critical-load/heating constraints, NaN/Infinity (both command and, as of this session, observation), and agent timeout/exception/invalid-output. **Gaps confirmed by this audit**: stale-sensor handling (no explicit staleness check found — the safety layer trusts whatever `plant.observe()` returns, current or not), communication-failure and actuator-failure handling (no actuation RPC exists yet — Part 1 — so this is genuinely not yet applicable, not an oversight). **MUST HAVE**: a staleness check (timestamp-age bound on the observation, sanitizing to the same conservative fallback values the corrupted-observation fix already uses) — small, fits the existing `_sanitise_observation` pattern exactly. **SHOULD HAVE**: ramp-rate limits on genset setpoints between steps, if the physical hardware needs them (needs a real spec from `docs/calibration.md`'s sourcing, not an invented number).

### I. Fault injection / adversarial testing
Confirmed missing for physical/sensor faults (Part 1/3) — only adversarial *actions* are currently tested. **MUST HAVE**: extend `scripts/run_safety_audit.py`'s pattern (already proven) to inject faults at the **observation** layer (a genset's reported `can_start` forced `False` mid-episode, a battery's SOC frozen, PV/wind forced to zero) — this reuses the exact sanitization/reporting infrastructure this session's own safety fix built, and directly stresses the new observation-sanitization code path. **SHOULD HAVE**: comms-failure and AI-crash injection once the controller-switching mechanism (Part G) exists to have something to fail over to and measure recovery against.

### J. MILP baseline
**MUST HAVE, and the highest-leverage single addition in this plan** — right now there is *no* answer to "how far is RL from a strong optimization solution," which undermines every performance claim in Part 3. Recommend a rolling or full-horizon MILP over the same `PolarMicrogrid` state transitions: binary genset commitment with startup/shutdown costs and minimum up/down time (already modeled in the plant, just needs linearizing into constraints), battery charge/discharge/SOC linear dynamics, a linearized heat-recovery term, and the same reward-function prices as the RL objective (so the two are comparable on the same cost basis). Use `PuLP` or `python-mip` (CBC solver, no license cost) for feasibility at hackathon timescales — do not reach for a commercial solver. **Explicitly report the formulation's simplifications** (e.g. if wet-stacking deposit growth is dropped from the MILP objective because it's nonlinear) so "MILP-optimal" is never overclaimed.

### K. MPC baseline
**SHOULD HAVE**, built on the same digital twin and, ideally, the same linearized model as the MILP (reuse the formulation, just re-solve on a rolling horizon with forecast data instead of full-year lookahead). Use the existing (currently unwired) forecasting module (`allotrope/intelligence/forecasting/`) as the forecast source — this is exactly the integration point that module was built for and currently lacks. Report solve time per step explicitly, since that's the real edge-feasibility question (Part 5G/8).

### L. Fair controller benchmark
**MUST HAVE**, and mostly a matter of extending `evaluate_scenarios.py` (which already runs many-seed statistical comparisons) to include MILP/MPC once they exist, on the same held-out seeds already used for the RL numbers in Part 3.

---

## Part 6 — Resilience benchmark

**MUST HAVE.** None of the 15 scenario types listed in the brief exist as a suite today — only "normal, varying weather/demand" (`evaluate_scenarios.py`) and "adversarial action" (`run_safety_audit.py`) are covered. Recommend building the scenario suite as a thin layer over the **existing** `PolarMicrogrid` + a new `FaultInjector` (Part 5I) rather than a parallel simulator — reuse, don't rebuild.

**Resilience score** (defensible, not arbitrary): a weighted sum of already-measured quantities, each already in `plant.summary()` or newly available from the fault injector, using the **existing reward-function prices** for the terms that overlap (critical-unserved, unmet-heat, fuel, black carbon) so it's cost-consistent with everything else in this project rather than a fresh invented formula:

$$\text{Resilience cost} = \underbrace{w_{cu}\cdot E_{critical}}_{\text{same price as reward}} + \underbrace{w_h\cdot E_{heat}}_{\text{same}} + \underbrace{w_f\cdot(\text{fuel burned during recovery})}_{\text{same}} + w_r\cdot T_{recover} + w_s\cdot(\text{safety violations})$$

Report the raw components alongside any combined score — per this project's own established convention (`RewardBreakdown`) of never hiding a scalar's composition.

---

## Part 7 — Federated learning strategy

Audited honestly in Parts 1/3: the current implementation is a **real mechanism** (FedAvg, validation, rollback — all genuinely tested), evaluated at a scale (5k–10k local steps) far below what would let it compete with or complement the 500k-step single-station runs. **The question "can knowledge from one station help another without sharing raw data" is currently unanswered, not answered negatively** — 2 rounds is a smoke test, not an experiment. **SHOULD HAVE**, in order: (1) run federated training to a comparable step budget as the single-station numbers (500k total, split across rounds) before drawing any conclusion; (2) compare against local-Maitri, local-Bharati, and a naively-centralized (pooled-data, not federated) baseline on the same held-out seeds as Part 3's table. **NICE TO HAVE**: personalization (fine-tune the federated global model briefly on each station's own recent data) — worth trying only after the base comparison above exists. **DO NOT IMPLEMENT**: claiming a federated-learning result before that comparison is run — this is exactly the kind of invented-result risk this whole audit is structured to prevent.

---

## Part 8 — Edge deployment

Inference is already CPU-only and lightweight (small MLPs, not large models) — the hard requirements are operational, not computational. **MUST HAVE**: a watchdog around the inference process (restart-on-crash, feeding straight into the existing `GuardedController` fallback while restarting), local telemetry buffering when the (currently nonexistent) uplink is down. **SHOULD HAVE**: checkpoint integrity verification (hash check before load) and a simple version/rollback scheme (keep the last-known-good checkpoint path, swap back on a failed sanity check against a fixed validation scenario). **DO NOT IMPLEMENT**: a bespoke model-serving framework — a checkpoint file and a Python process are already sufficient for this workload's size.

---

## Part 9 — Digital twin strategy

Deterministic scenarios exist (fixed seed). Stochastic/Monte-Carlo exists (`evaluate_scenarios.py`, many seeds). **Missing**: fault injection (Part 5I/6), and a single reproducible-experiment record that ties together seed + config + model version + controller version + scenario + metrics + runtime — `ExperimentTracker` covers training runs; extending the same `runs/<id>/record.json` pattern to evaluation and resilience runs is a **MUST HAVE** and is small (the pattern and the code already exist; it just needs applying consistently to every new experiment type this plan adds).

---

## Part 10 — Explainability

**Honest framing, matching the brief's own instruction:** RL is not inherently explainable, and nothing here should pretend otherwise. What *is* available today, and underused: the safety layer already logs a human-readable intervention list (`SafetyPanel`'s `INTERVENTION_LABELS` in the frontend already translates codes like `raised_setpoint_to_cover_critical_load` into plain English) and `RewardBreakdown` already itemizes exactly what drove a step's cost. **MUST HAVE**: surface both of these together per decision — "the controller raised G2's setpoint because committed capacity (125 kW) fell below required capacity (65 kW) plus reserve margin (20 kW)" is a real, derivable sentence from data that already exists, not a generated rationalization. **SHOULD HAVE**: report which controller tier was active (Part 5G) and why (the OOD/confidence signal that triggered a switch), once that mechanism exists.

---

## Part 11 — Observability / dashboard

The existing Command Center already covers power balance, generator states/loading, battery SOC, renewables, safety interventions, fallback activations, fuel/emissions, and (as of this session) a historical trend chart. **Gaps against the brief's list**: predicted demand (blocked on wiring the forecasting module in — Part 5K creates the need), controller-currently-active + AI confidence (blocked on Part 5F/G existing), generator/battery health (the asset-health module exists but isn't in the API/frontend yet — **MUST HAVE**, it's the cheapest win in this entire plan since the backend module, tests, and docs already exist and only need one new API endpoint plus one new frontend panel).

---

## Part 12 — Experimental design

`evaluate_scenarios.py` already provides multi-seed, held-out-seed evaluation with mean/std/min/max/percentiles — the statistical machinery for Part 12 already exists. **MUST HAVE**: apply it consistently to every new controller (MILP, MPC, degradation-aware RL, etc.) on the *same* held-out seed set already used for Part 3's numbers, so every future comparison in this project is apples-to-apples by construction. **Ablation list from the brief, feasibility-checked**: RL-without/with-safety is directly measurable today (Part 3's guarded/unguarded columns already are exactly this ablation). RL-without/with-degradation, thermal-optimization, and uncertainty-handling all require Part 5's B/D/F to be built first — sequence the ablation study after implementation, not before.

---

## Part 13 — Ablation discipline

Applying the brief's own test to this plan's proposals: asset-health wiring into RL (Part 5B) is cheap and directly measurable against Part 3's baseline — keep if it moves fuel or wear-score down without hurting critical-load reliability, cut if not. MILP/MPC baselines (Part J/K) are pure measurement infrastructure — always keep, they cost nothing to the running system. Full five-tier controller switching (Part G) is exactly the kind of "collection of buzzwords" risk the brief warns about — build only the RL→fallback tier first, measure whether the OOD signal ever actually fires in practice, and let that measurement decide whether MPC/MILP tiers are worth adding as *live* fallbacks versus staying pure benchmarks.

---

## Part 14 — Implementation roadmap

| Phase | Objective | Key files | Depends on | Acceptance criteria | Complexity |
|---|---|---|---|---|---|
| 0 | Reproducibility: pin the RL retrain's real before/after numbers (in progress this session), fix the README's PyPSA claim | `README.md` | — | Retrain evaluated on held-out seed 1; README corrected | Low |
| 1 | Wire asset-health into observation + API + dashboard | `envs/polar_microgrid.py`, `api/app.py`, `frontend/src/components/` | Phase 0 | Wear score visible in UI; no regression in existing tests | Low |
| 2 | MILP baseline | new `allotrope/optimize/milp.py` | Phase 0 | Matches plant's cost basis; reported on same seeds as Part 3 | High |
| 3 | Generator/battery degradation in reward+observation | `envs/reward.py`, `envs/polar_microgrid.py` | Phase 1 | New reward term uses existing `RewardWeights` prices, not new arbitrary ones | Medium |
| 4 | Thermal-optimization metric + observation check | `envs/polar_microgrid.py`, `safety/projection.py` (read-only instrumentation) | Phase 0 | `forced_start_to_protect_heating` rate reported per run | Low |
| 5 | Long-horizon reserve-escalation rule | `safety/projection.py` | Phase 0 | New deterministic rule, unit-tested like every existing bound | Medium |
| 6 | Uncertainty: OOD flag + ensemble | new `allotrope/uncertainty/` | Phase 0 | Ensemble disagreement metric reported alongside Part 3 numbers | Medium |
| 7 | Controller switching (RL→fallback tier only) | `safety/fallback.py` | Phases 5, 6 | Switch event logged and explainable (Part 10) | Medium |
| 8 | Fault injection + resilience benchmark | new `allotrope/faults/`, `scripts/run_resilience_benchmark.py` | Phase 0 | 15-scenario suite runs, resilience score reported with components | Medium |
| 9 | Federated: run to comparable step budget, compare fairly | `federated/coordinator.py` (config only, likely no code change) | Phase 0 | Honest comparison table, positive or negative | Low (compute-bound, not code-bound) |
| 10 | Edge hardening: watchdog, checkpoint integrity | new `allotrope/deploy/` | Phase 0 | Crash-and-recover demonstrated in a test | Medium |
| 11 | Dashboard: predicted demand, controller/confidence panel | `frontend/src/components/` | Phases 6, 7, plus Phase 1's asset-health panel | Visually verified via Playwright screenshot, as this session did for PRs #9/#10 | Medium |
| 12 | MPC baseline + final fair benchmark + ablations | new `allotrope/optimize/mpc.py`, `evaluate_scenarios.py` extension | Phases 2, 6 | Full controller table (rule-based/MILP/MPC/DQN/SDDPG/hybrid/hybrid+safety) on held-out seeds | High |

Risks common to every phase: none of this should touch the safety projection's existing, audited bounds without re-running `tests/test_safety.py` and `scripts/run_safety_audit.py` — those are this project's actual guarantee, and every phase above is designed to extend around them, not through them.

---

## Part 15 — Team division (evaluated, not assumed)

The suggested four-way split maps cleanly onto the codebase's real module boundaries, confirmed by this audit:

- **Team 1 — Control & Optimization**: `allotrope/agents/`, `allotrope/envs/reward.py`, new `allotrope/optimize/` (MILP/MPC), controller-switching logic in `safety/fallback.py`.
- **Team 2 — Digital Twin & Physics**: `allotrope/sim/`, `allotrope/synth/`, `allotrope/config/stations/`, new `allotrope/faults/`.
- **Team 3 — Safety, Resilience & Deployment**: `allotrope/safety/`, new `allotrope/uncertainty/`, new `allotrope/deploy/`, `scripts/run_safety_audit.py` and its resilience-benchmark extension.
- **Team 4 — Platform, Data & Demonstration**: `allotrope/api/`, `allotrope/mqtt/`, `allotrope/controlplane/`, `frontend/`, `allotrope/intelligence/` (asset-health, forecasting — already platform-adjacent, not core control), `allotrope/evidence/`, `deploy/`.

**One real interface risk**: `allotrope/envs/polar_microgrid.py` sits between Team 1 (reads it for agent I/O) and Team 2 (owns the plant it wraps) — both teams will want to touch its observation-width logic (Team 1 for new degradation/forecast features, Team 2 for new plant state). **Recommendation**: Team 2 owns `_observation_width()` and `_observe()`; Team 1 only ever *reads* observation indices by name (already how `HybridAgent._encode` works) and requests new fields via an interface contract (a shared markdown doc listing exactly which keys `plant.observe()` promises), never edits the plant or env directly.

---

## Part 16 — Git / Claude Code workflow

This session already established a working pattern worth keeping: **one branch per component, PR-gated by the existing CI (`pytest -v` + frontend build/test), merged only after tests are green.** Extend it for four parallel teams:
- **Branch naming**: `team1/<feature>`, `team2/<feature>`, etc., so CI history and blame stay attributable.
- **Ownership boundaries**: per Part 15's file map — a PR touching another team's owned files needs that team's review, enforced by a lightweight CODEOWNERS file (not built yet — **MUST HAVE**, five minutes of work, prevents exactly the "two agents independently rewrite the same core file" failure mode the brief warns about).
- **Shared interfaces**: `plant.observe()`'s key set, `DispatchCommand`'s fields, and the reward's `RewardWeights` prices are the three contracts every team depends on — freeze their shape behind a version note in `docs/`, and require a cross-team sign-off (not just CI green) to change any of the three.
- **Merge order**: Phase-ordered per Part 14 — a team's PR that depends on an earlier phase should not merge before that phase does, even if CI passes, because CI can't catch "this observation field doesn't exist yet."
- **Experiment artifacts**: `runs/` is already gitignored — keep it that way; share results via committed markdown reports (this document's own convention) with the raw numbers, not raw `.pt` checkpoints, in git.

---

## Part 17 — SIH demonstration strategy

Reuse what already exists and is provably real, in this order, using the Command Center dashboard (already live, already screenshotted working this session):
1. Start the simulation normally (existing `Start` control) — narrate power balance, genset/storage coordination as it happens live.
2. Click through to the safety panel during a moment the projection actually intervenes (real, not staged — the panel already shows real intervention codes).
3. Trigger a fault (once Part 8's fault-injection framework exists) — a generator forced offline mid-demo — and show the safety layer and (once built) controller-switching respond in real time, not via a pre-recorded animation.
4. Show the AI-failure path: kill the inference process, show `GuardedController`'s deterministic fallback take over live (this already works today — no new code needed for this specific step).
5. Close with the fair-benchmark table (Part L) — rule-based vs. MILP vs. RL vs. RL+safety on the same held-out scenario, numbers pulled live from a committed `record.json`, not typed into a slide.

Every step above is either already real or depends only on Phases 2 and 8 from Part 14 — this bounds the demo's dependency on unfinished work to exactly two phases.

---

## Part 18 — Final classification

| Area | Classification |
|---|---|
| Wire asset-health into observation/API/dashboard (Part 5B, 11) | **MUST HAVE** |
| MILP baseline (Part 5J) | **MUST HAVE** |
| Fault injection + resilience benchmark (Part 5I, 6) | **MUST HAVE** |
| Long-horizon reserve-escalation rule (Part 5E) | **MUST HAVE** |
| Stale-sensor safety check (Part 5H) | **MUST HAVE** |
| CODEOWNERS + frozen interface contracts (Part 16) | **MUST HAVE** |
| Explainability from existing intervention/reward-breakdown data (Part 10) | **MUST HAVE** |
| Correct the README's PyPSA claim (Part 1) | **MUST HAVE** (trivial, high trust cost if left) |
| MPC baseline (Part 5K) | **SHOULD HAVE** |
| Battery cycling-cost reward term (Part 5C) | **SHOULD HAVE** |
| OOD flag + ensemble uncertainty (Part 5F) | **SHOULD HAVE** |
| RL→fallback controller switching (Part 5G, tier 1 only) | **SHOULD HAVE** |
| Fair honest federated-vs-local comparison at matched step budget (Part 7) | **SHOULD HAVE** |
| Edge watchdog + checkpoint integrity (Part 8) | **SHOULD HAVE** |
| Capacity-fade battery model affecting plant dynamics (Part 5C) | **NICE TO HAVE** |
| Federated personalization/fine-tuning (Part 7) | **NICE TO HAVE** |
| Full five-tier controller hierarchy including MILP/MPC as live fallbacks (Part 5G) | **NICE TO HAVE**, pending Part 13's ablation evidence |
| Learned/physics RUL estimator for generators | **DO NOT IMPLEMENT** — no data to validate against |
| Bespoke uncertainty architecture before ensemble baseline is measured | **DO NOT IMPLEMENT** |
| Multi-objective RL (MORL) / Pareto-frontier training | **DO NOT IMPLEMENT** — no evidence the reward shape is the bottleneck yet |
| Claiming a federated-learning win before the matched-budget comparison exists | **DO NOT IMPLEMENT** |
| Bespoke model-serving framework for edge inference | **DO NOT IMPLEMENT** — current checkpoint+process approach is already sufficient |

**Prioritized order for a team of four, starting now**: Phase 0 (fix PyPSA claim, land the in-progress retrain's real numbers) → Phase 1 (asset-health wiring, cheapest real win) → Phase 2 (MILP, highest-leverage missing baseline) in parallel with Phase 8 (fault injection, independent of Phase 2) → everything else per Part 14's dependency order.
