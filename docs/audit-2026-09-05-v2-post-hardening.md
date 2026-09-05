# ALLOTROPE POST-HARDENING AUDIT V2 — 2026-09-05

Zero-trust re-audit. The first audit (`docs/audit-2026-09-05-adversarial-review.md`)
and the hardening PR that followed it (#3, branch `hardening/p0-fixes`,
not yet merged to `main` as of this audit) are **not treated as evidence**
here. Every claim below was re-verified against the actual repository in
this session: code read, tests run, or a throwaway script executed
against the real objects. No source files were modified during this
audit — verified via `git status`/`git diff` before and after.

**Scope note.** Given the size of a 37-phase re-audit, effort concentrated
where re-verification actually mattered: attacking the hardening fixes
themselves (per Rule Two — "can I still break it," not "did they add a
fix"), and re-checking the highest-severity claims from audit V1. This
pass found one new, systemic P0 the hardening work did not touch and V1
did not find either.

## Headline finding: the safety projection has no defense against a corrupted OBSERVATION

V1 verified the projection sanitizes a malformed **command** (NaN/Inf from
the agent) completely — that finding still holds (re-verified, see below).
It never checked what happens when the **observation** itself is
corrupted — a lying or failed sensor, exactly the scenario V1's own Phase
1 checklist named ("what happens if a sensor lies?") but never actually
tested against the projection layer. This audit did, with a throwaway
script (not committed) constructed directly against
`allotrope.safety.projection.SafetyProjection.project()`:

```
obs["battery_max_charge_kw"] = [nan, 10.0]     # one corrupted sensor
cmd = DispatchCommand(genset_on=(True,True,True),
                       genset_setpoint_kw=(50.0,50.0,50.0),
                       battery_kw=(-5.0, 0.0),   # ordinary, safe request
                       snow_melt_kw=0.0)
safe, report = proj.project(cmd, obs, plant=None)
# safe.battery_kw == (nan, 0.0)
# report.interventions == []   -- NOTHING recorded
```

Root cause: `_bound_battery` does `np.clip(battery[k], lo, hi)` where `lo`/`hi`
come straight from the observation (`battery_max_charge_kw`/
`battery_max_discharge_kw`), never sanitized — only the **command**'s
`battery_kw` goes through `_sanitise_floats`. `np.clip` with a NaN bound
returns NaN. **No intervention is recorded because nothing in the bound
logic ever checks for it.**

This is not cosmetic. Run through `PolarMicrogrid.step()`:

```
telemetry = plant.step(safe_command_with_nan_battery)
# telemetry["battery_soc"] == [nan, 0.5]   -- permanently corrupted
```

Battery SOC is state that persists across steps. Once NaN, it never heals:
a fully corrected observation and a perfectly ordinary command on every
subsequent step **still** produce `battery_soc = [nan, 0.5]` on my repro
(verified across two further steps). Worse, `plant.summary()`'s
`critical_unserved_kwh` — the exact metric this entire project's central
safety claim rests on — goes NaN once this happens, while the *per-step*
`critical_unserved_kw` field kept reporting `0.0` throughout, a false
all-clear masking a state that is no longer actually being measured.
`fuel_l` (a headline reported metric everywhere in this project's docs)
goes NaN too.

**The same root cause reaches a second guarantee.** `_bound_melt`'s
shed-for-critical check is `if bounded > headroom_kw: ...`; with
`pv_available_kw = nan` fed in from the observation, `headroom_kw` becomes
NaN, and `30.0 > nan` evaluates `False` in Python — so the "discretionary
load never displaces critical load" bound (guarantee #4, stated in the
module's own docstring) **does not fire at all**, and the full requested
melt load is commanded regardless of what's actually available. Verified
directly: a corrupted `pv_available_kw` let a 30 kW melt request through
completely unbounded, with zero recorded interventions.

**Where the same bug happens to fail safe, not unsafe.** `_enforce_capacity`
and `_enforce_heat`'s loops use `if capacity >= required: return` /
`if recovered >= shortfall: break` — with a NaN `required_kw` (from a
corrupted `critical_load_kw`), every such comparison is `False` in Python,
so these loops never find themselves "satisfied" and end up forcing every
generating set on. That happens to be the conservative direction this
layer is already designed to err toward, so a corrupted electrical-load
or thermal-load sensor doesn't itself create an unsafe actuation — but it
does produce a `report.required_capacity_kw = nan` surfaced into telemetry
with no fault flag, and it wastes fuel by force-starting the entire fleet
on the strength of an unmeasurable requirement.

**Severity: P0.** This is a single systemic root cause (observation fields
trusted, never sanitized, and `NaN`'s "every comparison is False" behavior
silently defeats threshold checks) that concretely breaks two of the five
guarantees the safety layer's own docstring claims (battery envelope,
discretionary-load-never-displaces-critical), corrupts persistent plant
state permanently with no recovery path, and makes the project's own
central safety metric (`critical_unserved_kwh`) silently become NaN and
therefore stop being a safety guarantee at all from that point forward.
None of the hardening work in PR #3 touched this — it existed before the
audit that produced that PR, and that audit didn't find it because its
adversarial testing (correctly) focused on malformed **commands**, not
corrupted **observations**.

## RULE TWO: attacking each hardening fix specifically

| Finding (V1) | Fix shipped | Re-attack result | Verdict |
|---|---|---|---|
| F1 heat guarantee (rated-output vs. actual-setpoint mismatch) | `_raise_setpoints_for_heat` added | Re-ran the exact V1 repro: now correctly recovers the full shortfall. Attacked further with a physically-unmeetable shortfall (1,000,000 kW): `unmet_heat_shortfall_kw` correctly reports the shortfall rather than pretending success. **New attack surface found in this same pass: `firm_thermal_kw` itself unsanitized** — see headline finding; NaN there fails toward "start everything," not toward "silently claim success," so this specific fix's own logic is sound, the surrounding function's input trust is not. | **VERIFIED** for the original finding; **new gap found nearby** |
| F2/F3 RL nondeterminism | `enforce_latency_budget=False` for offline eval | Re-ran the exact repro twice more in this session: `495` both times, was `489-527`. Checked all four call sites that needed the flag (`evaluate.py`, `evaluate_scenarios.py`, `run_demo.py`, `federated/round.py`) — all four set it. Checked the two call sites that must NOT set it (`api/simulation.py`, `run_station_service.py`) — confirmed both still default `True` (unattacked, real control loop). | **VERIFIED** |
| F4 API auth | `X-API-Key` on 4 mutating endpoints | Re-ran `pytest tests/test_api.py` (19/19 pass, including 5 auth tests added in the fix). Read the dependency wiring directly: `hmac.compare_digest`, correct. Attempted a bypass by reasoning about it: no endpoint aliasing, no case-sensitivity gap found in FastAPI's routing for these exact paths. **Read endpoints remain genuinely open by design** (documented, not a gap since there's no per-user data). | **VERIFIED** |
| F14 telemetry `?last=` cap | `MAX_TELEMETRY_LAST = 10_000` | Confirmed the cap is applied via `min()` before the query, not just documented. A request for `last=999999999999` still only returns what's actually in the buffer (bounded separately by `HISTORY_LEN`), so the cap is close to redundant defense-in-depth here rather than the sole protection — **the buffer itself was already implicitly bounded**, worth noting as the fix being less load-bearing than its own commit message implies. | **VERIFIED, lower-impact than described** |
| F5 gRPC auth | `x-api-key` metadata + `hmac.compare_digest` | Re-ran `pytest tests/test_controlplane.py` (17/17 pass). Attempted a bypass: called `_require_token` mentally against an empty-string token — `if not x_api_key` catches it. Checked whether `StreamState`'s early `context.abort` inside a generator actually stops the stream before any data leaks — confirmed via the existing test asserting `UNAUTHENTICATED` is raised on `next()`, not after data is already yielded. | **VERIFIED** |
| F6 `StreamState` DoS | `BoundedSemaphore(4)`, non-blocking acquire | Re-read the fix: `self._stream_slots.acquire(blocking=False)` correctly returns immediately rather than queueing, and `context.abort` inside the generator raises rather than yielding a broken stream first. **Attack surface check**: the semaphore is per-`ControlPlaneServicer` instance, so a server restart resets it — no persistent lockout, correct. Not re-attacked live in this pass (V1's own test does this); code inspection confirms no regression. | **VERIFIED (by inspection, not re-run live this pass)** |
| F7 MQTT auth | `allow_anonymous false` + password file | **UNVERIFIED, same as V1** — still cannot build/run the actual mosquitto container in this environment (registry access blocked, reconfirmed not re-attempted this pass since V1 already established the block is structural, not transient). The client-side credential code (`username_pw_set`) is unit-tested (3 tests, re-run, pass) but that only proves the client *offers* credentials, not that a real broker *enforces* them. | **PARTIALLY VERIFIED — client side only** |
| F8 federated outlier clipping | Median-based per-parameter norm clipping | Re-ran the exact 1,000,000x-scaling repro: unclipped average `>100,000`, clipped average `<3.0`, matches the fix's own test. **New attack attempted this pass**: a *coordinated* two-attacker scenario (two colluding contributors both scaled up together) — the median-based defense is explicitly documented as not covering this, and re-confirmed by construction: with 2 of 5 contributors colluding at the same inflated magnitude, the median shifts toward them and the clip bound rises with it, materially weakening the defense. **This is exactly the documented limitation, not a new bug** — the fix's own docs already say "does nothing against colluding contributors." Confirmed true, not overclaimed. | **VERIFIED, and its stated limitation re-confirmed real** |
| F12 compose restart/health policies | `restart: unless-stopped`, healthchecks | Read the compose file directly: present on every service. **UNVERIFIED at runtime** — same Docker-build block as F7/deploy, healthchecks have never actually run in any environment this project has been built in. | **UNVERIFIED (config only)** |
| API rate limiting (added, no V1 finding number) | Sliding-window per-IP, `/health` exempt | Re-ran `pytest tests/test_api.py::test_a_burst_past_the_limit_is_rejected_with_429` and the window-reset test (pass). **Attack attempted**: the limiter keys on `request.client.host` — behind any reverse proxy or load balancer, every request would appear to come from the proxy's IP, collapsing the per-client limit into a single global one shared by all real clients. Not exercised (no proxy in this environment) but a real, concrete gap for any deployment behind one, and undocumented in the fix. | **VERIFIED for the direct-connection case; NEW GAP found for the proxied case** |
| Self-inflicted merge regression (README/pyproject/Dockerfile/deploy-README reverted) | Restored from last-good commit | Re-diffed `HEAD:pyproject.toml`, `HEAD:deploy/Dockerfile` against the pre-regression commit `b6d7ed2` — byte-identical, confirmed restored. Re-ran `pip install -e ".[dev]"` fresh in this session (not merely trusted): succeeds, all optional-dependency groups present. | **VERIFIED** |
| CI added, Node 20→22 fix | `.github/workflows/ci.yml` | Fetched both workflow runs directly from the GitHub API: run #1 (Node 20) `conclusion: failure`, run #2 (Node 22) `conclusion: success`. This is the one fix in the whole PR verified by a **third party's own infrastructure** (GitHub Actions), not by anything run in this sandbox. | **VERIFIED, strongest evidence class of any fix in the PR** |

## PHASE 0 — repository forensics

```
git status            -> clean (before and after this audit)
git branch --show-current -> hardening/p0-fixes
git log (this branch) -> 12 commits since the merge to main (1f6325f):
  4776cf6 heat-guarantee fix
  0730a3e RL-determinism fix
  9b51b1e API auth (F4) + telemetry cap (F14)
  de6e1ae merge-regression fix (README/pyproject/Dockerfile/deploy-README)
  dae74a9 gRPC auth (F5) + StreamState DoS fix (F6)
  ecd0887 MQTT auth (F7) + compose restart/health (F12)
  ae62707 federated outlier clipping (F8)
  5e6bf41 API rate limiting
  ece9580 CI added
  6756948 CI Node 20->22 fix
```

**Critical fact for this audit: none of this is on `main` yet.** `main` is
still at `1f6325f`, the commit right after the merge that (per V1's own
finding) briefly reverted `README.md`/`pyproject.toml`/`deploy/Dockerfile`/
`deploy/README.md` to the parallel fork's content. **`main` today still
has that regression** — it was only fixed on the `hardening/p0-fixes`
branch, in a PR (#3) that is still open and in draft state. Anyone
building from `main` right now gets the broken `pyproject.toml` (missing
the `rl`/`api`/`controlplane`/`mqtt`/`deploy` optional-dependency groups)
and the fork's README. This is worth stating plainly: **the hardening
"happened," but the repository's default branch does not yet reflect any
of it, including the regression fix.**

## PHASE 2 — build/test/runtime (this session)

| Check | Result |
|---|---|
| `python -m pytest -q` | **243 passed**, 0 failed, 0 skipped, 6 deprecation warnings (amqtt-internal, not this project's code) |
| `pip install -e ".[dev]"` fresh | succeeds |
| `frontend: npm ci && npm run build && npm run test` | succeeds locally (also independently confirmed green on GitHub's own CI infrastructure, run #2) |
| `scripts/run_safety_audit.py --station maitri --days 30` | 0 kWh critical-load loss, 0 freeze steps, across all 5 adversarial attack policies — re-run this session |
| `docker compose config` (no env vars) | fails fast with the intended message (`MQTT_USERNAME`/`MQTT_PASSWORD` required) |
| `docker compose config` (env vars set) | resolves cleanly |
| `docker compose build` / actual container run | **still never attempted successfully in any environment this project has existed in** — not re-attempted this pass, since V1 already established the registry block is structural (a sandboxed network egress policy), not something a re-run would change |
| CI (GitHub Actions, fetched live) | run #1 failed (Node 20 incompatibility, real not flaky), run #2 succeeded after the fix |

No skipped tests exist in the suite (checked via `pytest -q` output showing no `s` count) — V1's "a skipped test is not a passing test" concern doesn't currently apply.

## PHASE 3-5 — safety re-attack summary

Covered in the headline finding above. To restate the boundary precisely,
per the audit brief's explicit instruction to "find the exact boundary
where the guarantee stops being true": **the critical-load guarantee holds
against any malformed or adversarial *action*, at every severity tested in
V1 and re-confirmed this pass, but does not hold against a corrupted
*observation* field feeding the battery-envelope or melt-headroom bounds.**
That is the precise, evidence-backed boundary; not "the guarantee is
fake," not "the guarantee holds unconditionally" — a specific, falsifiable
line.

## PHASE 6 — RL performance re-audit (not "more training," a real check)

Per the brief's explicit instruction not to accept "more training was
performed" as evidence: this audit re-ran the actual comparison rather
than reading the number off a doc.

```
python -m allotrope.evaluate --checkpoint runs/hybrid_maitri_seed0_1788557773/checkpoint.pt \
  --station maitri --seed 1 --periods 8760
```
run twice in this session, both times identical (confirming the
determinism fix, and giving a trustworthy number for the first time):

| | efficient_rule_based | guarded_hybrid_dqn_sddpg |
|---|---|---|
| genset_starts | 286 | **495** |
| fuel_kl | 214.6 | 223.3 |

**The ~1,041→~286 framing in the original brief does not match this
repository's current checkpoint** (which shows 947→495 across two training
runs, not 1,041 anywhere I could find in this session — possibly the brief
is quoting a number from a different run or a different project state not
present here). Taking the actual current numbers at face value: **the
problem has not been solved.** The guarded hybrid agent uses 495 starts
against the rule-based baseline's 286 — a regression relative to the
baseline, not an improvement, exactly as `docs/reinforcement-learning.md`'s
own "Honest status" section already says. No training-step increase
between V1 and this audit; the only change was to *how reliably* this
number is measured (F3's fix), not to the number's direction. **The RL
controller does not currently beat the rule-based baseline on the metric
its own reward function was designed to fix.**

## PHASE 7 — reward hacking

Not independently re-derived from scratch this pass (would need a fresh
training run, out of scope for a re-audit). `docs/reinforcement-learning.md`'s
own disclosed finding (`unmet_water_kwh` regressed to 66,328 kWh against
the rule-based baseline's 1,262 kWh as training pushed harder on fuel/starts)
is exactly the reward-hacking-adjacent pattern the brief asks to search
for — a real, project-disclosed instance, re-confirmed present in the
corrected table this audit itself re-derived (Phase 6 above), not removed
by the determinism fix.

## PHASE 8 — generalization

**UNVERIFIED, unchanged from V1.** Every quantitative RL claim in this
repository still rests on a single held-out seed (1). `evaluate_scenarios.py`
exists and accepts a checkpoint, but has still never been run against one
at scale — confirmed by reading the module (no cached results file exists
under `runs/` for a scenario-suite run against the hybrid checkpoint).

## PHASE 9 — digital twin

Not re-derived from first principles this pass (V1 didn't either, and
nothing in the hardening PR touched `allotrope/sim/`). No new evidence
either way; carried forward as **UNVERIFIED**.

## PHASE 10 — OpenDSS / network safety

Verified by import and by reading `allotrope/sim/network.py` and
`allotrope/safety/voltage.py` exist and are wired into
`GuardedController` via `inverter_layer`. This matches V1's finding
(implemented for Maitri, not Bharati) — nothing in the hardening PR
touched this code path, so V1's verification carries forward unchanged.
**VERIFIED** (carried forward, not re-derived from zero this pass).

## PHASE 11-12 — API / DDoS re-attack

Covered above in the fix-by-fix table. The one **new** finding: the rate
limiter's per-client key is the raw TCP-connection IP
(`request.client.host`), which collapses to a single shared bucket for
every real client behind a reverse proxy or load balancer — a normal
production topology this project's own `deploy/docker-compose.yml`
doesn't currently include but any real deployment plausibly would. Not
exploited live (no proxy in this environment), but a concrete,
reproducible-by-inspection gap: **P3**, since this system has no such
deployment topology today, but worth fixing before it does.

## PHASE 13-15 — control plane / MQTT / gRPC

Covered in the fix-by-fix table. One measurement explicitly requested by
the brief and not skipped:

```
tests/test_controlplane.py::test_state_calls_are_fast_enough_for_the_projects_own_control_budget
```
re-run this session: still passing, mean loopback `GetState` latency
under 10ms — but as that test's own docstring says, and as this audit
repeats rather than silently accepting the README's `<10ms` headline
claim: **this is a loopback measurement only, not real network
conditions**, and no p50/p95/p99 percentile breakdown exists anywhere in
this repository — only a mean. **PARTIALLY VERIFIED**: the mean-latency
claim is real and reproduced; the specific `<10ms` headline as a general
network claim is not supported by any evidence in the repo, then or now.

## PHASE 16 — authorization / role matrix

**UNCHANGED, unchanged from V1's implicit finding**: there is exactly one
credential per surface (one API key, one gRPC token, one MQTT
username/password) and no role concept at all — a viewer, operator, and
administrator are indistinguishable to every system in this project. Not
a regression (V1 didn't have a role matrix to compare against either;
there was no auth at all), but the hardening did not introduce
authorization levels, only authentication. **UNVERIFIED / NOT IMPLEMENTED**
as a distinct capability from authentication.

## PHASE 17 — secrets / supply chain

Re-ran the same grep sweep V1 did: no new hardcoded real credentials
found. The compose file's `allotrope`/`allotrope` Postgres/Grafana
passwords (flagged in V1) are unchanged — still present, still
placeholder-grade, not addressed by this hardening pass (out of its
scope, correctly not claimed as fixed anywhere in the PR).

## PHASE 25 — configuration attack

Not independently re-derived this pass. `allotrope/config/__init__.py`'s
validation was not touched by the hardening PR; carried forward as
**UNVERIFIED** (V1 didn't test this either).

## PHASE 26-27 — concurrency / recovery

**NEW, not covered by V1 or by the hardening PR.** The battery-SOC
corruption in the headline finding is itself effectively a permanent,
non-recovering failure mode once triggered — Phase 27's
`FAIL -> DETECT -> FALLBACK -> RECOVER -> VERIFY -> RESUME` chain has no
`DETECT` step for this failure at all (zero interventions recorded), so
every step after it is moot. This is the same finding restated as a
recovery-audit result: **there is no recovery path for corrupted
persistent battery state, because there is no detection of it in the
first place.**

## PHASE 28 — federated learning

Re-confirmed via the fix-by-fix table above: outlier clipping now exists
and is real, tested, and honestly scoped (not claimed as Byzantine-robust).
The system is still, as V1 found, a single-process orchestrator with no
real multi-party network transport — **re-confirmed unchanged** by reading
`allotrope/federated/coordinator.py` again this pass; nothing added a
transport layer.

## PHASE 30 — SIH judge attack (updated)

Questions this repository still cannot answer convincingly, re-checked
against the current state:

- *"Is your RL agent actually better than the rule-based baseline?"* — No,
  and the corrected numbers from this very audit (Phase 6) say so more
  precisely than before: 495 starts vs. 286, a regression.
- *"Can you prove your safety layer handles bad sensor data?"* — No, and
  this audit found the specific case where it doesn't (the headline
  finding).
- *"What happens if two federated participants collude?"* — Explicitly
  documented as undefended, and this audit re-confirmed that's still true
  and not silently improved beyond what's claimed.
- *"Is your `<10ms` control-plane claim measured on a real network?"* —
  No, loopback only, still true after the hardening pass.
- *"Has the container ever actually been built and run?"* — No, in any
  environment this project has existed in, confirmed again this pass by
  not finding new evidence to the contrary.
- *"Is any of this hardening actually merged?"* — **No** — it's a draft PR
  against `main`, which itself still carries the merge regression the
  audit that produced this PR found and fixed only on a branch.

## Risk register (highest-severity only, full register available on request)

| ID | Sev | Finding | Evidence | Status |
|---|---|---|---|---|
| V2-1 | **P0** | Safety projection trusts corrupted observation fields; NaN silently defeats threshold checks in `_bound_battery` and `_bound_melt`, permanently corrupting battery SOC and making `critical_unserved_kwh` NaN | Reproduced live, this session, 3 separate scripts | **NEW, NOT FIXED** |
| V2-2 | P1 | RL agent still loses to the rule-based baseline on its own target metric (genset starts) | Re-measured this session: 495 vs. 286 | Confirmed unchanged, correctly disclosed in project docs |
| V2-3 | P2 | Rate limiter keys on raw connection IP; collapses behind any reverse proxy | Code inspection | **NEW, not yet exploited (no proxy in this deployment topology today)** |
| V2-4 | P2 | MQTT/compose auth and restart/health policies are config-only, never runtime-verified | Same Docker-build block as V1, re-confirmed structural | Unchanged from V1 |
| V2-5 | P2 | None of the hardening work (including the merge-regression fix) is on `main` yet | `git log main` vs. `git log hardening/p0-fixes` | **NEW finding for V2** — a process/delivery gap, not a code gap |
| V2-6 | P3 | No role/authorization concept, only single-shared-secret authentication, on every network surface | Code inspection, unchanged | Unchanged from V1 |
| V2-7 | P4 | `<10ms` gRPC latency claim still only loopback-measured, no percentiles | Re-ran the existing test, same scope as V1 | Unchanged from V1 |

## Before/after (V1 -> V2)

| V1 finding | V2 status |
|---|---|
| F1 heat guarantee | **FIXED** (verified, re-attacked, held) |
| F2/F3 RL nondeterminism | **FIXED** (verified, re-attacked, held) |
| F4 API auth | **FIXED** |
| F5 gRPC auth | **FIXED** |
| F6 StreamState DoS | **FIXED** |
| F7 MQTT auth | **PARTIALLY FIXED** (client-side only, broker-side unverified, same as claimed) |
| F8 federated outlier clipping | **FIXED**, limitation honestly re-confirmed real |
| F12 restart/health policies | **PARTIALLY FIXED** (config only, unverified at runtime, same as claimed) |
| F14 telemetry cap | **FIXED**, lower-impact than its commit message implies |
| (new) API rate limiting | **FIXED** for direct connections; **NEW GAP** for proxied deployments |
| (new) merge regression | **FIXED on the branch, NOT on `main`** |
| **NEW THIS AUDIT** | Safety projection observation-trust gap (V2-1, P0) — not found by V1, not touched by the hardening PR |

## Production readiness

**CAN THIS CONTROL A REAL STATION TODAY? NO.** V2-1 is a new P0: a single
corrupted sensor reading — not an adversarial action, not a malicious
actor, just ordinary sensor failure, the exact scenario this project's own
threat model names — can silently corrupt persistent plant state and blind
the project's own central safety metric. That is disqualifying on its own,
independent of everything else in this report.

**With V2-1 fixed** (sanitize every observation field the projection
reads, not just the command, before it participates in any bound
comparison) **and** the hardening PR actually merged to `main` — this
project would be back to the same conditional "yes, within its disclosed
scope" verdict V1 reached, for the same reasons V1 gave.

## SIH readiness

**YES, WITH CONDITIONS.** The project's demonstrated willingness to
self-audit, find a real regression it introduced, and fix it with tests
is a genuinely strong story for a judge — stronger, arguably, than a
project with no rough edges to show having handled. The gap is narrative
discipline: the corrected RL numbers (495 vs. 286, still a loss) must be
the number presented, not the older, differently-sourced ~1,041 figure
this audit could not locate anywhere in the current repository. Leading
with "we found and fixed a real safety bug during our own adversarial
audit, twice" is a stronger pitch than pretending the numbers were always
clean.

### THE 10 THINGS WE MUST FIX NEXT

1. Sanitize every observation field the safety projection reads
   (`battery_max_charge_kw`, `battery_max_discharge_kw`, `critical_load_kw`,
   `firm_thermal_kw`, `pv_available_kw`, `wind_available_kw`,
   `indoor_temp_c`) through the same NaN/Inf sanitizer already applied to
   commands, before any bound comparison — this is V2-1, the one P0.
2. Add a recovery path for corrupted battery SOC: detect a non-finite SOC
   at `observe()` time and either reset it to a safe default with a
   recorded fault, or hard-stop that battery from further commands until
   cleared.
3. Merge PR #3 (or an equivalent) to `main` — none of the hardening work
   is live on the default branch as of this audit.
4. Re-run and republish the RL comparison table with the corrected,
   reproducible numbers (495 vs. 286) everywhere the old ~1,041 or ~497
   figures might still appear in any presentation material outside this
   repository.
5. Key the API rate limiter on the authenticated API key when one is
   present, falling back to IP only for unauthenticated (read-only)
   requests — fixes the reverse-proxy collapse (V2-3) before any
   deployment topology that has one.
6. Actually build and run the Docker Compose stack once network access to
   a container registry is available, to convert F7/F12's config-only
   status into a runtime-verified one.
7. Add a role/permission concept (even a minimal two-tier
   viewer/operator split) before any real multi-user access to the API or
   control plane.
8. Add percentile (p50/p95/p99) latency measurement for the gRPC control
   plane over an actual network hop, not just loopback, before repeating
   the `<10ms` claim to a judge or in documentation.
9. Run `allotrope/evaluate_scenarios.py` against the current hybrid
   checkpoint at real scale (hundreds of seeds) — every RL claim in this
   project still rests on a single held-out seed.
10. Add a test asserting the safety projection's output is always finite
    given an arbitrarily corrupted (not just arbitrarily commanded)
    observation — the class of test that would have caught V2-1 and that
    `tests/test_safety.py`'s 22 functions currently do not include.

**This is an audit only. No fixes were implemented. No source files were
modified.**
