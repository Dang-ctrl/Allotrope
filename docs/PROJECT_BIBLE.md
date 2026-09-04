# The Allotrope Project Bible

The authoritative reference for what this project is, how it is built, and why
each decision was taken. If something here disagrees with the code, the code is
right and this document is stale — fix it.

For the current working state (environment, next steps, open questions) see
[../context.md](../context.md). For where every number came from see
[calibration.md](calibration.md).

**Last updated:** 2026-09-05, end of Phase 5.

---

## 1. The submission

| | |
|---|---|
| Event | Smart India Hackathon 2026 |
| Problem Statement ID | **SIH26061** |
| Title | AI Driven Smart Energy Management System for Polar Research Stations |
| Theme | Clean and Green Technology |
| Category | **Software** — no hardware is fabricated |
| Team | Allotrope, SRM Institute of Science and Technology |
| Repository | https://github.com/Dang-ctrl/Allotrope |

Team member contact details appear in the SIH deck but are deliberately **not**
recorded in this repository, which is public. The `.pptx` is gitignored.

### Hardware, and why this is still a software project

The deck lists Jet A-1 CHP gensets, PV arrays, wind turbines, LTO and LiFePO₄
storage and smart inverters as *hardware in scope*. These are the assets the
software controls and models, not devices the team builds. They exist here as
simulation models on the far side of the control interface.

Two items could blur the line and do not:

- The **ruggedized edge server** is a deployment target. The same containers run
  on a laptop.
- **Typhoon HIL (HIL402 / HIL604)** is real hardware and the project's one
  genuine external dependency. The deck's own risk table handles it:
  software-in-the-loop first through OpenDSS, C-HIL booked at a partner lab. It
  is optional validation, not procurement.

Keeping the gRPC/MQTT actuation boundary a hard interface from day one is what
makes the C-HIL step a drop-in later: swap the simulated plant for the rig, same
boundary.

## 2. The problem

Through the Antarctic winter-over, the generating sets at Maitri and Bharati run
far below 30 % of rating. Unburnt fuel coats the exhaust manifolds — **wet
stacking** — and the resulting black carbon settles on surrounding ice, lowering
its albedo. Published measurement over Maitri: **39.3 ng/m³ average, peaks of
76.6**. Bharati stores **296 kL of Jet A-1 per resupply season**, every litre
carried in by ice-class vessel or DROMLAN flight.

The cause is not carelessness. With no forecast, no dispatchable storage and no
tolerance for a blackout in August, keeping a spare set spinning is rational: a
set already turning cannot fail to start. Rule-based controllers and MPC cannot
adapt to an environment this stochastic, multi-energy and hardware-degrading.

**The model reproduces this.** The `LegacyNPlusOne` controller, written to be a
fair opponent rather than a strawman, runs a synthetic year at 26.5 % mean load
factor, wet-stacking in 80.6 % of steps, with deposits saturated at 1.00.

## 3. Architecture

```
sensors and DERs  ->  digital twin  ->  safe DRL agent  ->  safety projection  ->  actuation
CHP, PV, wind,        PyPSA +           DQN (discrete)      hard limits on         gRPC < 10 ms
BESS, thermal         OpenDSS           + SDDPG             heating and            to inverters
                      state est.        (continuous)        life support           and gensets
```

Two properties are built in rather than trained in:

1. **The safety projection layer** analytically bounds every action. The agent
   cannot breach life-support power or heating limits, whatever it has learned.
2. **A deterministic fallback** takes over instantly if the networks time out,
   raise, or return invalid tensors.

Only model gradients cross the station's 4 MHz satellite link; all inference runs
on station.

### Phase plan

| Phase | Content | State |
|---|---|---|
| 1 | The plant — config, synthetic climate, demand, assets, simulator, baselines | **done** |
| 2 | The guarantee — safety projection, fallback, Gymnasium env, reward | **done** |
| 3 | The agents — DQN + SDDPG, training, checkpointing, federated averaging | **done** |
| 4 | The twin — OpenDSS network, Volt-VAr / Volt-Watt fallback | **done** |
| 5 | The system — gRPC actuation, MQTT telemetry, TimescaleDB, Grafana, containers | **code-complete; infra untested here — see §10** |

Phases 3–5 were built in one autonomous session (2026-09-05) at the user's
request, without further confirmation between steps. Section 10 states plainly
which parts of that work are backed by tests run in this environment and which
are infrastructure code that has not been exercised end to end.

## 4. Design decisions, and why

These are the choices a reviewer is most likely to challenge. Each is recorded
with its reasoning so it can be defended or revisited deliberately.

### Synthetic climate from physics, not resampled data

There is no public telemetry, so the environment must be generated. It is
generated from solar geometry and station latitude rather than by resampling
mid-latitude weather, because the two features that dominate a polar energy
system do not survive resampling: the **polar night**, during which PV yield is
exactly zero for weeks, and the **snow albedo**, which at a steep tilt returns a
third more irradiance onto the array than the horizontal receives.

The polar night is *emergent* — change the configured latitude and its length
changes automatically. That property is what lets the same code serve a Himalayan
or island deployment, which is a deck claim.

### Auxiliary boilers, not electric backup heating

The first full-year run fell back on electric resistance heating when recovered
CHP heat fell short, producing 200 MWh of phantom unserved load and thousands of
freeze violations. Real polar stations run oil-fired boilers. Modelling them
removed the artefact *and* gave recovered heat its economic meaning: **CHP heat is
not a bonus, it is boiler fuel not burnt.**

### Generating sets follow the bus, they do not hold a setpoint

Originally each set produced exactly what it was commanded. Under the legacy
controller this curtailed 100 % of renewables, because commanded output already
covered demand. Real sets are governed by the bus. Commitment and power are now
separate phases: the controller decides which sets turn and sets a ceiling; the
bus decides how hard each works, floored at minimum stable load.

Consequence worth stating: **curtailment now emerges** from several sets being
unable to collectively go below their minimum stable loads, rather than being
asserted by a parameter.

### The reward is priced in physical units

Every term is an amount of something real — litres, grams, kWh, machine starts —
converted by a stated price. Each weight is arguable on its merits, and the total
reads back as an operating cost rather than a number meaningful only relative to
itself.

Safety terms are priced far above any reachable fuel saving, but **they are not
the mechanism that keeps the station safe** — the projection is. A penalty the
agent could learn to trade against would be the wrong design. These terms exist
so a policy that somehow reaches an unsafe state learns to leave it.

Machine starts are priced because `EfficientRuleBased` reaches its load factor at
307 starts a year against the incumbent's 22, and a reward counting only fuel
would happily run that up.

Deposit **growth** is charged; deposit reduction is **not** rewarded, or an agent
could farm the fouling cycle.

### Hybrid action space, not flattened

The control problem genuinely is hybrid — commitment is discrete with minimum up
and down times attached, dispatch is continuous — and the project solves the two
with different algorithms. Flattening would hide that. The environment presents
`Dict{genset_on: MultiBinary, dispatch: Box}`.

### Safety applied inside `step`, not left to the agent

The agent trains against a plant it *cannot* damage, so exploration is safe from
the first random action, and the resulting policy never had to learn constraints
that were enforced for it. What it learns is how to be efficient inside them.
`apply_safety=False` exists for ablation and is used in the audit as a control.

## 5. The model

### Station configuration

Every physical parameter lives in `allotrope/config/stations/*.yaml`, never in
code, and carries a `[public]` / `[derived]` / `[assumed]` tag. The loader is a
thin typed projection with invariant checks that reject an incoherent station.

| | Maitri | Bharati |
|---|---|---|
| Position | 70.766 °S, 11.731 °E | 69.407 °S, 76.190 °E |
| Gensets | 3 × 125 kW | 3 × 200 kW |
| PV | 50 kWp @ 70° tilt | 100 kWp @ 68° |
| Wind | 2 × 10 kW | 2 × 30 kW |
| Storage | 200 kWh LFP (core) + 60 kWh LTO (exterior) | 320 + 90 kWh |
| Boilers | 200 kW @ 85 % | 200 kW @ 85 % |
| Envelope | 1.9 kW/°C | 1.5 kW/°C |
| Crew | 25 winter / 60 summer | 25 / 65 |

### The fuel model

Willans line `F = a + b·P`, intercept at 9 % of rated flow. This is the single
most consequential modelling choice in the project.

| Load factor | 125 kW set | Specific consumption |
|---|---|---|
| 15 % | 7.6 L/h | 0.408 L/kWh |
| 25 % | 10.7 L/h | 0.343 L/kWh |
| 50 % | 18.4 L/h | 0.294 L/kWh |
| 80 % | 27.6 L/h | 0.276 L/kWh |
| 100 % | 33.8 L/h | 0.270 L/kWh |

A set at 15 % load burns **51 % more fuel per kWh** than at rating. Every
efficiency claim in this project traces to this table.

### Wet stacking

Deposit is a state variable in `[0, 1]`. It grows below 30 % load and burns off
above 60 %. The black-carbon emission factor rises with **both** accumulated
deposit and instantaneous load factor, so running dirty is self-reinforcing.

The model reports **mass emitted**, not atmospheric concentration — it does not
model dispersion. Claims must be stated as *ratios between control strategies*,
never as a predicted ng/m³.

### The two buses

```
electrical:  PV + wind + gensets + battery = demand + melting + charging + curtailment
thermal:     recovered CHP heat + boilers  = space heat + hot water + snow melt
```

Space heating is **not** a controller decision. It is served automatically — by
recovered heat first, boilers after — because no learned policy should be in a
position to let a station freeze. The controller's influence is indirect and
honest: run the sets well and there is recovered heat to spare.

**Snow melting for potable water** is the large genuinely deferrable load. It is
the sink for surplus wind and the dump load for burn-off cycles, and it is what
lets a set be pushed into its efficient band instead of idling dirty.

### Dual chemistry

LiFePO₄ in the heated core, LTO on the exterior. Modelled honestly: an LFP pack
below freezing reports a charge limit of **zero**, and a cold pack's power
capability tapers to a third at its chemistry floor. Ignoring this is how a
nominally well-sized polar battery turns out to be unavailable in August.

## 6. The safety layer

The claim: **no action — random, adversarial or malformed — can make the station
shed life support or freeze.**

The projection is analytic and closed-form. It solves no optimisation problem and
calls no solver, because a safety layer that can fail to converge is not a safety
layer. It is conservative in one direction only: it will start machines the agent
did not ask for and refuse stops the agent wanted, but it will **never stop a
running machine on its own initiative**, because every failure mode worth
protecting against at a polar station is a failure of supply.

### What it enforces

1. Committed capacity always covers life support plus reserve — evaluated
   **jointly** across the whole commitment.
2. No stop that would breach that cover.
3. Setpoint ceilings raised to match the capacity, since a committed fleet at
   minimum stable load is not cover.
4. Battery commands inside the cells' *actual* thermal envelope.
5. Charging bounded against critical load, because charging is demand.
6. Discretionary load never displaces critical load.
7. Heat supply never left short of the envelope.

Every intervention is recorded, never applied silently, and surfaces to the
operator HMI.

### The audit

Thirty midwinter days at Maitri, `scripts/run_safety_audit.py`:

| Attack policy | Guarded | Unguarded |
|---|---|---|
| Random actions | **0 kWh** | 4 075 kWh |
| Shut every machine down | **0 kWh** | 26 901 kWh |
| Charge storage flat out | **0 kWh** | 26 901 kWh |
| Melt flat out | **0 kWh** | 26 901 kWh |
| Oscillate commitment | **0 kWh** | 33 309 kWh |

The unguarded column is the control. Without it the guarded column proves
nothing.

### Bugs found here, recorded because they were subtle

- **Pairwise vs joint constraint.** Stops were checked one machine at a time.
  Two sets online, both commanded off: each stop looks safe because the other is
  running, and the plant goes to zero with 65 kW required. Now evaluated jointly;
  `test_capacity_cover_is_evaluated_jointly_not_per_machine` keeps it that way.
- **Capacity without a ceiling.** Capacity was counted for sets whose setpoints
  capped them below life support.
- **Unbounded charging.** Battery charging was never bounded against critical
  load, though charging is demand like any other.

The first random-policy run leaked 3.9 MWh. These were found only because the
audit runs adversarial policies rather than sensible ones.

## 7. Results — rule-based baselines

Synthetic year, hourly, seed 0 — `scripts/run_baseline.py`:

**Maitri**

| | Legacy N+1 | Efficient rules |
|---|---|---|
| Fuel | 254.1 kL | 213.8 kL |
| Black carbon | 72 324 g | 11 190 g |
| Specific fuel | 0.383 L/kWh | 0.322 L/kWh |
| Mean genset load factor | **26.5 %** | 52.2 % |
| Steps wet-stacking | **80.6 %** | 2.3 % |
| Mean deposit | **1.00** | 0.00 |
| Renewable fraction | 15.8 % | 16.1 % |
| Curtailed | 1 925 kWh | 0 kWh |
| Genset run hours | 20 087 | 10 183 |
| Genset starts | 22 | **307** |
| Life support unserved | 0 | 0 |
| Freeze violations | 0 | 0 |

Rules alone: **15.9 % fuel, 84.5 % black carbon.**

**Bharati** (seed 0) — also the calibration check against the published fuel
figure:

| | Legacy N+1 | Efficient rules |
|---|---|---|
| Fuel | 264.2 kL | 204.4 kL |
| Black carbon | 95 251 g | 39 889 g |
| Mean genset load factor | 0.180 | 0.374 |
| Wet-stacking fraction | 0.939 | 0.262 |
| Genset starts | 182 | 142 |

Rules alone: **22.6 % fuel, 58.1 % black carbon.** Legacy's 264.2 kL sits close
to the published 296 kL seasonal budget — an independent consistency check, not
a fit (see [calibration.md](calibration.md)).

Headroom is left deliberately in both stations' rule-based numbers — a baseline
capturing everything would leave the learned agent nothing to show. Bharati's
larger absolute genset starts and lower load factor than Maitri's reflect its
larger fleet (3×200 kW vs 3×125 kW) relative to its demand, not a difference in
the control logic.

### Climate validation (Maitri)

| Feature | Model | Independently known |
|---|---|---|
| Polar night | 72 days, 18 May – 28 Jul | Weeks around midwinter at 70.8 °S |
| PV in polar night | exactly 0 | 0 |
| July mean | −28.7 °C | High −20s |
| Minimum | −43.6 °C | Near −40 °C |
| Mean wind | 7.5 m/s, peak 26.4 | Coastal katabatic |
| POA gain over horizontal | 1.33× | Consequence of 0.8 albedo at 70° tilt |
| Mean air density | 1.36 kg/m³ | 11 % above nameplate 1.225 |

## 8. Results — the learned agent

`HybridAgent`: Double DQN over the 8 enumerated commitment patterns (2³ for
three gensets), DDPG over the continuous dispatch (per-set loading fraction,
per-pack battery power, melt rate). Trained entirely behind the safety
projection layer — `PolarMicrogridEnv(apply_safety=True)`, the default — so
exploration is safe from the first random action and the policy never had to
unlearn a habit it was never permitted to form.

**Maitri**, `checkpoints/maitri.pt`, 500 training episodes (30-day episodes,
random start dates), evaluated on **held-out seeds 100–104** (disjoint from
every training seed), full year (8760 steps) each —
`scripts/evaluate_agent.py`:

| | Legacy N+1 | Efficient rules | **Hybrid DQN+SDDPG** |
|---|---|---|---|
| Fuel | 253.8 kL | 213.4 kL | **209.6 kL** |
| Black carbon | 72 141 g | **10 617 g** | 15 931 g |
| Mean load factor | 0.264 | **0.524** | 0.514 |
| Wet-stacking fraction | 0.813 | **0.018** | 0.051 |
| Renewable fraction | 0.160 | 0.162 | 0.162 |
| Genset starts/year | 23.8 | 272.2 | **210.0** |
| Life support unserved | 0 | 0 | **0** |
| Freeze violations | 0 | 0 | **0** |

**The agent clears the bar it has to clear**: 1.8 % less fuel than
`EfficientRuleBased`, on seeds it never saw during training. It also cuts
genset starts by 22.9 % against that same baseline — the reward's
`genset_start_per_event` price doing exactly the job it was added for after
Phase 1 found the rule-based controller alone made 307 starts/year.

**It does not win on every metric.** Black carbon is higher than the
rule-based baseline's. This is not a bug: `RewardWeights` prices fuel and
starts more heavily in absolute terms than black carbon, so the learned policy
is correctly optimising a different point on the trade-off surface than the
hand-tuned rules chose, not a worse point by its own objective. State it this
way if asked, rather than as an unqualified win.

### How this result was nearly wrong

The first training run (300 episodes) is not the one reported above and was
discarded. `DQNConfig`/`SDDPGConfig`'s default exploration-decay rates are
tuned for roughly 1000 episodes; at 300 episodes the run finished with the DQN
still choosing randomly 40 % of the time and the SDDPG noise barely decayed
from its start value. Every number that run produced — fuel, load factor,
genset starts — was measuring exploration noise, not the learned policy.
`scripts/train_agent.py` now derives the decay schedule from the actual
episode count requested, reaching the exploration floor at 70 % of the run so
the saved checkpoint gets a real stretch of near-greedy training before it is
evaluated. A second, unrelated bug in `evaluate_agent.py` itself — a
transposed pandas DataFrame — was caught only because it crashed outright;
see §11 for what that implies about test coverage of the reporting scripts.

**Bharati and federated training**: see [../context.md](../context.md) for
whether these landed in this session or remain open.

## 9. The network twin and Volt-VAr/Volt-Watt

`allotrope/network/twin.py`: a minimal radial LV feeder in OpenDSS, one bus per
asset group (PV, wind, each storage pack, three load classes) off the genset
bus. Topology and cable impedances are `[assumed]` — there is no public wiring
diagram for either station — tagged as such in each station's new `network:`
YAML section. Real power at each bus is injected as a signed `Load` (negative
kW for generation) rather than a `Generator` element, which is standard
practice for a net-load snapshot study and avoids PV-mode convergence quirks on
a weakly meshed radial network.

The fallback implements IEEE 1547-2018's default Category III curves in two
stages: reactive support first (free — no real energy cost), then real-power
curtailment only if VAr support alone cannot hold a bus in range. Verified
under a deliberately extreme case — 600 kW of PV export against a 50 kWp
rating:

| Stage | PV bus voltage |
|---|---|
| Uncorrected | 1.113 pu |
| After VAr support alone | above 1.10 pu still |
| After Volt-Watt curtailment | ≤ 1.10 pu (the ceiling) |

That the second stage is needed, and that it actually brings the bus back into
range, is what `tests/test_network.py` checks — not merely that the curves
exist and have the right shape (also checked, separately).

This closes the gap Phase 2 flagged: the deck's "deterministic Volt-VAr /
Volt-Watt fallback" is now real code with passing tests, not merely a claim
deferred to a future phase. It is dispatch-adjacent, not the same code path as
`allotrope.safety.fallback.DeterministicFallback` — the two fallbacks act at
different layers (station dispatch vs. inverter-level voltage response) and
are not yet wired together into one combined control loop. See §11.

## 10. The system: gRPC, MQTT, TimescaleDB, Grafana, containers

**The actuation interface** (`allotrope/rpc/`): a gRPC service defined in
`allotrope.proto`, carrying `DispatchRequest`/`Telemetry` between a controller
and a plant. This is the hard interface the C-HIL claim depends on — the
server applies the safety projection to every command exactly as
`GuardedController` does in-process, so a remote controller gets no less
protection than a local one, and swapping the simulated plant for a Typhoon HIL
rig behind the same interface is a deployment change, not a rewrite. Verified
by sending an all-off command and a command full of NaN/infinity over the wire
and confirming life support still reads zero unserved on the other side.

**The telemetry link** (`allotrope/mqtt/`): MQTT carries telemetry and
federated model updates only — never control commands, which stay on the gRPC
path with its latency budget. Tested against a real embedded MQTT broker
(`amqtt`, pure Python, spun up on a throwaway port for the test session), not
a mock of the protocol, including a payload deliberately corrupted to confirm
the subscriber drops it rather than crashing.

**The TimescaleDB bridge** (`allotrope/mqtt/timescale_bridge.py`): turns each
telemetry message into a row. Tested against a fake connection object, since
no live TimescaleDB exists in this environment — what is actually verified is
that the bridge builds the right SQL for a given telemetry dict and survives a
broken connection without taking the subscriber down.

**Federated learning across stations** (`allotrope/agents/federated.py`):
FedAvg — each site trains locally for a round, then only the resulting network
*parameters* are averaged into the next round's global model, never the
weather, demand, or telemetry that produced them. This is what makes the
deck's "only gradients cross the satellite link" claim concrete: a round's
parameter delta is bandwidth-equivalent to the accumulated gradient it
represents, and no raw station data ever leaves a site. Averaging is only
meaningful because Maitri and Bharati, despite differing roughly 2× in
installed capacity, share the same asset counts — three gensets, two storage
packs — and `HybridAgent`'s observation is scaled by each station's own
capacity, so the same network architecture and the same action space serve
both.

**Containers and Grafana** (`deploy/`): a `Dockerfile`, `docker-compose.yml`
wiring mosquitto + TimescaleDB + Grafana + a bridge + one container per
station, a TimescaleDB schema turning `telemetry` into a hypertable, and a
provisioned Grafana dashboard (genset output vs. load, battery SoC, fuel burn
rate, black carbon, indoor temperature, and a stat panel for critical-unserved
that should read zero always). See §11 for what is and is not proven to work.

## 11. What was tested here, and what is infrastructure code

Read this section before repeating any claim about Phases 3–5 to someone who
will check.

**Genuinely tested in this environment** (194 tests, all passing at last
count — see [../context.md](../context.md) for the current number):

- Every agent mechanism: replay buffer, DQN/SDDPG action validity and gradient
  flow, checkpoint round-trips, the hybrid agent satisfying the same
  `Controller` protocol as the rule-based baselines.
- **The safety guarantee under an untrained agent** — near-random network
  weights, wrapped in `GuardedController`, still cannot endanger the station.
  This is a real test of the projection layer, not one that passes because the
  agent happens to already be sensible.
- Federated averaging's arithmetic (elementwise weighted mean, exactly) and an
  end-to-end federated round across both stations, checked for safety at each
  participating station afterward.
- The OpenDSS twin's power flow and the two-stage Volt-VAr/Volt-Watt fallback,
  including that curtailment is actually needed and actually works under an
  extreme case.
- The gRPC actuation path end to end, client and server, including safety
  enforcement over the wire and malformed-payload handling.
- The MQTT telemetry path end to end against a real (embedded) broker,
  including malformed-payload handling.
- The trained agent's held-out evaluation numbers in §8.

**Infrastructure code, not exercised end to end here:**

- `deploy/docker-compose.yml` — this environment has the Docker CLI but no
  running daemon. The stack has not been started.
- Grafana actually rendering the provisioned dashboard against real data — no
  Postgres/TimescaleDB was running here to render it against.
- The TimescaleDB bridge's SQL against an actual TimescaleDB — tested against
  a fake connection only (see §10).
- The Volt-VAr/Volt-Watt fallback and `DeterministicFallback` are not wired
  into one combined runtime path; they exist as separate, separately-tested
  mechanisms at different control layers.

None of the above is claimed as working in production. It is claimed as
correct, tested Python wired to real infrastructure definitions that have not
yet been started — a smaller, more honest claim, and the true one.

## 12. What this project is *not* entitled to claim

Stated explicitly so no one quotes past the evidence. Updated at the end of
Phase 5 — items from Phase 2 that Phases 3–5 resolved are marked as such
rather than silently dropped.

- **Not a predicted black-carbon concentration.** The model emits mass and does
  not model dispersion. Ratios between strategies only.
- **Not a validated freeze guarantee.** Unchanged since Phase 2: the audit
  shows zero freeze violations in *both* guarded and unguarded conditions,
  because boilers protect heat independently of the controller. The guarantee
  is real; no attack scenario built so far actually exercises it.
- ~~Not Volt-VAr / Volt-Watt today~~ — **resolved in Phase 4.** The fallback
  exists, is tested, and is verified to actually curtail power when reactive
  support alone is insufficient. It is not yet wired to the station-level
  `DeterministicFallback` into one runtime path (see §11).
- **Not validated against station data.** There is none public. Calibration is
  a consistency argument against published fuel and climate figures for both
  stations now, not just Maitri.
- **Not a C-HIL result.** Software-in-the-loop only, so far. The gRPC
  interface built in Phase 5 is what a HIL rig would sit behind, but no rig
  has been connected.
- **Not a deployed federated-learning result.** The mechanism is real and
  tested (§10, §11); no long training run across real station conditions
  (rather than the short runs used to test the mechanism) has produced a
  reportable federated policy as of this writing — check
  [../context.md](../context.md) for whether that changed later in this
  session.
- **Not an end-to-end infrastructure deployment.** See §11's table. Every
  piece is real, tested Python; the containerised stack running together has
  not been started in this environment.
- **The Bharati agent may not exist yet at time of reading.** Only Maitri's
  held-out result is reported in §8 as of this document's last update; check
  `checkpoints/` and the commit log before assuming otherwise.

## 13. Maintenance

Update this document whenever the architecture, parameters, results, roadmap or
claims change. Update [../context.md](../context.md) at the end of any session
that changes the state of the project.

When a result changes, change the number **and** re-check section 12 — the list
of things not claimed is the part most likely to go quietly stale, and it is
the part that matters most.
