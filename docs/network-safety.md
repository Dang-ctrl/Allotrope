# Network twin and inverter-level safety

`allotrope.sim.plant.PolarMicrogrid` is a power-balance model: kW in, kW
out, no bus, no line, no volt. That's sufficient for
`allotrope.safety.projection.SafetyProjection`'s guarantees (capacity,
reserve, setpoints, battery envelope, discretionary load), but there is one
failure mode it structurally cannot see: inverter-driven **overvoltage**
when renewable export outruns local load on the feeder. This is the piece
that fills that gap — and, stated as plainly as every other honest-status
section in this project, the piece of it that does **not** exist yet.

## What's implemented

- **`allotrope.sim.network.DistributionNetwork`** — a single-feeder radial
  LV network, solved with `opendssdirect.py` (already a project dependency,
  previously unused). Given each non-source bus's net real-power injection,
  it returns real per-unit bus voltages from an actual OpenDSS power flow —
  not a lookup table, not a linearised approximation. `tests/test_network.py`
  checks the physics: renewable export raises the exporting bus's voltage,
  heavy load depresses the loaded bus's, the source bus holds its 1.0 pu
  reference regardless, and a solve is fast enough (well under 50 ms mean,
  rebuilding the circuit from scratch every call) to run inside a per-step
  control loop.
- **`allotrope.safety.voltage.VoltWattCurve`** — the IEEE 1547-2018 default
  Category I/II Volt-Watt curve: full output below 1.06 pu, linearly
  curtailed to a 0.2 floor by 1.10 pu. A pure function, property-tested
  (`tests/test_voltage_safety.py`) for the two things that actually matter —
  output stays bounded in `[p2_frac, 1.0]`, and it's monotonic non-increasing
  in voltage — across 100 random voltages, not a handful of hand-picked ones.
- **`allotrope.safety.voltage.InverterVoltageLayer`** — solves one voltage
  snapshot per step from the plant's current renewable availability and
  electrical load, and applies the curve to the renewables bus. Composed
  into `GuardedController` as an **optional** constructor argument
  (`inverter_layer=`); a controller built without one — every existing
  caller in this codebase before this change — behaves exactly as before.
  `allotrope.api.simulation.default_controller` wires one in automatically
  for any station whose config declares a `network:` section.
- **Maitri has a network config; Bharati does not.** This is the honest
  state of a synthetic, [assumed] single-feeder layout (see the `network:`
  section of `allotrope/config/stations/maitri.yaml` — cable lengths and
  ratings are notional, chosen to produce a plausible small-feeder voltage
  range under stress, not measured). Extending it to Bharati is copying
  that section's shape with Bharati's own plausible numbers, not new code.
- **Curtailing renewables can never reopen the critical-load guarantee.**
  `SafetyProjection` already excludes renewables from its own capacity
  requirement by design ("committing against the wind is exactly the
  mistake that leaves a station dark when the wind drops" —
  `allotrope/safety/projection.py`). The inverter layer runs strictly
  *after* the projection for exactly this reason: whatever it curtails was
  never counted toward keeping life support covered, so curtailing it
  further can only ever increase reliance on the already-guaranteed firm
  capacity, never reduce it. `tests/test_voltage_safety.py`'s
  `test_guarded_controller_curtails_renewables_without_touching_critical_load`
  runs both layers together over real weather and checks
  `critical_unserved_kw == 0.0` on every step regardless of whether
  curtailment fired that step.

## What's explicitly not implemented — Volt-VAr

**Volt-VAr does not exist in this codebase, and building a curve for it
without the model behind it would be exactly the kind of claim this
project's own rules forbid.** IEEE 1547-2018's inverter response is really
two curves: Volt-Watt (curtail real power) and Volt-VAr (inject or absorb
reactive power). `allotrope.sim.plant` has no reactive-power balance
anywhere — every load, generator, and battery in this project is modelled
in kW only, with no kVAr, no power factor, no Q. A `VoltVarCurve` function
could be written in an afternoon; it would have nothing real to act on,
because the plant it would feed into cannot represent what it changes.
Closing this gap means adding a reactive-power balance to the plant itself
— genuinely new physics in `allotrope.sim`, not an extension of this
module — which is future work, not something this pass claims to have done
by only writing half of it.

## Try it

```bash
python -c "
from allotrope.config import load_station
from allotrope.sim.network import DistributionNetwork

cfg = load_station('maitri')
net = DistributionNetwork(cfg.network)
print(net.solve({'renewables': 250.0, 'load': -60.0}).bus_voltage_pu)
"
```

Or watch it live: `GET /stations/maitri/safety` (see
[docs/api.md](api.md)) returns a `voltage` field with the current bus
voltages and whether curtailment is active — `null` for Bharati, which has
no network model.
