# Calibration

There is no public operational telemetry from Maitri or Bharati. Every number in
this simulator is therefore either a published figure, something derived from
one, or an engineering assumption — and this document says which is which. The
station YAML files carry the same `[public]` / `[derived]` / `[assumed]` tags
inline, so a reviewer can audit any single parameter without reading code.

This matters more than it might seem. The project's headline claims are all
differences between two runs of this simulator. A reviewer is entitled to ask
whether those differences are physics or artefacts of a tuned parameter, and the
only honest answer is a model whose assumptions are visible and whose
qualitative behaviour is checked against things that are known independently.

## The anchor

The one hard public number about this energy system is Bharati's stored fuel:

> **296 kL of Jet A-1 per resupply season.**

Everything else is calibrated so that the incumbent control strategy reproduces
it. That single constraint pins down the plausible range of station load, since
fuel, efficiency and demand cannot be chosen independently:

| Quantity | Value | Source |
|---|---|---|
| Jet A-1 lower heating value | 34.7 MJ/L = 9.64 kWh/L | published |
| Season fuel energy | 296 kL × 9.64 = **2 853 MWh** | derived |
| Generating-set electrical efficiency at part load | ~26 % (see below) | derived from the Willans line |
| Implied electrical output | ~740 MWh/yr | derived |
| Implied mean station load | **~85 kW** | derived |

A mean load near 85 kW for a station wintering ~25 people, rising past 200 kW in
the summer campaign, is consistent with a plant of three sets in the 125–200 kW
class. That is the sizing the configurations use. The agreement is not a fit —
nothing was regressed — it is a consistency check that the assumed demand,
assumed efficiency and published fuel figure can all be true at once.

Running the incumbent controller for a synthetic year at Maitri produces
**254 kL**, against a 296 kL budget for the larger station. That the two land in
the same range from independent assumptions is the strongest validation
available without station data.

## The fuel model

Generating-set fuel flow follows a Willans line, `F = a + b·P`, with the
intercept `a` set to 9 % of the rated flow. This is the single most consequential
modelling choice in the project, because the intercept is what makes part-load
operation expensive:

| Load factor | Fuel flow (125 kW set) | Specific consumption |
|---|---|---|
| 15 % | 7.6 L/h | 0.408 L/kWh |
| 25 % | 10.7 L/h | 0.343 L/kWh |
| 50 % | 18.4 L/h | 0.294 L/kWh |
| 80 % | 27.6 L/h | 0.276 L/kWh |
| 100 % | 33.8 L/h | 0.270 L/kWh |

A set held at 15 % load burns **51 % more fuel per kWh** than the same set at its
rating. No efficiency claim in this project comes from anywhere else.

## Wet stacking and black carbon

Deposit accumulation is modelled as a state variable in `[0, 1]`: it grows while
a set runs below 30 % load, and burns off above 60 %. The emission factor rises
with both the accumulated deposit and the instantaneous load factor, so running
dirty is self-reinforcing — light load raises emissions directly *and* lays down
the deposits that raise them further.

The published measurement this is aimed at is black carbon over Maitri averaging
**39.3 ng/m³ with peaks of 76.6**. The simulator does not model atmospheric
dispersion and so cannot predict a concentration; it reports mass emitted, and
the relevant claim is the *ratio* between control strategies, not an absolute
concentration. Any figure quoted from this model should be stated that way.

## What is assumed, and how much it matters

| Assumption | Value | Sensitivity |
|---|---|---|
| Envelope heat-loss coefficient | 1.9 kW/°C (Maitri) | High — sized so recovered CHP heat covers most space heat, which is what makes CHP dispatch worth anything |
| Boiler efficiency | 85 % | Moderate — sets the value of each kW of recovered heat |
| Idle fuel fraction | 9 % of rated flow | **Highest** — directly scales every efficiency claim |
| Deposit accumulation rate | 0.055 /h at zero load | Moderate — sets the timescale of burn-off cycles, not their existence |
| Crew size | 25 winter / 60 summer | High — drives demand almost linearly |
| Snow-melt energy | 0.115 kWh/L | Low — a well-constrained thermodynamic quantity |

The two assumptions worth challenging first are the idle fuel fraction and the
envelope heat-loss coefficient. Both are stated in YAML and both can be swept
without touching code.

## The climate model

Weather is generated from physics and station latitude rather than resampled
from mid-latitude data, because the two features that dominate a polar energy
system do not survive resampling. Generated for Maitri (70.77 °S):

| Feature | Model output | Independently known |
|---|---|---|
| Polar night | 72 days, 18 May – 28 Jul | Sun below the horizon from roughly late May to late July at this latitude |
| PV yield in polar night | exactly 0 | 0 |
| Midwinter mean temperature | −28.7 °C (July) | Maitri winter means in the high −20s |
| Minimum temperature | −43.6 °C | Extremes near −40 °C |
| Mean wind speed | 7.5 m/s, peaks 26 m/s | Coastal katabatic regime |
| Plane-of-array gain over horizontal | 1.33× | Consequence of 0.8 snow albedo at 70° tilt |
| Mean air density | 1.36 kg/m³ | 11 % above the 1.225 kg/m³ nameplate assumption |

The polar night is emergent, not imposed: it falls out of solar geometry at the
configured latitude. Changing the station's latitude changes its length
automatically, which is the property that lets the same code serve a Himalayan
or island deployment.

## Reproducing these numbers

```bash
python scripts/run_baseline.py --station maitri --seed 0
```

Every figure above is regenerated by that command, and the invariants behind them
are asserted in `tests/`, so a change that quietly breaks the calibration fails
the suite rather than silently altering a headline claim.
