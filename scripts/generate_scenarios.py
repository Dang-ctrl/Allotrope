"""Regenerate the four scenarios shown in the scenario explorer artifact.

    python scripts/generate_scenarios.py

Writes scenarios.json with the exact data behind every number in the
"Allotrope scenario explorer" page. Each scenario is picked to exercise a
different part of the system, and each is a real run of the simulator -- not
narrated, not hand-tuned to look good.

A note on reproducibility that cost real debugging time to learn: `ClimateGenerator`
draws each weather quantity as one batch sized to `periods`, so the same seed at
the same start date still produces different weather if `periods` differs -- every
later random draw shifts to a different offset in the underlying stream. The
(station, start, periods, seed) tuple for each scenario below is therefore load
-bearing in full, not just the seed, and is kept identical between the script that
found the scenario and the one that (re)computes it.

  1. storm       -- a blizzard and a record cold snap landing in the same six
                     hours as an injected AI controller failure, replayed with
                     and without the safety layer (Maitri, seed 7).
  2. wetstack     -- the founding problem itself: one generator's exhaust
                     fouling under incumbent practice against staying clean
                     under disciplined dispatch, over two midwinter weeks
                     (Maitri, seed 3).
  3. freeenergy   -- a windy autumn day at the station with by far the larger
                     renewable fleet: incumbent practice wastes over a
                     megawatt-hour of free wind power in one day; disciplined
                     dispatch captures essentially all of it and runs the
                     station on wind alone for nine hours (Bharati, seed 0).
  4. gridstress   -- not a claim about today's grid, which never approaches
                     this limit -- a forward stress test of the OpenDSS twin's
                     Volt-VAr/Volt-Watt fallback as installed renewable capacity
                     is scaled up, the way a "Maitri II"-scale expansion might.
"""

from __future__ import annotations

import json

import pandas as pd

from allotrope.config import load_station
from allotrope.control.baseline import EfficientRuleBased, LegacyNPlusOne
from allotrope.network.twin import NetworkTwin
from allotrope.sim.plant import DispatchCommand
from allotrope.sim.runner import build_plant, run_episode

OUT_PATH = "scenarios.json"


def scenario_storm() -> list[dict]:
    cfg = load_station("maitri")
    # These four values must all match together -- see the module docstring.
    START, PERIODS, SEED = "2026-07-01", 24 * 30, 7
    BLIZZARD_START = pd.Timestamp("2026-07-20 14:00:00", tz="UTC")
    BLIZZARD_END = pd.Timestamp("2026-07-20 19:00:00", tz="UTC")
    WINDOW_START = BLIZZARD_START - pd.Timedelta(hours=6)
    WINDOW_END = BLIZZARD_END + pd.Timedelta(hours=6)

    class FailingDuringBlizzard:
        """Stands in for a learned agent that crashes for the storm's duration."""

        name = "agent_failing_at_the_worst_moment"

        def __init__(self, cfg):
            self.good_agent = EfficientRuleBased(cfg)

        def reset(self):
            self.good_agent.reset()

        def act(self, observation, plant):
            if BLIZZARD_START <= observation["timestamp"] <= BLIZZARD_END:
                raise RuntimeError("simulated: control network raised mid-inference")
            return self.good_agent.act(observation, plant)

    from allotrope.safety.fallback import GuardedController

    plant_g = build_plant(cfg, start=START, periods=PERIODS, seed=SEED)
    guard = GuardedController(cfg, agent=FailingDuringBlizzard(cfg))
    result_g = run_episode(plant_g, guard)
    tel_g = result_g.telemetry.loc[WINDOW_START:WINDOW_END]
    weather = plant_g.climate.to_frame().loc[WINDOW_START:WINDOW_END]

    plant_u = build_plant(cfg, start=START, periods=PERIODS, seed=SEED)
    plant_u.reset()
    failing = FailingDuringBlizzard(cfg)
    records = []
    for _ in range(plant_u.n_steps):
        obs = plant_u.observe()
        try:
            command = failing.act(obs, plant_u)
        except Exception:
            command = DispatchCommand.all_off(cfg)
        records.append(plant_u.step(command))
    uw = pd.DataFrame.from_records(records).set_index("timestamp").loc[WINDOW_START:WINDOW_END]

    rows, cum_crit_g, cum_crit_u, cum_shed_g = [], 0.0, 0.0, 0.0
    for ts in tel_g.index:
        crit_g = float(tel_g.loc[ts, "critical_unserved_kw"])
        crit_u = float(uw.loc[ts, "critical_unserved_kw"])
        shed_g = float(tel_g.loc[ts, "unserved_kw"])
        cum_crit_g += crit_g
        cum_crit_u += crit_u
        cum_shed_g += shed_g
        rows.append({
            "t": ts.strftime("%H:%M"), "date": ts.strftime("%b %d"),
            "temp": round(float(weather.loc[ts, "air_temp_c"]), 1),
            "wind": round(float(weather.loc[ts, "wind_speed_ms"]), 1),
            "blizzard": bool(weather.loc[ts, "blizzard"]),
            "aiFailed": bool(BLIZZARD_START <= ts <= BLIZZARD_END),
            "gensetG": round(float(tel_g.loc[ts, "genset_kw"]), 1),
            "gensetU": round(float(uw.loc[ts, "genset_kw"]), 1),
            "critG": round(crit_g, 1), "critU": round(crit_u, 1),
            "cumCritG": round(cum_crit_g, 1), "cumCritU": round(cum_crit_u, 1),
            "cumShedG": round(cum_shed_g, 1),
        })
    return rows


def scenario_wetstack() -> dict:
    cfg = load_station("maitri")
    START, PERIODS, SEED = "2026-06-01", 24 * 14, 3
    plant_l = build_plant(cfg, start=START, periods=PERIODS, seed=SEED)
    result_l = run_episode(plant_l, LegacyNPlusOne(cfg))
    plant_e = build_plant(cfg, start=START, periods=PERIODS, seed=SEED)
    result_e = run_episode(plant_e, EfficientRuleBased(cfg))

    agg = {"mean_deposit": "mean", "wet_stacking": "mean", "fuel_l": "sum",
           "black_carbon_mg": "sum", "mean_online_load_frac": "mean"}
    daily_l = result_l.telemetry.resample("1D").agg(agg)
    daily_e = result_e.telemetry.resample("1D").agg(agg)

    rows = []
    for i, (ts, r_l) in enumerate(daily_l.iterrows()):
        r_e = daily_e.iloc[i]
        rows.append({
            "day": i + 1, "date": ts.strftime("%b %d"),
            "depositLegacy": round(float(r_l["mean_deposit"]) * 100, 1),
            "depositEfficient": round(float(r_e["mean_deposit"]) * 100, 1),
            "wetStackLegacy": round(float(r_l["wet_stacking"]) * 100, 1),
            "wetStackEfficient": round(float(r_e["wet_stacking"]) * 100, 1),
            "fuelLegacy": round(float(r_l["fuel_l"]), 1),
            "fuelEfficient": round(float(r_e["fuel_l"]), 1),
            "bcLegacy": round(float(r_l["black_carbon_mg"]) / 1000, 1),
            "bcEfficient": round(float(r_e["black_carbon_mg"]) / 1000, 1),
            "loadLegacy": round(float(r_l["mean_online_load_frac"]) * 100, 1),
            "loadEfficient": round(float(r_e["mean_online_load_frac"]) * 100, 1),
        })
    return {
        "station": "Maitri", "rows": rows,
        "totalFuelLegacy": round(result_l.summary["fuel_l"], 0),
        "totalFuelEfficient": round(result_e.summary["fuel_l"], 0),
        "totalBcLegacy": round(result_l.summary["black_carbon_g"], 0),
        "totalBcEfficient": round(result_e.summary["black_carbon_g"], 0),
    }


def scenario_freeenergy() -> dict:
    cfg = load_station("bharati")
    START, PERIODS, SEED = "2026-01-01", 8760, 0
    DAY = pd.Timestamp("2026-03-14", tz="UTC")

    plant_l = build_plant(cfg, start=START, periods=PERIODS, seed=SEED)
    result_l = run_episode(plant_l, LegacyNPlusOne(cfg))
    plant_e = build_plant(cfg, start=START, periods=PERIODS, seed=SEED)
    result_e = run_episode(plant_e, EfficientRuleBased(cfg))

    tel_l = result_l.telemetry.loc[DAY:DAY + pd.Timedelta(hours=23)]
    tel_e = result_e.telemetry.loc[DAY:DAY + pd.Timedelta(hours=23)]
    idxs = [plant_l.index.get_loc(ts) for ts in tel_l.index]

    rows = []
    for k, ts in enumerate(tel_l.index):
        i = idxs[k]
        avail = float(plant_l.pv_available_kw[i] + plant_l.wind_available_kw[i])
        rows.append({
            "t": ts.strftime("%H:%M"), "renewAvail": round(avail, 1),
            "renewUsedLegacy": round(float(tel_l["renewable_used_kw"].iloc[k]), 1),
            "renewUsedEfficient": round(float(tel_e["renewable_used_kw"].iloc[k]), 1),
            "curtailedLegacy": round(float(tel_l["curtailed_kw"].iloc[k]), 1),
            "curtailedEfficient": round(float(tel_e["curtailed_kw"].iloc[k]), 1),
            "gensetLegacy": round(float(tel_l["genset_kw"].iloc[k]), 1),
            "gensetEfficient": round(float(tel_e["genset_kw"].iloc[k]), 1),
        })
    return {
        "station": "Bharati", "date": "March 14", "rows": rows,
        "totalCurtailedLegacy": round(tel_l["curtailed_kw"].sum(), 0),
        "totalCurtailedEfficient": round(tel_e["curtailed_kw"].sum(), 0),
        "gensetOffHoursEfficient": int((tel_e["genset_kw"] < 1.0).sum()),
    }


def scenario_gridstress() -> dict:
    cfg = load_station("bharati")
    plant = build_plant(cfg, start="2026-01-01", periods=8760, seed=0)
    twin = NetworkTwin(cfg)
    peak = pd.Timestamp("2026-03-14 07:00", tz="UTC")
    i = plant.index.get_loc(peak)
    pv0 = float(plant.pv_available_kw[i])
    wind0 = float(plant.wind_available_kw[i])

    rows = []
    for mult in range(1, 7):
        inv_rated = {
            "pv": cfg.pv.rated_kwp * mult, "wind": cfg.wind.rated_kw_total * mult,
            "bess_heated_core": 100.0, "bess_exterior": 90.0,
        }
        loads = {
            "pv": -pv0 * mult, "wind": -wind0 * mult, "bess_heated_core": 0.0, "bess_exterior": 0.0,
            "load_critical": float(plant.loads.critical_kw[i]),
            "load_general": float(plant.loads.electrical_kw[i] - plant.loads.critical_kw[i]),
            "load_melt": 10.0,
        }
        raw = twin.solve(loads)
        fb = twin.apply_volt_var_volt_watt(loads, inv_rated)
        rows.append({
            "mult": mult,
            "installedPv": round(cfg.pv.rated_kwp * mult, 0),
            "installedWind": round(cfg.wind.rated_kw_total * mult, 0),
            "vPvRaw": round(raw.voltages_pu["pv"], 4), "vWindRaw": round(raw.voltages_pu["wind"], 4),
            "vPvFallback": round(fb.voltages_pu["pv"], 4), "vWindFallback": round(fb.voltages_pu["wind"], 4),
            "curtailPv": round(fb.curtailment_fraction.get("pv", 1.0), 3),
            "curtailWind": round(fb.curtailment_fraction.get("wind", 1.0), 3),
            "intervened": fb.intervened_buses,
        })
    return {"station": "Bharati", "rows": rows, "realPv": round(pv0, 1), "realWind": round(wind0, 1)}


def main() -> None:
    scenarios = {
        "storm": scenario_storm(),
        "wetstack": scenario_wetstack(),
        "freeenergy": scenario_freeenergy(),
        "gridstress": scenario_gridstress(),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(scenarios, f, indent=None)
    print(f"wrote {OUT_PATH}")
    for name, data in scenarios.items():
        n = len(data) if isinstance(data, list) else len(data["rows"])
        print(f"  {name}: {n} rows")


if __name__ == "__main__":
    main()
