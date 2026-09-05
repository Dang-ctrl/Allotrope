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
import os

import numpy as np
import pandas as pd

from allotrope.agents.checkpoint import load as load_checkpoint, load_federated
from allotrope.config import load_station
from allotrope.control.baseline import EfficientRuleBased, LegacyNPlusOne
from allotrope.envs.polar_microgrid import PolarMicrogridEnv
from allotrope.network.twin import NetworkTwin
from allotrope.safety.fallback import GuardedController
from allotrope.sim.assets import Battery, BatteryState
from allotrope.sim.plant import DispatchCommand
from allotrope.sim.runner import build_plant, run_episode

OUT_PATH = "scenarios.json"
HELD_OUT_SEEDS = [100, 101, 102, 103, 104]


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


def _agent_eval_rows(cfg, agent_label, guarded_agent, seeds, periods=8760):
    """Runs legacy/efficient/agent on each held-out seed; returns per-seed rows.

    Mirrors scripts/evaluate_agent.py exactly -- same controllers, same seeds,
    same periods -- so this is the same held-out evaluation, not a rebuilt one.
    """
    rows = []
    for seed in seeds:
        results = {}
        for controller, label in (
            (LegacyNPlusOne(cfg), "legacy"),
            (EfficientRuleBased(cfg), "efficient"),
            (guarded_agent, "agent"),
        ):
            plant = build_plant(cfg, periods=periods, seed=seed)
            result = run_episode(plant, controller)
            results[label] = result.summary
        rows.append({
            "seed": seed,
            "fuelLegacy": round(results["legacy"]["fuel_l"] / 1000, 1),
            "fuelEfficient": round(results["efficient"]["fuel_l"] / 1000, 1),
            "fuelAgent": round(results["agent"]["fuel_l"] / 1000, 1),
            "bcLegacy": round(results["legacy"]["black_carbon_g"], 0),
            "bcEfficient": round(results["efficient"]["black_carbon_g"], 0),
            "bcAgent": round(results["agent"]["black_carbon_g"], 0),
            "startsLegacy": round(results["legacy"]["genset_starts"], 1),
            "startsEfficient": round(results["efficient"]["genset_starts"], 1),
            "startsAgent": round(results["agent"]["genset_starts"], 1),
            "critUnservedAgent": round(results["agent"]["critical_unserved_kwh"], 4),
            "freezeAgent": int(results["agent"]["freeze_violation_steps"]),
        })
    return rows


def scenario_agenteval(station_id: str, checkpoint_path: str) -> dict:
    cfg = load_station(station_id)
    agent = load_checkpoint(checkpoint_path, cfg)
    guarded = GuardedController(cfg, agent=agent)
    rows = _agent_eval_rows(cfg, "hybrid_dqn_sddpg", guarded, HELD_OUT_SEEDS)

    mean = lambda k: sum(r[k] for r in rows) / len(rows)
    fuel_vs_efficient = (mean("fuelEfficient") - mean("fuelAgent")) / mean("fuelEfficient")
    starts_vs_efficient = mean("startsAgent") - mean("startsEfficient")

    return {
        "station": cfg.site.name,
        "seeds": HELD_OUT_SEEDS,
        "rows": rows,
        "meanFuelLegacy": round(mean("fuelLegacy"), 1),
        "meanFuelEfficient": round(mean("fuelEfficient"), 1),
        "meanFuelAgent": round(mean("fuelAgent"), 1),
        "meanStartsLegacy": round(mean("startsLegacy"), 1),
        "meanStartsEfficient": round(mean("startsEfficient"), 1),
        "meanStartsAgent": round(mean("startsAgent"), 1),
        "meanBcLegacy": round(mean("bcLegacy"), 0),
        "meanBcEfficient": round(mean("bcEfficient"), 0),
        "meanBcAgent": round(mean("bcAgent"), 0),
        "fuelVsEfficientPct": round(fuel_vs_efficient * 100, 1),
        "startsVsEfficientDelta": round(starts_vs_efficient, 1),
        "maxCriticalUnservedAgent": max(r["critUnservedAgent"] for r in rows),
        "maxFreezeAgent": max(r["freezeAgent"] for r in rows),
    }


def scenario_federated(checkpoint_path: str) -> dict:
    """The federated checkpoint, evaluated the same way, against BOTH stations'
    held-out seeds and their own dedicated single-station checkpoints -- so the
    comparison is apples to apples, not just federated-vs-rules."""
    out = {}
    for station_id, own_ckpt in (("maitri", "checkpoints/maitri.pt"), ("bharati", "checkpoints/bharati.pt")):
        cfg = load_station(station_id)
        fed_agent = load_federated(checkpoint_path, cfg)
        fed_guarded = GuardedController(cfg, agent=fed_agent)
        fed_rows = _agent_eval_rows(cfg, "federated", fed_guarded, HELD_OUT_SEEDS)

        own_agent = load_checkpoint(own_ckpt, cfg)
        own_guarded = GuardedController(cfg, agent=own_agent)
        own_rows = _agent_eval_rows(cfg, "own", own_guarded, HELD_OUT_SEEDS)

        mean = lambda rows, k: sum(r[k] for r in rows) / len(rows)
        out[station_id] = {
            "station": cfg.site.name,
            "meanFuelEfficient": round(mean(fed_rows, "fuelEfficient"), 1),
            "meanFuelFederated": round(mean(fed_rows, "fuelAgent"), 1),
            "meanFuelOwnCheckpoint": round(mean(own_rows, "fuelAgent"), 1),
            "meanStartsEfficient": round(mean(fed_rows, "startsEfficient"), 1),
            "meanStartsFederated": round(mean(fed_rows, "startsAgent"), 1),
            "meanStartsOwnCheckpoint": round(mean(own_rows, "startsAgent"), 1),
            "maxCriticalUnservedFederated": max(r["critUnservedAgent"] for r in fed_rows),
            "maxFreezeFederated": max(r["freezeAgent"] for r in fed_rows),
        }
    return out


def scenario_safetyaudit(station_id: str = "maitri", days: int = 30, seed: int = 0) -> dict:
    """Reproduces scripts/run_safety_audit.py's policies and numbers exactly."""
    cfg = load_station(station_id)

    def random_policy(env, rng):
        return env.action_space.sample()

    def all_off_policy(env, rng):
        n_g, n_s = len(env.cfg.gensets), len(env.cfg.storage)
        return {"genset_on": np.zeros(n_g, dtype=np.int8), "dispatch": np.full(n_g + n_s + 1, -1.0, dtype=np.float32)}

    def max_charge_policy(env, rng):
        n_g, n_s = len(env.cfg.gensets), len(env.cfg.storage)
        dispatch = np.full(n_g + n_s + 1, -1.0, dtype=np.float32)
        dispatch[n_g:n_g + n_s] = -1.0
        return {"genset_on": np.zeros(n_g, dtype=np.int8), "dispatch": dispatch}

    def max_melt_policy(env, rng):
        n_g, n_s = len(env.cfg.gensets), len(env.cfg.storage)
        dispatch = np.full(n_g + n_s + 1, -1.0, dtype=np.float32)
        dispatch[-1] = 1.0
        return {"genset_on": np.zeros(n_g, dtype=np.int8), "dispatch": dispatch}

    def oscillating_policy(env, rng):
        flip = bool(rng.integers(0, 2))
        n_g, n_s = len(env.cfg.gensets), len(env.cfg.storage)
        return {
            "genset_on": np.full(n_g, int(flip), dtype=np.int8),
            "dispatch": np.full(n_g + n_s + 1, 1.0 if flip else -1.0, dtype=np.float32),
        }

    policies = {
        "random": random_policy,
        "shut everything down": all_off_policy,
        "charge flat out": max_charge_policy,
        "melt flat out": max_melt_policy,
        "oscillate commitment": oscillating_policy,
    }

    def run(policy, apply_safety):
        env = PolarMicrogridEnv(station=cfg, start="2026-06-01", periods=24 * days, seed=seed, apply_safety=apply_safety)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        rng = np.random.default_rng(seed)
        interventions: dict[str, int] = {}
        while True:
            _, _, terminated, truncated, info = env.step(policy(env, rng))
            for name in info.get("safety", {}).get("interventions", []):
                interventions[name] = interventions.get(name, 0) + 1
            if terminated or truncated:
                break
        return env.summary(), interventions

    rows = []
    all_interventions: dict[str, int] = {}
    for label, policy in policies.items():
        guarded, interventions = run(policy, apply_safety=True)
        unguarded, _ = run(policy, apply_safety=False)
        for name, count in interventions.items():
            all_interventions[name] = all_interventions.get(name, 0) + count
        rows.append({
            "attack": label,
            "critLostGuarded": round(guarded["critical_unserved_kwh"], 2),
            "critLostUnguarded": round(unguarded["critical_unserved_kwh"], 2),
            "freezeGuarded": int(guarded["freeze_violation_steps"]),
            "freezeUnguarded": int(unguarded["freeze_violation_steps"]),
        })

    return {
        "station": cfg.site.name,
        "days": days,
        "rows": rows,
        "interventionCounts": [{"name": n, "count": c} for n, c in sorted(all_interventions.items(), key=lambda kv: -kv[1])],
        "worstGuarded": max(r["critLostGuarded"] for r in rows),
        "worstFreezeGuarded": max(r["freezeGuarded"] for r in rows),
    }


def scenario_coldbattery(station_id: str = "maitri") -> dict:
    """Sweeps each storage pack's own asset model across temperature -- real
    numbers from allotrope.sim.assets.Battery, not a hand-drawn illustration."""
    cfg = load_station(station_id)
    temps = list(range(20, -36, -2))
    packs = {}
    for spec in cfg.storage:
        battery = Battery(spec=spec, state=BatteryState(soc=0.5))
        rows = []
        for t in temps:
            battery.state.temperature_c = float(t)
            rows.append({
                "tempC": t,
                "derate": round(battery.cold_derate(), 3),
                "maxChargeKw": round(battery.max_charge_kw(), 1),
                "maxDischargeKw": round(battery.max_discharge_kw(), 1),
            })
        packs[spec.id] = {
            "chemistry": spec.chemistry,
            "location": spec.location,
            "minOperatingTempC": spec.min_operating_temp_c,
            "maxChargeKwRated": spec.max_charge_kw,
            "maxDischargeKwRated": spec.max_discharge_kw,
            "capacityKwh": spec.capacity_kwh,
            "rows": rows,
        }
    return {"station": cfg.site.name, "packs": packs}


def main() -> None:
    scenarios = {
        "storm": scenario_storm(),
        "wetstack": scenario_wetstack(),
        "freeenergy": scenario_freeenergy(),
        "gridstress": scenario_gridstress(),
        "agentmaitri": scenario_agenteval("maitri", "checkpoints/maitri.pt"),
        "agentbharati": scenario_agenteval("bharati", "checkpoints/bharati.pt"),
        "safetyaudit": scenario_safetyaudit("maitri", days=30, seed=0),
        "coldbattery": scenario_coldbattery("maitri"),
    }
    if os.path.exists("checkpoints/federated.pt"):
        scenarios["federated"] = scenario_federated("checkpoints/federated.pt")
    else:
        print("checkpoints/federated.pt not found -- skipping the federated scenario "
              "(run scripts/run_federated.py first, then re-run this script)")

    with open(OUT_PATH, "w") as f:
        json.dump(scenarios, f, indent=None)
    print(f"wrote {OUT_PATH}")
    for name, data in scenarios.items():
        if isinstance(data, list):
            n = len(data)
        elif "rows" in data:
            n = len(data["rows"])
        else:
            n = len(data)
        print(f"  {name}: {n} rows")


if __name__ == "__main__":
    main()
