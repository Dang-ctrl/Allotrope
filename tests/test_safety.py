"""The safety layer's guarantee, tested the only way a guarantee can be tested.

A guarantee that holds for sensible actions is not a guarantee. These tests
attack the projection with random policies, adversarial policies that actively
try to hurt the station, malformed actions full of NaN, and agents that crash or
hang. The property asserted is always the same and is never relaxed:

    life support is served, and the station does not freeze.

If any test in this file fails, the central claim of the project is false, and no
efficiency result elsewhere in the repository is worth reporting.
"""

from __future__ import annotations

import numpy as np
import pytest

from allotrope.config import available_stations, load_station
from allotrope.control.baseline import EfficientRuleBased
from allotrope.envs.polar_microgrid import PolarMicrogridEnv
from allotrope.safety.fallback import (
    DeterministicFallback,
    FallbackReason,
    GuardedController,
)
from allotrope.safety.projection import Intervention, SafetyProjection
from allotrope.sim.plant import DispatchCommand
from allotrope.sim.runner import build_plant, run_episode

WINTER = "2026-06-01"
SUMMER = "2026-12-15"


def _env(station="maitri", start=WINTER, days=21, seed=0):
    return PolarMicrogridEnv(station=station, start=start, periods=24 * days, seed=seed)


def _run(env, policy, action_seed=0):
    """Drive an environment to the end under a policy, returning its summary."""
    env.reset(seed=env._seed)
    env.action_space.seed(action_seed)
    rng = np.random.default_rng(action_seed)
    while True:
        _, _, terminated, truncated, _ = env.step(policy(env, rng))
        if terminated or truncated:
            break
    return env.summary()


# -- the guarantee, under everything we can throw at it -----------------------


def _assert_station_safe(summary, label=""):
    assert summary["critical_unserved_kwh"] == pytest.approx(0.0, abs=1e-9), (
        f"{label}: life support went unserved"
    )
    assert summary["freeze_violation_steps"] == 0.0, f"{label}: the station froze"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("start", [WINTER, SUMMER])
def test_random_policy_never_endangers_the_station(seed, start):
    env = _env(start=start, seed=seed)
    summary = _run(env, lambda e, rng: e.action_space.sample(), action_seed=seed)
    _assert_station_safe(summary, f"random seed={seed} start={start}")


@pytest.mark.parametrize("station", available_stations())
def test_the_guarantee_holds_at_every_station(station):
    env = _env(station=station, days=14)
    summary = _run(env, lambda e, rng: e.action_space.sample(), action_seed=7)
    _assert_station_safe(summary, station)


def test_a_policy_that_shuts_everything_down_cannot_do_so():
    """The adversary that matters: an agent that simply stops every machine."""

    def all_off(env, rng):
        n_g = len(env.cfg.gensets)
        n_s = len(env.cfg.storage)
        return {
            "genset_on": np.zeros(n_g, dtype=np.int8),
            "dispatch": np.full(n_g + n_s + 1, -1.0, dtype=np.float32),
        }

    summary = _run(_env(days=30), all_off)
    _assert_station_safe(summary, "all-off adversary")
    assert summary["genset_run_hours"] > 0.0, "the projection never started anything"


def test_a_policy_that_charges_flat_out_cannot_starve_life_support():
    """Charging is demand. An agent that maximises it must not black out the station."""

    def max_charge(env, rng):
        n_g = len(env.cfg.gensets)
        n_s = len(env.cfg.storage)
        dispatch = np.full(n_g + n_s + 1, -1.0, dtype=np.float32)
        dispatch[n_g : n_g + n_s] = -1.0  # full charge
        return {"genset_on": np.zeros(n_g, dtype=np.int8), "dispatch": dispatch}

    summary = _run(_env(days=21), max_charge)
    _assert_station_safe(summary, "max-charge adversary")


def test_a_policy_that_melts_flat_out_cannot_starve_life_support():
    """Melting is the dump load, and therefore the obvious thing to over-commit."""

    def max_melt(env, rng):
        n_g = len(env.cfg.gensets)
        n_s = len(env.cfg.storage)
        dispatch = np.full(n_g + n_s + 1, -1.0, dtype=np.float32)
        dispatch[-1] = 1.0
        return {"genset_on": np.zeros(n_g, dtype=np.int8), "dispatch": dispatch}

    summary = _run(_env(days=21), max_melt)
    _assert_station_safe(summary, "max-melt adversary")


def test_an_oscillating_policy_cannot_endanger_the_station():
    """Commitment thrash is what anti-cycling and the projection must survive together."""
    state = {"flip": False}

    def oscillate(env, rng):
        state["flip"] = not state["flip"]
        n_g = len(env.cfg.gensets)
        n_s = len(env.cfg.storage)
        value = 1 if state["flip"] else 0
        return {
            "genset_on": np.full(n_g, value, dtype=np.int8),
            "dispatch": np.full(n_g + n_s + 1, -1.0 if state["flip"] else 1.0, dtype=np.float32),
        }

    summary = _run(_env(days=21), oscillate)
    _assert_station_safe(summary, "oscillating adversary")


# -- malformed actions --------------------------------------------------------


@pytest.fixture
def projection_setup():
    cfg = load_station("maitri")
    plant = build_plant(cfg, start=WINTER, periods=48, seed=0)
    plant.reset()
    return cfg, plant, SafetyProjection(cfg)


def test_non_finite_actions_are_sanitised(projection_setup):
    """A corrupted checkpoint produces NaN. It must not reach a generating set."""
    cfg, plant, projection = projection_setup
    command = DispatchCommand(
        genset_on=(True, False, True),
        genset_setpoint_kw=(float("nan"), float("inf"), -float("inf")),
        battery_kw=(float("nan"), 0.0),
        snow_melt_kw=float("nan"),
    )
    safe, report = projection.project(command, plant.observe(), plant)

    assert Intervention.SANITISED_NAN in report.interventions
    assert all(np.isfinite(v) for v in safe.genset_setpoint_kw)
    assert all(np.isfinite(v) for v in safe.battery_kw)
    assert np.isfinite(safe.snow_melt_kw)


def test_absurd_magnitudes_are_bounded(projection_setup):
    cfg, plant, projection = projection_setup
    command = DispatchCommand(
        genset_on=(True, True, True),
        genset_setpoint_kw=(1e9, -1e9, 1e9),
        battery_kw=(1e9, -1e9),
        snow_melt_kw=1e9,
    )
    safe, _ = projection.project(command, plant.observe(), plant)

    for k, g in enumerate(cfg.gensets):
        assert g.min_stable_kw <= safe.genset_setpoint_kw[k] <= g.rated_kw
    for k, s in enumerate(cfg.storage):
        assert -s.max_charge_kw <= safe.battery_kw[k] <= s.max_discharge_kw
    assert 0.0 <= safe.snow_melt_kw <= projection.melt_ceiling_kw()


def test_short_and_overlong_commands_are_reshaped(projection_setup):
    cfg, plant, projection = projection_setup
    command = DispatchCommand(
        genset_on=(True,),
        genset_setpoint_kw=(50.0, 50.0, 50.0, 50.0, 50.0),
        battery_kw=(),
        snow_melt_kw=5.0,
    )
    safe, _ = projection.project(command, plant.observe(), plant)

    assert len(safe.genset_on) == len(cfg.gensets)
    assert len(safe.genset_setpoint_kw) == len(cfg.gensets)
    assert len(safe.battery_kw) == len(cfg.storage)


def test_capacity_cover_is_evaluated_jointly_not_per_machine(projection_setup):
    """Two stops that are each individually safe must not be jointly permitted.

    This is the failure the projection was found to have: with two sets online
    and both commanded off, each stop looked safe because the other was still
    running, and the plant ended up with nothing turning.
    """
    cfg, plant, projection = projection_setup
    # Put two sets on the bus, each free to stop, and command both off at once.
    for g in plant.gensets[:2]:
        g.state.online = True
        g.state.minutes_in_state = 1e6

    observation = plant.observe()
    online = [k for k, flag in enumerate(observation["genset_online"]) if flag]
    assert len(online) == 2 and all(observation["genset_can_stop"][k] for k in online)

    safe, report = projection.project(
        DispatchCommand.all_off(cfg), observation, plant
    )
    capacity = sum(cfg.gensets[k].rated_kw for k, on in enumerate(safe.genset_on) if on)
    assert capacity >= report.required_capacity_kw
    assert Intervention.BLOCKED_STOP in report.interventions


def test_the_projection_never_stops_a_running_machine(projection_setup):
    """It may start what was not asked for; it must never stop what was."""
    cfg, plant, projection = projection_setup
    rng = np.random.default_rng(3)
    for _ in range(40):
        observation = plant.observe()
        command = DispatchCommand(
            genset_on=tuple(bool(v) for v in rng.integers(0, 2, len(cfg.gensets))),
            genset_setpoint_kw=tuple(rng.uniform(0, 200, len(cfg.gensets))),
            battery_kw=tuple(rng.uniform(-100, 100, len(cfg.storage))),
            snow_melt_kw=float(rng.uniform(0, 100)),
        )
        safe, _ = projection.project(command, observation, plant)
        for k in range(len(cfg.gensets)):
            if command.genset_on[k]:
                assert safe.genset_on[k], "the projection stopped a set the agent wanted running"
        plant.step(safe)


def test_a_set_that_cannot_start_is_not_counted_as_cover(projection_setup):
    """Cover that depends on a machine inside its minimum down time is not cover."""
    cfg, plant, projection = projection_setup
    observation = plant.observe()
    observation["genset_online"] = [False] * len(cfg.gensets)
    observation["genset_can_start"] = [False] * len(cfg.gensets)
    observation["genset_can_stop"] = [False] * len(cfg.gensets)

    _, report = projection.project(DispatchCommand.all_off(cfg), observation, plant)
    assert report.committed_capacity_kw == 0.0
    assert report.required_capacity_kw > 0.0


def test_a_frozen_battery_is_never_commanded_to_charge(projection_setup):
    cfg, plant, projection = projection_setup
    observation = plant.observe()
    observation["battery_max_charge_kw"] = [0.0] * len(cfg.storage)

    command = DispatchCommand(
        genset_on=(True, True, True),
        genset_setpoint_kw=tuple(g.rated_kw for g in cfg.gensets),
        battery_kw=tuple(-100.0 for _ in cfg.storage),
        snow_melt_kw=0.0,
    )
    safe, report = projection.project(command, observation, plant)
    assert all(p >= 0.0 for p in safe.battery_kw)
    assert Intervention.CLIPPED_BATTERY in report.interventions


# -- the fallback and the guard -----------------------------------------------


def test_the_fallback_alone_keeps_the_station_safe():
    """The fallback is what runs when everything else has failed. It must suffice."""
    cfg = load_station("maitri")
    plant = build_plant(cfg, start=WINTER, periods=24 * 21, seed=5)
    result = run_episode(plant, GuardedController(cfg, agent=None))
    _assert_station_safe(result.summary, "fallback only")
    assert result.summary["genset_run_hours"] > 0.0


def test_the_fallback_is_deterministic():
    cfg = load_station("maitri")
    summaries = []
    for _ in range(2):
        plant = build_plant(cfg, start=WINTER, periods=24 * 7, seed=5)
        summaries.append(run_episode(plant, DeterministicFallback(cfg)).summary["fuel_l"])
    assert summaries[0] == pytest.approx(summaries[1])


def test_a_crashing_agent_is_replaced_not_propagated():
    cfg = load_station("maitri")

    class Crashing:
        name = "crashing"

        def act(self, observation, plant):
            raise RuntimeError("checkpoint is corrupt")

    guard = GuardedController(cfg, agent=Crashing())
    plant = build_plant(cfg, start=WINTER, periods=24 * 7, seed=6)
    result = run_episode(plant, guard)

    _assert_station_safe(result.summary, "crashing agent")
    assert guard.stats.fallback_rate == 1.0
    assert guard.stats.reasons[FallbackReason.AGENT_RAISED.value] == guard.stats.steps


def test_a_slow_agent_is_treated_as_a_failed_agent():
    """A late answer is a wrong answer when the control path budgets 10 ms."""
    cfg = load_station("maitri")

    class Slow:
        name = "slow"

        def act(self, observation, plant):
            import time

            time.sleep(0.02)
            return DispatchCommand.all_off(cfg)

    guard = GuardedController(cfg, agent=Slow(), latency_budget_ms=1.0)
    plant = build_plant(cfg, start=WINTER, periods=24, seed=6)
    run_episode(plant, guard)

    assert guard.stats.reasons.get(FallbackReason.AGENT_TIMED_OUT.value, 0) > 0


def test_a_malformed_agent_output_is_rejected():
    cfg = load_station("maitri")

    class Malformed:
        name = "malformed"

        def act(self, observation, plant):
            return "not a dispatch command"

    guard = GuardedController(cfg, agent=Malformed())
    plant = build_plant(cfg, start=WINTER, periods=24 * 3, seed=6)
    result = run_episode(plant, guard)

    _assert_station_safe(result.summary, "malformed agent")
    assert guard.stats.reasons[FallbackReason.AGENT_RETURNED_INVALID.value] > 0


def test_a_healthy_agent_is_left_alone():
    """The guard must not be so eager that it displaces a working controller."""
    cfg = load_station("maitri")
    guard = GuardedController(cfg, agent=EfficientRuleBased(cfg), latency_budget_ms=1000.0)
    plant = build_plant(cfg, start=WINTER, periods=24 * 7, seed=6)
    run_episode(plant, guard)

    assert guard.stats.fallbacks == 0
    assert guard.stats.projection_rate < 0.5


def test_every_intervention_is_reported(projection_setup):
    """Nothing is overridden silently; the operator sees which bound bit."""
    cfg, plant, projection = projection_setup
    command = DispatchCommand(
        genset_on=tuple(False for _ in cfg.gensets),
        genset_setpoint_kw=tuple(float("nan") for _ in cfg.gensets),
        battery_kw=tuple(-1e6 for _ in cfg.storage),
        snow_melt_kw=1e6,
    )
    _, report = projection.project(command, plant.observe(), plant)

    assert report.intervened
    payload = report.as_dict()
    assert payload["interventions"]
    assert all(isinstance(name, str) for name in payload["interventions"])
    assert payload["required_capacity_kw"] > 0.0
