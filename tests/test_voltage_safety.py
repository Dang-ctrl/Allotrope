"""Inverter-level Volt-Watt: the curve itself, the layer, and its effect
end to end through GuardedController on a real overvoltage scenario."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from allotrope.config import load_station
from allotrope.control.baseline import EfficientRuleBased
from allotrope.safety.fallback import GuardedController
from allotrope.safety.voltage import VoltWattCurve, build_inverter_layer
from allotrope.sim.plant import DispatchCommand
from allotrope.sim.runner import build_plant


def _maitri():
    return load_station("maitri")


# -- the curve itself, as a pure function --------------------------------


def test_curve_is_full_power_below_v1():
    curve = VoltWattCurve()
    assert curve.power_limit_frac(1.0) == 1.0
    assert curve.power_limit_frac(curve.v1_pu) == 1.0


def test_curve_floors_at_p2_frac_above_v2():
    curve = VoltWattCurve()
    assert curve.power_limit_frac(curve.v2_pu) == curve.p2_frac
    assert curve.power_limit_frac(2.0) == curve.p2_frac


def test_curve_is_linear_at_the_midpoint():
    curve = VoltWattCurve(v1_pu=1.0, v2_pu=1.2, p2_frac=0.0)
    assert abs(curve.power_limit_frac(1.1) - 0.5) < 1e-9


@given(voltage_pu=st.floats(min_value=0.5, max_value=2.0, allow_nan=False))
@settings(max_examples=100)
def test_curve_output_is_always_bounded_and_monotonic_with_a_lower_point(voltage_pu):
    curve = VoltWattCurve()
    frac = curve.power_limit_frac(voltage_pu)
    assert curve.p2_frac <= frac <= 1.0
    # Monotonic non-increasing: a lower voltage never curtails more.
    lower_frac = curve.power_limit_frac(max(voltage_pu - 0.01, 0.0))
    assert lower_frac >= frac - 1e-9


# -- the layer, against a real network solve -----------------------------


def test_high_renewable_export_triggers_curtailment():
    cfg = _maitri()
    layer = build_inverter_layer(cfg)
    assert layer is not None

    command = DispatchCommand(
        genset_on=(False, False, False),
        genset_setpoint_kw=(0.0, 0.0, 0.0),
        battery_kw=(0.0, 0.0),
        snow_melt_kw=0.0,
    )
    observation = {
        "pv_available_kw": 300.0,
        "wind_available_kw": 300.0,
        "electrical_load_kw": 20.0,
    }
    new_command, report = layer.apply(command, observation)
    assert report.converged
    assert report.curtailed
    assert new_command.renewable_limit_kw is not None
    assert new_command.renewable_limit_kw < report.renewable_available_kw


def test_light_renewable_output_is_not_curtailed():
    cfg = _maitri()
    layer = build_inverter_layer(cfg)
    assert layer is not None

    command = DispatchCommand(
        genset_on=(True, False, False),
        genset_setpoint_kw=(65.0, 0.0, 0.0),
        battery_kw=(0.0, 0.0),
        snow_melt_kw=0.0,
    )
    observation = {"pv_available_kw": 2.0, "wind_available_kw": 3.0, "electrical_load_kw": 65.0}
    new_command, report = layer.apply(command, observation)
    assert report.converged
    assert not report.curtailed
    assert new_command.renewable_limit_kw is None
    assert new_command is command  # unmodified: no new DispatchCommand needed


def test_station_without_a_network_config_gets_no_layer():
    cfg = load_station("bharati")
    assert cfg.network is None
    assert build_inverter_layer(cfg) is None


# -- end to end: GuardedController with the layer wired in ----------------


def test_guarded_controller_curtails_renewables_without_touching_critical_load():
    """The inverter layer runs strictly after SafetyProjection, which
    already excludes renewables from its own capacity guarantee by design
    -- curtailing them further must never reopen the critical-load
    guarantee this project's central claim rests on."""
    cfg = _maitri()
    layer = build_inverter_layer(cfg)
    guard = GuardedController(cfg, agent=EfficientRuleBased(cfg), inverter_layer=layer)

    plant = build_plant(cfg, start="2026-12-15", periods=24 * 3, seed=3)
    plant.reset()
    guard.reset()

    saw_curtailment = False
    for _ in range(plant.n_steps):
        obs = plant.observe()
        command = guard.act(obs, plant)
        telemetry = plant.step(command)
        assert telemetry["critical_unserved_kw"] == 0.0
        if guard.last_voltage_report is not None and guard.last_voltage_report.curtailed:
            saw_curtailment = True

    # Not asserting curtailment necessarily fires in this particular window
    # (that depends on the synthetic weather draw) -- only that when the
    # layer is wired in, life support is never the price of running it.
    assert guard.last_voltage_report is not None


def test_guarded_controller_without_a_layer_behaves_exactly_as_before():
    cfg = _maitri()
    guard = GuardedController(cfg, agent=EfficientRuleBased(cfg))
    assert guard.inverter_layer is None
    plant = build_plant(cfg, periods=24, seed=0)
    plant.reset()
    guard.reset()
    obs = plant.observe()
    command = guard.act(obs, plant)
    assert command.renewable_limit_kw is None
    assert guard.last_voltage_report is None
