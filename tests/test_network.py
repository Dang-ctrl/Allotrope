"""The network twin: does the feeder solve sensibly, and does the fallback fire.

`opendssdirect` binds to a single C++ engine instance per process, so these
tests each build their own `NetworkTwin` rather than sharing one across test
functions -- state from one circuit must not leak into another's assertions.
"""

from __future__ import annotations

import pytest

from allotrope.config import available_stations, load_station
from allotrope.network.twin import (
    INVERTER_BUSES,
    NetworkTwin,
    VOLT_VAR_POINTS,
    VOLT_WATT_POINTS,
    volt_var_fraction,
    volt_watt_fraction,
)


@pytest.fixture
def twin():
    return NetworkTwin(load_station("maitri"))


def _balanced_loads(twin: NetworkTwin) -> dict[str, float]:
    return {
        "pv": -20.0,
        "wind": -5.0,
        "bess_heated_core": 0.0,
        "bess_exterior": 0.0,
        "load_critical": 45.0,
        "load_general": 30.0,
        "load_melt": 10.0,
    }


# -- the curves themselves -----------------------------------------------


def test_volt_var_has_a_deadband_around_nominal_voltage():
    assert volt_var_fraction(1.00) == pytest.approx(0.0)
    assert volt_var_fraction(0.99) == pytest.approx(0.0)
    assert volt_var_fraction(1.01) == pytest.approx(0.0)


def test_volt_var_supports_low_voltage_and_absorbs_high_voltage():
    assert volt_var_fraction(0.85) > 0  # inject, to raise a sagging bus
    assert volt_var_fraction(1.10) < 0  # absorb, to pull down an overvoltage bus


def test_volt_var_saturates_at_its_rated_fraction():
    low_point = VOLT_VAR_POINTS[0][1]
    high_point = VOLT_VAR_POINTS[-1][1]
    assert volt_var_fraction(0.5) == pytest.approx(low_point)
    assert volt_var_fraction(1.5) == pytest.approx(high_point)


def test_volt_var_is_monotonically_non_increasing_in_voltage():
    voltages = [0.80 + 0.01 * i for i in range(50)]
    fractions = [volt_var_fraction(v) for v in voltages]
    assert all(a >= b - 1e-12 for a, b in zip(fractions, fractions[1:]))


def test_volt_watt_does_not_curtail_within_normal_range():
    assert volt_watt_fraction(1.00) == pytest.approx(1.0)
    assert volt_watt_fraction(VOLT_WATT_POINTS[0][0]) == pytest.approx(1.0)


def test_volt_watt_fully_curtails_far_above_its_ceiling():
    assert volt_watt_fraction(1.5) == pytest.approx(0.0)


def test_volt_watt_ramps_between_its_breakpoints():
    v0, p0 = VOLT_WATT_POINTS[0]
    v1, p1 = VOLT_WATT_POINTS[1]
    mid_v = (v0 + v1) / 2
    mid_p = volt_watt_fraction(mid_v)
    assert p1 < mid_p < p0


# -- the circuit ------------------------------------------------------------


def test_the_circuit_builds_and_solves_for_every_station():
    for name in available_stations():
        cfg = load_station(name)
        if cfg.network is None:
            continue
        t = NetworkTwin(cfg)
        solution = t.solve(_balanced_loads(t))
        assert solution.converged


def test_a_station_without_a_network_section_raises_clearly():
    cfg = load_station("maitri")
    stripped = cfg.__class__(**{**cfg.__dict__, "network": None})
    with pytest.raises(ValueError, match="no \\[network\\] section"):
        NetworkTwin(stripped)


def test_normal_operating_load_stays_within_ansi_range(twin):
    solution = twin.solve(_balanced_loads(twin))
    for name, v in solution.voltages_pu.items():
        assert 0.95 <= v <= 1.05, f"{name} sits at {v:.4f} pu under ordinary load"


def test_every_configured_branch_reports_a_voltage(twin):
    solution = twin.solve(_balanced_loads(twin))
    assert set(solution.voltages_pu) == set(twin.net.branches)


def test_heavier_export_raises_the_exporting_buss_voltage(twin):
    light = twin.solve({**_balanced_loads(twin), "pv": -5.0})
    heavy = twin.solve({**_balanced_loads(twin), "pv": -300.0})
    assert heavy.voltages_pu["pv"] > light.voltages_pu["pv"]


def test_solving_is_deterministic(twin):
    loads = _balanced_loads(twin)
    a = twin.solve(loads)
    b = twin.solve(loads)
    assert a.voltages_pu == pytest.approx(b.voltages_pu)


# -- the fallback -------------------------------------------------------------


@pytest.fixture
def inverter_ratings():
    return {"pv": 50.0, "wind": 20.0, "bess_heated_core": 60.0, "bess_exterior": 60.0}


def test_the_fallback_is_a_no_op_under_ordinary_conditions(twin, inverter_ratings):
    result = twin.apply_volt_var_volt_watt(_balanced_loads(twin), inverter_ratings)
    assert result.intervened_buses == []
    assert all(f == pytest.approx(1.0) for f in result.curtailment_fraction.values())


def test_a_stressed_bus_receives_reactive_support_before_curtailment(twin, inverter_ratings):
    """VAr support should measurably reduce voltage even where it does not fully fix it."""
    stressed = {**_balanced_loads(twin), "pv": -600.0}
    raw = twin.solve(stressed)
    result = twin.apply_volt_var_volt_watt(stressed, inverter_ratings)

    assert result.reactive_kvar["pv"] < 0.0, "an overvoltage bus should absorb reactive power"
    assert result.voltages_pu["pv"] < raw.voltages_pu["pv"]


def test_extreme_export_triggers_real_curtailment(twin, inverter_ratings):
    """Reactive support alone cannot hold an extreme overvoltage; power must be cut."""
    extreme = {**_balanced_loads(twin), "pv": -600.0}
    result = twin.apply_volt_var_volt_watt(extreme, inverter_ratings)

    assert "pv" in result.intervened_buses
    assert result.curtailment_fraction["pv"] < 1.0
    assert result.voltages_pu["pv"] <= 1.10 + 1e-6, (
        "curtailment must actually bring the bus back under the Volt-Watt ceiling"
    )


def test_fallback_only_acts_on_inverter_interfaced_buses(twin, inverter_ratings):
    stressed = {**_balanced_loads(twin), "pv": -600.0, "load_general": 500.0}
    result = twin.apply_volt_var_volt_watt(stressed, inverter_ratings)
    assert set(result.reactive_kvar) <= set(INVERTER_BUSES)
    assert "load_general" not in result.reactive_kvar
    assert "load_critical" not in result.curtailment_fraction
