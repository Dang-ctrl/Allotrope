"""The distribution network twin: real OpenDSS power flows, not a stub.

Every assertion here is about physical behaviour a radial LV feeder
actually exhibits -- export raising voltage, load depressing it -- not
about a hand-picked number this module was tuned to produce.
"""

from __future__ import annotations

import time

from allotrope.config import load_station
from allotrope.sim.network import DistributionNetwork


def _network():
    cfg = load_station("maitri")
    assert cfg.network is not None
    return DistributionNetwork(cfg.network), cfg.network


def test_no_injection_or_load_leaves_every_bus_near_one_pu():
    net, cfg = _network()
    solution = net.solve({})
    assert solution.converged
    for bus in cfg.bus_ids():
        assert abs(solution.voltage_pu(bus) - 1.0) < 0.01


def test_renewable_export_raises_the_renewables_bus_voltage():
    net, _ = _network()
    baseline = net.solve({})
    exporting = net.solve({"renewables": 200.0})
    assert exporting.converged
    assert exporting.voltage_pu("renewables") > baseline.voltage_pu("renewables")


def test_heavy_load_depresses_the_load_bus_voltage():
    net, _ = _network()
    baseline = net.solve({})
    loaded = net.solve({"load": -150.0})
    assert loaded.converged
    assert loaded.voltage_pu("load") < baseline.voltage_pu("load")


def test_source_bus_stays_at_the_reference_regardless_of_other_buses():
    net, cfg = _network()
    solution = net.solve({"renewables": 300.0, "load": -300.0})
    assert abs(solution.voltage_pu(cfg.source_bus) - 1.0) < 0.02


def test_a_solve_is_fast_enough_for_an_hourly_control_step():
    """Rebuilding the circuit every call (see the class docstring) must stay
    fast -- this is meant to run inside a per-step safety layer, not a
    batch analysis tool."""
    net, _ = _network()
    start = time.perf_counter()
    for _ in range(20):
        net.solve({"renewables": 50.0, "load": -80.0})
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / 20
    assert elapsed_ms < 50.0, f"mean solve time {elapsed_ms:.2f} ms is too slow for a control step"


def test_repeated_solves_do_not_interfere_with_each_other():
    """OpenDSS's global circuit state is exactly the hazard `solve` rebuilds
    around every call -- confirm two different inputs never see each
    other's leftovers."""
    net, _ = _network()
    light = net.solve({"renewables": 10.0, "load": -20.0})
    heavy = net.solve({"renewables": 400.0, "load": -400.0})
    light_again = net.solve({"renewables": 10.0, "load": -20.0})
    assert abs(light.voltage_pu("renewables") - light_again.voltage_pu("renewables")) < 1e-9
    assert heavy.voltage_pu("renewables") != light.voltage_pu("renewables")
