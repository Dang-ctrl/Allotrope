"""The gRPC actuation interface: correctness over the wire, and a latency reading.

Latency here is informative, not a pass/fail gate. A loopback call inside a
shared test-runner sandbox is not the deployment the 10 ms control budget is
written for -- that number is a design target for a station's own LAN, enforced
in practice by `GuardedController` treating a slow reply as a failed one. These
tests report the number and assert it is finite and positive; they do not
assert it is under any particular threshold, because a flaky timing assertion
here would be testing the test runner, not the interface.
"""

from __future__ import annotations

import pytest

from allotrope.config import load_station
from allotrope.rpc.client import ActuationClient
from allotrope.rpc.server import serve
from allotrope.sim.plant import DispatchCommand
from allotrope.sim.runner import build_plant


@pytest.fixture
def server_and_client():
    cfg = load_station("maitri")
    plant = build_plant(cfg, start="2026-06-01", periods=48, seed=0)
    plant.reset()
    server = serve(plant)
    client = ActuationClient(server.bound_address)
    yield plant, server, client
    client.close()
    server.stop(grace=None)


def test_observe_matches_the_plants_own_observation(server_and_client):
    plant, server, client = server_and_client
    local = plant.observe()
    remote = client.observe()

    assert remote.electrical_load_kw == pytest.approx(local["electrical_load_kw"])
    assert remote.critical_load_kw == pytest.approx(local["critical_load_kw"])
    assert list(remote.genset_online) == list(local["genset_online"])


def test_dispatch_advances_the_plant_and_returns_its_telemetry(server_and_client):
    plant, server, client = server_and_client
    cfg = plant.cfg
    command = DispatchCommand(
        genset_on=tuple(True for _ in cfg.gensets),
        genset_setpoint_kw=tuple(g.rated_kw for g in cfg.gensets),
        battery_kw=tuple(0.0 for _ in cfg.storage),
        snow_melt_kw=0.0,
    )
    result = client.dispatch(command)

    assert plant.state.step_index == 1
    assert result.telemetry.fuel_l > 0.0
    assert result.latency_ms > 0.0
    import math

    assert math.isfinite(result.latency_ms)


def test_the_server_applies_the_safety_projection_over_the_wire(server_and_client):
    """A remote controller gets no less protection than an in-process one."""
    plant, server, client = server_and_client
    cfg = plant.cfg
    all_off = DispatchCommand(
        genset_on=tuple(False for _ in cfg.gensets),
        genset_setpoint_kw=tuple(0.0 for _ in cfg.gensets),
        battery_kw=tuple(0.0 for _ in cfg.storage),
        snow_melt_kw=0.0,
    )
    result = client.dispatch(all_off)

    assert result.safety.intervened
    assert result.telemetry.critical_unserved_kw == pytest.approx(0.0, abs=1e-9)
    assert any(g for g in result.telemetry.genset_online), (
        "the safety layer should have started a set the client refused to"
    )


def test_malformed_wire_values_are_still_sanitised(server_and_client):
    plant, server, client = server_and_client
    cfg = plant.cfg
    command = DispatchCommand(
        genset_on=tuple(True for _ in cfg.gensets),
        genset_setpoint_kw=tuple(float("nan") for _ in cfg.gensets),
        battery_kw=tuple(1e9 for _ in cfg.storage),
        snow_melt_kw=1e9,
    )
    result = client.dispatch(command)
    assert result.safety.intervened
    assert result.telemetry.critical_unserved_kw == pytest.approx(0.0, abs=1e-9)


def test_repeated_dispatch_calls_advance_the_plant_step_by_step(server_and_client):
    plant, server, client = server_and_client
    cfg = plant.cfg
    command = DispatchCommand(
        genset_on=(True,) + tuple(False for _ in cfg.gensets[1:]),
        genset_setpoint_kw=(cfg.gensets[0].rated_kw,) + tuple(0.0 for _ in cfg.gensets[1:]),
        battery_kw=tuple(0.0 for _ in cfg.storage),
        snow_melt_kw=0.0,
    )
    for i in range(5):
        client.dispatch(command)
    assert plant.state.step_index == 5


def test_dispatch_past_the_end_of_weather_is_reported_as_an_error(server_and_client):
    import grpc

    plant, server, client = server_and_client
    cfg = plant.cfg
    command = DispatchCommand(
        genset_on=tuple(True for _ in cfg.gensets),
        genset_setpoint_kw=tuple(g.rated_kw for g in cfg.gensets),
        battery_kw=tuple(0.0 for _ in cfg.storage),
        snow_melt_kw=0.0,
    )
    for _ in range(48):
        client.dispatch(command)

    with pytest.raises(grpc.RpcError):
        client.dispatch(command)
