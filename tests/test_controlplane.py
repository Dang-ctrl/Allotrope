"""The gRPC control plane: a real server on a real (ephemeral) port, a real
client -- no in-memory fakes standing in for either half of the wire."""

from __future__ import annotations

import time

import grpc
import pytest

from allotrope.controlplane import allotrope_pb2 as pb2
from allotrope.controlplane import allotrope_pb2_grpc as pb2_grpc
from allotrope.controlplane.server import ControlPlaneServicer, serve


@pytest.fixture()
def running_server():
    server, port = serve(port=0)  # port 0: OS assigns an ephemeral free port
    try:
        yield port
    finally:
        server.stop(None)


@pytest.fixture()
def client(running_server):
    channel = grpc.insecure_channel(f"localhost:{running_server}")
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        yield pb2_grpc.ControlPlaneStub(channel)
    finally:
        channel.close()


def test_heartbeat_reports_real_uptime_and_stations(client):
    resp = client.Heartbeat(pb2.HeartbeatRequest())
    assert resp.ok is True
    assert resp.uptime_s >= 0.0
    assert set(resp.stations) == {"maitri", "bharati"}
    assert len(resp.model_version) > 0


def test_get_state_returns_real_observation_values(client):
    state = client.GetState(pb2.StateRequest(station_id="maitri"))
    assert state.station_id == "maitri"
    assert state.sequence_number >= 1
    assert state.electrical_load_kw > 0.0
    assert len(state.genset_online) == 3
    assert len(state.battery_soc) == 2
    assert state.controller.type == "GuardedController"


def test_get_state_reports_stale_quality_when_not_running(client):
    state = client.GetState(pb2.StateRequest(station_id="maitri"))
    assert state.quality == pb2.STALE
    assert state.sim_running is False


def test_unknown_station_is_not_found_not_a_default_state(client):
    with pytest.raises(grpc.RpcError) as exc_info:
        client.GetState(pb2.StateRequest(station_id="nonexistent"))
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


def test_sequence_number_advances_on_each_call(client):
    a = client.GetState(pb2.StateRequest(station_id="maitri"))
    b = client.GetState(pb2.StateRequest(station_id="maitri"))
    assert b.sequence_number > a.sequence_number


def test_stream_state_yields_multiple_states_with_advancing_sequence(client):
    seen = []
    for state in client.StreamState(pb2.StateRequest(station_id="maitri")):
        seen.append(state.sequence_number)
        if len(seen) >= 3:
            break
    assert seen == sorted(seen)
    assert len(set(seen)) == 3


def test_stream_state_on_unknown_station_is_not_found(client):
    with pytest.raises(grpc.RpcError) as exc_info:
        list(client.StreamState(pb2.StateRequest(station_id="nonexistent")))
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


def test_client_can_reconnect_after_cancelling_a_stream(client):
    """No per-client session state should leak between stream calls."""
    first = client.StreamState(pb2.StateRequest(station_id="maitri"))
    next(first)
    first.cancel()

    second = client.StreamState(pb2.StateRequest(station_id="maitri"))
    state = next(second)
    assert state.station_id == "maitri"
    second.cancel()


def test_a_short_client_deadline_is_honoured_as_deadline_exceeded(client):
    """Malformed/slow-path handling: a client that sets an already-expired
    deadline gets DEADLINE_EXCEEDED, not a hang or a silent default. A
    negative timeout is unambiguously already in the past, which is what
    makes this deterministic rather than a race against how fast the
    loopback call happens to complete (a `timeout=0` was observed to still
    occasionally complete before expiry checking under pytest specifically)."""
    with pytest.raises(grpc.RpcError) as exc_info:
        client.GetState(pb2.StateRequest(station_id="maitri"), timeout=-1)
    assert exc_info.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED


def test_state_calls_are_fast_enough_for_the_projects_own_control_budget(client):
    """README.md's architecture names a <10 ms control-path target. This
    measures loopback GetState latency only -- not network conditions,
    which this test explicitly does not and cannot claim to represent."""
    n = 20
    start = time.perf_counter()
    for _ in range(n):
        client.GetState(pb2.StateRequest(station_id="maitri"))
    mean_ms = (time.perf_counter() - start) * 1000.0 / n
    assert mean_ms < 10.0, f"mean loopback GetState latency {mean_ms:.2f} ms"


def test_reports_invalid_quality_once_the_run_is_out_of_data():
    """A dedicated short-run station (24 steps, not a full 8760-step year --
    only whether the run is *over* matters here), isolated from the shared
    `client` fixture's simulations so driving it to completion doesn't
    affect other tests."""
    from allotrope.api.simulation import StationSimulation, default_controller
    from allotrope.config import load_station

    cfg = load_station("maitri")
    sim = StationSimulation(
        station_id="maitri", cfg=cfg, controller=default_controller(cfg), periods=24
    )
    servicer = ControlPlaneServicer(stations={"maitri": sim})
    for _ in range(sim.plant.n_steps):
        sim.step()
    assert sim.plant.done

    class _Ctx:
        def abort(self, *a, **k):
            raise AssertionError("should not abort for a known station")

    state = servicer.GetState(pb2.StateRequest(station_id="maitri"), _Ctx())
    assert state.quality == pb2.INVALID
