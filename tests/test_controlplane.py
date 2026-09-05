"""The gRPC control plane: a real server on a real (ephemeral) port, a real
client -- no in-memory fakes standing in for either half of the wire."""

from __future__ import annotations

import time

import grpc
import pytest

from allotrope.controlplane import allotrope_pb2 as pb2
from allotrope.controlplane import allotrope_pb2_grpc as pb2_grpc
from allotrope.controlplane.server import MAX_CONCURRENT_STREAMS, ControlPlaneServicer, serve

TEST_TOKEN = "test-only-token-not-a-secret"


class _AddMetadataInterceptor(grpc.UnaryUnaryClientInterceptor, grpc.UnaryStreamClientInterceptor):
    """Attaches `x-api-key` to every outgoing call. `grpc.composite_channel_credentials`
    needs TLS channel credentials as its base and so can't attach call
    credentials to a plaintext test channel; an interceptor is the
    plaintext-compatible equivalent for driving the real auth code path in
    tests rather than skipping it."""

    def __init__(self, token: str) -> None:
        self._metadata = (("x-api-key", token),)

    def _add(self, client_call_details):
        metadata = list(client_call_details.metadata or [])
        metadata.extend(self._metadata)
        return client_call_details._replace(metadata=metadata)

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return continuation(self._add(client_call_details), request)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        return continuation(self._add(client_call_details), request)


@pytest.fixture()
def running_server():
    server, port, token = serve(port=0, token=TEST_TOKEN)  # port 0: OS assigns an ephemeral free port
    try:
        yield port
    finally:
        server.stop(None)


@pytest.fixture()
def client(running_server):
    raw_channel = grpc.insecure_channel(f"localhost:{running_server}")
    channel = grpc.intercept_channel(raw_channel, _AddMetadataInterceptor(TEST_TOKEN))
    try:
        grpc.channel_ready_future(raw_channel).result(timeout=5)
        yield pb2_grpc.ControlPlaneStub(channel)
    finally:
        raw_channel.close()


@pytest.fixture()
def unauthenticated_client(running_server):
    """The same server, but a channel that never attaches x-api-key."""
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
    servicer = ControlPlaneServicer(stations={"maitri": sim}, token=TEST_TOKEN)
    for _ in range(sim.plant.n_steps):
        sim.step()
    assert sim.plant.done

    class _Ctx:
        def invocation_metadata(self):
            return (("x-api-key", TEST_TOKEN),)

        def abort(self, *a, **k):
            raise AssertionError("should not abort for a known station")

    state = servicer.GetState(pb2.StateRequest(station_id="maitri"), _Ctx())
    assert state.quality == pb2.INVALID


# -- authentication (F5) and stream-pool exhaustion (F6), from the adversarial audit --


def test_get_state_without_a_token_is_unauthenticated(unauthenticated_client):
    with pytest.raises(grpc.RpcError) as exc_info:
        unauthenticated_client.GetState(pb2.StateRequest(station_id="maitri"))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_heartbeat_without_a_token_is_unauthenticated(unauthenticated_client):
    with pytest.raises(grpc.RpcError) as exc_info:
        unauthenticated_client.Heartbeat(pb2.HeartbeatRequest())
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_stream_state_without_a_token_is_unauthenticated(unauthenticated_client):
    with pytest.raises(grpc.RpcError) as exc_info:
        next(unauthenticated_client.StreamState(pb2.StateRequest(station_id="maitri")))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_a_wrong_token_is_also_rejected(running_server):
    channel = grpc.insecure_channel(f"localhost:{running_server}")
    channel = grpc.intercept_channel(channel, _AddMetadataInterceptor("wrong-token"))
    stub = pb2_grpc.ControlPlaneStub(channel)
    with pytest.raises(grpc.RpcError) as exc_info:
        stub.GetState(pb2.StateRequest(station_id="maitri"))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_an_unset_token_is_generated_not_left_open():
    """No ALLOTROPE_CONTROLPLANE_TOKEN and no explicit token= must still
    result in a real, working credential requirement."""
    server, port, token = serve(port=0)
    try:
        assert token
        channel = grpc.insecure_channel(f"localhost:{port}")
        try:
            grpc.channel_ready_future(channel).result(timeout=5)
            stub = pb2_grpc.ControlPlaneStub(channel)
            with pytest.raises(grpc.RpcError) as exc_info:
                stub.Heartbeat(pb2.HeartbeatRequest())
            assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED

            authed_channel = grpc.intercept_channel(channel, _AddMetadataInterceptor(token))
            authed_stub = pb2_grpc.ControlPlaneStub(authed_channel)
            assert authed_stub.Heartbeat(pb2.HeartbeatRequest()).ok is True
        finally:
            channel.close()
    finally:
        server.stop(None)


def test_a_stream_flood_cannot_starve_unary_rpcs(running_server):
    """The concrete F6 DoS: previously, MAX_CONCURRENT_STREAMS+ clients each
    opening StreamState (and never disconnecting) would eventually exhaust
    the entire gRPC thread pool, starving even Heartbeat. Now, calls past
    the cap get RESOURCE_EXHAUSTED immediately, and a plain unary RPC keeps
    working throughout."""
    channel = grpc.insecure_channel(f"localhost:{running_server}")
    channel = grpc.intercept_channel(channel, _AddMetadataInterceptor(TEST_TOKEN))
    stub = pb2_grpc.ControlPlaneStub(channel)

    streams = []
    try:
        for _ in range(MAX_CONCURRENT_STREAMS):
            call = stub.StreamState(pb2.StateRequest(station_id="maitri"))
            next(call)  # block until the server has actually started this stream
            streams.append(call)

        with pytest.raises(grpc.RpcError) as exc_info:
            overflow = stub.StreamState(pb2.StateRequest(station_id="maitri"))
            next(overflow)
        assert exc_info.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED

        # The unary RPC path is untouched by the stream flood.
        assert stub.Heartbeat(pb2.HeartbeatRequest()).ok is True
    finally:
        for call in streams:
            call.cancel()
        channel.close()
