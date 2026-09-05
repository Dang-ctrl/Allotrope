"""The gRPC control-plane server: implements `ControlPlane` from allotrope.proto.

Wraps the same `allotrope.api.simulation.StationSimulation` objects the REST
API serves -- one process, one set of simulations, two transports. Nothing
here re-derives state independently; `_build_state` reads the same
`plant.observe()` and `GuardedController` attributes `allotrope.api.
simulation.StationSimulation.state()`/`.safety()` do.

Two things this project's own adversarial audit found and this module now
fixes:

- **F5, no authentication.** Every RPC now checks an `x-api-key` metadata
  entry against `self.token` (constant-time comparison). `serve()` reads
  `ALLOTROPE_CONTROLPLANE_TOKEN`, or generates and logs one if unset -- the
  same pattern `allotrope.api.app` uses, so there is never a silently-open
  default. This is still `add_insecure_port` (plaintext), not TLS/mTLS --
  the token stops an unauthenticated read, not eavesdropping on the wire,
  and that gap is recorded rather than implied away.
- **F6, `StreamState` worker-pool exhaustion.** A long-lived stream
  occupies one thread-pool worker for its entire connection lifetime; with
  the previous 8-worker pool, 8 concurrent streaming clients starved every
  other RPC, including `Heartbeat`. `_stream_slots`, a bounded semaphore
  acquired non-blockingly, now caps how many `StreamState` calls can be in
  flight at once (independent of the unary-RPC pool), and a client past
  that cap gets `RESOURCE_EXHAUSTED` immediately rather than queuing
  forever behind one that never disconnects.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
import threading
import time
from concurrent import futures
from typing import Iterator

import grpc

from allotrope.api.simulation import StationSimulation, build_simulation
from allotrope.config import available_stations
from allotrope.controlplane import allotrope_pb2 as pb2
from allotrope.controlplane import allotrope_pb2_grpc as pb2_grpc
from allotrope.experiment import git_commit
from allotrope.observability import configure_logging, log_event

STREAM_INTERVAL_S = 1.0
MAX_CONCURRENT_STREAMS = 4
"""Independent of the unary-RPC thread pool -- see the module docstring's F6."""

_logger = configure_logging()
_auth_logger = logging.getLogger("allotrope.controlplane")


class ControlPlaneServicer(pb2_grpc.ControlPlaneServicer):
    """Serves StationState and Heartbeat over gRPC. See allotrope.proto for the contract."""

    def __init__(
        self,
        stations: dict[str, StationSimulation] | None = None,
        token: str | None = None,
        max_concurrent_streams: int = MAX_CONCURRENT_STREAMS,
    ) -> None:
        self.stations = (
            stations if stations is not None else {sid: build_simulation(sid) for sid in available_stations()}
        )
        self._sequence: dict[str, int] = dict.fromkeys(self.stations, 0)
        self._started_at = time.monotonic()
        self.model_version = git_commit()

        self.token = token or os.environ.get("ALLOTROPE_CONTROLPLANE_TOKEN")
        if not self.token:
            self.token = secrets.token_urlsafe(32)
            _auth_logger.warning(
                "ALLOTROPE_CONTROLPLANE_TOKEN not set -- generated a token for this process "
                "only: %s (set ALLOTROPE_CONTROLPLANE_TOKEN to use a stable one across restarts)",
                self.token,
            )
        self._stream_slots = threading.BoundedSemaphore(max_concurrent_streams)

    # -- RPCs ---------------------------------------------------------------

    def GetState(self, request: pb2.StateRequest, context: grpc.ServicerContext) -> pb2.StationState:
        self._require_token(context)
        sim = self._require_station(request.station_id, context)
        return self._build_state(sim)

    def StreamState(
        self, request: pb2.StateRequest, context: grpc.ServicerContext
    ) -> Iterator[pb2.StationState]:
        self._require_token(context)
        sim = self._require_station(request.station_id, context)
        if not self._stream_slots.acquire(blocking=False):
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"already at the limit of {MAX_CONCURRENT_STREAMS} concurrent StreamState calls",
            )
            return
        try:
            while context.is_active():
                yield self._build_state(sim)
                time.sleep(STREAM_INTERVAL_S)
        finally:
            self._stream_slots.release()

    def Heartbeat(self, request: pb2.HeartbeatRequest, context: grpc.ServicerContext) -> pb2.HeartbeatResponse:
        self._require_token(context)
        return pb2.HeartbeatResponse(
            ok=True,
            uptime_s=time.monotonic() - self._started_at,
            model_version=self.model_version,
            stations=list(self.stations.keys()),
        )

    # -- internals ------------------------------------------------------------

    def _require_token(self, context: grpc.ServicerContext) -> None:
        metadata = dict(context.invocation_metadata())
        supplied = metadata.get("x-api-key", "")
        if not supplied or not hmac.compare_digest(supplied, self.token):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "missing or invalid x-api-key metadata")

    def _require_station(self, station_id: str, context: grpc.ServicerContext) -> StationSimulation:
        sim = self.stations.get(station_id)
        if sim is None:
            log_event(_logger, "controlplane.unknown_station", station_id=station_id)
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown station {station_id!r}")
            raise AssertionError("unreachable: context.abort raises")
        return sim

    def _build_state(self, sim: StationSimulation) -> pb2.StationState:
        self._sequence[sim.station_id] += 1
        obs = sim.plant.observe()
        guard = sim.controller
        stats = getattr(guard, "stats", None)
        last_report = getattr(guard, "last_report", None)
        last_fallback = getattr(guard, "last_fallback_reason", None)
        status = sim.controller_status()

        if sim.plant.done:
            quality = pb2.INVALID  # no more data: the run is over
        elif not sim.running:
            quality = pb2.STALE  # valid, but not currently advancing
        else:
            quality = pb2.GOOD

        return pb2.StationState(
            station_id=sim.station_id,
            sequence_number=self._sequence[sim.station_id],
            timestamp_unix_ms=int(time.time() * 1000),
            quality=quality,
            sim_step=sim.step_count,
            sim_n_steps=sim.plant.n_steps,
            sim_running=sim.running,
            electrical_load_kw=obs["electrical_load_kw"],
            critical_load_kw=obs["critical_load_kw"],
            pv_available_kw=obs["pv_available_kw"],
            wind_available_kw=obs["wind_available_kw"],
            genset_online=list(obs["genset_online"]),
            genset_power_kw=list(obs["genset_power_kw"]),
            battery_soc=list(obs["battery_soc"]),
            safety=pb2.SafetyStatus(
                intervened_last_step=bool(last_report.intervened) if last_report else False,
                interventions=[i.value for i in last_report.interventions] if last_report else [],
                fallback_reason=last_fallback.value if last_fallback else "",
                fallback_rate=getattr(stats, "fallback_rate", 0.0),
                projection_rate=getattr(stats, "projection_rate", 0.0),
                max_latency_ms=getattr(stats, "max_latency_ms", 0.0),
            ),
            controller=pb2.ControllerStatus(
                name=status["name"],
                type=status["type"],
                model_version=self.model_version,
            ),
        )


def serve(port: int = 50051, max_workers: int = 8, token: str | None = None) -> tuple[grpc.Server, int, str]:
    """Start the control-plane server; return it (already `.start()`ed), its
    bound port, and the token every RPC now requires as `x-api-key` metadata.

    Pass `port=0` to let the OS assign a free ephemeral port -- the actual
    port is only known after binding, hence returning it rather than
    trusting the caller's input back. The caller owns the returned server's
    lifecycle: call `.wait_for_termination()` to block, or
    `.stop(grace_period_s)` to shut it down (tests use the latter).
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = ControlPlaneServicer(token=token)
    pb2_grpc.add_ControlPlaneServicer_to_server(servicer, server)
    bound_port = server.add_insecure_port(f"[::]:{port}")
    server.start()
    log_event(_logger, "controlplane.started", port=bound_port)
    return server, bound_port, servicer.token


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()
    server, bound_port, _token = serve(port=args.port)
    print(f"control plane listening on :{bound_port} (see the log line above for the x-api-key token)")
    server.wait_for_termination()


if __name__ == "__main__":
    main()


__all__ = ["ControlPlaneServicer", "serve", "main"]
