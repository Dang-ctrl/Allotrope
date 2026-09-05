"""The gRPC control-plane server: implements `ControlPlane` from allotrope.proto.

Wraps the same `allotrope.api.simulation.StationSimulation` objects the REST
API serves -- one process, one set of simulations, two transports. Nothing
here re-derives state independently; `_build_state` reads the same
`plant.observe()` and `GuardedController` attributes `allotrope.api.
simulation.StationSimulation.state()`/`.safety()` do.
"""

from __future__ import annotations

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
_logger = configure_logging()


class ControlPlaneServicer(pb2_grpc.ControlPlaneServicer):
    """Serves StationState and Heartbeat over gRPC. See allotrope.proto for the contract."""

    def __init__(self, stations: dict[str, StationSimulation] | None = None) -> None:
        self.stations = (
            stations if stations is not None else {sid: build_simulation(sid) for sid in available_stations()}
        )
        self._sequence: dict[str, int] = dict.fromkeys(self.stations, 0)
        self._started_at = time.monotonic()
        self.model_version = git_commit()

    # -- RPCs ---------------------------------------------------------------

    def GetState(self, request: pb2.StateRequest, context: grpc.ServicerContext) -> pb2.StationState:
        sim = self._require_station(request.station_id, context)
        return self._build_state(sim)

    def StreamState(
        self, request: pb2.StateRequest, context: grpc.ServicerContext
    ) -> Iterator[pb2.StationState]:
        sim = self._require_station(request.station_id, context)
        while context.is_active():
            yield self._build_state(sim)
            time.sleep(STREAM_INTERVAL_S)

    def Heartbeat(self, request: pb2.HeartbeatRequest, context: grpc.ServicerContext) -> pb2.HeartbeatResponse:
        return pb2.HeartbeatResponse(
            ok=True,
            uptime_s=time.monotonic() - self._started_at,
            model_version=self.model_version,
            stations=list(self.stations.keys()),
        )

    # -- internals ------------------------------------------------------------

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


def serve(port: int = 50051, max_workers: int = 8) -> tuple[grpc.Server, int]:
    """Start the control-plane server; return it (already `.start()`ed) and its bound port.

    Pass `port=0` to let the OS assign a free ephemeral port -- the actual
    port is only known after binding, hence returning it rather than
    trusting the caller's input back. The caller owns the returned server's
    lifecycle: call `.wait_for_termination()` to block, or
    `.stop(grace_period_s)` to shut it down (tests use the latter).
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb2_grpc.add_ControlPlaneServicer_to_server(ControlPlaneServicer(), server)
    bound_port = server.add_insecure_port(f"[::]:{port}")
    server.start()
    log_event(_logger, "controlplane.started", port=bound_port)
    return server, bound_port


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()
    server, bound_port = serve(port=args.port)
    print(f"control plane listening on :{bound_port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()


__all__ = ["ControlPlaneServicer", "serve", "main"]
