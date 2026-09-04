"""The plant-side actuation server.

This is what a real deployment's edge server would run: it owns the plant --
today `PolarMicrogrid`, tomorrow a Typhoon HIL rig behind the same interface --
and applies the safety projection to every command that arrives over the wire,
exactly as `GuardedController` does for an in-process controller. A remote
controller gets no less protection than a local one.
"""

from __future__ import annotations

from concurrent import futures

import grpc

from allotrope.rpc import allotrope_pb2 as pb
from allotrope.rpc import allotrope_pb2_grpc as pb_grpc
from allotrope.rpc.convert import (
    command_from_proto,
    observation_to_proto,
    safety_report_to_proto,
    telemetry_to_proto,
)
from allotrope.safety.projection import SafetyProjection
from allotrope.sim.plant import PolarMicrogrid


class ActuationServicer(pb_grpc.ActuationServicer):
    """Wraps a `PolarMicrogrid` and its safety projection behind gRPC."""

    def __init__(self, plant: PolarMicrogrid) -> None:
        self.plant = plant
        self.projection = SafetyProjection(plant.cfg)

    def Dispatch(self, request: pb.DispatchRequest, context) -> pb.DispatchResponse:
        if self.plant.done:
            context.abort(grpc.StatusCode.OUT_OF_RANGE, "plant has run past the end of its weather")

        proposed = command_from_proto(request)
        observation = self.plant.observe()
        safe, report = self.projection.project(proposed, observation, self.plant)
        telemetry = self.plant.step(safe)

        return pb.DispatchResponse(
            telemetry=telemetry_to_proto(telemetry),
            safety=safety_report_to_proto(report),
        )

    def Observe(self, request: pb.ObserveRequest, context) -> pb.Observation:
        return observation_to_proto(self.plant.observe())


def serve(plant: PolarMicrogrid, address: str = "localhost:0", max_workers: int = 4) -> grpc.Server:
    """Start a server and return it already running; the caller stops it.

    `address` with port 0 lets the OS assign a free port -- the normal choice
    for tests, which must never depend on a fixed port being free.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb_grpc.add_ActuationServicer_to_server(ActuationServicer(plant), server)
    bound_port = server.add_insecure_port(address)
    server.bound_address = f"localhost:{bound_port}"
    server.start()
    return server


__all__ = ["ActuationServicer", "serve"]
