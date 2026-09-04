"""The controller-side actuation client.

A `Controller` (rule-based, or a `HybridAgent`) that wants to act over the wire
uses this instead of calling `plant.step` directly. Latency is measured on every
call and reported, not merely hoped for: the deck's <10 ms control budget is
enforced by `GuardedController` treating a slow answer as a failed one, and this
client is what supplies the number that decision is made on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import grpc

from allotrope.rpc import allotrope_pb2 as pb
from allotrope.rpc import allotrope_pb2_grpc as pb_grpc
from allotrope.rpc.convert import command_to_proto
from allotrope.sim.plant import DispatchCommand


@dataclass
class DispatchResult:
    telemetry: pb.Telemetry
    safety: pb.SafetyReport
    latency_ms: float


class ActuationClient:
    """A thin, timed wrapper around the generated gRPC stub."""

    def __init__(self, address: str, timeout_s: float = 1.0) -> None:
        self.address = address
        self.timeout_s = timeout_s
        self._channel = grpc.insecure_channel(address)
        self._stub = pb_grpc.ActuationStub(self._channel)

    def dispatch(self, command: DispatchCommand) -> DispatchResult:
        start = time.perf_counter()
        response = self._stub.Dispatch(command_to_proto(command), timeout=self.timeout_s)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return DispatchResult(telemetry=response.telemetry, safety=response.safety, latency_ms=latency_ms)

    def observe(self) -> pb.Observation:
        return self._stub.Observe(pb.ObserveRequest(), timeout=self.timeout_s)

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "ActuationClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["ActuationClient", "DispatchResult"]
