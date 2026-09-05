"""Holds the latest live state per station and fans it out to WebSockets.

MQTT callbacks (`TelemetrySubscriber`, `SafetySubscriber`) fire on paho's own
background thread, never on the FastAPI event loop -- so `broadcast` schedules
itself onto that loop via `run_coroutine_threadsafe` rather than assuming it
can safely touch a WebSocket directly from another thread.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

from fastapi import WebSocket

_SAFETY_HISTORY = 50


@dataclass
class StationLiveState:
    latest_telemetry: dict | None = None
    latest_observation: dict | None = None
    safety_events: deque = field(default_factory=lambda: deque(maxlen=_SAFETY_HISTORY))
    mqtt_connected: bool = False
    grpc_connected: bool = False
    sockets: set = field(default_factory=set)


class LiveState:
    """One `StationLiveState` per station id, plus the broadcast plumbing."""

    def __init__(self, station_ids: list[str]) -> None:
        self._stations = {sid: StationLiveState() for sid in station_ids}
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def get(self, station_id: str) -> StationLiveState:
        return self._stations[station_id]

    def has_station(self, station_id: str) -> bool:
        return station_id in self._stations

    def all_stations(self) -> dict[str, StationLiveState]:
        return self._stations

    def snapshot(self, station_id: str) -> dict:
        s = self.get(station_id)
        return {
            "type": "snapshot",
            "telemetry": s.latest_telemetry,
            "observation": s.latest_observation,
            "safety_events": list(s.safety_events),
        }

    def on_telemetry(self, station_id: str, telemetry: dict) -> None:
        s = self.get(station_id)
        s.latest_telemetry = telemetry
        s.mqtt_connected = True
        self._schedule_broadcast(station_id, {"type": "telemetry", "data": telemetry, "ts": time.time()})

    def on_safety(self, station_id: str, report: dict) -> None:
        s = self.get(station_id)
        event = {**report, "ts": time.time()}
        s.safety_events.append(event)
        self._schedule_broadcast(station_id, {"type": "safety", "data": report, "ts": event["ts"]})

    async def on_observation(self, station_id: str, observation: dict) -> None:
        s = self.get(station_id)
        s.latest_observation = observation
        s.grpc_connected = True
        await self._broadcast(station_id, {"type": "observation", "data": observation, "ts": time.time()})

    def _schedule_broadcast(self, station_id: str, message: dict) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(station_id, message), self._loop)

    async def _broadcast(self, station_id: str, message: dict) -> None:
        s = self.get(station_id)
        dead: set[WebSocket] = set()
        for ws in s.sockets:
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        s.sockets -= dead


__all__ = ["LiveState", "StationLiveState"]
