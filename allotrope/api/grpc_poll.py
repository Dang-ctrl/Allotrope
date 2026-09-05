"""Polls a station's gRPC `Observe` endpoint for the web API's live feed.

`Observe` only ever returns `plant.observe()` (see `allotrope.rpc.server`) --
it never advances the plant, unlike `Dispatch`. That is what makes it safe for
a bystander like this poller to call on its own schedule: the station service
already drives `Dispatch` itself, once per its own control loop, and a second
caller of `Dispatch` would double-step the plant.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from allotrope.rpc.client import ActuationClient
from allotrope.rpc.convert import observation_to_dict

ObservationCallback = Callable[[str, dict], Awaitable[None] | None]


class ObservationPoller:
    """Polls one station's `Observe` RPC on a fixed interval."""

    def __init__(self, station_id: str, address: str, interval_s: float = 1.5) -> None:
        self.station_id = station_id
        self.interval_s = interval_s
        self._client = ActuationClient(address)
        self._callbacks: list[ObservationCallback] = []
        self._task: asyncio.Task | None = None

    def on_observation(self, callback: ObservationCallback) -> None:
        self._callbacks.append(callback)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            try:
                obs = await asyncio.to_thread(self._client.observe)
                data = observation_to_dict(obs)
                for callback in self._callbacks:
                    result = callback(self.station_id, data)
                    if asyncio.iscoroutine(result):
                        await result
            except Exception:
                # A station that hasn't started its gRPC server yet, or a
                # transient network hiccup, must not take the poller down --
                # it just tries again next tick.
                pass
            await asyncio.sleep(self.interval_s)

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
        self._client.close()


__all__ = ["ObservationPoller"]
