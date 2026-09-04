"""A real, in-process MQTT broker for tests -- not a mock of the protocol.

Uses `amqtt`, a pure-Python MQTT broker, so `TelemetryPublisher` and
`TelemetrySubscriber` are exercised against an actual TCP MQTT server, on a
throwaway port, with no external service (no Docker, no installed mosquitto)
required. Production would point at a real broker; this proves the client code
speaks the protocol correctly, not that a mock agrees with itself.
"""

from __future__ import annotations

import asyncio
import threading
import time


class EmbeddedBroker:
    def __init__(self, port: int) -> None:
        self.port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop_event: asyncio.Event | None = None

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()

        def runner() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run())

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("embedded MQTT broker did not start in time")
        time.sleep(0.2)  # let the listener finish binding before clients connect

    async def _run(self) -> None:
        from amqtt.broker import Broker

        config = {
            "listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{self.port}"}},
            "sys_interval": 0,
            "auth": {"allow-anonymous": True},
        }
        broker = Broker(config)
        await broker.start()
        self._stop_event = asyncio.Event()
        self._ready.set()
        await self._stop_event.wait()
        await broker.shutdown()

    def stop(self) -> None:
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=5)


__all__ = ["EmbeddedBroker"]
