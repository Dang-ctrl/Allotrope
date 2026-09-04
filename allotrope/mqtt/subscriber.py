"""Subscribes to station telemetry, for a monitoring bridge or the operator HMI.

This is the last stage before Grafana: in a full deployment, a bridge built on
this subscriber would write each decoded record into TimescaleDB, and Grafana
would query that store, matching the deck's "TimescaleDB time-series store" and
"Grafana dashboards" pairing. Standing up TimescaleDB and Grafana themselves is
infrastructure, not this project's Python -- see `deploy/docker-compose.yml`
for how they are wired together at deploy time; this class is the code that
would feed them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import paho.mqtt.client as mqtt

from allotrope.mqtt.codec import decode_telemetry
from allotrope.mqtt.topics import telemetry_topic

TelemetryCallback = Callable[[str, dict], None]


@dataclass
class SubscriberStats:
    received: int = 0
    dropped_malformed: int = 0


class TelemetrySubscriber:
    """Subscribes to one or more stations' telemetry topics.

    A malformed payload -- truncated by the satellite link, or from a
    mismatched software version -- is dropped and counted, never raised. A
    monitoring bridge that crashed on one bad packet would be worse than useless
    on a link this unreliable.
    """

    def __init__(self, station_ids: list[str], host: str, port: int = 1883) -> None:
        self.station_ids = station_ids
        self.stats = SubscriberStats()
        self._callbacks: list[TelemetryCallback] = []
        self._topic_to_station = {telemetry_topic(sid): sid for sid in station_ids}

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_message = self._on_message
        self._client.connect(host, port, keepalive=60)
        for topic in self._topic_to_station:
            self._client.subscribe(topic)
        self._client.loop_start()

    def on_telemetry(self, callback: TelemetryCallback) -> None:
        self._callbacks.append(callback)

    def _on_message(self, client, userdata, message) -> None:
        station_id = self._topic_to_station.get(message.topic)
        if station_id is None:
            return
        telemetry = decode_telemetry(message.payload)
        if telemetry is None:
            self.stats.dropped_malformed += 1
            return
        self.stats.received += 1
        for callback in self._callbacks:
            callback(station_id, telemetry)

    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def __enter__(self) -> "TelemetrySubscriber":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["TelemetrySubscriber", "SubscriberStats", "TelemetryCallback"]
