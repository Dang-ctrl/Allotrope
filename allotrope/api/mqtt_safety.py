"""Subscribes to station safety reports, for the web API's live feed.

A structural mirror of `allotrope.mqtt.subscriber.TelemetrySubscriber` --
same paho pattern, same defensive decoding -- built on the safety topic
instead. Nothing in the rest of the codebase subscribes to that topic today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import paho.mqtt.client as mqtt

from allotrope.mqtt.codec import decode_telemetry
from allotrope.mqtt.topics import safety_topic

SafetyCallback = Callable[[str, dict], None]


@dataclass
class SafetySubscriberStats:
    received: int = 0
    dropped_malformed: int = 0


class SafetySubscriber:
    """Subscribes to one or more stations' safety topics.

    Safety reports are event-driven -- published only when `intervened` is
    true, never as a heartbeat -- so a long quiet period here means the
    projection layer has not had to act, not that the link is down.
    """

    def __init__(self, station_ids: list[str], host: str, port: int = 1883) -> None:
        self.station_ids = station_ids
        self.stats = SafetySubscriberStats()
        self._callbacks: list[SafetyCallback] = []
        self._topic_to_station = {safety_topic(sid): sid for sid in station_ids}

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.connect(host, port, keepalive=60)
        self._client.loop_start()

    def on_safety(self, callback: SafetyCallback) -> None:
        self._callbacks.append(callback)

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        # Re-subscribing here, not only in __init__, is what makes this
        # survive a broker restart -- see the identical comment in
        # allotrope.mqtt.subscriber.TelemetrySubscriber.
        for topic in self._topic_to_station:
            client.subscribe(topic)

    def _on_message(self, client, userdata, message) -> None:
        station_id = self._topic_to_station.get(message.topic)
        if station_id is None:
            return
        report = decode_telemetry(message.payload)
        if report is None:
            self.stats.dropped_malformed += 1
            return
        self.stats.received += 1
        for callback in self._callbacks:
            callback(station_id, report)

    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def __enter__(self) -> "SafetySubscriber":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["SafetySubscriber", "SafetySubscriberStats", "SafetyCallback"]
