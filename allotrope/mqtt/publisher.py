"""Publishes plant telemetry and safety reports to the station's MQTT broker."""

from __future__ import annotations

import paho.mqtt.client as mqtt

from allotrope.mqtt.codec import encode
from allotrope.mqtt.topics import safety_topic, telemetry_topic


class TelemetryPublisher:
    def __init__(self, station_id: str, host: str, port: int = 1883, qos: int = 0) -> None:
        self.station_id = station_id
        self.qos = qos
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.connect(host, port, keepalive=60)
        self._client.loop_start()

    def publish_telemetry(self, telemetry: dict) -> None:
        self._client.publish(
            telemetry_topic(self.station_id), encode(telemetry), qos=self.qos
        )

    def publish_safety_report(self, report: dict) -> None:
        self._client.publish(safety_topic(self.station_id), encode(report), qos=self.qos)

    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def __enter__(self) -> "TelemetryPublisher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["TelemetryPublisher"]
