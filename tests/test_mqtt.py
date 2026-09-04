"""The MQTT telemetry link, against a real embedded broker."""

from __future__ import annotations

import time

import pytest

from allotrope.mqtt.codec import decode_telemetry, encode
from allotrope.mqtt.publisher import TelemetryPublisher
from allotrope.mqtt.subscriber import TelemetrySubscriber
from allotrope.mqtt.topics import model_update_topic, safety_topic, telemetry_topic
from tests.mqtt_broker import EmbeddedBroker

TEST_PORT = 18831


@pytest.fixture(scope="module")
def broker():
    b = EmbeddedBroker(TEST_PORT)
    b.start()
    yield b
    b.stop()


def _wait_until(predicate, timeout=3.0, interval=0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# -- topics -------------------------------------------------------------------


def test_topics_are_namespaced_per_station():
    assert telemetry_topic("maitri") != telemetry_topic("bharati")
    assert telemetry_topic("maitri") != safety_topic("maitri")
    assert "federated" in model_update_topic("maitri")


# -- codec ----------------------------------------------------------------


def test_encode_decode_round_trips_a_plain_dict():
    payload = {"fuel_l": 12.5, "genset_online": [True, False, True]}
    assert decode_telemetry(encode(payload)) == payload


def test_decode_rejects_garbage_without_raising():
    assert decode_telemetry(b"\xff\xfe not json at all") is None
    assert decode_telemetry(b"[1, 2, 3]") is None  # valid JSON, wrong shape
    assert decode_telemetry(b"") is None


def test_encode_handles_a_dataclass():
    from dataclasses import dataclass

    @dataclass
    class Sample:
        fuel_l: float
        online: bool

    decoded = decode_telemetry(encode(Sample(fuel_l=1.5, online=True)))
    assert decoded == {"fuel_l": 1.5, "online": True}


# -- publisher / subscriber, against the real broker --------------------------


def test_published_telemetry_reaches_the_subscriber(broker):
    received = []
    sub = TelemetrySubscriber(["maitri"], host="127.0.0.1", port=TEST_PORT)
    sub.on_telemetry(lambda station, t: received.append((station, t)))
    time.sleep(0.3)

    pub = TelemetryPublisher("maitri", host="127.0.0.1", port=TEST_PORT)
    pub.publish_telemetry({"fuel_l": 42.0, "genset_kw": 100.0})

    assert _wait_until(lambda: len(received) == 1)
    assert received[0] == ("maitri", {"fuel_l": 42.0, "genset_kw": 100.0})

    pub.close()
    sub.close()


def test_subscriber_only_hears_its_own_stations(broker):
    received = []
    sub = TelemetrySubscriber(["maitri"], host="127.0.0.1", port=TEST_PORT)
    sub.on_telemetry(lambda station, t: received.append(station))
    time.sleep(0.3)

    other = TelemetryPublisher("bharati", host="127.0.0.1", port=TEST_PORT)
    other.publish_telemetry({"fuel_l": 1.0})
    time.sleep(0.5)

    assert received == []
    other.close()
    sub.close()


def test_multiple_callbacks_all_fire(broker):
    seen_a, seen_b = [], []
    sub = TelemetrySubscriber(["maitri"], host="127.0.0.1", port=TEST_PORT)
    sub.on_telemetry(lambda s, t: seen_a.append(t))
    sub.on_telemetry(lambda s, t: seen_b.append(t))
    time.sleep(0.3)

    pub = TelemetryPublisher("maitri", host="127.0.0.1", port=TEST_PORT)
    pub.publish_telemetry({"x": 1})

    assert _wait_until(lambda: seen_a and seen_b)
    pub.close()
    sub.close()


def test_a_corrupted_payload_is_dropped_not_raised(broker):
    """The subscriber must survive a payload the satellite link mangled."""
    sub = TelemetrySubscriber(["maitri"], host="127.0.0.1", port=TEST_PORT)
    sub.on_telemetry(lambda s, t: None)
    time.sleep(0.3)

    import paho.mqtt.client as mqtt

    raw = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    raw.connect("127.0.0.1", TEST_PORT, keepalive=60)
    raw.loop_start()
    raw.publish(telemetry_topic("maitri"), b"not valid json{{{")
    time.sleep(0.5)

    assert sub.stats.dropped_malformed >= 1
    raw.loop_stop()
    raw.disconnect()
    sub.close()


def test_safety_reports_publish_on_their_own_topic(broker):
    received = []
    import paho.mqtt.client as mqtt

    watcher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    watcher.on_message = lambda c, u, m: received.append(m.payload)
    watcher.connect("127.0.0.1", TEST_PORT, keepalive=60)
    watcher.subscribe(safety_topic("maitri"))
    watcher.loop_start()
    time.sleep(0.3)

    pub = TelemetryPublisher("maitri", host="127.0.0.1", port=TEST_PORT)
    pub.publish_safety_report({"intervened": True, "interventions": ["blocked_stop"]})

    assert _wait_until(lambda: len(received) == 1)
    assert decode_telemetry(received[0])["intervened"] is True

    pub.close()
    watcher.loop_stop()
    watcher.disconnect()
