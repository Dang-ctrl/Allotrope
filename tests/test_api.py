"""The web API layer: config projection, gRPC->dict conversion, and the
safety subscriber (against a real embedded broker, same as test_mqtt.py).

`allotrope.api.db` is not tested here for the same reason
`allotrope.mqtt.timescale_bridge` is only tested against a fake connection in
test_timescale_bridge.py: there is no live TimescaleDB in this environment.
`allotrope.api.app`/`grpc_poll`/`live` wire real network clients together and
are exercised by the docker-compose stack itself, not by pytest.
"""

from __future__ import annotations

import time

import pytest

from allotrope.api.app import _parse_grpc_targets
from allotrope.api.config_view import station_to_dict
from allotrope.api.mqtt_safety import SafetySubscriber
from allotrope.config import load_station
from allotrope.rpc import allotrope_pb2 as pb
from allotrope.rpc.convert import observation_to_dict
from tests.mqtt_broker import EmbeddedBroker

TEST_PORT = 18841


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


# -- config_view ---------------------------------------------------------


def test_station_to_dict_covers_both_stations():
    for station_id in ("maitri", "bharati"):
        d = station_to_dict(load_station(station_id))
        assert d["id"] == station_id
        assert len(d["gensets"]) == 3
        assert len(d["storage"]) == 2
        assert d["total_genset_kw"] == sum(g["rated_kw"] for g in d["gensets"])


def test_station_to_dict_is_json_serializable():
    import json

    json.dumps(station_to_dict(load_station("maitri")))


# -- convert.observation_to_dict -------------------------------------------


def test_observation_to_dict_round_trips_repeated_fields():
    obs = pb.Observation(
        electrical_load_kw=100.0,
        critical_load_kw=45.0,
        genset_online=[True, False, True],
        genset_power_kw=[80.0, 0.0, 40.0],
        genset_deposit=[0.1, 0.3, 0.9],
        battery_soc=[0.5, 0.7],
    )
    d = observation_to_dict(obs)
    assert d["genset_online"] == [True, False, True]
    assert d["genset_power_kw"] == [80.0, 0.0, 40.0]
    assert d["genset_deposit"] == [0.1, 0.3, 0.9]
    assert d["battery_soc"] == [0.5, 0.7]
    assert d["electrical_load_kw"] == 100.0


# -- app._parse_grpc_targets ------------------------------------------------


def test_parse_grpc_targets():
    targets = _parse_grpc_targets("maitri=station-maitri:50051 bharati=station-bharati:50051")
    assert targets == {"maitri": "station-maitri:50051", "bharati": "station-bharati:50051"}


def test_parse_grpc_targets_empty_string():
    assert _parse_grpc_targets("") == {}


# -- SafetySubscriber, against the real broker ------------------------------


def test_safety_subscriber_receives_a_published_report(broker):
    received = []
    sub = SafetySubscriber(["maitri"], host="127.0.0.1", port=TEST_PORT)
    sub.on_safety(lambda station, report: received.append((station, report)))
    time.sleep(0.3)

    from allotrope.mqtt.publisher import TelemetryPublisher

    pub = TelemetryPublisher("maitri", host="127.0.0.1", port=TEST_PORT)
    pub.publish_safety_report({"intervened": True, "interventions": ["blocked_stop"]})

    assert _wait_until(lambda: len(received) == 1)
    assert received[0] == ("maitri", {"intervened": True, "interventions": ["blocked_stop"]})

    pub.close()
    sub.close()


def test_safety_subscriber_resubscribes_after_a_broker_restart():
    """Same regression as
    test_mqtt.test_subscriber_resubscribes_after_a_broker_restart, for the
    safety topic's own subscriber. Uses its own broker instance (restarted
    mid-test), not the module-scoped fixture the other tests share."""
    from allotrope.mqtt.publisher import TelemetryPublisher

    port = TEST_PORT + 1
    local_broker = EmbeddedBroker(port)
    local_broker.start()
    try:
        received = []
        sub = SafetySubscriber(["maitri"], host="127.0.0.1", port=port)
        sub.on_safety(lambda station, report: received.append(report))
        time.sleep(0.3)

        pub = TelemetryPublisher("maitri", host="127.0.0.1", port=port)
        pub.publish_safety_report({"intervened": True, "interventions": ["a"]})
        assert _wait_until(lambda: len(received) == 1)
        pub.close()

        local_broker.stop()
        local_broker = EmbeddedBroker(port)  # a new process, same port
        local_broker.start()

        assert _wait_until(lambda: sub._client.is_connected(), timeout=10.0)

        pub = TelemetryPublisher("maitri", host="127.0.0.1", port=port)
        pub.publish_safety_report({"intervened": True, "interventions": ["b"]})
        assert _wait_until(lambda: len(received) == 2, timeout=10.0)
        assert received[1] == {"intervened": True, "interventions": ["b"]}

        pub.close()
        sub.close()
    finally:
        local_broker.stop()


def test_safety_subscriber_drops_malformed_payloads(broker):
    sub = SafetySubscriber(["maitri"], host="127.0.0.1", port=TEST_PORT)
    sub.on_safety(lambda s, r: None)
    time.sleep(0.3)

    import paho.mqtt.client as mqtt

    from allotrope.mqtt.topics import safety_topic

    raw = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    raw.connect("127.0.0.1", TEST_PORT, keepalive=60)
    raw.loop_start()
    raw.publish(safety_topic("maitri"), b"not valid json{{{")
    time.sleep(0.5)

    assert sub.stats.dropped_malformed >= 1
    raw.loop_stop()
    raw.disconnect()
    sub.close()
