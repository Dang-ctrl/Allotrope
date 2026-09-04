"""The TimescaleDB bridge, tested against a fake connection.

There is no live TimescaleDB in this environment, so what is tested is the
bridge's own logic: it builds the row it should for a given telemetry dict, it
commits after every write, and a malformed message is dropped rather than
taking the bridge down. Whether `psycopg` itself can reach a real database is
not this project's code to test.
"""

from __future__ import annotations

import pytest

from allotrope.mqtt.timescale_bridge import TimescaleBridge, telemetry_to_row


class FakeCursor:
    def __init__(self, log: list) -> None:
        self.log = log

    def execute(self, query, params) -> None:
        self.log.append((query, params))


class FakeConnection:
    def __init__(self) -> None:
        self.log: list = []
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.log)

    def commit(self) -> None:
        self.commits += 1


class FakeSubscriber:
    """Stands in for TelemetrySubscriber: just records the registered callback."""

    def __init__(self) -> None:
        self.callback = None

    def on_telemetry(self, callback) -> None:
        self.callback = callback


SAMPLE_TELEMETRY = {
    "genset_kw": 80.0,
    "fuel_l": 12.3,
    "black_carbon_mg": 500.0,
    "renewable_used_kw": 10.0,
    "curtailed_kw": 0.0,
    "electrical_load_kw": 90.0,
    "melt_kw": 5.0,
    "unserved_kw": 0.0,
    "critical_unserved_kw": 0.0,
    "indoor_temp_c": 20.0,
    "air_temp_c": -25.0,
    "battery_soc": [0.6, 0.7],
}


def test_telemetry_to_row_includes_the_station_and_all_fields():
    row = telemetry_to_row("maitri", SAMPLE_TELEMETRY)
    assert row[0] == "maitri"
    assert row[1] == 80.0  # genset_kw
    assert row[-1] == pytest.approx(0.65)  # mean of battery_soc


def test_missing_fields_become_null_rather_than_raising():
    row = telemetry_to_row("maitri", {"fuel_l": 1.0})
    assert row[0] == "maitri"
    assert None in row


def test_a_write_inserts_and_commits():
    sub = FakeSubscriber()
    conn = FakeConnection()
    bridge = TimescaleBridge(sub, conn)

    sub.callback("maitri", SAMPLE_TELEMETRY)

    assert len(conn.log) == 1
    assert conn.commits == 1
    assert bridge.stats.written == 1
    assert bridge.stats.failed == 0
    query, params = conn.log[0]
    assert "INSERT INTO telemetry" in query
    assert params[0] == "maitri"


def test_a_failing_connection_is_counted_not_raised():
    class BrokenConnection(FakeConnection):
        def cursor(self):
            raise RuntimeError("connection lost")

    sub = FakeSubscriber()
    bridge = TimescaleBridge(sub, BrokenConnection())

    sub.callback("maitri", SAMPLE_TELEMETRY)  # must not raise

    assert bridge.stats.failed == 1
    assert bridge.stats.written == 0


def test_multiple_writes_accumulate_stats():
    sub = FakeSubscriber()
    bridge = TimescaleBridge(sub, FakeConnection())
    for _ in range(5):
        sub.callback("bharati", SAMPLE_TELEMETRY)
    assert bridge.stats.written == 5
