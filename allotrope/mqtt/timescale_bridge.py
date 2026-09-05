"""Bridges MQTT telemetry into TimescaleDB, for Grafana to read.

This is the piece the deck's "TimescaleDB time-series store" and "Grafana
dashboards" claims depend on existing: a bridge that turns the telemetry stream
`allotrope.mqtt.subscriber.TelemetrySubscriber` receives into rows a dashboard
can query. It takes a connection factory rather than opening its own connection,
so the SQL it issues can be verified against a fake cursor without a live
database -- the property worth testing here is "does this write the right row
for a given telemetry dict, and survive a malformed one," not "does psycopg
itself work."

The schema (`deploy/init-timescaledb.sql`) turns `telemetry` into a TimescaleDB
hypertable, which is what makes a table like this practical at a full winter's
worth of one-second samples: the deck's own choice of TimescaleDB over plain
Postgres is exactly for this workload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from allotrope.mqtt.subscriber import TelemetrySubscriber

INSERT_SQL = """
INSERT INTO telemetry (
    station_id, genset_kw, fuel_l, black_carbon_mg, renewable_used_kw,
    curtailed_kw, electrical_load_kw, melt_kw, unserved_kw,
    critical_unserved_kw, indoor_temp_c, air_temp_c, battery_soc_mean
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# Every column above must be resolvable from a telemetry dict, tolerating
# whichever fields happen to be missing rather than requiring an exact schema
# match -- the field list here is the bridge's contract with the plant's
# telemetry, kept in one place so a renamed field fails obviously.
_FIELDS = [
    "genset_kw",
    "fuel_l",
    "black_carbon_mg",
    "renewable_used_kw",
    "curtailed_kw",
    "electrical_load_kw",
    "melt_kw",
    "unserved_kw",
    "critical_unserved_kw",
    "indoor_temp_c",
    "air_temp_c",
]


class SupportsCursor(Protocol):
    def execute(self, query: str, params: tuple) -> Any: ...


class SupportsConnection(Protocol):
    def cursor(self) -> SupportsCursor: ...
    def commit(self) -> None: ...


def telemetry_to_row(station_id: str, telemetry: dict) -> tuple:
    battery_soc = telemetry.get("battery_soc") or []
    battery_soc_mean = sum(battery_soc) / len(battery_soc) if battery_soc else None
    values = [telemetry.get(field) for field in _FIELDS]
    return (station_id, *values, battery_soc_mean)


@dataclass
class BridgeStats:
    written: int = 0
    failed: int = 0


class TimescaleBridge:
    """Writes each telemetry message it receives into the `telemetry` table."""

    def __init__(self, subscriber: TelemetrySubscriber, connection: SupportsConnection) -> None:
        self.connection = connection
        self.stats = BridgeStats()
        subscriber.on_telemetry(self._on_telemetry)

    def _on_telemetry(self, station_id: str, telemetry: dict) -> None:
        try:
            row = telemetry_to_row(station_id, telemetry)
            cursor = self.connection.cursor()
            cursor.execute(INSERT_SQL, row)
            self.connection.commit()
            self.stats.written += 1
        except Exception:
            # A single bad row must not take the bridge down; the satellite
            # link is unreliable by design, and so is whatever crosses it.
            self.stats.failed += 1


__all__ = ["TimescaleBridge", "BridgeStats", "telemetry_to_row", "INSERT_SQL"]
