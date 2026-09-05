"""Reads from the `telemetry` table for the web API.

Deliberately the mirror image of `allotrope.mqtt.timescale_bridge`: that
module writes rows, this one reads them back. Connections are opened
short-lived per query rather than pooled -- at two stations and a handful of
concurrent demo viewers this is simple and fast enough, and a pool would be
solving a problem this deployment does not have.
"""

from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

_COLUMNS = [
    "time",
    "station_id",
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
    "battery_soc_mean",
]

_HISTORY_SQL = f"""
SELECT {", ".join(_COLUMNS)} FROM telemetry
WHERE station_id = %s AND time >= now() - (%s || ' minutes')::interval
ORDER BY time ASC
LIMIT %s
"""

_LATEST_SQL = f"""
SELECT {", ".join(_COLUMNS)} FROM telemetry
WHERE station_id = %s
ORDER BY time DESC
LIMIT 1
"""


def fetch_history(dsn: str, station_id: str, minutes: int = 60, limit: int = 3600) -> list[dict]:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(_HISTORY_SQL, (station_id, minutes, limit))
            return cur.fetchall()


def fetch_latest(dsn: str, station_id: str) -> dict | None:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(_LATEST_SQL, (station_id,))
            return cur.fetchone()


__all__ = ["fetch_history", "fetch_latest"]
