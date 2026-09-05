-- Schema for the telemetry hypertable TimescaleBridge writes into.
-- Applied automatically by the timescaledb container on first start
-- (mounted into /docker-entrypoint-initdb.d/).

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS telemetry (
    time                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    station_id            TEXT NOT NULL,
    genset_kw             DOUBLE PRECISION,
    fuel_l                DOUBLE PRECISION,
    black_carbon_mg       DOUBLE PRECISION,
    renewable_used_kw     DOUBLE PRECISION,
    curtailed_kw          DOUBLE PRECISION,
    electrical_load_kw    DOUBLE PRECISION,
    melt_kw               DOUBLE PRECISION,
    unserved_kw           DOUBLE PRECISION,
    critical_unserved_kw  DOUBLE PRECISION,
    indoor_temp_c         DOUBLE PRECISION,
    air_temp_c            DOUBLE PRECISION,
    battery_soc_mean      DOUBLE PRECISION
);

-- Turns `telemetry` into a hypertable, partitioned by time. This is the
-- specific reason the deck names TimescaleDB rather than plain Postgres: a
-- full winter-over at one-second resolution is tens of millions of rows, and
-- hypertable chunking is what keeps both writes and Grafana's range queries
-- fast at that scale.
SELECT create_hypertable('telemetry', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS telemetry_station_time_idx
    ON telemetry (station_id, time DESC);
