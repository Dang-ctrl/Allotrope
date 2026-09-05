"""Run the web API: station config, telemetry history, and a live feed.

    python scripts/run_api.py --host 0.0.0.0 --port 8000 \\
        --mqtt-host mosquitto \\
        --db-dsn postgresql://allotrope:allotrope@timescaledb:5432/allotrope \\
        --grpc-targets maitri=station-maitri:50051 bharati=station-bharati:50051
"""

from __future__ import annotations

import argparse

import uvicorn

from allotrope.api.app import ApiConfig, _parse_grpc_targets, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--db-dsn", default="postgresql://allotrope:allotrope@localhost:5432/allotrope")
    parser.add_argument(
        "--grpc-targets",
        nargs="+",
        default=[],
        help="station_id=host:port pairs, e.g. maitri=station-maitri:50051",
    )
    args = parser.parse_args()

    config = ApiConfig(
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        db_dsn=args.db_dsn,
        grpc_targets=_parse_grpc_targets(" ".join(args.grpc_targets)),
    )
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
