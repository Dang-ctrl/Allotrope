"""Subscribe to every station's telemetry and write it into TimescaleDB.

    python scripts/run_timescale_bridge.py --stations maitri bharati
"""

from __future__ import annotations

import argparse
import time

import psycopg

from allotrope.mqtt.subscriber import TelemetrySubscriber
from allotrope.mqtt.timescale_bridge import TimescaleBridge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", nargs="+", default=["maitri", "bharati"])
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--db-dsn", default="postgresql://allotrope:allotrope@localhost:5432/allotrope")
    args = parser.parse_args()

    connection = psycopg.connect(args.db_dsn, autocommit=False)
    subscriber = TelemetrySubscriber(args.stations, host=args.mqtt_host, port=args.mqtt_port)
    bridge = TimescaleBridge(subscriber, connection)

    print(f"bridging {args.stations} -> {args.db_dsn}")
    try:
        while True:
            time.sleep(5.0)
            print(f"written={bridge.stats.written} failed={bridge.stats.failed}")
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        subscriber.close()
        connection.close()


if __name__ == "__main__":
    main()
