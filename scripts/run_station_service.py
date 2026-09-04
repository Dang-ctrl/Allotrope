"""Run one station end to end: plant, gRPC actuation, MQTT telemetry.

    python scripts/run_station_service.py --station maitri --checkpoint checkpoints/maitri.pt

This is what a single edge server actually runs: the plant (today simulated,
tomorrow a Typhoon HIL rig behind the same `allotrope.rpc` interface) served
over gRPC, a controller acting as its own client against that interface exactly
as a remote HMI or a separate control process would, and every step's telemetry
published to MQTT for the TimescaleDB bridge and Grafana to pick up.

Running the "brain" and the plant's gRPC server in one process is a deployment
choice appropriate to a single ruggedized edge server, not a limitation of the
interface -- `allotrope.rpc.client.ActuationClient` talks to any address, local
or not.
"""

from __future__ import annotations

import argparse
import time

from allotrope.agents.checkpoint import load as load_agent
from allotrope.config import load_station
from allotrope.control.baseline import EfficientRuleBased
from allotrope.mqtt.publisher import TelemetryPublisher
from allotrope.rpc.client import ActuationClient
from allotrope.rpc.server import serve
from allotrope.sim.runner import build_plant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="maitri")
    parser.add_argument("--checkpoint", default=None, help="trained HybridAgent checkpoint")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--interval-s", type=float, default=1.0, help="wall-clock time per simulated step")
    parser.add_argument("--periods", type=int, default=8760)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_station(args.station)
    if args.checkpoint:
        controller = load_agent(args.checkpoint, cfg)
        print(f"loaded trained agent from {args.checkpoint}")
    else:
        controller = EfficientRuleBased(cfg)
        print("no checkpoint given; running the efficient rule-based controller")

    plant = build_plant(cfg, periods=args.periods, seed=args.seed)
    plant.reset()
    server = serve(plant, address="0.0.0.0:50051")
    print(f"actuation server listening on {server.bound_address}")

    client = ActuationClient(server.bound_address)
    publisher = TelemetryPublisher(cfg.site.id, host=args.mqtt_host, port=args.mqtt_port)
    print(f"publishing telemetry to MQTT at {args.mqtt_host}:{args.mqtt_port}")

    try:
        while True:
            if plant.done:
                print("weather series exhausted; looping back to the start")
                plant.reset()

            observation = plant.observe()
            command = controller.act(observation, plant)
            result = client.dispatch(command)

            record = {
                "genset_kw": result.telemetry.genset_kw,
                "fuel_l": result.telemetry.fuel_l,
                "black_carbon_mg": result.telemetry.black_carbon_mg,
                "renewable_used_kw": result.telemetry.renewable_used_kw,
                "curtailed_kw": result.telemetry.curtailed_kw,
                "electrical_load_kw": result.telemetry.electrical_load_kw,
                "melt_kw": result.telemetry.melt_kw,
                "unserved_kw": result.telemetry.unserved_kw,
                "critical_unserved_kw": result.telemetry.critical_unserved_kw,
                "indoor_temp_c": result.telemetry.indoor_temp_c,
                "air_temp_c": result.telemetry.air_temp_c,
                "battery_soc": list(result.telemetry.battery_soc),
                "dispatch_latency_ms": result.latency_ms,
            }
            publisher.publish_telemetry(record)
            if result.safety.intervened:
                publisher.publish_safety_report(
                    {
                        "intervened": True,
                        "interventions": list(result.safety.interventions),
                    }
                )

            time.sleep(args.interval_s)
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        publisher.close()
        client.close()
        server.stop(grace=2)


if __name__ == "__main__":
    main()
