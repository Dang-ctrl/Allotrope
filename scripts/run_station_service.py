"""Run one station end to end: the plant, its guarded controller, MQTT telemetry.

    python scripts/run_station_service.py --station maitri --checkpoint runs/hybrid_maitri_.../checkpoint.pt

This is what a single edge server actually runs. There is no remote
command-injection path here -- see `docs/control-plane.md`'s "what's
explicitly not implemented" for why: computing a command needs the full
`PolarMicrogrid` object, so the controller and the plant it commands stay in
the same process. What this script exposes over the network is the gRPC
state distribution `allotrope.controlplane` already provides (run that
separately if a remote HMI needs to observe this process) and, here, MQTT
telemetry for the TimescaleDB bridge and Grafana to pick up.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from allotrope.config import load_station
from allotrope.control.baseline import EfficientRuleBased
from allotrope.mqtt.publisher import TelemetryPublisher
from allotrope.safety.fallback import GuardedController
from allotrope.safety.voltage import build_inverter_layer
from allotrope.sim.runner import build_plant


def _load_controller(checkpoint: str | None, cfg):
    if not checkpoint:
        print("no checkpoint given; running the efficient rule-based controller")
        return GuardedController(cfg, agent=EfficientRuleBased(cfg), inverter_layer=build_inverter_layer(cfg))

    from allotrope.agents.hybrid import HybridAgent
    from allotrope.evaluate import load_checkpoint

    dqn, sddpg, _ = load_checkpoint(Path(checkpoint))
    hybrid = HybridAgent(cfg, dqn, sddpg, deterministic=True)
    print(f"loaded trained agent from {checkpoint}")
    return GuardedController(cfg, agent=hybrid, inverter_layer=build_inverter_layer(cfg))


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
    guard = _load_controller(args.checkpoint, cfg)

    plant = build_plant(cfg, periods=args.periods, seed=args.seed)
    plant.reset()
    guard.reset()

    publisher = TelemetryPublisher(cfg.site.id, host=args.mqtt_host, port=args.mqtt_port)
    print(f"publishing telemetry to MQTT at {args.mqtt_host}:{args.mqtt_port}")

    try:
        while True:
            if plant.done:
                print("weather series exhausted; looping back to the start")
                plant.reset()
                guard.reset()

            observation = plant.observe()
            start = time.perf_counter()
            command = guard.act(observation, plant)
            telemetry = plant.step(command)
            latency_ms = (time.perf_counter() - start) * 1000.0

            record = {
                "genset_kw": telemetry["genset_kw"],
                "fuel_l": telemetry["fuel_l"],
                "black_carbon_mg": telemetry["black_carbon_mg"],
                "renewable_used_kw": telemetry["renewable_used_kw"],
                "curtailed_kw": telemetry["curtailed_kw"],
                "electrical_load_kw": telemetry["electrical_load_kw"],
                "melt_kw": telemetry["melt_kw"],
                "unserved_kw": telemetry["unserved_kw"],
                "critical_unserved_kw": telemetry["critical_unserved_kw"],
                "indoor_temp_c": telemetry["indoor_temp_c"],
                "air_temp_c": telemetry["air_temp_c"],
                "battery_soc": list(telemetry["battery_soc"]),
                "dispatch_latency_ms": latency_ms,
            }
            publisher.publish_telemetry(record)
            if guard.last_report is not None and guard.last_report.intervened:
                publisher.publish_safety_report(
                    {
                        "intervened": True,
                        "interventions": [i.value for i in guard.last_report.interventions],
                    }
                )

            time.sleep(args.interval_s)
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        publisher.close()


if __name__ == "__main__":
    main()
