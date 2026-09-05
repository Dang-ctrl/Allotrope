"""The FastAPI app factory: wires MQTT subscribers and gRPC pollers into `LiveState`.

Everything that touches the network is started here, on FastAPI's own
`lifespan`, and torn down the same way -- there is no other place in this
package that opens a socket.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from allotrope.api.grpc_poll import ObservationPoller
from allotrope.api.live import LiveState
from allotrope.api.mqtt_safety import SafetySubscriber
from allotrope.api.routes.live import router as live_router
from allotrope.api.routes.stations import router as stations_router
from allotrope.api.routes.telemetry import router as telemetry_router
from allotrope.config import available_stations
from allotrope.mqtt.subscriber import TelemetrySubscriber


@dataclass
class ApiConfig:
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    db_dsn: str = "postgresql://allotrope:allotrope@localhost:5432/allotrope"
    grpc_targets: dict[str, str] | None = None  # station_id -> "host:port"


def _parse_grpc_targets(raw: str) -> dict[str, str]:
    """Parses "maitri=host:port bharati=host:port" into a dict."""
    targets: dict[str, str] = {}
    for pair in raw.split():
        station_id, _, address = pair.partition("=")
        if station_id and address:
            targets[station_id] = address
    return targets


def create_app(config: ApiConfig | None = None) -> FastAPI:
    config = config or ApiConfig()
    station_ids = available_stations()
    grpc_targets = config.grpc_targets or {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        live = app.state.live
        live.bind_loop(asyncio.get_running_loop())

        telemetry_sub = TelemetrySubscriber(station_ids, host=config.mqtt_host, port=config.mqtt_port)
        telemetry_sub.on_telemetry(live.on_telemetry)

        safety_sub = SafetySubscriber(station_ids, host=config.mqtt_host, port=config.mqtt_port)
        safety_sub.on_safety(live.on_safety)

        pollers = []
        for station_id in station_ids:
            address = grpc_targets.get(station_id)
            if address is None:
                continue
            poller = ObservationPoller(station_id, address)
            poller.on_observation(live.on_observation)
            poller.start()
            pollers.append(poller)

        yield

        telemetry_sub.close()
        safety_sub.close()
        for poller in pollers:
            poller.stop()

    app = FastAPI(title="Allotrope API", lifespan=lifespan)
    app.state.live = LiveState(station_ids)
    app.state.db_dsn = config.db_dsn

    app.include_router(stations_router)
    app.include_router(telemetry_router)
    app.include_router(live_router)

    return app


__all__ = ["create_app", "ApiConfig", "_parse_grpc_targets"]
