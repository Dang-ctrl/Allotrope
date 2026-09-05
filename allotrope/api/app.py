"""The backend API: real simulator state over HTTP, nothing invented.

This serves exactly one thing honestly: `allotrope.api.simulation`'s
in-process `PolarMicrogrid` + `GuardedController` instances, one per station.
There is no physical link to Maitri or Bharati, so every response is
simulation output, never presented as anything else -- see
`StationSimulation.state()`'s `"mode": "simulation"` field, which every
consumer (including the frontend this API exists for) should surface rather
than hide.

Endpoints implemented (the ones with real backing state; see docs/api.md for
what is deliberately not here yet and why):

    GET  /stations
    GET  /stations/{id}
    GET  /stations/{id}/state
    GET  /stations/{id}/telemetry
    GET  /stations/{id}/safety
    GET  /stations/{id}/controller
    POST /stations/{id}/simulation/start   (requires X-API-Key)
    POST /stations/{id}/simulation/stop    (requires X-API-Key)
    POST /stations/{id}/simulation/reset   (requires X-API-Key)
    POST /stations/{id}/simulation/step    (requires X-API-Key)

Authentication: the four `POST` endpoints above are the closest thing this
system has to an actuator surface -- they control the simulation loop a
"live" deployment of this API would be observing -- and are the ones this
project's own adversarial audit (F4) flagged as reachable without any
credential at all. They now require an `X-API-Key` header matching
`app.state.api_key`. Read-only `GET` endpoints stay open: this is a
single-tenant demo/simulation backend with no per-user data to protect,
and gating every read behind the same key would just push the secret into
the frontend bundle -- making the frontend the security boundary, which
this project treats as a standing rule to avoid rather than a convenience
to take. `ALLOTROPE_API_KEY` sets the key explicitly (for a real
deployment); if unset, `create_app()` generates one and logs it once at
startup, the same pattern Jupyter's notebook server uses -- so nothing is
ever hardcoded, and there is no unauthenticated default the way there was
before this change.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from allotrope.config import available_stations, load_station
from allotrope.api.simulation import StationSimulation, build_simulation

DEFAULT_STEP_INTERVAL_S = 0.25
"""Wall-clock seconds between steps in an auto-running simulation loop."""

MAX_TELEMETRY_LAST = 10_000
"""Upper bound on `?last=` for the telemetry endpoint (F14): without one, a
single request can force serialisation of the entire in-memory telemetry
buffer, a cheap and repeatable memory/CPU amplification."""

DEFAULT_RATE_LIMIT_REQUESTS = 120
DEFAULT_RATE_LIMIT_WINDOW_S = 10.0
"""Found in this project's own adversarial audit: no endpoint had any
concurrency or rate limit at all, so a client could flood any of them --
particularly `/simulation/step`, which does real per-request CPU work.
120 requests / 10s per client is generous for a single operator or the
frontend's poll loop, and small enough to blunt a naive flood."""

_logger = logging.getLogger("allotrope.api")


class RateLimiter:
    """A plain sliding-window counter per client IP, in-process.

    Not a substitute for a real edge/infrastructure rate limiter (an
    actual volumetric flood needs blocking upstream of this process, not
    inside it -- this only protects one process's own CPU/memory from a
    naive per-client flood, which is the gap this project's audit actually
    found: zero limiting of any kind). State is per-`RateLimiter` instance
    (one per `create_app()` call), so tests get isolated state rather than
    sharing a global counter across every app built in the same process.
    """

    def __init__(self, max_requests: int, window_s: float) -> None:
        self.max_requests = max_requests
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client_key: str) -> bool:
        now = time.monotonic()
        hits = self._hits[client_key]
        while hits and now - hits[0] > self.window_s:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        return True


def _too_many_requests_response() -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})


class SimulationManager:
    """Owns one `StationSimulation` per known station and its background loop task."""

    def __init__(self) -> None:
        self.stations: dict[str, StationSimulation] = {
            station_id: build_simulation(station_id) for station_id in available_stations()
        }
        self._tasks: dict[str, asyncio.Task] = {}

    def get(self, station_id: str) -> StationSimulation:
        sim = self.stations.get(station_id)
        if sim is None:
            raise HTTPException(status_code=404, detail=f"unknown station {station_id!r}")
        return sim

    async def start(self, station_id: str, interval_s: float) -> None:
        sim = self.get(station_id)
        if sim.running:
            return
        sim.running = True

        async def _run() -> None:
            try:
                while sim.running:
                    record = await asyncio.to_thread(sim.step)
                    if record is None:
                        sim.running = False
                        break
                    await asyncio.sleep(interval_s)
            finally:
                sim.running = False

        self._tasks[station_id] = asyncio.create_task(_run())

    async def stop(self, station_id: str) -> None:
        sim = self.get(station_id)
        sim.running = False
        task = self._tasks.pop(station_id, None)
        if task is not None:
            task.cancel()

    async def reset(self, station_id: str) -> None:
        await self.stop(station_id)
        self.get(station_id).reset()


class StartRequest(BaseModel):
    interval_s: float = Field(default=DEFAULT_STEP_INTERVAL_S, gt=0.0, le=60.0)


def create_app(
    api_key: str | None = None,
    rate_limit_requests: int = DEFAULT_RATE_LIMIT_REQUESTS,
    rate_limit_window_s: float = DEFAULT_RATE_LIMIT_WINDOW_S,
) -> FastAPI:
    """Build the app. `api_key` is exposed as a parameter (rather than only
    read from the environment) so tests can construct an app with a known
    key without setting process-wide environment state; the rate-limit
    parameters exist so tests can use a small window instead of waiting
    out the real one."""
    app = FastAPI(
        title="Allotrope",
        description=(
            "Simulation state for the Allotrope polar-microgrid controller. "
            "All data is synthetic-simulation output -- see README.md 'On the data'."
        ),
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    manager = SimulationManager()
    app.state.manager = manager
    app.state.started_at = time.monotonic()
    rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window_s)

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        # /health excluded: an orchestrator's own liveness probe hitting
        # it repeatedly must never be the thing that trips a client's limit.
        if request.url.path != "/health":
            client_key = request.client.host if request.client else "unknown"
            if not rate_limiter.allow(client_key):
                return _too_many_requests_response()
        return await call_next(request)

    app.state.api_key = api_key or os.environ.get("ALLOTROPE_API_KEY")
    if not app.state.api_key:
        app.state.api_key = secrets.token_urlsafe(32)
        _logger.warning(
            "ALLOTROPE_API_KEY not set -- generated a key for this process only: %s "
            "(set ALLOTROPE_API_KEY to use a stable one across restarts)",
            app.state.api_key,
        )

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        """Guards the simulation-control endpoints (F4 from the adversarial
        audit) -- the closest thing this API has to an actuator surface.
        `hmac.compare_digest` avoids leaking the key length/prefix via a
        timing side channel."""
        if not x_api_key or not hmac.compare_digest(x_api_key, app.state.api_key):
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "uptime_s": round(time.monotonic() - app.state.started_at, 3),
            "stations": list(manager.stations.keys()),
        }

    @app.get("/stations")
    def list_stations() -> list[dict[str, Any]]:
        return [
            {
                "id": sid,
                "name": load_station(sid).site.name,
                "running": sim.running,
                "step": sim.step_count,
                "n_steps": sim.plant.n_steps,
            }
            for sid, sim in manager.stations.items()
        ]

    @app.get("/stations/{station_id}")
    def get_station(station_id: str) -> dict[str, Any]:
        sim = manager.get(station_id)
        cfg = sim.cfg
        return {
            "id": station_id,
            "name": cfg.site.name,
            "latitude_deg": cfg.site.latitude_deg,
            "longitude_deg": cfg.site.longitude_deg,
            "gensets": [{"id": g.id, "rated_kw": g.rated_kw} for g in cfg.gensets],
            "storage": [{"id": s.id, "capacity_kwh": s.capacity_kwh} for s in cfg.storage],
            "controller": sim.controller_status(),
        }

    @app.get("/stations/{station_id}/state")
    def get_state(station_id: str) -> dict[str, Any]:
        return manager.get(station_id).state()

    @app.get("/stations/{station_id}/telemetry")
    def get_telemetry(station_id: str, last: int | None = None) -> list[dict[str, Any]]:
        if last is not None:
            last = min(last, MAX_TELEMETRY_LAST)
        return manager.get(station_id).telemetry(last_n=last)

    @app.get("/stations/{station_id}/metrics")
    def get_metrics(station_id: str) -> dict[str, float]:
        return manager.get(station_id).summary()

    @app.get("/stations/{station_id}/safety")
    def get_safety(station_id: str) -> dict[str, Any]:
        return manager.get(station_id).safety()

    @app.get("/stations/{station_id}/controller")
    def get_controller(station_id: str) -> dict[str, Any]:
        return manager.get(station_id).controller_status()

    @app.post("/stations/{station_id}/simulation/start", dependencies=[Depends(require_api_key)])
    async def start_simulation(station_id: str, body: StartRequest = StartRequest()) -> dict[str, Any]:
        manager.get(station_id)  # 404s before scheduling a task for a bad id
        await manager.start(station_id, body.interval_s)
        return manager.get(station_id).state()

    @app.post("/stations/{station_id}/simulation/stop", dependencies=[Depends(require_api_key)])
    async def stop_simulation(station_id: str) -> dict[str, Any]:
        await manager.stop(station_id)
        return manager.get(station_id).state()

    @app.post("/stations/{station_id}/simulation/reset", dependencies=[Depends(require_api_key)])
    async def reset_simulation(station_id: str) -> dict[str, Any]:
        await manager.reset(station_id)
        return manager.get(station_id).state()

    @app.post("/stations/{station_id}/simulation/step", dependencies=[Depends(require_api_key)])
    def step_simulation(station_id: str) -> dict[str, Any]:
        sim = manager.get(station_id)
        if sim.running:
            raise HTTPException(
                status_code=409, detail="simulation is auto-running; stop it before single-stepping"
            )
        record = sim.step()
        if record is None:
            raise HTTPException(status_code=410, detail="simulation has reached the end of its data")
        return sim.state() | {"last_telemetry": manager.get(station_id).latest_telemetry()}

    return app


app = create_app()

__all__ = ["app", "create_app", "SimulationManager"]
