"""Historical telemetry, read from TimescaleDB."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from allotrope.api import db

router = APIRouter()


@router.get("/api/stations/{station_id}/telemetry/history")
def telemetry_history(station_id: str, request: Request, minutes: int = 60, limit: int = 3600) -> dict:
    rows = db.fetch_history(request.app.state.db_dsn, station_id, minutes=minutes, limit=limit)
    return {"station_id": station_id, "rows": rows}


@router.get("/api/stations/{station_id}/telemetry/latest")
def telemetry_latest(station_id: str, request: Request, response: Response):
    row = db.fetch_latest(request.app.state.db_dsn, station_id)
    if row is None:
        response.status_code = 204
        return None
    return row


__all__ = ["router"]
