"""Static station configuration, projected from YAML."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from allotrope.api.config_view import station_to_dict
from allotrope.config import ConfigError, available_stations, load_station

router = APIRouter()


@router.get("/api/stations")
def list_stations() -> list[dict]:
    return [station_to_dict(load_station(sid)) for sid in available_stations()]


@router.get("/api/stations/{station_id}")
def get_station(station_id: str) -> dict:
    try:
        return station_to_dict(load_station(station_id))
    except ConfigError:
        raise HTTPException(status_code=404, detail=f"unknown station {station_id!r}")


__all__ = ["router"]
