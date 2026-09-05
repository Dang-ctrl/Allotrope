"""The live WebSocket feed and a health endpoint for on-stage troubleshooting."""

from __future__ import annotations

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/stations/{station_id}")
async def station_feed(websocket: WebSocket, station_id: str) -> None:
    live = websocket.app.state.live
    if not live.has_station(station_id):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    state = live.get(station_id)
    state.sockets.add(websocket)
    try:
        await websocket.send_json(live.snapshot(station_id))
        while True:
            # This feed is server -> browser only; block here so a client
            # disconnect is detected without polling.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        state.sockets.discard(websocket)


@router.get("/api/health")
def health(request: Request) -> dict:
    live = request.app.state.live
    stations = live.all_stations()
    return {
        "status": "ok",
        "mqtt": {sid: s.mqtt_connected for sid, s in stations.items()},
        "grpc": {sid: s.grpc_connected for sid, s in stations.items()},
    }


__all__ = ["router"]
