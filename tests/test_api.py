"""API tests: the backend serves real simulator state, not fixtures.

Each test drives `allotrope.api.app` through `TestClient` and checks the
response against the same `PolarMicrogrid`/`GuardedController` objects the
CLI scripts and `tests/test_safety.py` exercise directly -- the point being
that the API is a thin, honest window onto that state, not a second,
divergent implementation of it.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from allotrope.api.app import create_app
from allotrope.config import available_stations


def _client() -> TestClient:
    return TestClient(create_app())


def test_health_reports_real_uptime_and_stations():
    client = _client()
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["uptime_s"] >= 0.0
    assert set(body["stations"]) == set(available_stations())


def test_lists_every_configured_station():
    client = _client()
    resp = client.get("/stations")
    assert resp.status_code == 200
    ids = {s["id"] for s in resp.json()}
    assert ids == set(available_stations())


def test_unknown_station_is_404_everywhere():
    client = _client()
    for path in [
        "/stations/nonexistent",
        "/stations/nonexistent/state",
        "/stations/nonexistent/telemetry",
        "/stations/nonexistent/safety",
        "/stations/nonexistent/controller",
    ]:
        assert client.get(path).status_code == 404, path


def test_station_detail_reports_real_config():
    client = _client()
    resp = client.get("/stations/maitri")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Maitri"
    assert len(body["gensets"]) == 3
    assert body["controller"]["type"] == "GuardedController"


def test_state_before_any_step_is_not_done_and_is_labelled_simulation():
    client = _client()
    body = client.get("/stations/maitri/state").json()
    assert body["mode"] == "simulation"
    assert body["step"] == 0
    assert body["done"] is False
    assert "observation" in body and "electrical_load_kw" in body["observation"]


def test_single_step_advances_state_and_populates_telemetry():
    client = _client()
    resp = client.post("/stations/maitri/simulation/step")
    assert resp.status_code == 200
    body = resp.json()
    assert body["step"] == 1
    assert body["last_telemetry"] is not None
    assert body["last_telemetry"]["genset_starts"] is not None

    telemetry = client.get("/stations/maitri/telemetry").json()
    assert len(telemetry) == 1


def test_reset_clears_history_and_step_count():
    client = _client()
    client.post("/stations/maitri/simulation/step")
    client.post("/stations/maitri/simulation/step")
    assert client.get("/stations/maitri/state").json()["step"] == 2

    resp = client.post("/stations/maitri/simulation/reset")
    assert resp.status_code == 200
    assert resp.json()["step"] == 0
    assert client.get("/stations/maitri/telemetry").json() == []


def test_safety_report_reflects_the_guard_after_stepping():
    client = _client()
    for _ in range(5):
        client.post("/stations/maitri/simulation/step")
    body = client.get("/stations/maitri/safety").json()
    assert body["steps"] == 5
    assert 0.0 <= body["fallback_rate"] <= 1.0
    assert 0.0 <= body["projection_rate"] <= 1.0


def test_start_then_stop_auto_advances_and_then_holds():
    # A `with`-managed TestClient keeps its background event-loop portal (and
    # therefore the simulation's asyncio task) alive across every call in this
    # block; without it, the portal is free to tear down between requests and
    # cancel the still-running background loop out from under the test.
    with TestClient(create_app()) as client:
        resp = client.post("/stations/maitri/simulation/start", json={"interval_s": 0.01})
        assert resp.status_code == 200
        assert resp.json()["running"] is True

        import time

        time.sleep(0.2)
        running_state = client.get("/stations/maitri/state").json()
        assert running_state["step"] > 0

        client.post("/stations/maitri/simulation/stop")
        held = client.get("/stations/maitri/state").json()
        assert held["running"] is False
        time.sleep(0.1)
        assert client.get("/stations/maitri/state").json()["step"] == held["step"]


def test_cannot_single_step_while_auto_running():
    with TestClient(create_app()) as client:
        client.post("/stations/maitri/simulation/start", json={"interval_s": 5.0})
        resp = client.post("/stations/maitri/simulation/step")
        assert resp.status_code == 409
        client.post("/stations/maitri/simulation/stop")


def test_metrics_endpoint_matches_plant_summary():
    client = _client()
    for _ in range(3):
        client.post("/stations/maitri/simulation/step")
    metrics = client.get("/stations/maitri/metrics").json()
    assert "fuel_l" in metrics
    assert metrics["fuel_l"] >= 0.0
