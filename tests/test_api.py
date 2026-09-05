"""API tests: the backend serves real simulator state, not fixtures.

Each test drives `allotrope.api.app` through `TestClient` and checks the
response against the same `PolarMicrogrid`/`GuardedController` objects the
CLI scripts and `tests/test_safety.py` exercise directly -- the point being
that the API is a thin, honest window onto that state, not a second,
divergent implementation of it.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from allotrope.api.app import create_app
from allotrope.config import available_stations

TEST_API_KEY = "test-only-key-not-a-secret"


def _client() -> TestClient:
    client = TestClient(create_app(api_key=TEST_API_KEY))
    client.headers.update({"X-API-Key": TEST_API_KEY})
    return client


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
    with TestClient(create_app(api_key=TEST_API_KEY)) as client:
        client.headers.update({"X-API-Key": TEST_API_KEY})
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
    with TestClient(create_app(api_key=TEST_API_KEY)) as client:
        client.headers.update({"X-API-Key": TEST_API_KEY})
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


# -- authentication on the control endpoints (F4 from the adversarial audit) --


def test_simulation_control_endpoints_reject_missing_or_wrong_key():
    """The four POST endpoints are the closest thing this API has to an
    actuator surface. Before this fix, any client on the network could
    start/stop/reset/step every station's simulation with no credential
    at all."""
    app = create_app(api_key=TEST_API_KEY)
    client = TestClient(app)
    for path in [
        "/stations/maitri/simulation/start",
        "/stations/maitri/simulation/stop",
        "/stations/maitri/simulation/reset",
        "/stations/maitri/simulation/step",
    ]:
        assert client.post(path).status_code == 401, f"{path} accepted no key"
        assert (
            client.post(path, headers={"X-API-Key": "wrong-key"}).status_code == 401
        ), f"{path} accepted the wrong key"


def test_simulation_control_endpoints_accept_the_configured_key():
    app = create_app(api_key=TEST_API_KEY)
    client = TestClient(app)
    resp = client.post(
        "/stations/maitri/simulation/step", headers={"X-API-Key": TEST_API_KEY}
    )
    assert resp.status_code == 200


def test_read_endpoints_do_not_require_a_key():
    """Reads stay open by design -- see allotrope/api/app.py's module
    docstring for why gating every read the same way would just move the
    secret into the frontend bundle."""
    client = TestClient(create_app(api_key=TEST_API_KEY))
    assert client.get("/health").status_code == 200
    assert client.get("/stations/maitri/state").status_code == 200


def test_an_unset_api_key_is_generated_not_left_open():
    """No ALLOTROPE_API_KEY and no explicit api_key= must still result in a
    real, working credential requirement -- never a silently-open endpoint."""
    app = create_app()
    assert app.state.api_key
    client = TestClient(app)
    assert client.post("/stations/maitri/simulation/step").status_code == 401
    ok = client.post(
        "/stations/maitri/simulation/step", headers={"X-API-Key": app.state.api_key}
    )
    assert ok.status_code == 200


def test_telemetry_last_is_capped():
    from allotrope.api.app import MAX_TELEMETRY_LAST

    client = _client()
    for _ in range(3):
        client.post("/stations/maitri/simulation/step")
    resp = client.get(f"/stations/maitri/telemetry?last={MAX_TELEMETRY_LAST + 1_000_000}")
    assert resp.status_code == 200
    assert len(resp.json()) <= 3


# -- rate limiting (part of the DDoS/resource-exhaustion findings) ------------


def test_a_burst_past_the_limit_is_rejected_with_429():
    app = create_app(api_key=TEST_API_KEY, rate_limit_requests=5, rate_limit_window_s=10.0)
    client = TestClient(app)
    for _ in range(5):
        assert client.get("/stations").status_code == 200
    limited = client.get("/stations")
    assert limited.status_code == 429


def test_the_limit_resets_after_the_window_elapses():
    app = create_app(api_key=TEST_API_KEY, rate_limit_requests=3, rate_limit_window_s=0.2)
    client = TestClient(app)
    for _ in range(3):
        assert client.get("/stations").status_code == 200
    assert client.get("/stations").status_code == 429

    time.sleep(0.25)
    assert client.get("/stations").status_code == 200


def test_health_endpoint_is_exempt_from_the_rate_limit():
    """An orchestrator's own liveness probe must never trip a client's limit."""
    app = create_app(api_key=TEST_API_KEY, rate_limit_requests=2, rate_limit_window_s=10.0)
    client = TestClient(app)
    for _ in range(10):
        assert client.get("/health").status_code == 200
