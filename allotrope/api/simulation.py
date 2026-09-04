"""One steppable simulation per station, held in process memory.

This is the thing the API serves. It is real simulation state -- the same
`PolarMicrogrid`, the same `GuardedController`, the same reward-free
telemetry the CLI scripts and the test suite use -- not a mock, not a fixture,
not a number typed into a response body. When the API says a genset is
online, it is online in an actual `PolarMicrogrid` instance stepping through
an actual synthetic weather year.

What it is *not*: physical station telemetry. There is no live link to
Maitri or Bharati (`README.md`, "On the data"), so everything this module
produces is `SIMULATION` -- one of the modes `StationSimulation.mode` can
report -- and every response in `allotrope.api.app` is honest about that
rather than letting a client infer otherwise.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol

from allotrope.config import StationConfig, load_station
from allotrope.control.baseline import EfficientRuleBased
from allotrope.safety.fallback import GuardedController
from allotrope.sim.plant import DispatchCommand, PolarMicrogrid
from allotrope.sim.runner import build_plant

HISTORY_LEN = 500
"""How many past steps' telemetry the API keeps in memory for charting."""


class Controller(Protocol):
    name: str

    def act(self, observation: dict, plant: PolarMicrogrid) -> DispatchCommand: ...
    def reset(self) -> None: ...


@dataclass
class StationSimulation:
    """A running (or stopped) simulation of one station, safety-guarded.

    One instance per station lives for the life of the API process. It owns
    its own `PolarMicrogrid`, is stepped either on demand (`step()`, for a
    single-shot API call) or by a background loop (`allotrope.api.app`'s
    runner task, for `POST /stations/{id}/simulation/start`), and remembers
    the last `HISTORY_LEN` steps of telemetry so a client can chart a trend
    without replaying the whole run.
    """

    station_id: str
    cfg: StationConfig
    controller: Controller
    seed: int = 0
    start: str = "2026-01-01"
    periods: int = 8760
    freq: str = "1h"

    plant: PolarMicrogrid = field(init=False)
    running: bool = field(default=False, init=False)
    history: deque[dict[str, Any]] = field(init=False)
    lock: Lock = field(default_factory=Lock, init=False)
    step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.history = deque(maxlen=HISTORY_LEN)
        self.plant = build_plant(self.cfg, self.start, self.periods, self.freq, seed=self.seed)
        self.plant.reset()
        self.controller.reset()

    # -- stepping -----------------------------------------------------------

    def step(self) -> dict[str, Any] | None:
        """Advance one dispatch interval. Returns None once the run is over."""
        with self.lock:
            if self.plant.done:
                self.running = False
                return None
            observation = self.plant.observe()
            command = self.controller.act(observation, self.plant)
            telemetry = self.plant.step(command)
            record = _flatten(telemetry)
            record["safety"] = getattr(self.controller, "last_report", None)
            record["fallback_reason"] = getattr(self.controller, "last_fallback_reason", None)
            self.history.append(record)
            self.step_count += 1
            return record

    def reset(self) -> None:
        with self.lock:
            self.plant.reset()
            self.controller.reset()
            self.history.clear()
            self.step_count = 0
            self.running = False

    # -- reporting ------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        with self.lock:
            obs = self.plant.observe()
            return {
                "station_id": self.station_id,
                "mode": "simulation",
                "running": self.running,
                "step": self.step_count,
                "n_steps": self.plant.n_steps,
                "done": self.plant.done,
                "timestamp": obs["timestamp"].isoformat(),
                "observation": _jsonable(obs),
            }

    def telemetry(self, last_n: int | None = None) -> list[dict[str, Any]]:
        with self.lock:
            items = list(self.history)
        if last_n is not None:
            items = items[-last_n:]
        return [_jsonable(item) for item in items]

    def latest_telemetry(self) -> dict[str, Any] | None:
        with self.lock:
            return _jsonable(self.history[-1]) if self.history else None

    def summary(self) -> dict[str, float]:
        with self.lock:
            return self.plant.summary()

    def safety(self) -> dict[str, Any]:
        guard = self.controller
        stats = getattr(guard, "stats", None)
        last_report = getattr(guard, "last_report", None)
        last_fallback = getattr(guard, "last_fallback_reason", None)
        return {
            "last_report": last_report.as_dict() if last_report is not None else None,
            "last_fallback_reason": last_fallback.value if last_fallback is not None else None,
            "steps": getattr(stats, "steps", 0),
            "fallbacks": getattr(stats, "fallbacks", 0),
            "projections": getattr(stats, "projections", 0),
            "fallback_rate": getattr(stats, "fallback_rate", 0.0),
            "projection_rate": getattr(stats, "projection_rate", 0.0),
            "fallback_reasons": dict(getattr(stats, "reasons", {})),
            "max_latency_ms": getattr(stats, "max_latency_ms", 0.0),
        }

    def controller_status(self) -> dict[str, Any]:
        return {
            "name": getattr(self.controller, "name", type(self.controller).__name__),
            "type": type(self.controller).__name__,
            "wrapped_agent": type(getattr(self.controller, "agent", None)).__name__
            if getattr(self.controller, "agent", None) is not None
            else None,
        }


def _flatten(telemetry: dict[str, Any]) -> dict[str, Any]:
    """Same policy as `allotrope.sim.runner._flatten`: lists become indexed columns."""
    flat: dict[str, Any] = {}
    for key, value in telemetry.items():
        if isinstance(value, list):
            for unit, item in enumerate(value):
                flat[f"{key}_{unit}"] = item
        else:
            flat[key] = value
    return flat


def _jsonable(value: Any) -> Any:
    """Recursively convert pandas/numpy timestamps and scalars to JSON-safe types."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def default_controller(cfg: StationConfig) -> GuardedController:
    """The best controller available today, safety-guarded.

    The learned agents in `allotrope.agents` are not yet trained to a
    checkpoint this API loads by default (`docs/reinforcement-learning.md`,
    "Honest status") -- so the default is the best *rule-based* policy the
    project has, `EfficientRuleBased`, wrapped in the same `GuardedController`
    every other controller in this project is judged through. A trained
    checkpoint can be swapped in later without changing this module's
    interface.
    """
    return GuardedController(cfg, agent=EfficientRuleBased(cfg))


def build_simulation(station_id: str, seed: int = 0) -> StationSimulation:
    cfg = load_station(station_id)
    return StationSimulation(station_id=station_id, cfg=cfg, controller=default_controller(cfg), seed=seed)


__all__ = ["StationSimulation", "default_controller", "build_simulation", "HISTORY_LEN"]
