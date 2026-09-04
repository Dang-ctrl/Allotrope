"""Local experiment tracking: enough to make a run reproducible, nothing more.

No MLflow server is reachable from an edge-first, offline-capable project's
CI or this environment, so runs are recorded as one JSON file per run under
`runs/<run_id>/`. Every field needed to reproduce a run -- git commit, agent,
station, seed, config, hyperparameters, metrics, checkpoint path -- is
written there. This is a deliberately small abstraction: swapping it for a
hosted MLflow tracking server later means changing `ExperimentTracker`, not
every call site.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RUNS_DIR = Path("runs")


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=5
        )
        return bool(out.stdout.strip())
    except Exception:
        return True


@dataclass
class RunRecord:
    run_id: str
    agent: str
    station: str
    seed: int
    git_commit: str
    git_dirty: bool
    config: dict[str, Any]
    started_at: float
    finished_at: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)
    checkpoint_path: str | None = None
    env_version: str = "polar_microgrid_v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentTracker:
    """One run, one directory: `runs/<run_id>/record.json` plus checkpoints."""

    def __init__(self, agent: str, station: str, seed: int, config: dict[str, Any],
                 runs_dir: Path = RUNS_DIR) -> None:
        self.runs_dir = runs_dir
        run_id = f"{agent}_{station}_seed{seed}_{int(time.time())}"
        self.dir = self.runs_dir / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.record = RunRecord(
            run_id=run_id,
            agent=agent,
            station=station,
            seed=seed,
            git_commit=git_commit(),
            git_dirty=git_dirty(),
            config=config,
            started_at=time.time(),
        )
        self._save()

    def log(self, step: int, metrics: dict[str, float]) -> None:
        self.record.history.append({"step": step, **metrics})
        self._save()

    def finish(self, metrics: dict[str, float], checkpoint_path: str | None = None) -> None:
        self.record.metrics = metrics
        self.record.checkpoint_path = checkpoint_path
        self.record.finished_at = time.time()
        self._save()

    def _save(self) -> None:
        (self.dir / "record.json").write_text(json.dumps(self.record.as_dict(), indent=2))

    @staticmethod
    def load(run_dir: Path) -> dict[str, Any]:
        return json.loads((Path(run_dir) / "record.json").read_text())


__all__ = ["ExperimentTracker", "RunRecord", "git_commit", "git_dirty"]
