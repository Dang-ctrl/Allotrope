"""Provenance for headline metrics: which run produced this number, exactly.

`allotrope.experiment.ExperimentTracker` records *training* runs. Nothing
recorded the other kind of run this project produces constantly -- a baseline
comparison, an evaluation against a checkpoint, an ad hoc scenario sweep --
each of which prints a fuel-saved or genset-starts number to stdout and then
forgets everything about how it was computed. That is exactly the gap that
lets a headline number in a slide deck become untraceable: nobody can later
answer "which station, which seed, which git commit, compared against what."

An `EvidenceRecord` is a small, durable answer to that question, one JSON
file per record under `runs/evidence/<experiment_id>.json`, following the
same file-based, no-server convention as `ExperimentTracker`. It deliberately
does not aggregate across runs or compute statistics -- a record reports
exactly what one script measured, on exactly the seed(s) it used, and no
more. If only one seed was run, the record says `seed=0`, not "validated";
inflating that distinction is the failure mode this module exists to prevent.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from allotrope.experiment import git_commit, git_dirty

EVIDENCE_DIR = Path("runs/evidence")


@dataclass
class EvidenceRecord:
    """One measured metric, traced back to the exact run that produced it.

    `seeds` is always a list, even for a single-seed run -- that keeps
    "how many seeds" an honest, inspectable fact rather than something a
    reader has to infer from a scalar's absence. `baseline_name`/
    `baseline_value` are optional because not every recorded metric is a
    comparison (e.g. a standalone evaluation number), but when they are
    present they must be values this run actually computed, not looked up
    or assumed.
    """

    experiment_id: str
    timestamp: float
    git_commit: str
    git_dirty: bool
    station: str
    seeds: list[int]
    metric_name: str
    metric_value: float
    description: str
    baseline_name: str | None = None
    baseline_value: float | None = None
    model_checkpoint: str | None = None
    source_script: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_experiment_id(station: str, metric_name: str) -> str:
    return f"{station}_{metric_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def record_result(
    station: str,
    seeds: int | list[int],
    metric_name: str,
    metric_value: float,
    description: str,
    baseline_name: str | None = None,
    baseline_value: float | None = None,
    model_checkpoint: str | None = None,
    source_script: str | None = None,
    extra: dict[str, Any] | None = None,
    experiment_id: str | None = None,
    evidence_dir: Path = EVIDENCE_DIR,
) -> EvidenceRecord:
    """Build an `EvidenceRecord` from a script's own measured result and save it.

    Any script that produces a headline number should call this once it has
    that number in hand -- `seeds` may be a single int for a single-seed run
    (the common case today) or a list if the caller genuinely ran several,
    in which case `metric_value` must already be whatever aggregate the
    caller computed over them; this function performs no aggregation itself.
    """
    seed_list = [seeds] if isinstance(seeds, int) else list(seeds)
    record = EvidenceRecord(
        experiment_id=experiment_id or _default_experiment_id(station, metric_name),
        timestamp=time.time(),
        git_commit=git_commit(),
        git_dirty=git_dirty(),
        station=station,
        seeds=seed_list,
        metric_name=metric_name,
        metric_value=metric_value,
        description=description,
        baseline_name=baseline_name,
        baseline_value=baseline_value,
        model_checkpoint=model_checkpoint,
        source_script=source_script,
        extra=extra or {},
    )
    save_evidence(record, evidence_dir)
    return record


def save_evidence(record: EvidenceRecord, evidence_dir: Path = EVIDENCE_DIR) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / f"{record.experiment_id}.json"
    path.write_text(json.dumps(record.as_dict(), indent=2))
    return path


def load_evidence(experiment_id: str, evidence_dir: Path = EVIDENCE_DIR) -> EvidenceRecord:
    path = evidence_dir / f"{experiment_id}.json"
    data = json.loads(path.read_text())
    return EvidenceRecord(**data)


def list_evidence(evidence_dir: Path = EVIDENCE_DIR) -> list[EvidenceRecord]:
    """All records currently on disk, newest first -- the closest thing this
    project has to a results ledger, until something reads it into a report."""
    if not evidence_dir.exists():
        return []
    records = [
        load_evidence(p.stem, evidence_dir) for p in evidence_dir.glob("*.json")
    ]
    return sorted(records, key=lambda r: r.timestamp, reverse=True)


__all__ = [
    "EvidenceRecord",
    "record_result",
    "save_evidence",
    "load_evidence",
    "list_evidence",
    "EVIDENCE_DIR",
]
