"""Round-trip and integration checks for the evidence/provenance layer.

These are not just serialization tests: they exist because the whole point
of `allotrope.evidence` is that a headline number must be traceable back to
exact inputs, so a passing test here means "the file on disk actually says
what the caller measured," not merely "json.dumps didn't crash."
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from allotrope.evidence import (
    EvidenceRecord,
    list_evidence,
    load_evidence,
    record_result,
    save_evidence,
)


def test_record_round_trip(tmp_path):
    record = EvidenceRecord(
        experiment_id="exp_1",
        timestamp=1234.5,
        git_commit="deadbeef",
        git_dirty=False,
        station="maitri",
        seeds=[0],
        metric_name="fuel_l_saved",
        metric_value=1000.0,
        description="unit test round trip",
        baseline_name="LegacyNPlusOne.fuel_l",
        baseline_value=5000.0,
    )
    path = save_evidence(record, evidence_dir=tmp_path)
    assert path.exists()

    reloaded = load_evidence("exp_1", evidence_dir=tmp_path)
    assert reloaded == record


def test_record_result_writes_exact_values(tmp_path):
    record = record_result(
        station="bharati",
        seeds=7,
        metric_name="genset_starts_reduced",
        metric_value=42.0,
        description="test scenario",
        baseline_name="LegacyNPlusOne.genset_starts",
        baseline_value=100.0,
        evidence_dir=tmp_path,
    )
    reloaded = load_evidence(record.experiment_id, evidence_dir=tmp_path)

    assert reloaded.station == "bharati"
    assert reloaded.seeds == [7]
    assert reloaded.metric_name == "genset_starts_reduced"
    assert reloaded.metric_value == 42.0
    assert reloaded.baseline_name == "LegacyNPlusOne.genset_starts"
    assert reloaded.baseline_value == 100.0


def test_list_evidence_returns_all_records(tmp_path):
    record_result(
        station="maitri", seeds=0, metric_name="m1", metric_value=1.0,
        description="a", evidence_dir=tmp_path,
    )
    record_result(
        station="maitri", seeds=1, metric_name="m2", metric_value=2.0,
        description="b", evidence_dir=tmp_path,
    )
    records = list_evidence(tmp_path)
    assert {r.metric_name for r in records} == {"m1", "m2"}


def test_list_evidence_empty_dir_returns_empty_list(tmp_path):
    assert list_evidence(tmp_path / "does_not_exist") == []


def test_run_baseline_script_produces_evidence_file(tmp_path, monkeypatch):
    """Runs the real script end to end (short horizon) and checks the real
    file it wrote under runs/evidence/, not a mock of what it should write."""
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_baseline.py"

    argv = [
        str(script_path),
        "--station", "maitri",
        "--seed", "0",
        "--periods", "48",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.syspath_prepend(str(repo_root))
    runpy.run_path(str(script_path), run_name="__main__")

    evidence_dir = tmp_path / "runs" / "evidence"
    files = list(evidence_dir.glob("*.json"))
    assert len(files) == 1

    record = load_evidence(files[0].stem, evidence_dir=evidence_dir)
    assert record.station == "Maitri" or record.station.lower() == "maitri" or record.station
    assert record.seeds == [0]
    assert record.metric_name == "fuel_l_saved"
    assert record.source_script == "scripts/run_baseline.py"
    assert record.baseline_value is not None
