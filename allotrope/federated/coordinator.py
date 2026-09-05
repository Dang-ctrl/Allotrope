"""The federated coordinator: local updates, aggregation, validation, rollback.

    python -m allotrope.federated.coordinator --rounds 3 --local-steps 5000

One round:

    local training (per station, from the current global model)
        -> local checkpoints
        -> allotrope.federated.aggregate.fedavg_checkpoint
        -> candidate global checkpoint
        -> allotrope.federated.round.default_validator
        -> accepted: becomes the new global model, carried into the next round
           rejected: kept on disk for provenance, never promoted; the next
                     round starts from the last *accepted* global model

Every round's full record -- participating stations, per-station steps and
checkpoint paths, aggregation weights, validation results, accept/reject,
git commit, timestamps -- is written to `round_record.json` before this
function returns, whether the round was accepted or not. Nothing about a
rejected round is deleted: `docs/federated-learning.md` explains why
provenance for a rejected round matters as much as for an accepted one.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch

from allotrope.experiment import git_commit
from allotrope.federated.aggregate import fedavg_checkpoint
from allotrope.federated.round import LocalUpdateResult, ValidationResult, default_validator, run_local_update

Validator = Callable[[Path, list[str]], ValidationResult]


@dataclass
class RoundRecord:
    round_num: int
    stations: list[str]
    local_updates: list[dict]
    aggregation_weights: dict[str, float]
    candidate_checkpoint_path: str
    global_checkpoint_path: str | None
    """None when the round was rejected -- no checkpoint was promoted."""
    previous_global_checkpoint_path: str | None
    validation: dict
    accepted: bool
    git_commit: str
    started_at: float
    finished_at: float
    config: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def run_round(
    round_num: int,
    stations: list[str],
    local_steps: int,
    current_global_checkpoint: Path | None,
    seed_base: int,
    runs_dir: Path,
    episode_steps: int = 336,
    warmup_steps: int = 200,
    buffer_capacity: int = 20_000,
    validator: Validator = default_validator,
) -> RoundRecord:
    """Run one federated round and return its full, machine-readable record."""
    started_at = time.time()
    round_dir = runs_dir / "federated" / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)

    local_results: list[LocalUpdateResult] = []
    for i, station in enumerate(stations):
        result = run_local_update(
            station=station,
            steps=local_steps,
            seed=seed_base + i,
            episode_steps=episode_steps,
            warmup_steps=warmup_steps,
            buffer_capacity=buffer_capacity,
            init_checkpoint=current_global_checkpoint,
            runs_dir=round_dir / "local",
        )
        local_results.append(result)

    checkpoints = [
        torch.load(r.checkpoint_path, map_location="cpu", weights_only=True) for r in local_results
    ]
    weights = [float(r.steps) for r in local_results]
    aggregated = fedavg_checkpoint(checkpoints, weights)

    candidate_path = round_dir / "global_checkpoint_candidate.pt"
    torch.save(aggregated, candidate_path)

    validation = validator(candidate_path, stations)

    global_path: Path | None = None
    if validation.accepted:
        global_path = round_dir / "global_checkpoint.pt"
        candidate_path.replace(global_path)

    record = RoundRecord(
        round_num=round_num,
        stations=stations,
        local_updates=[r.as_dict() for r in local_results],
        aggregation_weights=dict(zip(stations, weights)),
        candidate_checkpoint_path=str(candidate_path if global_path is None else global_path),
        global_checkpoint_path=str(global_path) if global_path else None,
        previous_global_checkpoint_path=str(current_global_checkpoint) if current_global_checkpoint else None,
        validation=validation.as_dict(),
        accepted=validation.accepted,
        git_commit=git_commit(),
        started_at=started_at,
        finished_at=time.time(),
        config={
            "local_steps": local_steps,
            "episode_steps": episode_steps,
            "warmup_steps": warmup_steps,
            "buffer_capacity": buffer_capacity,
        },
    )
    (round_dir / "round_record.json").write_text(json.dumps(record.as_dict(), indent=2))
    return record


def run_rounds(
    n_rounds: int,
    stations: list[str],
    local_steps: int,
    runs_dir: Path,
    seed_base: int = 0,
    initial_checkpoint: Path | None = None,
    **round_kwargs,
) -> list[RoundRecord]:
    """Run `n_rounds` in sequence. A rejected round's checkpoint is never
    carried forward -- the following round starts again from the last
    *accepted* global model (or `initial_checkpoint`/scratch if none has
    been accepted yet)."""
    current_checkpoint = initial_checkpoint
    records: list[RoundRecord] = []
    for round_num in range(1, n_rounds + 1):
        record = run_round(
            round_num=round_num,
            stations=stations,
            local_steps=local_steps,
            current_global_checkpoint=current_checkpoint,
            seed_base=seed_base + round_num * len(stations),
            runs_dir=runs_dir,
            **round_kwargs,
        )
        records.append(record)
        if record.accepted:
            current_checkpoint = Path(record.global_checkpoint_path)
        print(
            f"round {round_num}/{n_rounds}: "
            f"{'accepted' if record.accepted else 'REJECTED'} -- {record.validation['reason']}"
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--stations", default="maitri,bharati")
    parser.add_argument("--local-steps", type=int, default=5_000)
    parser.add_argument("--episode-steps", type=int, default=336)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--buffer-capacity", type=int, default=20_000)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--init-checkpoint", default=None)
    args = parser.parse_args()

    stations = [s.strip() for s in args.stations.split(",") if s.strip()]
    records = run_rounds(
        n_rounds=args.rounds,
        stations=stations,
        local_steps=args.local_steps,
        runs_dir=Path(args.runs_dir),
        seed_base=args.seed_base,
        initial_checkpoint=Path(args.init_checkpoint) if args.init_checkpoint else None,
        episode_steps=args.episode_steps,
        warmup_steps=args.warmup_steps,
        buffer_capacity=args.buffer_capacity,
    )
    accepted = [r for r in records if r.accepted]
    print(f"\n{len(accepted)}/{len(records)} rounds accepted")
    if accepted:
        print(f"final global checkpoint: {accepted[-1].global_checkpoint_path}")


if __name__ == "__main__":
    main()


__all__ = ["RoundRecord", "run_round", "run_rounds", "main"]
