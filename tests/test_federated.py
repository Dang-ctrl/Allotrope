"""Federated learning: aggregation as a pure function, then the full local
-> aggregate -> validate -> (accept | rollback) loop against real, if
small, training runs.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from allotrope.config import load_station
from allotrope.federated.aggregate import average_state_dict, fedavg_checkpoint
from allotrope.federated.coordinator import run_round, run_rounds
from allotrope.federated.round import LocalUpdateResult, ValidationResult, run_local_update
from allotrope.train import train

# -- average_state_dict: a pure function, tested without training anything --


def test_average_state_dict_computes_the_exact_weighted_mean():
    a = OrderedDict(w=torch.tensor([1.0, 2.0]), b=torch.tensor(0.0))
    b = OrderedDict(w=torch.tensor([3.0, 4.0]), b=torch.tensor(10.0))
    result = average_state_dict([a, b], weights=[1.0, 1.0])
    assert torch.allclose(result["w"], torch.tensor([2.0, 3.0]))
    assert torch.allclose(result["b"], torch.tensor(5.0))


def test_average_state_dict_respects_unequal_weights():
    a = OrderedDict(w=torch.tensor([0.0]))
    b = OrderedDict(w=torch.tensor([10.0]))
    result = average_state_dict([a, b], weights=[3.0, 1.0])  # 75%/25%
    assert torch.allclose(result["w"], torch.tensor([2.5]))


def test_average_state_dict_rejects_mismatched_keys():
    a = OrderedDict(w=torch.tensor([1.0]))
    b = OrderedDict(x=torch.tensor([1.0]))
    with pytest.raises(ValueError, match="parameter keys"):
        average_state_dict([a, b], weights=[1.0, 1.0])


def test_average_state_dict_rejects_mismatched_shapes():
    a = OrderedDict(w=torch.tensor([1.0, 2.0]))
    b = OrderedDict(w=torch.tensor([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="shape mismatch"):
        average_state_dict([a, b], weights=[1.0, 1.0])


def test_average_state_dict_rejects_nonpositive_total_weight():
    a = OrderedDict(w=torch.tensor([1.0]))
    with pytest.raises(ValueError, match="positive"):
        average_state_dict([a, a], weights=[0.0, 0.0])


# -- fedavg_checkpoint: the same, at the checkpoint-dict level -------------


def _tiny_checkpoint(seed: int, obs_dim=5, n_g=3, dispatch_dim=6) -> dict:
    from allotrope.agents.dqn import BranchingDQN, DQNConfig
    from allotrope.agents.sddpg import SDDPG, SDDPGConfig

    dqn = BranchingDQN(obs_dim, n_g, DQNConfig(seed=seed))
    sddpg = SDDPG(obs_dim, dispatch_dim, SDDPGConfig(seed=seed))
    return {
        "obs_dim": obs_dim,
        "n_gensets": n_g,
        "dispatch_dim": dispatch_dim,
        "dqn": dqn.state_dict(),
        "sddpg": sddpg.state_dict(),
    }


def test_fedavg_checkpoint_averages_real_network_weights():
    c1 = _tiny_checkpoint(seed=1)
    c2 = _tiny_checkpoint(seed=2)
    merged = fedavg_checkpoint([c1, c2], weights=[1.0, 1.0])

    # Spot-check one real parameter tensor against the literal weighted mean.
    key = next(iter(c1["dqn"]["online"]))
    expected = (c1["dqn"]["online"][key] + c2["dqn"]["online"][key]) / 2.0
    assert torch.allclose(merged["dqn"]["online"][key], expected)

    # The merged checkpoint must be loadable into fresh agents of the same shape.
    from allotrope.agents.dqn import BranchingDQN, DQNConfig
    from allotrope.agents.sddpg import SDDPG, SDDPGConfig

    dqn = BranchingDQN(5, 3, DQNConfig())
    sddpg = SDDPG(5, 6, SDDPGConfig())
    dqn.load_state_dict(merged["dqn"])
    sddpg.load_state_dict(merged["sddpg"])

    import numpy as np

    obs = np.zeros(5, dtype=np.float32)
    action = dqn.act(obs, deterministic=True)
    assert action.shape == (3,)
    dispatch = sddpg.act(obs, deterministic=True)
    assert np.all(np.isfinite(dispatch))


def test_fedavg_checkpoint_sums_step_counters_rather_than_averaging_them():
    c1 = _tiny_checkpoint(seed=1)
    c1["dqn"]["env_steps"] = 100
    c2 = _tiny_checkpoint(seed=2)
    c2["dqn"]["env_steps"] = 300
    merged = fedavg_checkpoint([c1, c2], weights=[1.0, 1.0])
    assert merged["dqn"]["env_steps"] == 400


def test_fedavg_checkpoint_rejects_architecture_mismatch():
    c1 = _tiny_checkpoint(seed=1, n_g=3)
    c2 = _tiny_checkpoint(seed=2, n_g=4)
    with pytest.raises(ValueError, match="n_gensets"):
        fedavg_checkpoint([c1, c2], weights=[1.0, 1.0])


# -- warm-start: allotrope.train.train's init_checkpoint -------------------


def test_warm_started_training_with_no_update_reproduces_the_initial_weights(tmp_path):
    """With warmup_steps far above total_steps, no gradient update happens,
    so the saved checkpoint's weights must exactly equal what was loaded in."""
    source_dir = train(
        agent_kind="hybrid",
        station="maitri",
        total_steps=5,
        seed=0,
        episode_steps=24,
        warmup_steps=1000,
        buffer_capacity=100,
        runs_dir=tmp_path / "source",
    )
    source_checkpoint = source_dir / "checkpoint.pt"

    warm_dir = train(
        agent_kind="hybrid",
        station="maitri",
        total_steps=1,
        seed=42,  # different seed: only init_checkpoint should determine the weights
        episode_steps=24,
        warmup_steps=1000,
        buffer_capacity=100,
        runs_dir=tmp_path / "warm",
        init_checkpoint=source_checkpoint,
    )

    source_state = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    warm_state = torch.load(warm_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    key = next(iter(source_state["dqn"]["online"]))
    assert torch.equal(source_state["dqn"]["online"][key], warm_state["dqn"]["online"][key])


# -- the full round: local update -> aggregate -> validate -----------------

ROUND_KWARGS = dict(episode_steps=48, warmup_steps=50, buffer_capacity=500)


def test_a_real_round_produces_an_accepted_global_checkpoint(tmp_path):
    record = run_round(
        round_num=1,
        stations=["maitri", "bharati"],
        local_steps=100,
        current_global_checkpoint=None,
        seed_base=0,
        runs_dir=tmp_path,
        **ROUND_KWARGS,
    )
    assert record.accepted, record.validation["reason"]
    assert record.global_checkpoint_path is not None
    assert Path(record.global_checkpoint_path).exists()
    assert set(record.aggregation_weights) == {"maitri", "bharati"}
    for station in ("maitri", "bharati"):
        assert record.validation["per_station"][station]["critical_unserved_kwh"] == pytest.approx(0.0)

    record_file = tmp_path / "federated" / "round_1" / "round_record.json"
    assert record_file.exists()


def test_a_second_round_can_start_from_the_first_rounds_global_checkpoint(tmp_path):
    first = run_round(
        round_num=1,
        stations=["maitri", "bharati"],
        local_steps=100,
        current_global_checkpoint=None,
        seed_base=0,
        runs_dir=tmp_path,
        **ROUND_KWARGS,
    )
    assert first.accepted
    second = run_round(
        round_num=2,
        stations=["maitri", "bharati"],
        local_steps=100,
        current_global_checkpoint=Path(first.global_checkpoint_path),
        seed_base=10,
        runs_dir=tmp_path,
        **ROUND_KWARGS,
    )
    assert second.previous_global_checkpoint_path == first.global_checkpoint_path
    assert second.accepted, second.validation["reason"]


def test_a_rejected_round_is_never_promoted_and_is_kept_for_provenance(tmp_path):
    def always_reject(checkpoint_path: Path, stations: list[str]) -> ValidationResult:
        return ValidationResult(accepted=False, reason="forced rejection for this test", per_station={})

    record = run_round(
        round_num=1,
        stations=["maitri"],
        local_steps=50,
        current_global_checkpoint=None,
        seed_base=0,
        runs_dir=tmp_path,
        validator=always_reject,
        **ROUND_KWARGS,
    )
    assert not record.accepted
    assert record.global_checkpoint_path is None
    assert Path(record.candidate_checkpoint_path).exists()  # kept, not deleted


def test_run_rounds_carries_forward_only_the_last_accepted_checkpoint(tmp_path):
    calls = {"n": 0}

    def reject_first_accept_rest(checkpoint_path: Path, stations: list[str]) -> ValidationResult:
        calls["n"] += 1
        return ValidationResult(accepted=calls["n"] > 1, reason="test", per_station={})

    records = run_rounds(
        n_rounds=2,
        stations=["maitri"],
        local_steps=50,
        runs_dir=tmp_path,
        validator=reject_first_accept_rest,
        **ROUND_KWARGS,
    )
    assert not records[0].accepted
    assert records[1].accepted
    # Round 2 must not have been warm-started from round 1's (rejected) output.
    assert records[1].previous_global_checkpoint_path is None
