"""FedAvg: weighted parameter averaging across stations, nothing more exotic.

Each station's synthetic weather and demand realisation never leaves the
process that generates it -- `allotrope.federated.round` trains locally
against each station's own `PolarMicrogridEnv` and only ever passes *model
weights* (a `allotrope.train.train`-shaped checkpoint) between stations.
This module is the aggregation step: it does not see, and has no interface
to see, any per-station observation, reward, or weather value.

FedAvg only makes sense when every participant's network has the same
shape, which is a real constraint this module checks and raises on rather
than silently averaging mismatched tensors -- see `fedavg_checkpoint`'s
architecture check. It happens to hold for Maitri and Bharati today (both
three gensets, two storage packs), which is why this is buildable now; a
station with a different fleet size would need either a compatible network
architecture or a federation scheme this module doesn't implement
(personalisation layers, cross-architecture distillation, ...).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

# Sub-dicts of a allotrope.train checkpoint that are themselves flat
# {param_name: Tensor} state_dicts, averaged directly.
_DQN_TENSOR_KEYS = ("online", "target")
_SDDPG_TENSOR_KEYS = ("actor", "actor_target", "critic", "critic_target")


def average_state_dict(
    state_dicts: list[OrderedDict[str, torch.Tensor]], weights: list[float]
) -> OrderedDict[str, torch.Tensor]:
    """Weighted average of N flat tensor state_dicts with identical keys and shapes.

    Raises ValueError on a key or shape mismatch rather than averaging
    tensors that don't actually correspond to the same parameter --
    exactly the failure this function exists to catch before it becomes a
    silently corrupted model.
    """
    if len(state_dicts) != len(weights):
        raise ValueError("one weight per state_dict is required")
    if not state_dicts:
        raise ValueError("cannot average zero state_dicts")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("weights must sum to a positive number")
    normalised = [w / total for w in weights]

    reference_keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != reference_keys:
            raise ValueError(f"state_dict {i} has different parameter keys than state_dict 0")

    averaged: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in state_dicts[0]:
        tensors = [sd[key] for sd in state_dicts]
        shape = tensors[0].shape
        for i, t in enumerate(tensors[1:], start=1):
            if t.shape != shape:
                raise ValueError(f"parameter {key!r}: shape mismatch ({shape} vs {t.shape} at index {i})")
        stacked = torch.stack([t.float() * w for t, w in zip(tensors, normalised)], dim=0)
        averaged[key] = stacked.sum(dim=0).to(tensors[0].dtype)
    return averaged


def fedavg_checkpoint(local_checkpoints: list[dict[str, Any]], weights: list[float]) -> dict[str, Any]:
    """Aggregate N local `allotrope.train.train`-format checkpoints into one global checkpoint.

    Weighted by `weights` (FedAvg's usual choice: each participant's local
    step/sample count, so a station that trained on more experience this
    round counts for more). Validates every local checkpoint declares the
    same `obs_dim`/`n_gensets`/`dispatch_dim` before touching a single
    tensor -- see the module docstring.
    """
    if not local_checkpoints:
        raise ValueError("cannot aggregate zero local checkpoints")

    for key in ("obs_dim", "n_gensets", "dispatch_dim"):
        values = {c[key] for c in local_checkpoints}
        if len(values) > 1:
            raise ValueError(
                f"cannot FedAvg checkpoints with different {key} ({sorted(values)}) -- "
                "participating stations must share the same network architecture"
            )

    dqn_agg = {k: average_state_dict([c["dqn"][k] for c in local_checkpoints], weights) for k in _DQN_TENSOR_KEYS}
    sddpg_agg = {
        k: average_state_dict([c["sddpg"][k] for c in local_checkpoints], weights) for k in _SDDPG_TENSOR_KEYS
    }

    return {
        "agent_kind": "hybrid",
        "obs_dim": local_checkpoints[0]["obs_dim"],
        "n_gensets": local_checkpoints[0]["n_gensets"],
        "dispatch_dim": local_checkpoints[0]["dispatch_dim"],
        "dqn": {
            **dqn_agg,
            # Counters, not parameters: summed, since FedAvg pools experience
            # across participants rather than averaging a step count.
            "env_steps": sum(c["dqn"].get("env_steps", 0) for c in local_checkpoints),
            "train_steps": sum(c["dqn"].get("train_steps", 0) for c in local_checkpoints),
        },
        "sddpg": {
            **sddpg_agg,
            "train_steps": sum(c["sddpg"].get("train_steps", 0) for c in local_checkpoints),
        },
    }


__all__ = ["average_state_dict", "fedavg_checkpoint"]
