"""Save and load a trained HybridAgent, with just enough metadata to be honest.

A single-station checkpoint records the exact station config it was trained
against, so loading it anywhere else fails loudly instead of producing
silently wrong-scaled dispatch. A federated checkpoint is deliberately weaker:
it is meant to be deployed at *any* station whose asset counts match, so it
records only the shape its network was built for (genset and storage counts),
not one station's full configuration -- see `save_federated`/`load_federated`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from allotrope.agents.hybrid import HybridAgent
from allotrope.config import StationConfig


def _config_hash(cfg: StationConfig) -> str:
    return hashlib.sha256(json.dumps(cfg.raw, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _shape(cfg: StationConfig) -> tuple[int, int]:
    return (len(cfg.gensets), len(cfg.storage))


def save(agent: HybridAgent, path: str | Path) -> None:
    """Save a checkpoint tied to the exact station configuration it trained on."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "single_station",
            "station_id": agent.cfg.site.id,
            "config_hash": _config_hash(agent.cfg),
            "state": agent.state_dict(),
        },
        path,
    )


def load(path: str | Path, cfg: StationConfig) -> HybridAgent:
    """Load a single-station checkpoint. Refuses a federated one -- use `load_federated`."""
    checkpoint = torch.load(Path(path), weights_only=False)
    if checkpoint.get("kind", "single_station") != "single_station":
        raise ValueError(f"{path} is a federated checkpoint; use load_federated() instead")
    if checkpoint["station_id"] != cfg.site.id:
        raise ValueError(
            f"checkpoint was trained on {checkpoint['station_id']!r}, "
            f"cannot load against {cfg.site.id!r}"
        )
    if checkpoint["config_hash"] != _config_hash(cfg):
        raise ValueError(
            "checkpoint's station configuration has changed since training; "
            "the network's normalisation assumptions no longer match"
        )
    agent = HybridAgent(cfg)
    agent.load_state_dict(checkpoint["state"])
    return agent


def save_federated(agent: HybridAgent, station_ids: list[str], path: str | Path) -> None:
    """Save a checkpoint meant to be deployed at any of several stations.

    Only the network *shape* is recorded as a compatibility check, not any one
    station's full configuration -- the whole point of a federated checkpoint
    is that it was not trained against one station's specifics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "federated",
            "trained_on_stations": list(station_ids),
            "shape": list(_shape(agent.cfg)),
            "state": agent.state_dict(),
        },
        path,
    )


def load_federated(path: str | Path, cfg: StationConfig) -> HybridAgent:
    """Load a federated checkpoint against any station whose asset counts match."""
    checkpoint = torch.load(Path(path), weights_only=False)
    if checkpoint.get("kind") != "federated":
        raise ValueError(f"{path} is not a federated checkpoint; use load() instead")
    expected = tuple(checkpoint["shape"])
    if _shape(cfg) != expected:
        raise ValueError(
            f"federated checkpoint expects (gensets, storage)={expected}, "
            f"but {cfg.site.id!r} has {_shape(cfg)} -- the action space would not match"
        )
    agent = HybridAgent(cfg)
    agent.load_state_dict(checkpoint["state"])
    return agent


__all__ = ["save", "load", "save_federated", "load_federated"]
