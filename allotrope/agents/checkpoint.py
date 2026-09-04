"""Save and load a trained HybridAgent, with just enough metadata to be honest.

The checkpoint records the station and a hash of its config alongside the
weights, so loading a Maitri-trained policy against a Bharati environment fails
loudly instead of producing silently wrong-scaled dispatch.
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


def save(agent: HybridAgent, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "station_id": agent.cfg.site.id,
            "config_hash": _config_hash(agent.cfg),
            "state": agent.state_dict(),
        },
        path,
    )


def load(path: str | Path, cfg: StationConfig) -> HybridAgent:
    checkpoint = torch.load(Path(path), weights_only=False)
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


__all__ = ["save", "load"]
