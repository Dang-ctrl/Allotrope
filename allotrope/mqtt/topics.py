"""Topic naming, kept in one place so publisher and subscriber cannot drift."""

from __future__ import annotations


def telemetry_topic(station_id: str) -> str:
    return f"allotrope/{station_id}/telemetry"


def safety_topic(station_id: str) -> str:
    return f"allotrope/{station_id}/safety"


def model_update_topic(station_id: str) -> str:
    """Where a station's federated round contribution is published.

    Only model parameters travel here (see `allotrope.agents.federated`), never
    station telemetry or weather -- this is the channel the deck's "only
    gradients cross the satellite link" claim refers to.
    """
    return f"allotrope/{station_id}/federated/update"


__all__ = ["telemetry_topic", "safety_topic", "model_update_topic"]
