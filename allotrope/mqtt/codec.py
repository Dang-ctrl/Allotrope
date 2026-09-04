"""JSON encoding for the telemetry link.

Only telemetry crosses MQTT in this design -- commands travel over the gRPC
actuation path (`allotrope.rpc`), which carries the sub-10ms control budget and
the safety projection. MQTT here plays the role the deck assigns it: the
low-bandwidth, high-latency-tolerant channel a station's 4 MHz satellite link
actually is, carrying monitoring data and (in `allotrope.agents.federated`)
model weights, never raw control commands.

Decoding is defensive on principle: a payload arriving over a satellite link,
possibly corrupted or from a mismatched software version, must not raise an
exception that takes a subscriber down. `decode_telemetry` returns `None` on
anything it cannot parse rather than propagating the error.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any


def encode(payload: Any) -> bytes:
    if is_dataclass(payload) and not isinstance(payload, type):
        payload = asdict(payload)
    return json.dumps(payload, default=str).encode("utf-8")


def decode_telemetry(raw: bytes) -> dict | None:
    """Parse a telemetry payload, or return None if it cannot be trusted."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


__all__ = ["encode", "decode_telemetry"]
