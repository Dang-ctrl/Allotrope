"""Structured logging: one JSON object per line, so a real log pipeline can read it.

No observability platform (Grafana, an OTel collector) is reachable from this
environment or guaranteed to exist at edge deployment, mirroring the same
constraint `allotrope/experiment.py` documents for experiment tracking. What
this module gives instead is the thing every such platform actually
ingests: structured, timestamped, machine-parseable log lines to stdout.
Swapping this for a shipped OTel/structlog pipeline later is a change to
this module, not to every call site that logs an event.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

EVENT_LOGGER_NAME = "allotrope.events"


class JsonFormatter(logging.Formatter):
    """One JSON object per log record: timestamp, level, logger, message, and
    whatever structured fields the call site attached via `extra={"fields": ...}`."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach a JSON-formatted stdout handler to the shared event logger.

    Idempotent: calling this more than once (the API and a CLI script both
    importing this module, say) does not duplicate handlers or duplicate log
    lines.
    """
    logger = logging.getLogger(EVENT_LOGGER_NAME)
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one structured event: `log_event(logger, "safety.intervened", station="maitri", ...)`."""
    logger.log(level, event, extra={"fields": {"event": event, **fields}})


__all__ = ["configure_logging", "log_event", "JsonFormatter", "EVENT_LOGGER_NAME"]
