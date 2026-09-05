"""Structured logging: every event carries valid, well-formed structured fields.

Attaches a small in-memory handler directly to the logger rather than
capturing stdout, since pytest's own log-capture machinery intercepts
record emission before it reaches a real stream handler in a test process.
"""

from __future__ import annotations

import json
import logging

import pytest

from allotrope.observability import EVENT_LOGGER_NAME, JsonFormatter, configure_logging, log_event


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(JsonFormatter())
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@pytest.fixture()
def logger_and_handler():
    logger = configure_logging(level=logging.DEBUG)
    handler = _ListHandler()
    logger.addHandler(handler)
    try:
        yield logger, handler
    finally:
        logger.removeHandler(handler)


def test_log_event_emits_one_json_object_with_the_attached_fields(logger_and_handler):
    logger, handler = logger_and_handler
    log_event(logger, "safety.intervened", station="maitri", step=5, interventions=["blocked_stop"])

    assert len(handler.lines) == 1
    payload = json.loads(handler.lines[0])
    assert payload["event"] == "safety.intervened"
    assert payload["station"] == "maitri"
    assert payload["step"] == 5
    assert payload["interventions"] == ["blocked_stop"]
    assert payload["level"] == "INFO"
    assert "ts" in payload


def test_log_event_respects_level(logger_and_handler):
    logger, handler = logger_and_handler
    log_event(logger, "controller.fallback", level=logging.WARNING, station="bharati")
    payload = json.loads(handler.lines[0])
    assert payload["level"] == "WARNING"


def test_configure_logging_is_idempotent():
    logger = logging.getLogger(EVENT_LOGGER_NAME)
    logger.handlers.clear()
    configure_logging()
    n_handlers = len(logger.handlers)
    configure_logging()
    configure_logging()
    assert len(logger.handlers) == n_handlers
    assert n_handlers == 1
