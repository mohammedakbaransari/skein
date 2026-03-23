"""
framework/observability/logging.py
====================================
Structured JSON logging with correlation context propagation.

Every log record emitted by SKEIN agents includes:
  - trace_id, span_id: OpenTelemetry-compatible distributed trace context
  - agent_id, agent_type: which agent produced the record
  - session_id: which user session
  - level, message, timestamp: standard log fields
  - extra: arbitrary structured data

This makes every log record directly queryable in ELK, Datadog, Splunk,
CloudWatch, or any structured log aggregator without post-processing.

THREAD SAFETY
=============
ContextVar stores per-thread/per-async-task context automatically.
No global mutable state other than the handler registry.

USAGE
=====
    # At app startup
    setup_logging(level="INFO", json_output=True)

    # In an agent
    log = get_logger(__name__, agent_id="agent-abc", agent_type="SupplierStressAgent")
    log.info("Processing task", extra={"task_id": "task-xyz", "supplier_count": 12})

    # Inject context for a request scope
    with correlation_context(trace_id="abc123", span_id="def456"):
        log.info("Inside traced request")  # will include trace_id automatically
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from framework.core.types import CorrelationContext


# ---------------------------------------------------------------------------
# Thread-local / context-var correlation context storage
# ---------------------------------------------------------------------------

_CORRELATION_VAR: ContextVar[Optional[CorrelationContext]] = ContextVar(
    "_skein_correlation", default=None
)


class correlation_context:
    """
    Context manager that injects a CorrelationContext into the current scope.

    Thread-safe: uses ContextVar (works with asyncio and threads).

    Usage:
        ctx = CorrelationContext.new(user_id="u123")
        with correlation_context(ctx):
            log.info("This record will include trace_id")

        # Or from raw IDs:
        with correlation_context(trace_id="abc", span_id="def"):
            ...
    """

    def __init__(
        self,
        context: Optional[CorrelationContext] = None,
        *,
        trace_id:  Optional[str] = None,
        span_id:   Optional[str] = None,
        **baggage: str,
    ) -> None:
        if context is not None:
            self._ctx = context
        else:
            self._ctx = CorrelationContext(
                trace_id=trace_id or CorrelationContext().trace_id,
                span_id=span_id or CorrelationContext().span_id,
                baggage=baggage,
            )
        self._token = None

    def __enter__(self) -> CorrelationContext:
        self._token = _CORRELATION_VAR.set(self._ctx)
        return self._ctx

    def __exit__(self, *_: Any) -> None:
        if self._token is not None:
            _CORRELATION_VAR.reset(self._token)


def get_current_context() -> Optional[CorrelationContext]:
    """Return the CorrelationContext active in the current scope, if any."""
    return _CORRELATION_VAR.get()


# ---------------------------------------------------------------------------
# JSON log formatter
# ---------------------------------------------------------------------------

class SKEINJsonFormatter(logging.Formatter):
    """
    Formats every log record as a single-line JSON object.

    Standard fields:
        timestamp, level, logger, message

    Auto-injected from correlation context (when available):
        trace_id, span_id, parent_span_id, baggage.*

    From 'extra' kwarg on log calls:
        agent_id, agent_type, session_id, task_id, and any other keys

    Exception fields:
        error_type, error_message, traceback
    """

    def format(self, record: logging.LogRecord) -> str:
        ctx = _CORRELATION_VAR.get()

        record_dict: Dict[str, Any] = {
            "timestamp":  datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level":      record.levelname,
            "logger":     record.name,
            "message":    record.getMessage(),
        }

        # Inject correlation context
        if ctx:
            record_dict["trace_id"]  = ctx.trace_id
            record_dict["span_id"]   = ctx.span_id
            if ctx.parent_span_id:
                record_dict["parent_span_id"] = ctx.parent_span_id
            for k, v in ctx.baggage.items():
                record_dict[f"baggage.{k}"] = v

        # Inject extra fields passed by the caller
        for key in (
            "agent_id", "agent_type", "session_id", "task_id",
            "workflow_id", "attempt", "duration_ms", "error",
        ):
            val = getattr(record, key, None)
            if val is not None:
                record_dict[key] = val

        # Any other extra keys
        skip = set(logging.LogRecord.__dict__) | {
            "message", "asctime", "exc_info", "exc_text",
            "stack_info", "msg", "args",
        }
        for key, val in record.__dict__.items():
            if key not in skip and not key.startswith("_") and key not in record_dict:
                try:
                    json.dumps(val)  # only include JSON-serialisable extras
                    record_dict[key] = val
                except (TypeError, ValueError):
                    record_dict[key] = str(val)

        # Exception info
        if record.exc_info:
            exc_type, exc_val, exc_tb = record.exc_info
            record_dict["error_type"]    = exc_type.__name__ if exc_type else "UnknownError"
            record_dict["error_message"] = str(exc_val)
            record_dict["traceback"]     = "".join(
                traceback.format_exception(exc_type, exc_val, exc_tb)
            ).strip()

        try:
            return json.dumps(record_dict, ensure_ascii=False, default=str)
        except Exception:
            return json.dumps({"level": "ERROR", "message": "log serialisation failed"})


# ---------------------------------------------------------------------------
# Text formatter (development/console mode)
# ---------------------------------------------------------------------------

class SKEINTextFormatter(logging.Formatter):
    """
    Human-readable formatter for local development.

    Format:
        2026-03-15T12:34:56Z INFO  [agent=agent-abc] [task=task-xyz] Message
    """

    LEVEL_COLOURS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colour: bool = True) -> None:
        super().__init__()
        self._colour = use_colour and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        ctx   = _CORRELATION_VAR.get()
        ts    = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        level = record.levelname.ljust(8)
        if self._colour:
            colour = self.LEVEL_COLOURS.get(record.levelname, "")
            level  = f"{colour}{level}{self.RESET}"

        parts = [ts, level, record.name]

        if ctx:
            parts.append(f"[trace={ctx.trace_id[:8]}]")
        agent_id = getattr(record, "agent_id", None)
        if agent_id:
            parts.append(f"[agent={agent_id}]")
        task_id = getattr(record, "task_id", None)
        if task_id:
            parts.append(f"[task={task_id}]")

        parts.append(record.getMessage())

        if record.exc_info:
            parts.append("\n" + self.formatException(record.exc_info))

        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

class AgentLogger(logging.LoggerAdapter):
    """
    Logger adapter that automatically injects agent-level context
    into every log record without callers needing to pass extra= every time.

    Usage:
        log = get_logger(__name__, agent_id="agent-abc", agent_type="SupplierStress")
        log.info("Processing %d suppliers", 5)
        # Produces: {..., "agent_id": "agent-abc", "agent_type": "SupplierStress", ...}
    """

    def process(self, msg: Any, kwargs: Dict[str, Any]) -> tuple:
        extra = kwargs.setdefault("extra", {})
        for key, val in self.extra.items():
            if key not in extra and val is not None:
                extra[key] = val
        return msg, kwargs


def get_logger(
    name: str,
    agent_id:   Optional[str] = None,
    agent_type: Optional[str] = None,
    session_id: Optional[str] = None,
) -> AgentLogger:
    """
    Return an AgentLogger for the given module name.
    Optional agent context is injected into every record automatically.
    """
    base = logging.getLogger(name)
    extra: Dict[str, Any] = {}
    if agent_id:   extra["agent_id"]   = str(agent_id)
    if agent_type: extra["agent_type"] = agent_type
    if session_id: extra["session_id"] = str(session_id)
    return AgentLogger(base, extra)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure root logger for the SKEIN framework.

    Args:
        level:       Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
        json_output: True for structured JSON (production). False for text (dev).
        log_file:    Optional file path for log output in addition to stderr.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    formatter = SKEINJsonFormatter() if json_output else SKEINTextFormatter()

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(SKEINJsonFormatter())  # always JSON to file
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
