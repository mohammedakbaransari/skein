"""Local reference sink for tests and development."""

from __future__ import annotations

import threading
from typing import List

from framework.adapters.audit.interfaces import AuditEvent


class InMemoryAuditSink:
    """Thread-safe sink useful for local development and contract tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[AuditEvent] = []

    def emit(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(event)

    def events(self) -> List[AuditEvent]:
        with self._lock:
            return list(self._events)