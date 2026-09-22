"""Dead-letter capture for permanently-failed tasks (R10)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DeadLetterEntry:
    task_id: str
    agent_type: str
    tenant_id: Optional[str]
    attempts: int
    error: Optional[str]
    payload: Dict[str, Any]
    failed_at: str = field(default_factory=_now)


class DeadLetterQueue:
    """Thread-safe in-memory dead-letter queue with replay support.

    Deployment-scale durability (a real queue/table) can replace this
    behind the same interface; the orchestrator only depends on
    `capture`/`list_entries`/`replay`.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: List[DeadLetterEntry] = []

    def capture(self, task: Any, error: Optional[str]) -> DeadLetterEntry:
        entry = DeadLetterEntry(
            task_id=str(task.task_id), agent_type=task.agent_type,
            tenant_id=str(task.tenant_id) if task.tenant_id else None,
            attempts=task.attempt_number, error=error, payload=dict(task.payload),
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    def list_entries(self, tenant_id: Optional[str] = None) -> List[DeadLetterEntry]:
        with self._lock:
            entries = list(self._entries)
        if tenant_id is not None:
            entries = [e for e in entries if e.tenant_id == tenant_id]
        return entries

    def replay(self, task_id: str, orchestrator: Any) -> Any:
        """Re-submit a dead-lettered task's payload as a fresh task via
        the orchestrator; caller supplies the orchestrator because the
        queue itself has no agent-registry dependency."""
        with self._lock:
            entry = next((e for e in self._entries if e.task_id == task_id), None)
        if entry is None:
            raise KeyError(f"no dead-letter entry for task_id={task_id!r}")
        from framework.core.types import Task, TenantId
        task = Task.create(
            agent_type=entry.agent_type, payload=entry.payload,
            tenant_id=TenantId(entry.tenant_id) if entry.tenant_id else None,
        )
        return orchestrator.run_task(task)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
