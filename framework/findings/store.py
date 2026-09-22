"""In-process, queryable findings store (R9).

Persists every finding produced by an agent run so callers can query
"all HIGH-severity findings for tenant X in the last 30 days" instead of
only receiving findings once, synchronously. Deployment-scale storage
(a real database) can replace this store behind the same interface.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class FindingRecord:
    finding_id: str
    tenant_id: str
    task_id: str
    agent_type: str
    severity: str
    finding_type: str
    summary: str
    entity_id: Optional[str]
    confidence_score: float
    recorded_at: str = field(default_factory=_now)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id, "tenant_id": self.tenant_id,
            "task_id": self.task_id, "agent_type": self.agent_type,
            "severity": self.severity, "finding_type": self.finding_type,
            "summary": self.summary, "entity_id": self.entity_id,
            "confidence_score": self.confidence_score,
            "recorded_at": self.recorded_at, "evidence": self.evidence,
        }


class FindingsStore:
    """Thread-safe, queryable, tenant-scoped findings store."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: List[FindingRecord] = []

    def add(self, record: FindingRecord) -> None:
        with self._lock:
            self._records.append(record)

    def add_result(self, tenant_id: str, task_id: str, agent_type: str, result: Any) -> None:
        """Persist every finding from an AgentResult under the owning tenant."""
        for finding in getattr(result, "findings", []):
            self.add(FindingRecord(
                finding_id=finding.finding_id, tenant_id=tenant_id, task_id=task_id,
                agent_type=agent_type, severity=finding.severity.value,
                finding_type=finding.finding_type, summary=finding.summary,
                entity_id=finding.entity_id, confidence_score=finding.confidence_score,
                evidence=dict(finding.evidence),
            ))

    def query(
        self, tenant_id: Optional[str] = None, severity: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[FindingRecord]:
        with self._lock:
            records = list(self._records)
        if tenant_id is not None:
            records = [r for r in records if r.tenant_id == tenant_id]
        if severity is not None:
            records = [r for r in records if r.severity == severity]
        if since is not None:
            records = [r for r in records if r.recorded_at >= since]
        return records

    def get(self, finding_id: str) -> Optional[FindingRecord]:
        with self._lock:
            return next((r for r in self._records if r.finding_id == finding_id), None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
