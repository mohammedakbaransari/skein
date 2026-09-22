"""Vendor-neutral audit contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Protocol


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AuditEvent:
    """Versioned application event independent of any SIEM transport."""

    event_id: str
    event_type: str
    tenant_id: str
    action: str
    result: str
    principal_id: str = ""
    workflow_id: str = ""
    task_id: str = ""
    agent_id: str = ""
    resource: str = ""
    correlation_id: str = ""
    trace_id: str = ""
    data_classification: str = "internal"
    previous_hash: str = ""
    event_hash: str = ""
    event_version: str = "1.0"
    timestamp: str = field(default_factory=_now)
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditSink(Protocol):
    """Destination-neutral sink for already-normalized audit events."""

    def emit(self, event: AuditEvent) -> None:
        ...