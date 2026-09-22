"""Provider-neutral workflow execution contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol


class WorkflowState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class WorkflowDefinition:
    workflow_id: str
    name: str
    tasks: Any
    session_id: Any
    max_workers: int = 4
    timeout_seconds: int = 600
    cancel_on_failure: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowRun:
    run_id: str
    workflow_id: str
    state: WorkflowState
    result: Any = None


class WorkflowEngine(Protocol):
    def start(self, definition: WorkflowDefinition, input_data: Mapping[str, Any] | None = None) -> WorkflowRun:
        ...

    def get_status(self, run_id: str) -> WorkflowRun:
        ...

    def cancel(self, run_id: str, reason: str = "") -> None:
        ...