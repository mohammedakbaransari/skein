"""Reference workflow adapter backed by the existing in-process orchestrator."""

from __future__ import annotations

import threading
import uuid
from dataclasses import replace
from typing import Any, Mapping

from framework.adapters.workflow.interfaces import (
    WorkflowDefinition,
    WorkflowRun,
    WorkflowState,
)
from framework.orchestration.orchestrator import Workflow


class InMemoryWorkflowEngine:
    """Adapter that preserves the current runner behind the portable contract."""

    def __init__(self, orchestrator) -> None:
        self._orchestrator = orchestrator
        self._lock = threading.RLock()
        self._runs: dict[str, WorkflowRun] = {}

    def start(
        self,
        definition: WorkflowDefinition,
        input_data: Mapping[str, Any] | None = None,
    ) -> WorkflowRun:
        del input_data
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        self._set_run(WorkflowRun(run_id, definition.workflow_id, WorkflowState.RUNNING))
        workflow = Workflow(
            workflow_id=definition.workflow_id,
            name=definition.name,
            session_id=definition.session_id,
            tasks=list(definition.tasks),
            max_workers=definition.max_workers,
            timeout_seconds=definition.timeout_seconds,
            cancel_on_failure=definition.cancel_on_failure,
        )
        result = self._orchestrator.run_workflow(workflow)
        state = WorkflowState.SUCCEEDED if result.succeeded else WorkflowState.FAILED
        completed = WorkflowRun(run_id, definition.workflow_id, state, result)
        self._set_run(completed)
        return completed

    def get_status(self, run_id: str) -> WorkflowRun:
        with self._lock:
            try:
                return self._runs[run_id]
            except KeyError as exc:
                raise KeyError(f"unknown workflow run {run_id!r}") from exc

    def cancel(self, run_id: str, reason: str = "") -> None:
        del reason
        with self._lock:
            current = self.get_status(run_id)
            if current.state in (WorkflowState.RUNNING, WorkflowState.PENDING):
                self._runs[run_id] = replace(current, state=WorkflowState.CANCELLED)

    def _set_run(self, run: WorkflowRun) -> None:
        with self._lock:
            self._runs[run.run_id] = run