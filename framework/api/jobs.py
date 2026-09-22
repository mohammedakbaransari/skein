"""Async task submission job store (R7).

Provides `submit -> job id -> poll` semantics on top of the existing
synchronous `TaskOrchestrator.run_task` without requiring a durable
workflow engine — a thread pool executes the task in the background and
the caller polls `JobStore.get_status()` for the result.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    tenant_id: Optional[str]
    state: JobState = JobState.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=_now)
    completed_at: Optional[str] = None


class JobStore:
    """Thread-safe async job registry backed by a bounded thread pool."""

    def __init__(self, max_workers: int = 8) -> None:
        self._lock = threading.RLock()
        self._jobs: Dict[str, Job] = {}
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="skein-async-task")
        self._usage_ledger = None

    def configure_usage_ledger(self, usage_ledger: Any) -> None:
        self._usage_ledger = usage_ledger

    def submit(self, orchestrator: Any, task: Any) -> Job:
        job = Job(
            job_id=f"job-{uuid.uuid4().hex[:12]}",
            tenant_id=str(task.tenant_id) if task.tenant_id else None,
        )
        with self._lock:
            self._jobs[job.job_id] = job

        def _run() -> None:
            with self._lock:
                job.state = JobState.RUNNING
            try:
                result = orchestrator.run_task(task)
                if self._usage_ledger is not None and task.tenant_id:
                    self._usage_ledger.record(
                        str(task.tenant_id), result.llm_tokens_used or max(1, len(str(task.payload)) // 4),
                        str(task.task_id),
                    )
                with self._lock:
                    job.result = result
                    job.state = JobState.SUCCEEDED if result.succeeded else JobState.FAILED
                    job.error = None if result.succeeded else result.error
                    job.completed_at = _now()
            except Exception as exc:  # pragma: no cover - defensive
                with self._lock:
                    job.state = JobState.FAILED
                    job.error = str(exc)
                    job.completed_at = _now()

        self._pool.submit(_run)
        return job

    def submit_workflow(self, orchestrator: Any, workflow: Any) -> Job:
        """Async workflow-level submission (R7 remainder) — same job lifecycle
        as `submit`, executed through the `WorkflowEngine` adapter (R0) rather
        than calling the orchestrator's workflow runner directly."""
        tenant_id = next(
            (str(t.tenant_id) for t in workflow.tasks if t.tenant_id), None
        )
        job = Job(job_id=f"job-{uuid.uuid4().hex[:12]}", tenant_id=tenant_id)
        with self._lock:
            self._jobs[job.job_id] = job

        def _run() -> None:
            with self._lock:
                job.state = JobState.RUNNING
            try:
                from framework.adapters.workflow import InMemoryWorkflowEngine, WorkflowDefinition
                definition = WorkflowDefinition(
                    workflow_id=workflow.workflow_id, name=workflow.name, tasks=workflow.tasks,
                    session_id=workflow.session_id, max_workers=workflow.max_workers,
                    timeout_seconds=workflow.timeout_seconds, cancel_on_failure=workflow.cancel_on_failure,
                )
                result = InMemoryWorkflowEngine(orchestrator).start(definition).result
                if self._usage_ledger is not None and job.tenant_id:
                    tokens = sum(
                        r.llm_tokens_used or max(1, len(str(t.payload)) // 4)
                        for t, r in zip(workflow.tasks, result.task_results.values())
                    )
                    self._usage_ledger.record(job.tenant_id, tokens, job.job_id)
                with self._lock:
                    job.result = result
                    job.state = JobState.SUCCEEDED if result.succeeded else JobState.FAILED
                    job.completed_at = _now()
            except Exception as exc:  # pragma: no cover - defensive
                with self._lock:
                    job.state = JobState.FAILED
                    job.error = str(exc)
                    job.completed_at = _now()

        self._pool.submit(_run)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
