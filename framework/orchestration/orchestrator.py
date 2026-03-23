"""
framework/orchestration/orchestrator.py
=========================================
Task Orchestrator with production-grade resilience.

CHANGES FROM v1
===============
- Uses AgentPoolManager for agent acquisition (back-pressure, isolation)
- Per-task retry: tasks retry up to task.retry_config.max_attempts
- Correlation context propagated through every task in a workflow
- Timeout enforcement with proper cancellation (not blocking future.result)
- Structured logging with workflow-level trace_id
- Metrics: workflow duration and status recorded
- WorkflowBuilder step-label resolution fixed (TaskId-based, not string match)
- cancel_on_failure flag: workflows can opt to fail-fast or continue on error
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from framework.core.types import (
    AgentResult, CorrelationContext, RetryConfig, SessionId,
    Task, TaskId, TaskStatus,
)
from framework.core.registry import AgentRegistry
from framework.observability.logging import correlation_context, get_logger
from framework.observability.metrics import get_metrics

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()



log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@dataclass
class Workflow:
    """
    Directed acyclic graph (DAG) of Tasks.

    New in v2:
        context: CorrelationContext shared across all tasks
        cancel_on_failure: if True, cancel pending tasks when one fails
    """
    workflow_id:       str
    name:              str
    session_id:        SessionId
    context:           CorrelationContext = field(default_factory=CorrelationContext.new)
    tasks:             List[Task] = field(default_factory=list)
    max_workers:       int = 4
    timeout_seconds:   int = 600
    cancel_on_failure: bool = True

    def add_task(self, task: Task) -> "Workflow":
        self.tasks.append(task)
        return self

    def get_task(self, task_id: TaskId) -> Optional[Task]:
        return next((t for t in self.tasks if t.task_id.value == task_id.value), None)

    def validate_dag(self) -> None:
        """Validate DAG: all depends_on references exist and no cycles."""
        task_ids = {t.task_id.value for t in self.tasks}
        for task in self.tasks:
            for dep_id in task.depends_on:
                if dep_id.value not in task_ids:
                    raise ValueError(
                        f"Task {task.task_id} depends on unknown task {dep_id}"
                    )
        self._topological_order()  # raises on cycle

    def _topological_order(self) -> List[Task]:
        """Kahn's algorithm. Raises ValueError on cycle."""
        in_degree:  Dict[str, int]       = {t.task_id.value: 0 for t in self.tasks}
        dependents: Dict[str, List[str]] = {t.task_id.value: [] for t in self.tasks}
        for task in self.tasks:
            for dep in task.depends_on:
                in_degree[task.task_id.value] += 1
                dependents[dep.value].append(task.task_id.value)
        queue = [t for t in self.tasks if in_degree[t.task_id.value] == 0]
        ordered: List[Task] = []
        task_map = {t.task_id.value: t for t in self.tasks}
        while queue:
            task = queue.pop(0)
            ordered.append(task)
            for dep_id in dependents[task.task_id.value]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(task_map[dep_id])
        if len(ordered) != len(self.tasks):
            raise ValueError(
                f"Workflow '{self.name}' has a dependency cycle."
            )
        return ordered


# ---------------------------------------------------------------------------
# WorkflowResult
# ---------------------------------------------------------------------------

@dataclass
class WorkflowResult:
    """Aggregated result of all tasks in a workflow."""
    workflow_id:     str
    workflow_name:   str
    session_id:      SessionId
    context:         CorrelationContext = field(default_factory=CorrelationContext.new)
    task_results:    Dict[str, AgentResult] = field(default_factory=dict)
    succeeded:       bool = True
    failed_tasks:    List[str] = field(default_factory=list)
    cancelled_tasks: List[str] = field(default_factory=list)
    retried_tasks:   List[str] = field(default_factory=list)
    duration_ms:     Optional[float] = None
    completed_at:    Optional[str] = None

    @property
    def all_findings(self):
        findings = []
        for result in self.task_results.values():
            if result and result.succeeded:
                findings.extend(result.findings)
        return findings


# ---------------------------------------------------------------------------
# TaskOrchestrator
# ---------------------------------------------------------------------------

class TaskOrchestrator:
    """
    Orchestrates multi-agent workflows with pool management and retry.

    Agents are acquired from AgentPoolManager (provides back-pressure).
    Tasks with retry_config will be retried on failure before cancelling
    dependent tasks.
    Correlation context is propagated through every task execution.
    """

    def __init__(
        self,
        registry:        AgentRegistry,
        config:          Any,
        pool_manager=    None,   # Optional[AgentPoolManager]
        event_callback:  Optional[Callable] = None,
    ) -> None:
        self._registry      = registry
        self._config        = config
        self._pool_manager  = pool_manager
        self._event_callback = event_callback

    # ------------------------------------------------------------------
    # Single task
    # ------------------------------------------------------------------

    def run_task(self, task: Task) -> AgentResult:
        """
        Execute a single task, acquiring an agent from the pool.
        Retries per task.retry_config before returning failure.
        """
        self._emit_event("task_started", {
            "task_id": str(task.task_id),
            "agent_type": task.agent_type,
            "attempt": task.attempt_number,
            "trace_id": task.context.trace_id,
        })

        current_task = task
        last_result: Optional[AgentResult] = None
        cfg = task.retry_config

        for attempt in range(1, cfg.max_attempts + 1):
            agent = self._acquire_agent(current_task.agent_type)
            try:
                result = agent.run(current_task)
                last_result = result
                if result.succeeded:
                    self._emit_event("task_completed", {
                        "task_id": str(task.task_id),
                        "succeeded": True,
                        "attempt": attempt,
                    })
                    return result
                # Failed — check if retryable
                if attempt < cfg.max_attempts:
                    delay = self._compute_retry_delay(cfg, attempt)
                    log.info(
                        "Task failed on attempt %d/%d — retrying in %.1fs",
                        attempt, cfg.max_attempts, delay,
                        extra={"task_id": str(task.task_id)},
                    )
                    time.sleep(delay)
                    current_task = task.for_retry()

            finally:
                self._release_agent(current_task.agent_type, agent)

        # All attempts exhausted
        self._emit_event("task_completed", {
            "task_id":   str(task.task_id),
            "succeeded": False,
            "attempts":  cfg.max_attempts,
        })
        return last_result or AgentResult(
            task_id=task.task_id,
            agent_id=self._registry.get_or_create(task.agent_type, self._config).agent_id,
            agent_name=task.agent_type,
            agent_version="unknown",
            session_id=task.session_id,
            succeeded=False,
            error=f"All {cfg.max_attempts} attempts failed",
        )

    def _acquire_agent(self, agent_type: str):
        if self._pool_manager:
            return self._pool_manager.acquire(agent_type)
        return self._registry.get_or_create(agent_type, self._config)

    def _release_agent(self, agent_type: str, agent) -> None:
        if self._pool_manager:
            self._pool_manager.release(agent_type, agent)

    @staticmethod
    def _compute_retry_delay(cfg: RetryConfig, attempt: int) -> float:
        import random
        base  = min(
            cfg.initial_delay_s * (cfg.backoff_factor ** (attempt - 1)),
            cfg.max_delay_s,
        )
        jitter = base * cfg.jitter_factor
        return base + random.uniform(-jitter, jitter)

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------

    def run_workflow(self, workflow: Workflow) -> WorkflowResult:
        """
        Execute all tasks in a workflow, respecting dependencies.
        Uses parallel execution within each dependency-resolved batch.
        """
        t_start = time.monotonic()
        workflow.validate_dag()

        wf_ctx = workflow.context.child(
            workflow_id=workflow.workflow_id,
            workflow_name=workflow.name,
        )
        wf_result = WorkflowResult(
            workflow_id=workflow.workflow_id,
            workflow_name=workflow.name,
            session_id=workflow.session_id,
            context=wf_ctx,
        )
        completed_ids: Set[str] = set()
        failed_ids:    Set[str] = set()
        results:       Dict[str, AgentResult] = {}
        results_lock   = threading.Lock()

        ordered = workflow._topological_order()

        self._emit_event("workflow_started", {
            "workflow_id": workflow.workflow_id,
            "task_count":  len(ordered),
            "trace_id":    wf_ctx.trace_id,
        })

        log.info(
            "Workflow started: %s (%d tasks)",
            workflow.name, len(ordered),
            extra={"workflow_id": workflow.workflow_id, "trace_id": wf_ctx.trace_id},
        )

        with correlation_context(wf_ctx):
            with ThreadPoolExecutor(max_workers=workflow.max_workers) as pool:
                remaining = list(ordered)

                while remaining:
                    # Cancel tasks whose dependencies failed
                    if workflow.cancel_on_failure:
                        cancellable = [
                            t for t in remaining
                            if any(d.value in failed_ids for d in t.depends_on)
                        ]
                        for t in cancellable:
                            t.status = TaskStatus.CANCELLED
                            wf_result.cancelled_tasks.append(t.task_id.value)
                            remaining.remove(t)
                            log.warning(
                                "Task %s cancelled (dependency failed)", t.task_id,
                                extra={"workflow_id": workflow.workflow_id},
                            )

                    # Ready: all deps completed successfully
                    ready = [
                        t for t in remaining
                        if all(d.value in completed_ids for d in t.depends_on)
                    ]
                    if not ready:
                        break

                    # Inject dependency results into task payloads
                    for task in ready:
                        task.context = wf_ctx.child(
                            task_id=str(task.task_id),
                            agent_type=task.agent_type,
                        )
                        for dep_id in task.depends_on:
                            dep_result = results.get(dep_id.value)
                            if dep_result:
                                task.payload[f"_dep_{dep_id.value}"] = dep_result.to_dict()

                    # Submit ready tasks concurrently
                    futures: Dict[Future, Task] = {
                        pool.submit(self.run_task, t): t
                        for t in ready
                    }
                    for future in as_completed(futures, timeout=workflow.timeout_seconds):
                        task = futures[future]
                        remaining.remove(task)
                        try:
                            result = future.result()
                            with results_lock:
                                results[task.task_id.value] = result
                            if result.succeeded:
                                completed_ids.add(task.task_id.value)
                            else:
                                failed_ids.add(task.task_id.value)
                                wf_result.failed_tasks.append(task.task_id.value)
                        except Exception as exc:
                            log.error(
                                "Task %s raised exception: %s", task.task_id, exc,
                                extra={"workflow_id": workflow.workflow_id},
                                exc_info=True,
                            )
                            failed_ids.add(task.task_id.value)
                            wf_result.failed_tasks.append(task.task_id.value)

        duration_ms = round((time.monotonic() - t_start) * 1000, 1)
        wf_result.task_results  = results
        wf_result.succeeded     = len(wf_result.failed_tasks) == 0
        wf_result.duration_ms   = duration_ms
        wf_result.completed_at  = _now()

        get_metrics().workflow_finished(
            workflow.name, wf_result.succeeded, duration_ms
        )
        self._emit_event("workflow_completed", {
            "workflow_id":  workflow.workflow_id,
            "succeeded":    wf_result.succeeded,
            "duration_ms":  duration_ms,
            "failed_tasks": wf_result.failed_tasks,
            "trace_id":     wf_ctx.trace_id,
        })
        log.info(
            "Workflow %s: %s in %.0f ms — %d/%d tasks succeeded",
            workflow.name,
            "SUCCEEDED" if wf_result.succeeded else "FAILED",
            duration_ms,
            len(completed_ids), len(ordered),
            extra={"workflow_id": workflow.workflow_id},
        )
        return wf_result

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception as exc:
                log.warning("Event callback failed: %s", exc)


# ---------------------------------------------------------------------------
# WorkflowBuilder DSL — fixed step-label resolution
# ---------------------------------------------------------------------------

class WorkflowBuilder:
    """
    Fluent DSL for building procurement analysis workflows.

    Fixed in v2:
    - Step labels are now stored as TaskId values (not string keys)
    - then() correctly resolves the previous task's actual TaskId
    - parallel() does not create false dependencies between parallel steps
    """

    def __init__(self, name: str) -> None:
        self._name        = name
        self._workflow_id = f"wf-{uuid.uuid4().hex[:10]}"
        self._session_id: Optional[SessionId] = None
        self._context:    Optional[CorrelationContext] = None
        self._tasks:      List[Task] = []
        self._last_task_id: Optional[TaskId] = None

    def session(self, session_id: SessionId) -> "WorkflowBuilder":
        self._session_id = session_id
        return self

    def trace(self, context: CorrelationContext) -> "WorkflowBuilder":
        self._context = context
        return self

    def step(
        self,
        agent_type:  str,
        payload:     Dict[str, Any],
        depends_on:  Optional[List[TaskId]] = None,
        priority:    int = 50,
        timeout:     int = 300,
        retry:       Optional[RetryConfig] = None,
    ) -> "WorkflowBuilder":
        sid = self._session_id or SessionId.generate()
        ctx = self._context or CorrelationContext.new(session_id=str(sid))
        task = Task.create(
            agent_type=agent_type,
            payload=payload,
            session_id=sid,
            context=ctx,
            priority=priority,
            timeout_seconds=timeout,
            depends_on=depends_on or [],
            metadata={"step_index": len(self._tasks)},
            retry_config=retry or RetryConfig(),
        )
        self._tasks.append(task)
        self._last_task_id = task.task_id
        return self

    def then(
        self,
        agent_type: str,
        payload:    Dict[str, Any],
        **kwargs,
    ) -> "WorkflowBuilder":
        """Add a step that sequentially depends on the previous step."""
        prev = [self._last_task_id] if self._last_task_id else []
        return self.step(agent_type, payload, depends_on=prev, **kwargs)

    def parallel(self, *steps: Tuple[str, Dict]) -> "WorkflowBuilder":
        """Add multiple independent steps that run in parallel."""
        saved = self._last_task_id
        for agent_type, payload in steps:
            self.step(agent_type, payload, depends_on=None)
        # last_task_id points to the last parallel step added
        return self

    def build(self) -> Workflow:
        sid = self._session_id or SessionId.generate()
        ctx = self._context or CorrelationContext.new(session_id=str(sid))
        return Workflow(
            workflow_id=self._workflow_id,
            name=self._name,
            session_id=sid,
            context=ctx,
            tasks=self._tasks,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
