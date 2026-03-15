"""
framework/orchestration/orchestrator.py
=========================================
Task Orchestrator — coordinates multi-agent procurement workflows.

RESPONSIBILITIES:
  1. Accept task submissions (single or batch)
  2. Resolve dependency order (topological sort)
  3. Dispatch tasks to appropriate agents (registry-based routing)
  4. Execute independent tasks concurrently (ThreadPoolExecutor)
  5. Propagate results between dependent tasks
  6. Handle failures — retry or mark dependent tasks CANCELLED
  7. Emit lifecycle events to the governance layer

WORKFLOW CONCEPT:
  A Workflow is a DAG of Tasks.
  The orchestrator resolves the DAG, identifies parallelisable
  batches (tasks with all dependencies met), and executes each batch.

THREAD-SAFETY:
  - ThreadPoolExecutor handles concurrent task execution
  - Task status updates are serialised via a per-workflow lock
  - Results are written to a shared dict under lock before being read
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from framework.core.types import (
    AgentResult, SessionId, Task, TaskId, TaskStatus,
)
from framework.core.registry import AgentRegistry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workflow — a collection of tasks with dependency graph
# ---------------------------------------------------------------------------

@dataclass
class Workflow:
    """
    A directed acyclic graph (DAG) of Tasks.

    Attributes:
        workflow_id:  Unique identifier.
        name:         Human-readable label.
        session_id:   Caller session.
        tasks:        All tasks in this workflow.
        max_workers:  Parallelism limit (default: 4).
        timeout_seconds: Global workflow timeout.
    """
    workflow_id:      str
    name:             str
    session_id:       SessionId
    tasks:            List[Task] = field(default_factory=list)
    max_workers:      int = 4
    timeout_seconds:  int = 600

    def add_task(self, task: Task) -> "Workflow":
        """Fluent builder for adding tasks."""
        self.tasks.append(task)
        return self

    def get_task(self, task_id: TaskId) -> Optional[Task]:
        return next((t for t in self.tasks if t.task_id.value == task_id.value), None)

    def validate_dag(self) -> None:
        """
        Validate that the task dependency graph has no cycles.
        Raises ValueError if a cycle is detected.
        """
        task_ids: Set[str] = {t.task_id.value for t in self.tasks}
        # Check all depends_on references exist
        for task in self.tasks:
            for dep_id in task.depends_on:
                if dep_id.value not in task_ids:
                    raise ValueError(
                        f"Task {task.task_id} depends on unknown task {dep_id}"
                    )
        # Topological sort to detect cycles
        self._topological_order()  # raises ValueError on cycle

    def _topological_order(self) -> List[Task]:
        """Kahn's algorithm for topological sort. Raises ValueError on cycle."""
        in_degree: Dict[str, int] = {t.task_id.value: 0 for t in self.tasks}
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
            for dependent_id in dependents[task.task_id.value]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(task_map[dependent_id])

        if len(ordered) != len(self.tasks):
            raise ValueError(
                f"Workflow '{self.name}' contains a dependency cycle. "
                "Check task.depends_on references."
            )
        return ordered


# ---------------------------------------------------------------------------
# WorkflowResult — the outcome of a completed workflow
# ---------------------------------------------------------------------------

@dataclass
class WorkflowResult:
    """Aggregated result of all tasks in a workflow."""
    workflow_id:    str
    workflow_name:  str
    session_id:     SessionId
    task_results:   Dict[str, AgentResult] = field(default_factory=dict)
    succeeded:      bool = True
    failed_tasks:   List[str] = field(default_factory=list)
    cancelled_tasks: List[str] = field(default_factory=list)
    duration_ms:    Optional[float] = None
    completed_at:   Optional[str] = None

    @property
    def all_findings(self):
        """Flatten findings from all successful tasks."""
        from framework.core.types import Finding
        findings = []
        for result in self.task_results.values():
            if result and result.succeeded:
                findings.extend(result.findings)
        return findings


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TaskOrchestrator:
    """
    Orchestrates multi-agent procurement workflows.

    Usage:
        orchestrator = TaskOrchestrator(registry, config)

        # Single task
        result = orchestrator.run_task(task)

        # Multi-agent workflow
        workflow = Workflow(...)
        workflow.add_task(task1)
        workflow.add_task(task2)
        workflow_result = orchestrator.run_workflow(workflow)
    """

    def __init__(
        self,
        registry: AgentRegistry,
        config: Any,
        event_callback: Optional[Callable] = None,
    ) -> None:
        self._registry = registry
        self._config   = config
        self._event_callback = event_callback

    # ------------------------------------------------------------------
    # Single task execution
    # ------------------------------------------------------------------

    def run_task(self, task: Task) -> AgentResult:
        """
        Execute a single task synchronously.
        Finds a suitable agent from the registry and runs the task.
        """
        agent = self._registry.get_or_create(task.agent_type, self._config)
        self._emit_event("task_started", {"task_id": str(task.task_id),
                                           "agent_type": task.agent_type})
        result = agent.run(task)
        self._emit_event("task_completed", {"task_id": str(task.task_id),
                                             "succeeded": result.succeeded})
        return result

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------

    def run_workflow(self, workflow: Workflow) -> WorkflowResult:
        """
        Execute all tasks in a workflow, respecting dependencies.

        Algorithm:
          1. Validate the DAG
          2. Topologically sort tasks
          3. Execute tasks in batches — each batch contains all tasks
             whose dependencies have been satisfied
          4. Parallelise within each batch up to max_workers
          5. Cascade failures — cancel tasks that depend on a failed task
        """
        t_start = time.monotonic()
        workflow.validate_dag()

        wf_result = WorkflowResult(
            workflow_id=workflow.workflow_id,
            workflow_name=workflow.name,
            session_id=workflow.session_id,
        )
        completed_ids: Set[str] = set()
        failed_ids:    Set[str] = set()
        results:       Dict[str, AgentResult] = {}
        results_lock   = threading.Lock()

        ordered_tasks  = workflow._topological_order()

        self._emit_event("workflow_started", {
            "workflow_id": workflow.workflow_id,
            "task_count":  len(ordered_tasks),
        })

        with ThreadPoolExecutor(max_workers=workflow.max_workers) as pool:
            remaining = list(ordered_tasks)

            while remaining:
                # Identify tasks ready to run (all dependencies met)
                ready = [
                    t for t in remaining
                    if all(d.value in completed_ids for d in t.depends_on)
                    and all(d.value not in failed_ids for d in t.depends_on)
                ]
                # Cancel tasks that cannot run (dependency failed)
                cancellable = [
                    t for t in remaining
                    if any(d.value in failed_ids for d in t.depends_on)
                ]
                for t in cancellable:
                    t.status = TaskStatus.CANCELLED
                    wf_result.cancelled_tasks.append(t.task_id.value)
                    remaining.remove(t)
                    log.warning(
                        "[workflow=%s] Task %s cancelled due to failed dependency",
                        workflow.workflow_id, t.task_id,
                    )

                if not ready:
                    break

                # Inject shared results from dependencies into payload
                for task in ready:
                    for dep_id in task.depends_on:
                        dep_result = results.get(dep_id.value)
                        if dep_result:
                            task.payload[f"_dep_{dep_id.value}"] = dep_result.to_dict()

                # Submit ready tasks concurrently
                futures: Dict[Future, Task] = {
                    pool.submit(self.run_task, task): task
                    for task in ready
                }
                for future in as_completed(futures):
                    task = futures[future]
                    remaining.remove(task)
                    try:
                        result = future.result(
                            timeout=task.timeout_seconds
                        )
                        with results_lock:
                            results[task.task_id.value] = result
                        if result.succeeded:
                            completed_ids.add(task.task_id.value)
                        else:
                            failed_ids.add(task.task_id.value)
                            wf_result.failed_tasks.append(task.task_id.value)
                    except Exception as exc:
                        log.error(
                            "[workflow=%s] Task %s raised exception: %s",
                            workflow.workflow_id, task.task_id, exc,
                        )
                        failed_ids.add(task.task_id.value)
                        wf_result.failed_tasks.append(task.task_id.value)

        duration_ms = round((time.monotonic() - t_start) * 1000, 1)
        wf_result.task_results  = results
        wf_result.succeeded     = len(wf_result.failed_tasks) == 0
        wf_result.duration_ms   = duration_ms
        wf_result.completed_at  = _now()

        self._emit_event("workflow_completed", {
            "workflow_id":  workflow.workflow_id,
            "succeeded":    wf_result.succeeded,
            "duration_ms":  duration_ms,
            "failed_tasks": wf_result.failed_tasks,
        })
        log.info(
            "[workflow=%s] %s in %.0f ms — %d/%d tasks succeeded",
            workflow.workflow_id,
            "SUCCEEDED" if wf_result.succeeded else "FAILED",
            duration_ms,
            len(completed_ids), len(ordered_tasks),
        )
        return wf_result

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception as exc:
                log.warning("Event callback failed: %s", exc)


# ---------------------------------------------------------------------------
# Workflow builder DSL — fluent API for building workflows
# ---------------------------------------------------------------------------

class WorkflowBuilder:
    """
    Fluent DSL for constructing procurement analysis workflows.

    Usage:
        workflow = (
            WorkflowBuilder("supplier-portfolio-review")
            .session(session_id)
            .step("SupplierStressAgent", payload={"data": ...})
            .then("DecisionAuditAgent", depends_on=["step-0"])
            .parallel(
                ("TotalCostAgent",   {"data": tco_data}),
                ("BiasDetectorAgent",{"data": eval_data}),
            )
            .build()
        )
    """

    def __init__(self, name: str) -> None:
        import uuid as _uuid
        self._name       = name
        self._workflow_id = f"wf-{_uuid.uuid4().hex[:10]}"
        self._session_id: Optional[SessionId] = None
        self._tasks:      List[Task] = []
        self._step_ids:   List[str]  = []

    def session(self, session_id: SessionId) -> "WorkflowBuilder":
        self._session_id = session_id
        return self

    def step(
        self,
        agent_type: str,
        payload: Dict[str, Any],
        depends_on: Optional[List[str]] = None,
        priority:   int = 50,
        timeout:    int = 300,
    ) -> "WorkflowBuilder":
        """Add a task step. Returns self for chaining."""
        from framework.core.types import TaskId
        sid = self._session_id or SessionId.generate()
        dep_task_ids = [
            TaskId(dep_label) for dep_label in (depends_on or [])
            if dep_label in self._step_ids
        ]
        # Resolve labels to actual TaskIds stored in tasks
        actual_dep_ids = []
        for label in (depends_on or []):
            match = next((t.task_id for t in self._tasks
                          if t.metadata.get("step_label") == label), None)
            if match:
                actual_dep_ids.append(match)

        task = Task.create(
            agent_type=agent_type,
            payload=payload,
            session_id=sid,
            priority=priority,
            timeout_seconds=timeout,
            depends_on=actual_dep_ids,
            metadata={"step_label": f"step-{len(self._tasks)}"},
        )
        self._tasks.append(task)
        self._step_ids.append(f"step-{len(self._tasks)-1}")
        return self

    def then(
        self,
        agent_type: str,
        payload: Dict[str, Any],
        **kwargs,
    ) -> "WorkflowBuilder":
        """Add a step that depends on the previous step."""
        prev_label = self._step_ids[-1] if self._step_ids else None
        return self.step(
            agent_type,
            payload,
            depends_on=[prev_label] if prev_label else None,
            **kwargs,
        )

    def parallel(self, *steps: Tuple[str, Dict]) -> "WorkflowBuilder":
        """Add multiple independent steps that run in parallel."""
        for agent_type, payload in steps:
            self.step(agent_type, payload)
        return self

    def build(self) -> Workflow:
        return Workflow(
            workflow_id=self._workflow_id,
            name=self._name,
            session_id=self._session_id or SessionId.generate(),
            tasks=self._tasks,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

from typing import Tuple  # noqa — needed for WorkflowBuilder type hint
