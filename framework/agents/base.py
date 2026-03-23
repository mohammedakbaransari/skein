"""
framework/agents/base.py
=========================
BaseAgent hierarchy with structured logging, correlation context propagation,
per-agent metrics recording, and production-grade error handling.

CHANGES FROM v1
===============
- Structured logging via AgentLogger (JSON-serialisable, includes trace_id)
- Correlation context propagated through every execute() call
- Metrics recorded on every run (agent_run_started / agent_run_finished)
- First-person inconsistency fixed: all log messages reference agent_id
- StructuralAgent.execute() re-raises from inner block — no double-catch
- DecisionAgent.should_escalate() now configurable via subclass property
"""

from __future__ import annotations

import abc
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from framework.core.types import (
    AgentCapability, AgentId, AgentMetadata, AgentResult, AgentStatus,
    CorrelationContext, DecisionAuthority, Finding, SessionId,
    Severity, Task, TaskId, TaskStatus,
)
from framework.observability.logging import get_logger, correlation_context
from framework.observability.metrics import get_metrics

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()



if TYPE_CHECKING:
    from framework.core.registry import AgentRegistry
    from framework.memory.store import MemoryStore
    from framework.reasoning.engine import ReasoningEngine
    from framework.governance.logger import GovernanceLogger

MiddlewareFn = Callable[["BaseAgent", Task, Callable], AgentResult]


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent(abc.ABC):
    """
    Root agent class. All framework agents inherit from this.

    Provides:
      - Identity (agent_id, name, version via METADATA)
      - Structured logging with agent context auto-injected
      - Correlation context propagation through run()
      - Metrics recording (start / finish counts, duration)
      - Lifecycle hooks (startup, shutdown, task start/complete)
      - Middleware stack (composable execution wrappers)
      - Thread-safe task counter
      - Plugin interface (memory, reasoning, governance)

    Subclasses MUST define:
      - METADATA: AgentMetadata  (class attribute)
      - execute(task) → AgentResult
    """

    METADATA: AgentMetadata  # Required on every subclass

    def __init__(
        self,
        agent_id:            Optional[AgentId] = None,
        config:              Optional[Any] = None,
        memory:              Optional["MemoryStore"] = None,
        reasoning_engine:    Optional["ReasoningEngine"] = None,
        governance_logger:   Optional["GovernanceLogger"] = None,
    ) -> None:
        self.agent_id:   AgentId = agent_id or AgentId.generate()
        self.config:     Any     = config
        self.memory:     Optional["MemoryStore"]     = memory
        self.reasoning:  Optional["ReasoningEngine"] = reasoning_engine
        self.governance: Optional["GovernanceLogger"] = governance_logger

        self._middleware_stack: List[MiddlewareFn] = []
        self._task_lock:        threading.Lock = threading.Lock()
        self._active_task_count: int = 0

        # Structured logger with agent context pre-injected
        self._log = get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
            agent_id=str(self.agent_id),
            agent_type=self.agent_type,
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.METADATA.display_name

    @property
    def agent_type(self) -> str:
        return self.METADATA.agent_type

    @property
    def version(self) -> str:
        return self.METADATA.version

    @property
    def capabilities(self) -> tuple:
        return self.METADATA.capabilities

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def execute(self, task: Task) -> AgentResult:
        """
        Execute one task and return a structured AgentResult.

        Contract:
          - MUST return AgentResult (never raise — catch internally)
          - MUST set result.succeeded = False on error
          - MUST NOT modify task.payload
          - SHOULD propagate task.context into result.context
        """

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_startup(self) -> None: pass
    def on_shutdown(self) -> None: pass
    def on_task_start(self, task: Task) -> None: pass
    def on_task_complete(self, task: Task, result: AgentResult) -> None: pass

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    def use(self, middleware: MiddlewareFn) -> "BaseAgent":
        self._middleware_stack.append(middleware)
        return self

    def _build_pipeline(self) -> Callable[[Task], AgentResult]:
        pipeline: Callable[[Task], AgentResult] = self.execute
        for mw in reversed(self._middleware_stack):
            outer_mw = mw
            inner_fn = pipeline

            def make_wrapped(m=outer_mw, f=inner_fn):
                def wrapped(task: Task) -> AgentResult:
                    return m(self, task, f)
                return wrapped

            pipeline = make_wrapped()
        return pipeline

    # ------------------------------------------------------------------
    # Orchestrated run — called by the orchestrator
    # ------------------------------------------------------------------

    def run(self, task: Task) -> AgentResult:
        """
        Orchestrated task execution with full lifecycle management.

        Injects correlation context so all log records within this call
        include trace_id and span_id automatically.
        """
        t_start = time.monotonic()
        task.status     = TaskStatus.RUNNING
        task.started_at = _now()

        with self._task_lock:
            self._active_task_count += 1

        metrics = get_metrics()
        metrics.agent_run_started(self.agent_type)

        # Inject task's correlation context into logging scope
        child_ctx = task.context.child(
            agent_type=self.agent_type,
            task_id=str(task.task_id),
        )

        self._log.info(
            "Starting task",
            extra={
                "task_id":    str(task.task_id),
                "session_id": str(task.session_id),
                "attempt":    task.attempt_number,
            },
        )

        try:
            with correlation_context(child_ctx):
                self.on_task_start(task)
                pipeline = self._build_pipeline()
                result   = pipeline(task)

        except Exception as exc:
            self._log.error(
                "Unhandled exception in agent execution",
                extra={"task_id": str(task.task_id), "error": str(exc)},
                exc_info=True,
            )
            result = AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_name=self.name,
                agent_version=self.version,
                session_id=task.session_id,
                context=child_ctx,
                succeeded=False,
                error=f"{type(exc).__name__}: {exc}",
                attempt_number=task.attempt_number,
            )
        finally:
            elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
            result.duration_ms     = elapsed_ms
            result.context         = child_ctx
            result.attempt_number  = task.attempt_number
            task.status            = TaskStatus.SUCCEEDED if result.succeeded else TaskStatus.FAILED
            task.completed_at      = _now()
            task.result            = result

            with self._task_lock:
                self._active_task_count = max(0, self._active_task_count - 1)

            # Record metrics
            findings_by_severity = {}
            for f in result.findings:
                findings_by_severity[f.severity.value] = (
                    findings_by_severity.get(f.severity.value, 0) + 1
                )
            metrics.agent_run_finished(
                self.agent_type,
                succeeded=result.succeeded,
                duration_ms=elapsed_ms,
                findings=findings_by_severity,
            )

            self._log.info(
                "%s in %.0f ms — %d findings",
                "SUCCEEDED" if result.succeeded else "FAILED",
                elapsed_ms, len(result.findings),
                extra={"task_id": str(task.task_id), "duration_ms": elapsed_ms},
            )
            self.on_task_complete(task, result)
            self._log_to_governance(task, result)

        return result

    # ------------------------------------------------------------------
    # Governance logging
    # ------------------------------------------------------------------

    def _log_to_governance(self, task: Task, result: AgentResult) -> None:
        if self.governance:
            try:
                self.governance.record_execution(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    task=task,
                    result=result,
                )
            except Exception as exc:
                self._log.warning("Governance logging failed: %s", exc)

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def recall(self, key: str, session_id: Optional[SessionId] = None) -> Optional[Any]:
        if self.memory:
            return self.memory.get(key, session_id=session_id)
        return None

    def remember(
        self, key: str, value: Any, session_id: Optional[SessionId] = None
    ) -> None:
        if self.memory:
            self.memory.set(key, value, session_id=session_id, agent_id=self.agent_id)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def active_task_count(self) -> int:
        with self._task_lock:
            return self._active_task_count

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id}, type={self.agent_type}, v={self.version})"
        )

    @staticmethod
    def _parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
        """Strip markdown fences and parse JSON from LLM output."""
        import json as _json, re
        clean = text.strip() if text else ""
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", clean, re.DOTALL)
        if fence:
            clean = fence.group(1)
        try:
            result = _json.loads(clean)
            return result if isinstance(result, dict) else {"_value": result}
        except _json.JSONDecodeError:
            return None


# ---------------------------------------------------------------------------
# StructuralAgent — observe / reason / parse_findings pipeline
# ---------------------------------------------------------------------------

class StructuralAgent(BaseAgent, abc.ABC):
    """
    Domain base for structural intelligence agents.

    Pipeline: observe() → reason() → parse_findings()

    observe() is a pure function (no LLM, no I/O) — fully unit-testable.
    reason() calls the pluggable ReasoningEngine.
    parse_findings() converts LLM text to typed Finding objects.

    Subclasses implement the three abstract methods.
    execute() is implemented here — do not override.
    """

    def execute(self, task: Task) -> AgentResult:
        observations: Dict[str, Any] = {}
        reasoning_text: str = ""

        try:
            self._log.debug("OBSERVE phase", extra={"task_id": str(task.task_id)})
            observations = self.observe(task)

            self._log.debug("REASON phase", extra={"task_id": str(task.task_id)})
            reasoning_text = self.reason(observations, task)

            self._log.debug("PARSE phase", extra={"task_id": str(task.task_id)})
            findings = self.parse_findings(observations, reasoning_text, task)

            # Cache observations for downstream agents
            self.remember(
                f"observations:{task.task_id}",
                observations,
                session_id=task.session_id,
            )

            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_name=self.name,
                agent_version=self.version,
                session_id=task.session_id,
                findings=findings,
                summary=self._build_summary(findings),
                reasoning_trace=reasoning_text,
                observations=observations,
                succeeded=True,
            )

        except Exception as exc:
            self._log.error(
                "StructuralAgent pipeline failed",
                extra={"task_id": str(task.task_id), "error": str(exc)},
                exc_info=True,
            )
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_name=self.name,
                agent_version=self.version,
                session_id=task.session_id,
                succeeded=False,
                error=f"{type(exc).__name__}: {exc}",
                observations=observations,
                reasoning_trace=reasoning_text,
            )

    @abc.abstractmethod
    def observe(self, task: Task) -> Dict[str, Any]:
        """Extract and structure data from task.payload. No LLM calls."""

    @abc.abstractmethod
    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        """Call ReasoningEngine with structured observations. Return raw text."""

    @abc.abstractmethod
    def parse_findings(
        self,
        observations: Dict[str, Any],
        reasoning:    str,
        task:         Task,
    ) -> List[Finding]:
        """Parse LLM output into typed Finding objects."""

    def _build_summary(self, findings: List[Finding]) -> str:
        if not findings:
            return "No findings produced."
        critical = sum(1 for f in findings if f.severity == Severity.CRITICAL)
        high     = sum(1 for f in findings if f.severity == Severity.HIGH)
        top      = findings[0].summary[:120] if findings else ""
        return f"{len(findings)} findings: {critical} critical, {high} high. Top: {top}"

    def _make_finding(
        self,
        finding_type:       str,
        severity:           Severity,
        summary:            str,
        detail:             str = "",
        entity_id:          Optional[str] = None,
        entity_name:        Optional[str] = None,
        recommended_action: str = "",
        evidence:           Optional[Dict[str, Any]] = None,
        confidence:         float = 1.0,
    ) -> Finding:
        return Finding(
            finding_type=finding_type,
            severity=severity,
            entity_id=entity_id,
            entity_name=entity_name,
            summary=summary,
            detail=detail,
            evidence=evidence or {},
            recommended_action=recommended_action,
            decision_authority=self._authority_for_severity(severity),
            confidence_score=max(0.0, min(1.0, confidence)),
            tags=self.METADATA.tags,
        )

    @staticmethod
    def _authority_for_severity(severity: Severity) -> DecisionAuthority:
        return {
            Severity.CRITICAL: DecisionAuthority.ESCALATE,
            Severity.HIGH:     DecisionAuthority.RECOMMEND,
            Severity.MEDIUM:   DecisionAuthority.ADVISE,
            Severity.LOW:      DecisionAuthority.ADVISE,
            Severity.INFO:     DecisionAuthority.ADVISE,
        }[severity]


# ---------------------------------------------------------------------------
# ToolAgent
# ---------------------------------------------------------------------------

class ToolAgent(BaseAgent, abc.ABC):
    """Agent that executes external tools (ERP, market data, document parsers)."""

    @abc.abstractmethod
    def get_tool_calls(self, task: Task) -> List["ToolCall"]:
        """Determine which tools to call."""

    @abc.abstractmethod
    def aggregate_results(
        self, tool_results: List["ToolResult"], task: Task
    ) -> AgentResult:
        """Aggregate tool results into a single AgentResult."""

    def execute(self, task: Task) -> AgentResult:
        calls   = self.get_tool_calls(task)
        results = []
        for call in calls:
            self._log.debug("Calling tool %s", call.tool_name)
            results.append(self._execute_tool(call))
        return self.aggregate_results(results, task)

    def _execute_tool(self, call: "ToolCall") -> "ToolResult":
        from framework.tools.base import ToolRegistry
        tool = ToolRegistry.get(call.tool_name)
        if tool is None:
            return ToolResult(tool_name=call.tool_name, succeeded=False,
                              error=f"Tool '{call.tool_name}' not registered")
        try:
            output = tool.invoke(call.arguments)
            return ToolResult(tool_name=call.tool_name, output=output, succeeded=True)
        except Exception as exc:
            return ToolResult(tool_name=call.tool_name, succeeded=False, error=str(exc))


# ---------------------------------------------------------------------------
# DecisionAgent
# ---------------------------------------------------------------------------

class DecisionAgent(StructuralAgent, abc.ABC):
    """
    Agent with formal decision-making authority.

    Extends StructuralAgent with:
      - Decision records for Mystery 13 accountability
      - Human-in-the-loop escalation hooks
      - Configurable confidence gating

    MIN_CONFIDENCE_TO_DECIDE: class attribute, override in subclasses.
    """

    MIN_CONFIDENCE_TO_DECIDE: float = 0.7

    def execute(self, task: Task) -> AgentResult:
        result = super().execute(task)
        if not result.succeeded:
            return result
        if self.should_escalate(result.findings):
            self._log.info(
                "Escalating to human review",
                extra={"task_id": str(task.task_id),
                       "findings_count": len(result.findings)},
            )
            return self.escalate(task, result)
        self._record_decision(task, result)
        return result

    def should_escalate(self, findings: List[Finding]) -> bool:
        if any(f.severity == Severity.CRITICAL for f in findings):
            return True
        if findings:
            avg = sum(f.confidence_score for f in findings) / len(findings)
            if avg < self.MIN_CONFIDENCE_TO_DECIDE:
                return True
        return False

    def escalate(self, task: Task, result: AgentResult) -> AgentResult:
        result.metadata["escalated"] = True
        result.metadata["escalation_reason"] = (
            "Critical findings or low confidence triggered human review"
        )
        self._log.warning(
            "Decision escalated",
            extra={"task_id": str(task.task_id)},
        )
        if self.governance:
            try:
                self.governance.record_escalation(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    task=task,
                    result=result,
                    reason=result.metadata["escalation_reason"],
                )
            except Exception as exc:
                self._log.warning("Escalation governance logging failed: %s", exc)
        return result

    def _record_decision(self, task: Task, result: AgentResult) -> None:
        import hashlib
        record = {
            "task_id":           str(task.task_id),
            "agent_type":        self.agent_type,
            "session_id":        str(task.session_id),
            "trace_id":          task.context.trace_id,
            "findings_count":    len(result.findings),
            "finding_severities": [f.severity.value for f in result.findings],
            "reasoning_hash":    hashlib.sha256(
                result.reasoning_trace.encode()
            ).hexdigest()[:16] if result.reasoning_trace else "",
            "decided_at":        _now(),
        }
        self.remember(f"decision:{task.task_id}", record, session_id=task.session_id)
        if self.governance:
            self.governance.record_decision(
                agent_id=self.agent_id,
                task=task,
                result=result,
                decision_record=record,
            )


# ---------------------------------------------------------------------------
# Supporting types for ToolAgent
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    tool_name:  str
    arguments:  Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    tool_name: str
    succeeded: bool = True
    output:    Any = None
    error:     Optional[str] = None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
