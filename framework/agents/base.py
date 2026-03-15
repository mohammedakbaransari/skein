"""
framework/agents/base.py
=========================
BaseAgent — root of the SKEIN framework agent hierarchy.

WHY THIS MATTERS:
  A production framework separates these into:
    - BaseAgent:         Lifecycle, identity, plugin hooks
    - ProcurementAgent:  Domain-specific patterns for procurement tasks
    - ToolAgent:         Agents that execute external tools
    - DecisionAgent:     Agents with structured decision authority

AGENT HIERARCHY:
    BaseAgent
    ├── ProcurementAgent    (domain: procurement analysis)
    │   ├── SupplierStressAgent
    │   ├── ShouldCostAgent
    │   ├── BiasDetectorAgent
    │   └── ... (all 15 mystery agents)
    ├── ToolAgent           (domain: external tool execution)
    │   ├── ERPConnectorAgent
    │   └── MarketDataAgent
    └── DecisionAgent       (domain: structured decision support)
        └── StrategicProcurementAgent

LIFECYCLE HOOKS:
    on_startup()      — called once after construction
    on_shutdown()     — called before termination
    on_task_start()   — called before each task
    on_task_complete()— called after each task

PLUGIN POINTS:
    Middleware stack: list of callables wrapping execute()
    Pre/post hooks per lifecycle event
"""

from __future__ import annotations

import abc
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from framework.core.types import (
    AgentCapability, AgentId, AgentMetadata, AgentResult, AgentStatus,
    Finding, SessionId, Severity, Task, TaskId, TaskStatus,
)

if TYPE_CHECKING:
    from framework.core.registry import AgentRegistry
    from framework.memory.store import MemoryStore
    from framework.reasoning.engine import ReasoningEngine
    from framework.governance.logger import GovernanceLogger

log = logging.getLogger(__name__)

# Middleware type: wraps execute() calls
MiddlewareFn = Callable[["BaseAgent", Task, Callable], AgentResult]


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent(abc.ABC):
    """
    Root agent class. All framework agents inherit from this.

    Provides:
      - Identity (agent_id, name, version via METADATA)
      - Lifecycle management (startup, shutdown, task hooks)
      - Middleware stack (composable execution wrappers)
      - Thread-safe task counter
      - Plugin interface (memory, reasoning, governance)

    Subclasses MUST define:
      - METADATA: AgentMetadata   (class attribute)
      - execute(task) → AgentResult

    Subclasses MAY override lifecycle hooks.
    """

    # Every subclass must declare METADATA at class level
    METADATA: AgentMetadata  # abstract class attribute — enforced at registry.register_class

    def __init__(
        self,
        agent_id: Optional[AgentId] = None,
        config: Optional[Any] = None,
        memory: Optional["MemoryStore"] = None,
        reasoning_engine: Optional["ReasoningEngine"] = None,
        governance_logger: Optional["GovernanceLogger"] = None,
    ) -> None:
        self.agent_id:   AgentId  = agent_id or AgentId.generate()
        self.config:     Any      = config
        self.memory:     Optional["MemoryStore"] = memory
        self.reasoning:  Optional["ReasoningEngine"] = reasoning_engine
        self.governance: Optional["GovernanceLogger"] = governance_logger

        self._middleware_stack: List[MiddlewareFn] = []
        self._task_lock: threading.Lock = threading.Lock()
        self._active_task_count: int = 0
        self._log = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
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
    # Abstract interface — subclasses implement this
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def execute(self, task: Task) -> AgentResult:
        """
        Execute one task and return a structured AgentResult.

        Contract:
          - MUST return an AgentResult (never raise — catch internally)
          - MUST set result.succeeded = False on error
          - MUST NOT modify task.payload
          - SHOULD set result.reasoning_trace for audit
        """

    # ------------------------------------------------------------------
    # Lifecycle hooks — override in subclasses as needed
    # ------------------------------------------------------------------

    def on_startup(self) -> None:
        """Called once after the agent is constructed and registered."""
        pass

    def on_shutdown(self) -> None:
        """Called before the agent is terminated."""
        pass

    def on_task_start(self, task: Task) -> None:
        """Called immediately before execute(). Override for pre-task setup."""
        pass

    def on_task_complete(self, task: Task, result: AgentResult) -> None:
        """Called immediately after execute(). Override for post-task cleanup."""
        pass

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    def use(self, middleware: MiddlewareFn) -> "BaseAgent":
        """
        Add a middleware function to the stack.
        Middleware wraps execute() and can inspect/modify task/result.
        Returns self for fluent chaining.

        Middleware signature:
            def my_middleware(agent, task, next_fn) -> AgentResult:
                # pre-processing
                result = next_fn(task)
                # post-processing
                return result
        """
        self._middleware_stack.append(middleware)
        return self

    def _build_pipeline(self) -> Callable[[Task], AgentResult]:
        """
        Compose middleware stack into a single callable.
        Innermost = execute(), middleware wrap from outside in.
        """
        pipeline: Callable[[Task], AgentResult] = self.execute
        for mw in reversed(self._middleware_stack):
            outer_mw = mw
            inner_fn = pipeline

            def make_wrapped(mw=outer_mw, fn=inner_fn):
                def wrapped(task: Task) -> AgentResult:
                    return mw(self, task, fn)
                return wrapped

            pipeline = make_wrapped()
        return pipeline

    # ------------------------------------------------------------------
    # Orchestrated run — called by the orchestrator
    # ------------------------------------------------------------------

    def run(self, task: Task) -> AgentResult:
        """
        Orchestrated task execution with full lifecycle management.

        Called by the orchestrator (not directly by callers).
        Handles:
          - Pre/post lifecycle hooks
          - Thread-safe task counter
          - Timing
          - Exception boundary (never propagates)
          - Governance logging
        """
        t_start = time.monotonic()
        task.status     = TaskStatus.RUNNING
        task.started_at = _now()

        with self._task_lock:
            self._active_task_count += 1

        self._log.info(
            "[agent=%s] [task=%s] [session=%s] Starting task",
            self.agent_id, task.task_id, task.session_id,
        )

        try:
            self.on_task_start(task)
            pipeline = self._build_pipeline()
            result   = pipeline(task)
        except Exception as exc:
            self._log.error(
                "[agent=%s] [task=%s] Unhandled exception: %s",
                self.agent_id, task.task_id, exc, exc_info=True,
            )
            result = AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_name=self.name,
                agent_version=self.version,
                session_id=task.session_id,
                succeeded=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
            result.duration_ms = elapsed_ms
            task.status       = TaskStatus.SUCCEEDED if result.succeeded else TaskStatus.FAILED
            task.completed_at = _now()
            task.result       = result

            with self._task_lock:
                self._active_task_count = max(0, self._active_task_count - 1)

            self._log.info(
                "[agent=%s] [task=%s] %s in %.0f ms, %d findings",
                self.agent_id, task.task_id,
                "SUCCEEDED" if result.succeeded else "FAILED",
                elapsed_ms, len(result.findings),
            )
            self.on_task_complete(task, result)
            self._log_to_governance(task, result)

        return result

    # ------------------------------------------------------------------
    # Governance logging
    # ------------------------------------------------------------------

    def _log_to_governance(self, task: Task, result: AgentResult) -> None:
        """Write execution record to governance layer if configured."""
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
        """Retrieve a value from memory if memory layer is configured."""
        if self.memory:
            return self.memory.get(key, session_id=session_id)
        return None

    def remember(self, key: str, value: Any, session_id: Optional[SessionId] = None) -> None:
        """Store a value in memory if memory layer is configured."""
        if self.memory:
            self.memory.set(key, value, session_id=session_id)

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
            f"id={self.agent_id}, "
            f"type={self.agent_type}, "
            f"v={self.version})"
        )

    @staticmethod
    def _parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Strip markdown fences and parse JSON from LLM output.

        Handles:
          - Raw JSON
          - ```json ... ``` fences
          - ``` ... ``` fences

        Returns:
          Parsed dict, or None if parsing fails.
          Wraps non-dict JSON (list, scalar) in {"_value": ...}.
        """
        import json as _json
        import re
        clean = text.strip() if text else ""
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", clean, re.DOTALL)
        if fence:
            clean = fence.group(1)
        try:
            result = _json.loads(clean)
            if isinstance(result, dict):
                return result
            return {"_value": result}
        except _json.JSONDecodeError:
            return None


# ---------------------------------------------------------------------------
# ProcurementAgent — domain base for all procurement intelligence agents
# ---------------------------------------------------------------------------

class ProcurementAgent(BaseAgent, abc.ABC):
    """
    Domain-specific base for procurement intelligence agents.

    Adds on top of BaseAgent:
      - observe() / reason() / parse_findings() pipeline
        (the Observe→Reason→Act pattern — now in the right layer)
      - Structured output enforcement via reasoning engine
      - Procurement-specific Finding construction helpers

    Subclasses implement:
      observe(task)  → Dict[str, Any]
      reason(observations, task) → str    (LLM reasoning text)
      parse_findings(observations, reasoning, task) → List[Finding]

    The execute() method is implemented here — it orchestrates the pipeline.
    Subclasses do NOT override execute().
    """

    def execute(self, task: Task) -> AgentResult:
        """
        Orchestrate observe → reason → parse_findings.
        Override the three abstract methods below instead.
        """
        observations: Dict[str, Any] = {}
        reasoning_text: str = ""

        try:
            self._log.info(
                "[agent=%s] [task=%s] OBSERVE", self.agent_id, task.task_id
            )
            observations = self.observe(task)

            self._log.info(
                "[agent=%s] [task=%s] REASON", self.agent_id, task.task_id
            )
            reasoning_text = self.reason(observations, task)

            self._log.info(
                "[agent=%s] [task=%s] PARSE", self.agent_id, task.task_id
            )
            findings = self.parse_findings(observations, reasoning_text, task)

            # Cache observations in memory for downstream agents
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
                "[agent=%s] [task=%s] ProcurementAgent pipeline failed: %s",
                self.agent_id, task.task_id, exc, exc_info=True,
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
        """Call the LLM with structured observations. Return raw reasoning text."""

    @abc.abstractmethod
    def parse_findings(
        self,
        observations: Dict[str, Any],
        reasoning: str,
        task: Task,
    ) -> List[Finding]:
        """Parse LLM output into typed Finding objects."""

    def _build_summary(self, findings: List[Finding]) -> str:
        """Build a brief summary from findings list."""
        if not findings:
            return "No findings produced."
        critical = sum(1 for f in findings if f.severity == Severity.CRITICAL)
        high     = sum(1 for f in findings if f.severity == Severity.HIGH)
        return (
            f"{len(findings)} findings: "
            f"{critical} critical, {high} high. "
            f"Top: {findings[0].summary[:120] if findings else ''}"
        )

    def _make_finding(
        self,
        finding_type: str,
        severity: Severity,
        summary: str,
        detail: str = "",
        entity_id: Optional[str] = None,
        entity_name: Optional[str] = None,
        recommended_action: str = "",
        evidence: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
    ) -> Finding:
        """Convenience factory for creating well-formed Finding objects."""
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
    def _authority_for_severity(severity: Severity) -> "DecisionAuthority":
        from framework.core.types import DecisionAuthority
        return {
            Severity.CRITICAL: DecisionAuthority.ESCALATE,
            Severity.HIGH:     DecisionAuthority.RECOMMEND,
            Severity.MEDIUM:   DecisionAuthority.ADVISE,
            Severity.LOW:      DecisionAuthority.ADVISE,
            Severity.INFO:     DecisionAuthority.ADVISE,
        }[severity]


# ---------------------------------------------------------------------------
# ToolAgent — agents that execute external tools
# ---------------------------------------------------------------------------

class ToolAgent(BaseAgent, abc.ABC):
    """
    Agent that executes one or more registered tools.

    ToolAgents do not call the LLM directly — they call tools
    (ERP connectors, market data APIs, file parsers) and return
    structured results.

    Tool orchestration strategies:
      - Sequential:  execute tools one by one
      - Parallel:    execute independent tools concurrently
      - Conditional: execute tool B only if tool A returns X

    Subclasses implement:
      get_tool_calls(task) → List[ToolCall]
      aggregate_results(tool_results, task) → AgentResult
    """

    @abc.abstractmethod
    def get_tool_calls(self, task: Task) -> List["ToolCall"]:
        """Determine which tools to call and with what arguments."""

    @abc.abstractmethod
    def aggregate_results(
        self,
        tool_results: List["ToolResult"],
        task: Task,
    ) -> AgentResult:
        """Aggregate tool results into a single AgentResult."""

    def execute(self, task: Task) -> AgentResult:
        """Execute tool calls sequentially and aggregate."""
        tool_calls  = self.get_tool_calls(task)
        tool_results = []
        for call in tool_calls:
            self._log.debug(
                "[agent=%s] Calling tool %s", self.agent_id, call.tool_name
            )
            result = self._execute_tool(call)
            tool_results.append(result)
        return self.aggregate_results(tool_results, task)

    def _execute_tool(self, call: "ToolCall") -> "ToolResult":
        """Execute one tool call. Override for custom tool execution."""
        from framework.tools.base import ToolRegistry
        tool = ToolRegistry.get(call.tool_name)
        if tool is None:
            return ToolResult(
                tool_name=call.tool_name,
                succeeded=False,
                error=f"Tool '{call.tool_name}' not found in registry",
            )
        try:
            output = tool.invoke(call.arguments)
            return ToolResult(tool_name=call.tool_name, output=output, succeeded=True)
        except Exception as exc:
            return ToolResult(
                tool_name=call.tool_name, succeeded=False, error=str(exc)
            )


# ---------------------------------------------------------------------------
# DecisionAgent — agents with structured decision authority
# ---------------------------------------------------------------------------

class DecisionAgent(ProcurementAgent, abc.ABC):
    """
    Agent with formal decision-making authority.

    Extends ProcurementAgent with:
      - Structured decision records (not just findings)
      - Decision rationale capture (for Mystery 13 compliance)
      - Human-in-the-loop escalation hooks
      - Decision confidence gating

    Subclasses additionally implement:
      should_escalate(findings) → bool
      escalate(task, findings)  → AgentResult  (human handoff)
    """

    # Minimum confidence required to proceed without escalation
    MIN_CONFIDENCE_TO_DECIDE: float = 0.7

    def execute(self, task: Task) -> AgentResult:
        """
        ProcurementAgent pipeline + decision authority layer.
        Low-confidence or critical findings trigger escalation.
        """
        result = super().execute(task)
        if not result.succeeded:
            return result

        # Check escalation criteria
        if self.should_escalate(result.findings):
            self._log.info(
                "[agent=%s] [task=%s] Escalating to human review",
                self.agent_id, task.task_id,
            )
            return self.escalate(task, result)

        # Log decision record for accountability (Mystery 13)
        self._record_decision(task, result)
        return result

    def should_escalate(self, findings: List[Finding]) -> bool:
        """
        Return True to trigger human escalation.
        Default: escalate if any CRITICAL finding or avg confidence < threshold.
        Override in subclasses with domain-specific rules.
        """
        if any(f.severity == Severity.CRITICAL for f in findings):
            return True
        if findings:
            avg_conf = sum(f.confidence_score for f in findings) / len(findings)
            if avg_conf < self.MIN_CONFIDENCE_TO_DECIDE:
                return True
        return False

    def escalate(self, task: Task, result: AgentResult) -> AgentResult:
        """
        Human escalation handoff.
        Default: return result with escalation metadata.
        Override to integrate with ticketing, Slack, email, etc.
        """
        result.metadata["escalated"] = True
        result.metadata["escalation_reason"] = (
            "Critical findings or low confidence triggered human review"
        )
        self._log.warning(
            "[agent=%s] [task=%s] Decision escalated to human",
            self.agent_id, task.task_id,
        )
        return result

    def _record_decision(self, task: Task, result: AgentResult) -> None:
        """Record decision rationale for Mystery 13 accountability."""
        decision_record = {
            "task_id":          str(task.task_id),
            "agent_type":       self.agent_type,
            "session_id":       str(task.session_id),
            "findings_count":   len(result.findings),
            "finding_severities": [f.severity.value for f in result.findings],
            "reasoning_hash":   _sha256_short(result.reasoning_trace),
            "decided_at":       _now(),
        }
        self.remember(
            f"decision:{task.task_id}",
            decision_record,
            session_id=task.session_id,
        )
        if self.governance:
            self.governance.record_decision(
                agent_id=self.agent_id,
                task=task,
                result=result,
                decision_record=decision_record,
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
# Utilities
# ---------------------------------------------------------------------------

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _sha256_short(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]
