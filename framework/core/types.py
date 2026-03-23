"""
framework/core/types.py
========================
Canonical domain types for SKEIN — Structural Knowledge and Enterprise
Intelligence Network.

Changes from v1:
  - CorrelationContext: carries trace_id, span_id, parent_span_id through
    every layer for distributed tracing (OpenTelemetry compatible)
  - Task.context: injects CorrelationContext for end-to-end tracing
  - AgentResult.context: propagates context through result chain
  - RetryConfig: standardised retry policy type used by resilience layer
  - All types remain JSON-serialisable for inter-process communication
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()




# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AgentStatus(Enum):
    REGISTERED  = "registered"
    IDLE        = "idle"
    RUNNING     = "running"
    SUSPENDED   = "suspended"
    FAILED      = "failed"
    TERMINATED  = "terminated"


class TaskStatus(Enum):
    PENDING   = "pending"
    QUEUED    = "queued"
    RUNNING   = "running"
    SUCCEEDED = "succeeded"
    FAILED    = "failed"
    CANCELLED = "cancelled"
    TIMEOUT   = "timeout"
    RETRYING  = "retrying"


class Severity(Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    INFO     = "info"


class DecisionAuthority(Enum):
    RECOMMEND = "recommend"
    ADVISE    = "advise"
    EXECUTE   = "execute"
    ESCALATE  = "escalate"


class ReasoningStrategy(Enum):
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT            = "react"
    PLAN_AND_EXECUTE = "plan_and_execute"
    REFLEXION        = "reflexion"
    STRUCTURED       = "structured"


class CircuitState(Enum):
    CLOSED    = "closed"    # Normal operation
    OPEN      = "open"      # Rejecting calls
    HALF_OPEN = "half_open" # Probing recovery


# ---------------------------------------------------------------------------
# Typed value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentId:
    value: str

    @classmethod
    def generate(cls) -> "AgentId":
        return cls(f"agent-{uuid.uuid4().hex[:12]}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class TaskId:
    value: str

    @classmethod
    def generate(cls) -> "TaskId":
        return cls(f"task-{uuid.uuid4().hex[:12]}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SessionId:
    value: str

    @classmethod
    def generate(cls) -> "SessionId":
        return cls(f"session-{uuid.uuid4().hex[:12]}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SpanId:
    value: str

    @classmethod
    def generate(cls) -> "SpanId":
        return cls(uuid.uuid4().hex[:16])

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# CorrelationContext — distributed tracing (OTel-compatible)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorrelationContext:
    """
    Carries trace context through the entire agent call chain.

    Compatible with OpenTelemetry trace propagation format.
    Injected into every Task and propagated through AgentResult.

    Fields:
        trace_id:       Unique identifier for the entire request tree.
        span_id:        Identifier for this specific operation.
        parent_span_id: Span ID of the calling operation (None for root).
        baggage:        Arbitrary key-value context (user_id, workflow_id, etc.)
    """
    trace_id:       str = field(default_factory=lambda: uuid.uuid4().hex)
    span_id:        str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: Optional[str] = None
    baggage:        Dict[str, str] = field(default_factory=dict)

    def child(self, **extra_baggage: str) -> "CorrelationContext":
        """Create a child span context inheriting this span as parent."""
        merged = {**self.baggage, **extra_baggage}
        return CorrelationContext(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=self.span_id,
            baggage=merged,
        )

    def to_headers(self) -> Dict[str, str]:
        """Serialise to HTTP headers for outgoing requests."""
        headers = {
            "X-Trace-Id":  self.trace_id,
            "X-Span-Id":   self.span_id,
        }
        if self.parent_span_id:
            headers["X-Parent-Span-Id"] = self.parent_span_id
        for k, v in self.baggage.items():
            headers[f"X-Baggage-{k}"] = v
        return headers

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> "CorrelationContext":
        """Reconstruct context from incoming HTTP headers."""
        baggage = {
            k.replace("X-Baggage-", "").lower(): v
            for k, v in headers.items()
            if k.startswith("X-Baggage-")
        }
        return cls(
            trace_id=headers.get("X-Trace-Id", uuid.uuid4().hex),
            span_id=headers.get("X-Span-Id", uuid.uuid4().hex[:16]),
            parent_span_id=headers.get("X-Parent-Span-Id"),
            baggage=baggage,
        )

    @classmethod
    def new(cls, **baggage: str) -> "CorrelationContext":
        """Create a fresh root context."""
        return cls(baggage=baggage)


# ---------------------------------------------------------------------------
# RetryConfig — standardised retry policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetryConfig:
    """
    Retry policy used by the resilience layer and reasoning engine.

    max_attempts:      Total attempts including the first try.
    initial_delay_s:   Delay before second attempt.
    backoff_factor:    Multiplier applied per retry (exponential backoff).
    max_delay_s:       Cap on delay between retries.
    jitter_factor:     Fraction of delay to randomise (0.0–1.0).
    retryable_errors:  Exception types that trigger retry. None = all.
    """
    max_attempts:     int   = 3
    initial_delay_s:  float = 1.0
    backoff_factor:   float = 2.0
    max_delay_s:      float = 30.0
    jitter_factor:    float = 0.2
    retryable_errors: Optional[Tuple[type, ...]] = None

    @classmethod
    def aggressive(cls) -> "RetryConfig":
        """For critical operations — retry quickly."""
        return cls(max_attempts=5, initial_delay_s=0.5, max_delay_s=10.0)

    @classmethod
    def conservative(cls) -> "RetryConfig":
        """For expensive LLM calls — retry slowly."""
        return cls(max_attempts=3, initial_delay_s=2.0, max_delay_s=60.0)

    @classmethod
    def no_retry(cls) -> "RetryConfig":
        return cls(max_attempts=1)


# ---------------------------------------------------------------------------
# Task — the unit of work dispatched to agents
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """
    A unit of work submitted to the orchestration layer.

    Mutable: status and result fields are updated during execution.

    New in v2:
        context: CorrelationContext for distributed tracing
        retry_config: override default retry policy per task
        attempt_number: current attempt (1-based, incremented on retry)
    """
    task_id:         TaskId
    agent_type:      str
    payload:         Dict[str, Any]
    session_id:      SessionId
    context:         CorrelationContext = field(default_factory=CorrelationContext.new)
    priority:        int  = 50
    timeout_seconds: int  = 300
    retry_config:    RetryConfig = field(default_factory=RetryConfig)
    parent_task_id:  Optional[TaskId] = None
    depends_on:      List[TaskId] = field(default_factory=list)
    metadata:        Dict[str, Any] = field(default_factory=dict)
    # Mutable lifecycle fields
    status:          TaskStatus = TaskStatus.PENDING
    result:          Optional["AgentResult"] = None
    error:           Optional[str] = None
    attempt_number:  int = 1
    created_at:      str = field(default_factory=_now)
    started_at:      Optional[str] = None
    completed_at:    Optional[str] = None

    @classmethod
    def create(
        cls,
        agent_type: str,
        payload: Dict[str, Any],
        session_id: Optional[SessionId] = None,
        context: Optional[CorrelationContext] = None,
        **kwargs,
    ) -> "Task":
        sid = session_id or SessionId.generate()
        ctx = context or CorrelationContext.new(session_id=str(sid))
        return cls(
            task_id=TaskId.generate(),
            agent_type=agent_type,
            payload=payload,
            session_id=sid,
            context=ctx,
            **kwargs,
        )

    def for_retry(self) -> "Task":
        """Return a copy of this task for re-execution."""
        import copy
        t = copy.copy(self)
        t.task_id = TaskId.generate()
        t.attempt_number = self.attempt_number + 1
        t.status = TaskStatus.PENDING
        t.result = None
        t.error = None
        t.started_at = None
        t.completed_at = None
        t.context = self.context.child(attempt=str(t.attempt_number))
        return t


# ---------------------------------------------------------------------------
# Finding — atomic insight produced by an agent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    """
    A single insight, risk, or recommendation produced by an agent.
    Immutable — findings are records, never modified after creation.
    """
    finding_id:         str = field(default_factory=lambda: f"fnd-{uuid.uuid4().hex[:10]}")
    finding_type:       str = ""
    severity:           Severity = Severity.INFO
    entity_id:          Optional[str] = None
    entity_name:        Optional[str] = None
    summary:            str = ""
    detail:             str = ""
    evidence:           Dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""
    decision_authority: DecisionAuthority = DecisionAuthority.RECOMMEND
    confidence_score:   float = 1.0
    tags:               Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"]           = self.severity.value
        d["decision_authority"] = self.decision_authority.value
        d["tags"]               = list(self.tags)
        return d


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """
    Complete output from one agent task execution.

    New in v2:
        context: propagated CorrelationContext for tracing
        attempt_number: which retry attempt produced this result
        circuit_breaker_state: state of circuit breaker at time of call
    """
    task_id:               TaskId
    agent_id:              AgentId
    agent_name:            str
    agent_version:         str
    session_id:            SessionId
    context:               CorrelationContext = field(default_factory=CorrelationContext.new)
    findings:              List[Finding] = field(default_factory=list)
    summary:               str = ""
    reasoning_trace:       str = ""
    observations:          Dict[str, Any] = field(default_factory=dict)
    metadata:              Dict[str, Any] = field(default_factory=dict)
    duration_ms:           Optional[float] = None
    llm_tokens_used:       Optional[int] = None
    succeeded:             bool = True
    error:                 Optional[str] = None
    attempt_number:        int = 1
    produced_at:           str = field(default_factory=_now)

    @property
    def critical_findings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == Severity.CRITICAL]

    @property
    def high_findings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == Severity.HIGH]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id":         str(self.task_id),
            "agent_id":        str(self.agent_id),
            "agent_name":      self.agent_name,
            "agent_version":   self.agent_version,
            "session_id":      str(self.session_id),
            "trace_id":        self.context.trace_id,
            "span_id":         self.context.span_id,
            "findings":        [f.to_dict() for f in self.findings],
            "summary":         self.summary,
            "observations":    self.observations,
            "metadata":        self.metadata,
            "duration_ms":     self.duration_ms,
            "llm_tokens_used": self.llm_tokens_used,
            "succeeded":       self.succeeded,
            "error":           self.error,
            "attempt_number":  self.attempt_number,
            "produced_at":     self.produced_at,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Capability and Metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentCapability:
    name:         str
    description:  str
    input_types:  Tuple[str, ...] = field(default_factory=tuple)
    output_types: Tuple[str, ...] = field(default_factory=tuple)
    authority:    DecisionAuthority = DecisionAuthority.RECOMMEND


@dataclass(frozen=True)
class AgentMetadata:
    agent_type:   str
    display_name: str
    description:  str
    version:      str
    capabilities: Tuple[AgentCapability, ...]
    tags:         Tuple[str, ...] = field(default_factory=tuple)
    author:       str = "Mohammed Akbar Ansari"
    mystery_refs: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
