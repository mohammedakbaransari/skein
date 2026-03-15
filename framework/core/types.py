"""
framework/core/types.py
========================
Canonical domain types for the SKEIN framework.

These types are the shared language across all layers:
  core → agents → reasoning → orchestration → governance

Design principles:
  - Immutable where shared across threads (frozen dataclasses)
  - Serialisable to JSON for persistence and inter-process comms
  - Typed with full annotations — no Dict[str, Any] at boundaries
  - Extensible via Protocol rather than inheritance where possible
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AgentStatus(Enum):
    """Lifecycle status of a registered agent."""
    REGISTERED  = "registered"   # In registry, not yet started
    IDLE        = "idle"         # Started, waiting for tasks
    RUNNING     = "running"      # Actively processing a task
    SUSPENDED   = "suspended"    # Temporarily paused
    FAILED      = "failed"       # Unrecoverable error
    TERMINATED  = "terminated"   # Gracefully shut down


class TaskStatus(Enum):
    """Status of a task in the orchestration pipeline."""
    PENDING     = "pending"
    QUEUED      = "queued"
    RUNNING     = "running"
    SUCCEEDED   = "succeeded"
    FAILED      = "failed"
    CANCELLED   = "cancelled"
    TIMEOUT     = "timeout"


class Severity(Enum):
    """Standardised severity levels for actions and findings."""
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    INFO     = "info"


class DecisionAuthority(Enum):
    """
    Authority level an agent holds for a given decision type.

    RECOMMEND:  Agent proposes; human must approve.
    ADVISE:     Agent provides analysis; human decides.
    EXECUTE:    Agent may act autonomously within defined guardrails.
    ESCALATE:   Agent flags for senior human review.
    """
    RECOMMEND = "recommend"
    ADVISE    = "advise"
    EXECUTE   = "execute"
    ESCALATE  = "escalate"


class ReasoningStrategy(Enum):
    """Available reasoning strategy types."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT            = "react"           # Reason + Act loop
    PLAN_AND_EXECUTE = "plan_and_execute"
    REFLEXION        = "reflexion"       # Self-critique loop
    STRUCTURED       = "structured"      # JSON schema-constrained output


# ---------------------------------------------------------------------------
# Core value objects (immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentId:
    """Typed wrapper for agent identifiers — prevents stringly-typed bugs."""
    value: str

    @classmethod
    def generate(cls) -> "AgentId":
        return cls(f"agent-{uuid.uuid4().hex[:12]}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class TaskId:
    """Typed wrapper for task identifiers."""
    value: str

    @classmethod
    def generate(cls) -> "TaskId":
        return cls(f"task-{uuid.uuid4().hex[:12]}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SessionId:
    """Typed wrapper for session identifiers."""
    value: str

    @classmethod
    def generate(cls) -> "SessionId":
        return cls(f"session-{uuid.uuid4().hex[:12]}")

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Task — the unit of work dispatched to agents
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """
    A unit of work submitted to the orchestration layer.

    Mutable: status and result fields are updated during execution.
    NOT frozen — orchestrator updates lifecycle fields in place.

    Fields:
        task_id:         Unique identifier.
        agent_type:      Target agent class name or capability tag.
        payload:         Input data for the agent.
        session_id:      Caller session for tracing.
        priority:        0 = highest, 100 = lowest.
        timeout_seconds: Max allowed execution time.
        parent_task_id:  Set when this task was spawned by another task.
        depends_on:      Task IDs that must complete before this task starts.
        metadata:        Arbitrary key-value pairs for routing / filtering.
        status:          Lifecycle status (updated by orchestrator).
        result:          Output once completed.
        error:           Error message if failed.
        created_at:      Creation timestamp.
        started_at:      Execution start timestamp.
        completed_at:    Execution completion timestamp.
    """
    task_id:          TaskId
    agent_type:       str
    payload:          Dict[str, Any]
    session_id:       SessionId
    priority:         int = 50
    timeout_seconds:  int = 300
    parent_task_id:   Optional[TaskId] = None
    depends_on:       List[TaskId] = field(default_factory=list)
    metadata:         Dict[str, Any] = field(default_factory=dict)
    # Mutable lifecycle fields
    status:           TaskStatus = TaskStatus.PENDING
    result:           Optional["AgentResult"] = None
    error:            Optional[str] = None
    created_at:       str = field(default_factory=lambda: _now())
    started_at:       Optional[str] = None
    completed_at:     Optional[str] = None

    @classmethod
    def create(
        cls,
        agent_type: str,
        payload: Dict[str, Any],
        session_id: Optional[SessionId] = None,
        **kwargs,
    ) -> "Task":
        return cls(
            task_id=TaskId.generate(),
            agent_type=agent_type,
            payload=payload,
            session_id=session_id or SessionId.generate(),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Finding — atomic insight produced by an agent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    """
    A single insight, risk, or recommendation produced by an agent.

    frozen=True — findings are immutable records; never modify after creation.
    """
    finding_id:         str = field(default_factory=lambda: f"fnd-{uuid.uuid4().hex[:10]}")
    finding_type:       str = ""          # e.g. "supplier_stress", "cost_gap"
    severity:           Severity = Severity.INFO
    entity_id:          Optional[str] = None    # e.g. supplier ID, contract ID
    entity_name:        Optional[str] = None
    summary:            str = ""
    detail:             str = ""
    evidence:           Dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""
    decision_authority: DecisionAuthority = DecisionAuthority.RECOMMEND
    confidence_score:   float = 1.0       # 0.0–1.0
    tags:               tuple = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"]           = self.severity.value
        d["decision_authority"] = self.decision_authority.value
        d["tags"]               = list(self.tags)
        return d


# ---------------------------------------------------------------------------
# AgentResult — what an agent returns from a task
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """
    Complete output from one agent task execution.

    Contains findings (typed domain objects) plus the raw LLM reasoning
    for audit purposes.
    """
    task_id:          TaskId
    agent_id:         AgentId
    agent_name:       str
    agent_version:    str
    session_id:       SessionId
    findings:         List[Finding] = field(default_factory=list)
    summary:          str = ""
    reasoning_trace:  str = ""          # raw LLM output for audit
    observations:     Dict[str, Any] = field(default_factory=dict)
    metadata:         Dict[str, Any] = field(default_factory=dict)
    duration_ms:      Optional[float] = None
    llm_tokens_used:  Optional[int] = None
    succeeded:        bool = True
    error:            Optional[str] = None
    produced_at:      str = field(default_factory=lambda: _now())

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
            "findings":        [f.to_dict() for f in self.findings],
            "summary":         self.summary,
            "observations":    self.observations,
            "metadata":        self.metadata,
            "duration_ms":     self.duration_ms,
            "llm_tokens_used": self.llm_tokens_used,
            "succeeded":       self.succeeded,
            "error":           self.error,
            "produced_at":     self.produced_at,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# AgentCapability — declared by agents in the registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentCapability:
    """
    Declared capability of an agent.
    Used by the registry and orchestrator for routing.
    """
    name:        str              # e.g. "supplier_stress_detection"
    description: str
    input_types: tuple = field(default_factory=tuple)   # accepted payload keys
    output_types: tuple = field(default_factory=tuple)  # produced finding types
    authority:   DecisionAuthority = DecisionAuthority.RECOMMEND


# ---------------------------------------------------------------------------
# AgentMetadata — static description of an agent class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentMetadata:
    """
    Static metadata about an agent class, registered once at startup.
    Separate from runtime state (status, active tasks).
    """
    agent_type:    str
    display_name:  str
    description:   str
    version:       str
    capabilities:  tuple               # Tuple[AgentCapability, ...]
    tags:          tuple = field(default_factory=tuple)
    author:        str = "Mohammed Akbar Ansari"
    mystery_refs:  tuple = field(default_factory=tuple)  # e.g. ("mystery_02",)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
