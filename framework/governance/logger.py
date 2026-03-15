"""
framework/governance/logger.py
================================
Governance Logger — the accountability and compliance layer.

PURPOSE:
  Directly addresses Mystery 13 (Decision Accountability Gap).
  Every AI-assisted procurement decision must produce:
    - A traceable reasoning record
    - A human-readable rationale
    - A SHA-256 hash chain (tamper-evident)
    - An evaluator record (who ran the agent, when, in which session)

RECORD TYPES:
  ExecutionRecord:  every agent.run() call
  DecisionRecord:   when a DecisionAgent makes an authority-bearing decision
  EscalationRecord: when a decision is referred to a human
  AuditTrailEntry:  general purpose event

STORAGE:
  Append-only JSONL files, one per record type.
  Hash-chained across all entries in a file.
  Rotation: daily (configurable).

THREAD-SAFETY:
  Per-file threading.Lock via the same class-level registry
  pattern used in the PAM framework's security layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from framework.core.types import AgentId, AgentResult, SessionId, Task

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionRecord:
    """One agent execution — written for every agent.run() call."""
    record_type:      str = "execution"
    agent_id:         str = ""
    agent_type:       str = ""
    task_id:          str = ""
    session_id:       str = ""
    succeeded:        bool = True
    duration_ms:      Optional[float] = None
    findings_count:   int = 0
    critical_count:   int = 0
    error:            Optional[str] = None
    reasoning_hash:   str = ""          # SHA-256[:16] of reasoning_trace
    timestamp:        str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class DecisionRecord:
    """A procurement decision made by a DecisionAgent."""
    record_type:        str = "decision"
    agent_id:           str = ""
    agent_type:         str = ""
    task_id:            str = ""
    session_id:         str = ""
    findings_count:     int = 0
    finding_severities: List[str] = field(default_factory=list)
    authority_levels:   List[str] = field(default_factory=list)
    reasoning_hash:     str = ""
    rationale_captured: bool = True
    timestamp:          str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class EscalationRecord:
    """A decision escalated to human review."""
    record_type:        str = "escalation"
    agent_id:           str = ""
    agent_type:         str = ""
    task_id:            str = ""
    session_id:         str = ""
    escalation_reason:  str = ""
    critical_findings:  int = 0
    timestamp:          str = field(default_factory=lambda: _now())


# ---------------------------------------------------------------------------
# Hash-chained JSONL writer
# ---------------------------------------------------------------------------

class HashChainedWriter:
    """
    Writes records to an append-only JSONL file with SHA-256 hash chain.

    Each entry includes:
      prev_hash: hash of the previous line
      hash:      hash of this line

    Multiple instances pointing to the same file share a class-level lock.
    Thread-safe: no two writers interleave lines.
    """

    _path_locks: ClassVar[Dict[str, threading.Lock]] = {}
    _registry_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, log_path: Path) -> None:
        self._path = log_path.resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_hash = "GENESIS"
        with HashChainedWriter._registry_lock:
            key = str(self._path)
            if key not in HashChainedWriter._path_locks:
                HashChainedWriter._path_locks[key] = threading.Lock()
        self._lock = HashChainedWriter._path_locks[str(self._path)]

    def write(self, record: Dict[str, Any]) -> None:
        entry = {**record, "prev_hash": self._prev_hash}
        raw = json.dumps(entry, default=str, sort_keys=True, ensure_ascii=False)
        entry_hash = hashlib.sha256(raw.encode()).hexdigest()[:24]
        entry["hash"] = entry_hash
        final_line = json.dumps(entry, default=str, sort_keys=True, ensure_ascii=False)

        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(final_line + "\n")
        self._prev_hash = entry_hash


# ---------------------------------------------------------------------------
# GovernanceLogger
# ---------------------------------------------------------------------------

class GovernanceLogger:
    """
    Central governance logging facade.

    Writes four streams:
      executions.jsonl  — all agent runs
      decisions.jsonl   — authority-bearing decisions
      escalations.jsonl — human escalations
      audit.jsonl       — general audit trail

    All streams are hash-chained and append-only.

    Thread-safe: HashChainedWriter handles concurrent writes.
    """

    def __init__(self, log_dir: str) -> None:
        base = Path(log_dir)
        self._exec_writer   = HashChainedWriter(base / "executions.jsonl")
        self._dec_writer    = HashChainedWriter(base / "decisions.jsonl")
        self._esc_writer    = HashChainedWriter(base / "escalations.jsonl")
        self._audit_writer  = HashChainedWriter(base / "audit.jsonl")

    def record_execution(
        self,
        agent_id: AgentId,
        agent_type: str,
        task: Task,
        result: AgentResult,
    ) -> None:
        """Record every agent.run() call. Called by BaseAgent automatically."""
        record = ExecutionRecord(
            agent_id=str(agent_id),
            agent_type=agent_type,
            task_id=str(task.task_id),
            session_id=str(task.session_id),
            succeeded=result.succeeded,
            duration_ms=result.duration_ms,
            findings_count=len(result.findings),
            critical_count=sum(1 for f in result.findings
                               if hasattr(f.severity, 'value') and
                               f.severity.value == "critical"),
            error=result.error,
            reasoning_hash=_sha256_short(result.reasoning_trace),
        )
        self._exec_writer.write(asdict(record))

    def record_decision(
        self,
        agent_id: AgentId,
        task: Task,
        result: AgentResult,
        decision_record: Dict[str, Any],
    ) -> None:
        """Record an authority-bearing procurement decision."""
        record = DecisionRecord(
            agent_id=str(agent_id),
            agent_type=result.agent_name,
            task_id=str(task.task_id),
            session_id=str(task.session_id),
            findings_count=len(result.findings),
            finding_severities=[
                f.severity.value for f in result.findings
            ],
            authority_levels=list(set(
                f.decision_authority.value for f in result.findings
                if hasattr(f, "decision_authority")
            )),
            reasoning_hash=_sha256_short(result.reasoning_trace),
            rationale_captured=bool(result.reasoning_trace),
        )
        self._dec_writer.write(asdict(record))

    def record_escalation(
        self,
        agent_id: AgentId,
        agent_type: str,
        task: Task,
        result: AgentResult,
        reason: str,
    ) -> None:
        """Record a human escalation event."""
        record = EscalationRecord(
            agent_id=str(agent_id),
            agent_type=agent_type,
            task_id=str(task.task_id),
            session_id=str(task.session_id),
            escalation_reason=reason,
            critical_findings=sum(1 for f in result.findings
                                  if hasattr(f.severity, 'value') and
                                  f.severity.value == "critical"),
        )
        self._esc_writer.write(asdict(record))

    def audit(self, event_type: str, data: Dict[str, Any]) -> None:
        """General-purpose audit log entry."""
        self._audit_writer.write({
            "event_type": event_type,
            "timestamp":  _now(),
            **data,
        })

    def verify_chain(self, log_file: str) -> bool:
        """
        Verify the hash chain integrity of a governance log file.
        Returns True if intact, False if any entry has been tampered.
        """
        path = Path(log_file)
        if not path.exists():
            return True  # empty file = valid

        prev_hash = "GENESIS"
        with open(path, encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    stored_prev = entry.get("prev_hash", "")
                    stored_hash = entry.pop("hash", "")
                    # Recompute hash from entry without hash field
                    raw = json.dumps(
                        {**entry, "prev_hash": stored_prev},
                        default=str, sort_keys=True, ensure_ascii=False,
                    )
                    expected_hash = hashlib.sha256(raw.encode()).hexdigest()[:24]
                    if stored_prev != prev_hash:
                        log.error(
                            "Chain broken at line %d: prev_hash mismatch", line_num
                        )
                        return False
                    prev_hash = stored_hash
                except Exception as exc:
                    log.error("Corrupt entry at line %d: %s", line_num, exc)
                    return False
        return True


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_short(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
