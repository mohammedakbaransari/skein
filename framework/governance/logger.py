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
from framework.governance.hashchain import compute_chained_entry, verify_chained_entries
from framework.multitenancy.isolation import tenant_scoped_path

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
    principal_id:     str = ""
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

    Multiple instances pointing to the same file share a class-level lock
    AND a class-level chain-state (prev_hash) keyed by resolved path.

    THREAD-SAFETY / RESTART-SAFETY (fixed):
      Chain state used to be a per-instance attribute read before the write
      lock and written back after it — two threads could read the same
      prev_hash and both append valid-looking lines that break the chain,
      and a new instance always started from "GENESIS" even if the file
      already contained entries. Both are fixed by:
        1. Tracking prev_hash per-path at the CLASS level (shared by every
           instance/thread writing to the same file in this process).
        2. Seeding it once, on first construction for a given path, from
           the last valid hash already persisted in the file (so a new
           instance/process resumes the real chain instead of restarting
           it at GENESIS).
        3. Performing the read-compute-write-update of prev_hash entirely
           inside the same per-path lock that guards the file append, so
           the whole operation is atomic per line.
    """

    _path_locks:     ClassVar[Dict[str, threading.Lock]] = {}
    _path_prev_hash: ClassVar[Dict[str, str]] = {}
    _registry_lock:  ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, log_path: Path) -> None:
        self._path = log_path.resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._key = str(self._path)
        with HashChainedWriter._registry_lock:
            if self._key not in HashChainedWriter._path_locks:
                HashChainedWriter._path_locks[self._key] = threading.Lock()
            if self._key not in HashChainedWriter._path_prev_hash:
                HashChainedWriter._path_prev_hash[self._key] = self._read_last_hash()
        self._lock = HashChainedWriter._path_locks[self._key]

    def _read_last_hash(self) -> str:
        """Resume the chain from the last persisted entry's hash, if any."""
        if not self._path.exists():
            return "GENESIS"
        last_hash = "GENESIS"
        try:
            with open(self._path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    h = entry.get("hash")
                    if h:
                        last_hash = h
        except OSError:
            pass
        return last_hash

    def write(self, record: Dict[str, Any]) -> None:
        with self._lock:
            prev_hash = HashChainedWriter._path_prev_hash[self._key]
            entry = compute_chained_entry(record, prev_hash)
            final_line = json.dumps(entry, default=str, sort_keys=True, ensure_ascii=False)

            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(final_line + "\n")
            HashChainedWriter._path_prev_hash[self._key] = entry["hash"]


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

    def __init__(self, log_dir: str, audit_sink=None) -> None:
        base = Path(log_dir)
        self._exec_writer   = HashChainedWriter(base / "executions.jsonl")
        self._dec_writer    = HashChainedWriter(base / "decisions.jsonl")
        self._esc_writer    = HashChainedWriter(base / "escalations.jsonl")
        self._audit_writer  = HashChainedWriter(base / "audit.jsonl")
        # Optional pluggable AuditSink (R15) — emits the versioned,
        # vendor-neutral AuditEvent contract alongside the local
        # hash-chained JSONL record; failures here never break governance.
        self._audit_sink = audit_sink

    @classmethod
    def for_tenant(cls, base_dir: str, tenant_id: str, audit_sink=None) -> "GovernanceLogger":
        """Create a logger whose four hash chains live under one tenant path."""
        return cls(tenant_scoped_path(base_dir, tenant_id), audit_sink=audit_sink)

    def _emit_audit_event(self, event_type: str, tenant_id: str, action: str,
                          result: str, **fields) -> None:
        if self._audit_sink is None:
            return
        try:
            from framework.adapters.audit import AuditEvent
            import uuid
            self._audit_sink.emit(AuditEvent(
                event_id=f"evt-{uuid.uuid4().hex[:12]}", event_type=event_type,
                tenant_id=tenant_id, action=action, result=result, **fields,
            ))
        except Exception as exc:
            log.warning("Audit sink emission failed: %s", exc)

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
            principal_id=getattr(task, "principal_id", "") or "",
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
        self._emit_audit_event(
            "workflow.task.completed", str(task.tenant_id) if task.tenant_id else "",
            "completed" if result.succeeded else "failed",
            "success" if result.succeeded else "failure",
            principal_id=record.principal_id, task_id=record.task_id, agent_id=record.agent_id,
        )

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

        entries = []
        with open(path, encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception as exc:
                    log.error("Corrupt entry at line %d: %s", line_num, exc)
                    return False

        if not verify_chained_entries(entries):
            log.error("Hash chain verification failed for %s", path)
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
