"""
framework/memory/store.py
==========================
Memory layer for the SKEIN framework.

WHY MEMORY MATTERS FOR PROCUREMENT AI:
  The original POC had zero memory — every agent run started cold.
  Real procurement intelligence requires:
    - Supplier history across multiple analysis runs
    - Learned negotiation patterns per counterparty
    - Cross-agent knowledge sharing within a session
    - Persistent institutional knowledge (Mystery 01)

MEMORY TIERS:
  1. WorkingMemory   — in-process, session-scoped, TTL-based
                       (fast, volatile, good for within-session sharing)
  2. ContextMemory   — cross-session within a task chain
                       (medium-lived, scoped to orchestration workflow)
  3. InstitutionalMemory — persistent, knowledge base
                       (long-lived, shared across all sessions)

All three implement the MemoryStore protocol.
Agents receive a MemoryStore via dependency injection.
They never create or own memory directly.

THREAD-SAFETY:
  WorkingMemory: threading.RLock per store instance
  All put/get operations are atomic
  Session namespace prevents cross-session data leakage
"""

from __future__ import annotations

import abc
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from framework.core.types import AgentId, SessionId

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MemoryEntry — stored with TTL and provenance
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """One item in the memory store."""
    key:        str
    value:      Any
    session_id: str
    stored_by:  str              # agent_id.value
    stored_at:  float = field(default_factory=time.monotonic)
    ttl_seconds: Optional[float] = None   # None = no expiry

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return (time.monotonic() - self.stored_at) > self.ttl_seconds


# ---------------------------------------------------------------------------
# MemoryStore protocol
# ---------------------------------------------------------------------------

class MemoryStore(abc.ABC):
    """
    Abstract memory store interface.
    All implementations must satisfy this contract.
    """

    @abc.abstractmethod
    def set(
        self,
        key: str,
        value: Any,
        session_id: Optional[SessionId] = None,
        agent_id: Optional[AgentId] = None,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        """Store a value under the given key."""

    @abc.abstractmethod
    def get(
        self,
        key: str,
        session_id: Optional[SessionId] = None,
    ) -> Optional[Any]:
        """Retrieve a value by key. Returns None if not found or expired."""

    @abc.abstractmethod
    def delete(self, key: str, session_id: Optional[SessionId] = None) -> None:
        """Remove a key from the store."""

    @abc.abstractmethod
    def keys(self, session_id: Optional[SessionId] = None) -> List[str]:
        """List all non-expired keys for a session (or globally)."""

    def get_or_default(
        self, key: str, default: Any, session_id: Optional[SessionId] = None
    ) -> Any:
        """Return stored value or default if not found."""
        value = self.get(key, session_id)
        return value if value is not None else default


# ---------------------------------------------------------------------------
# WorkingMemory — in-process, session-isolated, TTL-enabled
# ---------------------------------------------------------------------------

class WorkingMemory(MemoryStore):
    """
    In-process memory with session isolation and TTL support.

    Storage: nested dict — session_id → {key → MemoryEntry}
    Thread-safe: RLock for all read/write operations.

    Use cases:
      - Observations from one agent shared with a downstream agent
      - Decision records (Mystery 13) within a session
      - Cached signal scores to avoid recomputation

    Limits:
      max_entries: Total entries across all sessions (prevents unbounded growth)
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._lock = threading.RLock()
        self._store: Dict[str, Dict[str, MemoryEntry]] = {}
        self._max_entries = max_entries

    def set(
        self,
        key: str,
        value: Any,
        session_id: Optional[SessionId] = None,
        agent_id: Optional[AgentId] = None,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        sid = str(session_id) if session_id else "_global"
        with self._lock:
            self._evict_expired()
            self._enforce_limit()
            if sid not in self._store:
                self._store[sid] = {}
            self._store[sid][key] = MemoryEntry(
                key=key,
                value=value,
                session_id=sid,
                stored_by=str(agent_id) if agent_id else "unknown",
                ttl_seconds=ttl_seconds,
            )

    def get(
        self,
        key: str,
        session_id: Optional[SessionId] = None,
    ) -> Optional[Any]:
        sid = str(session_id) if session_id else "_global"
        with self._lock:
            session_store = self._store.get(sid, {})
            entry = session_store.get(key)
            if entry is None:
                return None
            if entry.is_expired:
                del session_store[key]
                return None
            return entry.value

    def delete(
        self, key: str, session_id: Optional[SessionId] = None
    ) -> None:
        sid = str(session_id) if session_id else "_global"
        with self._lock:
            if sid in self._store:
                self._store[sid].pop(key, None)

    def keys(
        self, session_id: Optional[SessionId] = None
    ) -> List[str]:
        sid = str(session_id) if session_id else "_global"
        with self._lock:
            session_store = self._store.get(sid, {})
            return [
                k for k, entry in session_store.items()
                if not entry.is_expired
            ]

    def clear_session(self, session_id: SessionId) -> None:
        """Remove all entries for a session. Call at session end."""
        with self._lock:
            self._store.pop(str(session_id), None)

    def _evict_expired(self) -> None:
        """Remove expired entries. Called under lock."""
        for sid in list(self._store.keys()):
            session_store = self._store[sid]
            expired = [k for k, e in session_store.items() if e.is_expired]
            for k in expired:
                del session_store[k]
            if not session_store:
                del self._store[sid]

    def _enforce_limit(self) -> None:
        """Evict oldest entries if over limit. Called under lock."""
        total = sum(len(s) for s in self._store.values())
        if total <= self._max_entries:
            return
        # Collect all entries with timestamps, evict oldest 10%
        all_entries: List[Tuple[str, str, float]] = [
            (sid, key, entry.stored_at)
            for sid, session_store in self._store.items()
            for key, entry in session_store.items()
        ]
        all_entries.sort(key=lambda x: x[2])
        to_evict = len(all_entries) // 10
        for sid, key, _ in all_entries[:to_evict]:
            self._store.get(sid, {}).pop(key, None)

    @property
    def total_entries(self) -> int:
        with self._lock:
            return sum(len(s) for s in self._store.values())

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "sessions":      len(self._store),
                "total_entries": self.total_entries,
                "max_entries":   self._max_entries,
            }


# ---------------------------------------------------------------------------
# ContextMemory — wraps WorkingMemory with task-chain scoping
# ---------------------------------------------------------------------------

class ContextMemory(MemoryStore):
    """
    Memory scoped to an orchestration task chain.
    Shares the underlying WorkingMemory but namespaces keys
    by both session_id and a workflow_id — preventing cross-workflow leakage
    within the same session.
    """

    def __init__(self, working_memory: WorkingMemory, workflow_id: str) -> None:
        self._wm = working_memory
        self._workflow_id = workflow_id

    def _namespace(self, key: str) -> str:
        return f"workflow:{self._workflow_id}:{key}"

    def set(self, key, value, session_id=None, agent_id=None, ttl_seconds=None):
        self._wm.set(self._namespace(key), value, session_id, agent_id, ttl_seconds)

    def get(self, key, session_id=None):
        return self._wm.get(self._namespace(key), session_id)

    def delete(self, key, session_id=None):
        self._wm.delete(self._namespace(key), session_id)

    def keys(self, session_id=None):
        prefix = self._namespace("")
        return [
            k[len(prefix):]
            for k in self._wm.keys(session_id)
            if k.startswith(prefix)
        ]


# ---------------------------------------------------------------------------
# InstitutionalMemory — persistent across sessions (file-backed stub)
# ---------------------------------------------------------------------------

class InstitutionalMemory(MemoryStore):
    """
    Persistent knowledge store for institutional procurement intelligence.

    This is a simplified file-backed implementation.
    Production deployments would replace this with:
      - Vector database (Chroma, Pinecone, pgvector) for semantic search
      - Redis for fast persistent K/V
      - PostgreSQL for structured knowledge

    This stub provides the interface contract for those backends.
    Session_id is ignored — institutional memory is global.
    """

    def __init__(self, storage_path: Optional[str] = None) -> None:
        import json as _json
        from pathlib import Path

        self._lock = threading.RLock()
        self._path = Path(storage_path) if storage_path else None
        self._data: Dict[str, Any] = {}

        if self._path and self._path.exists():
            try:
                with open(self._path) as fh:
                    self._data = _json.load(fh)
                log.info("InstitutionalMemory loaded %d entries from %s",
                         len(self._data), self._path)
            except Exception as exc:
                log.warning("Could not load institutional memory: %s", exc)

    def set(self, key, value, session_id=None, agent_id=None, ttl_seconds=None):
        with self._lock:
            self._data[key] = {
                "value":     value,
                "stored_by": str(agent_id) if agent_id else "unknown",
                "stored_at": time.time(),
            }
            self._persist()

    def get(self, key, session_id=None):
        with self._lock:
            entry = self._data.get(key)
            return entry["value"] if entry else None

    def delete(self, key, session_id=None):
        with self._lock:
            self._data.pop(key, None)
            self._persist()

    def keys(self, session_id=None):
        with self._lock:
            return list(self._data.keys())

    def _persist(self) -> None:
        if self._path is None:
            return
        try:
            import json as _json
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as fh:
                _json.dump(self._data, fh, indent=2, default=str)
        except Exception as exc:
            log.warning("Could not persist institutional memory: %s", exc)
