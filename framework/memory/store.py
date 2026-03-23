"""
framework/memory/store.py
==========================
Memory layer with session isolation, LRU eviction, and metrics integration.

CHANGES FROM v1
===============
- WorkingMemory: LRU eviction (OrderedDict) replaces oldest-10% blunt eviction
- Session isolation: concurrent agents in different sessions cannot read
  each other's data even if keys collide
- Memory metrics: entry counts reported to Prometheus via get_metrics()
- InstitutionalMemory: atomic read-modify-write for concurrent agents
- ContextMemory: namespace collision protection
- All stores implement MemoryStore protocol identically (Liskov)

THREAD SAFETY
=============
WorkingMemory: single RLock covering all operations including eviction.
InstitutionalMemory: separate write lock with atomic read-modify-write.
No deadlock possible: neither store acquires the other's lock.
"""

from __future__ import annotations

import abc
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from framework.core.types import AgentId, SessionId
from framework.observability.metrics import get_metrics

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()



log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    key:         str
    value:       Any
    session_id:  str
    stored_by:   str
    stored_at:   float = field(default_factory=time.monotonic)
    ttl_seconds: Optional[float] = None

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return (time.monotonic() - self.stored_at) > self.ttl_seconds


# ---------------------------------------------------------------------------
# MemoryStore protocol
# ---------------------------------------------------------------------------

class MemoryStore(abc.ABC):

    @abc.abstractmethod
    def set(
        self, key: str, value: Any,
        session_id:  Optional[SessionId] = None,
        agent_id:    Optional[AgentId]   = None,
        ttl_seconds: Optional[float]     = None,
    ) -> None: ...

    @abc.abstractmethod
    def get(self, key: str, session_id: Optional[SessionId] = None) -> Optional[Any]: ...

    @abc.abstractmethod
    def delete(self, key: str, session_id: Optional[SessionId] = None) -> None: ...

    @abc.abstractmethod
    def keys(self, session_id: Optional[SessionId] = None) -> List[str]: ...

    def get_or_default(self, key: str, default: Any,
                       session_id: Optional[SessionId] = None) -> Any:
        v = self.get(key, session_id)
        return v if v is not None else default


# ---------------------------------------------------------------------------
# WorkingMemory — LRU, session-isolated
# ---------------------------------------------------------------------------

class WorkingMemory(MemoryStore):
    """
    In-process memory with session isolation and LRU eviction.

    Storage: nested OrderedDict — session_id → {key → MemoryEntry}
    OrderedDict preserves insertion order for LRU eviction.

    Session isolation guarantee:
        Two concurrent agents running in different sessions CANNOT read
        each other's data, even if they use identical keys.
        This is enforced by the session_id namespace — not just convention.

    Thread-safe: single RLock covering all read and write operations.
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._lock = threading.RLock()
        self._store: Dict[str, OrderedDict] = {}
        self._max_entries = max_entries

    # Namespace helper — every key is stored under (session_id, key) pair
    @staticmethod
    def _sid(session_id: Optional[SessionId]) -> str:
        return str(session_id) if session_id else "_global"

    def set(
        self, key: str, value: Any,
        session_id:  Optional[SessionId] = None,
        agent_id:    Optional[AgentId]   = None,
        ttl_seconds: Optional[float]     = None,
    ) -> None:
        sid = self._sid(session_id)
        with self._lock:
            self._evict_expired()
            if sid not in self._store:
                self._store[sid] = OrderedDict()
            session_store = self._store[sid]
            # LRU: move to end on update
            if key in session_store:
                session_store.move_to_end(key)
            session_store[key] = MemoryEntry(
                key=key, value=value, session_id=sid,
                stored_by=str(agent_id) if agent_id else "unknown",
                ttl_seconds=ttl_seconds,
            )
            # Enforce global limit via LRU eviction
            total = sum(len(s) for s in self._store.values())
            while total > self._max_entries:
                self._evict_lru_one()
                total -= 1
        get_metrics().memory_updated("working", self.total_entries)

    def get(self, key: str, session_id: Optional[SessionId] = None) -> Optional[Any]:
        sid = self._sid(session_id)
        with self._lock:
            session_store = self._store.get(sid)
            if session_store is None:
                return None
            entry = session_store.get(key)
            if entry is None:
                return None
            if entry.is_expired:
                del session_store[key]
                return None
            # LRU: mark as recently used
            session_store.move_to_end(key)
            return entry.value

    def delete(self, key: str, session_id: Optional[SessionId] = None) -> None:
        sid = self._sid(session_id)
        with self._lock:
            self._store.get(sid, {}).pop(key, None)

    def keys(self, session_id: Optional[SessionId] = None) -> List[str]:
        sid = self._sid(session_id)
        with self._lock:
            return [
                k for k, e in self._store.get(sid, {}).items()
                if not e.is_expired
            ]

    def clear_session(self, session_id: SessionId) -> None:
        """Remove all entries for a session. Call at session end."""
        with self._lock:
            self._store.pop(str(session_id), None)
        get_metrics().memory_updated("working", self.total_entries)

    def _evict_expired(self) -> None:
        """Remove all expired entries. Called under lock."""
        for sid in list(self._store):
            session_store = self._store[sid]
            expired = [k for k, e in session_store.items() if e.is_expired]
            for k in expired:
                del session_store[k]
            if not session_store:
                del self._store[sid]

    def _evict_lru_one(self) -> None:
        """Evict the globally oldest (LRU) entry. Called under lock."""
        oldest_sid = oldest_key = oldest_time = None
        for sid, session_store in self._store.items():
            if session_store:
                # First entry in OrderedDict is the least-recently-used
                k, entry = next(iter(session_store.items()))
                if oldest_time is None or entry.stored_at < oldest_time:
                    oldest_sid, oldest_key, oldest_time = sid, k, entry.stored_at
        if oldest_sid and oldest_key:
            del self._store[oldest_sid][oldest_key]
            if not self._store[oldest_sid]:
                del self._store[oldest_sid]

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
# ContextMemory — workflow-scoped namespace over WorkingMemory
# ---------------------------------------------------------------------------

class ContextMemory(MemoryStore):
    """
    Memory scoped to one workflow, preventing cross-workflow key collisions
    within the same session.

    Uses WorkingMemory as backend with {workflow_id}:{key} namespacing.
    """

    def __init__(self, working_memory: WorkingMemory, workflow_id: str) -> None:
        self._wm          = working_memory
        self._workflow_id = workflow_id

    def _ns(self, key: str) -> str:
        return f"wf:{self._workflow_id}:{key}"

    def set(self, key, value, session_id=None, agent_id=None, ttl_seconds=None):
        self._wm.set(self._ns(key), value, session_id, agent_id, ttl_seconds)

    def get(self, key, session_id=None):
        return self._wm.get(self._ns(key), session_id)

    def delete(self, key, session_id=None):
        self._wm.delete(self._ns(key), session_id)

    def keys(self, session_id=None):
        prefix = self._ns("")
        return [
            k[len(prefix):]
            for k in self._wm.keys(session_id)
            if k.startswith(prefix)
        ]


# ---------------------------------------------------------------------------
# InstitutionalMemory — persistent, JSON-backed
# ---------------------------------------------------------------------------

class InstitutionalMemory(MemoryStore):
    """
    Persistent knowledge store for cross-session institutional intelligence.

    File-backed implementation with atomic write-lock.
    Production deployments replace with:
      - Vector DB (Chroma, pgvector) for semantic search
      - Redis for fast persistent K/V
      - Databricks Delta Table (see platform/databricks adapter)

    Thread-safe: separate read lock and write lock with
    atomic read-modify-write pattern to prevent lost updates.
    """

    def __init__(self, storage_path: Optional[str] = None) -> None:
        import json as _json
        from pathlib import Path

        self._read_lock  = threading.RLock()
        self._write_lock = threading.Lock()
        self._path = Path(storage_path) if storage_path else None
        self._data: Dict[str, Any] = {}

        if self._path and self._path.exists():
            try:
                with open(self._path) as fh:
                    self._data = _json.load(fh)
                log.info(
                    "InstitutionalMemory: loaded %d entries from %s",
                    len(self._data), self._path,
                )
            except Exception as exc:
                log.warning("InstitutionalMemory: load failed: %s", exc)

    def set(self, key, value, session_id=None, agent_id=None, ttl_seconds=None):
        with self._write_lock:
            self._data[key] = {
                "value":     value,
                "stored_by": str(agent_id) if agent_id else "unknown",
                "stored_at": time.time(),
            }
            self._persist()
        get_metrics().memory_updated("institutional", len(self._data))

    def get(self, key, session_id=None):
        with self._read_lock:
            entry = self._data.get(key)
            return entry["value"] if entry else None

    def delete(self, key, session_id=None):
        with self._write_lock:
            self._data.pop(key, None)
            self._persist()

    def keys(self, session_id=None):
        with self._read_lock:
            return list(self._data.keys())

    def update(self, key: str, update_fn, default: Any = None) -> Any:
        """
        Atomic read-modify-write.
        update_fn receives current value (or default), returns new value.
        Prevents lost updates from concurrent agents.
        """
        with self._write_lock:
            current = self._data.get(key, {}).get("value", default)
            new_value = update_fn(current)
            self._data[key] = {
                "value":     new_value,
                "stored_by": "atomic_update",
                "stored_at": time.time(),
            }
            self._persist()
            return new_value

    def _persist(self) -> None:
        if self._path is None:
            return
        import json as _json
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w") as fh:
                _json.dump(self._data, fh, indent=2, default=str)
            tmp.replace(self._path)  # atomic rename
        except Exception as exc:
            log.warning("InstitutionalMemory: persist failed: %s", exc)
