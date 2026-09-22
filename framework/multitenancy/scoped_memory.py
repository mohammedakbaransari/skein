"""Tenant-scoped memory wrapper enforcing R13 isolation on shared stores.

Wraps any `MemoryStore` so every key is namespaced by tenant before
reaching the backing store — additive; does not modify `WorkingMemory`
itself. Deployments choosing logical isolation with a single shared
memory store use this wrapper instead of one instance per tenant.
"""

from __future__ import annotations

from typing import Any, List, Optional

from framework.multitenancy.isolation import scoped_key


class TenantScopedMemoryStore:
    """Delegates to a backing `MemoryStore`, prefixing every key with the tenant."""

    def __init__(self, backing_store: Any, tenant_id: str) -> None:
        self._backing = backing_store
        self._tenant_id = tenant_id

    def set(self, key: str, value: Any, session_id: Optional[Any] = None,
            agent_id: Optional[Any] = None, ttl_seconds: Optional[float] = None) -> None:
        self._backing.set(
            scoped_key(self._tenant_id, key), value,
            session_id=session_id, agent_id=agent_id, ttl_seconds=ttl_seconds,
        )

    def get(self, key: str, session_id: Optional[Any] = None) -> Optional[Any]:
        return self._backing.get(scoped_key(self._tenant_id, key), session_id=session_id)

    def delete(self, key: str, session_id: Optional[Any] = None) -> None:
        self._backing.delete(scoped_key(self._tenant_id, key), session_id=session_id)

    def keys(self, session_id: Optional[Any] = None) -> List[str]:
        prefix = scoped_key(self._tenant_id, "")
        return [k[len(prefix):] for k in self._backing.keys(session_id=session_id) if k.startswith(prefix)]

    def get_or_default(self, key: str, default: Any, session_id: Optional[Any] = None) -> Any:
        value = self.get(key, session_id)
        return value if value is not None else default
