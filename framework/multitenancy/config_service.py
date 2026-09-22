"""Per-tenant configuration service with an audit trail of changes (R16)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ConfigChangeRecord:
    tenant_id: str
    key: str
    old_value: Any
    new_value: Any
    changed_by: str
    changed_at: str = field(default_factory=_now)


class TenantConfigService:
    """Thread-safe per-tenant config overrides with full change history.

    Not a bigger YAML file — every mutation is recorded, and reads fall
    back to `defaults` when a tenant has no override for a key.
    """

    def __init__(self, defaults: Optional[Dict[str, Any]] = None) -> None:
        self._defaults: Dict[str, Any] = dict(defaults or {})
        self._lock = threading.RLock()
        self._overrides: Dict[str, Dict[str, Any]] = {}
        self._history: List[ConfigChangeRecord] = []

    def get(self, tenant_id: str, key: str) -> Any:
        with self._lock:
            tenant_overrides = self._overrides.get(tenant_id, {})
            if key in tenant_overrides:
                return tenant_overrides[key]
            return self._defaults.get(key)

    def get_all(self, tenant_id: str) -> Dict[str, Any]:
        with self._lock:
            merged = dict(self._defaults)
            merged.update(self._overrides.get(tenant_id, {}))
            return merged

    def set(self, tenant_id: str, key: str, value: Any, changed_by: str) -> None:
        if not changed_by.strip():
            raise ValueError("changed_by is required for an auditable config change")
        with self._lock:
            tenant_overrides = self._overrides.setdefault(tenant_id, {})
            old_value = tenant_overrides.get(key, self._defaults.get(key))
            tenant_overrides[key] = value
            self._history.append(ConfigChangeRecord(
                tenant_id=tenant_id, key=key, old_value=old_value,
                new_value=value, changed_by=changed_by,
            ))

    def history(self, tenant_id: Optional[str] = None) -> List[ConfigChangeRecord]:
        with self._lock:
            records = list(self._history)
        if tenant_id is not None:
            records = [r for r in records if r.tenant_id == tenant_id]
        return records
