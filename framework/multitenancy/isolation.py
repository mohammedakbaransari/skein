"""Logical tenant isolation enforcement helpers (R13).

Isolation must not mean merely a `tenant_id` column — every cache key,
log line, temp file, and memory namespace derived from a tenant-scoped
value must be tagged and verifiable. This module centralizes the scoped
key convention so callers cannot forget it.
"""

from __future__ import annotations

import re

_SCOPE_SEPARATOR = "::"
_SAFE_TENANT_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class CrossTenantAccessError(RuntimeError):
    """Raised when a scoped resource is accessed by the wrong tenant."""


def scoped_key(tenant_id: str, key: str) -> str:
    if not tenant_id:
        raise ValueError("tenant_id is required to build a scoped key")
    return f"tenant{_SCOPE_SEPARATOR}{tenant_id}{_SCOPE_SEPARATOR}{key}"


def tenant_of_scoped_key(scoped: str) -> str:
    parts = scoped.split(_SCOPE_SEPARATOR)
    if len(parts) < 3 or parts[0] != "tenant":
        raise ValueError(f"not a tenant-scoped key: {scoped!r}")
    return parts[1]


def assert_scoped_to_tenant(scoped: str, tenant_id: str) -> None:
    actual = tenant_of_scoped_key(scoped)
    if actual != tenant_id:
        raise CrossTenantAccessError(
            f"key scoped to tenant {actual!r} accessed by tenant {tenant_id!r}"
        )


class LogicalIsolationEnforcer:
    """Wraps a plain dict-like store to enforce tenant-scoped keys end to end."""

    def __init__(self, backing_store: dict) -> None:
        self._store = backing_store

    def put(self, tenant_id: str, key: str, value) -> None:
        self._store[scoped_key(tenant_id, key)] = value

    def get(self, tenant_id: str, key: str):
        return self._store.get(scoped_key(tenant_id, key))

    def keys_for_tenant(self, tenant_id: str) -> list:
        prefix = scoped_key(tenant_id, "")
        return [k for k in self._store if k.startswith(prefix)]


def tenant_scoped_path(base_dir: str, tenant_id: str) -> str:
    """Scoped subdirectory for per-tenant logs/temp-files under `base_dir`.

    Rejects path-traversal attempts in `tenant_id` (e.g. `../other`) —
    fails closed rather than silently sanitizing.
    """
    import os
    if not _SAFE_TENANT_ID.match(tenant_id or ""):
        raise ValueError(f"unsafe tenant_id for path scoping: {tenant_id!r}")
    return os.path.join(base_dir, f"tenant_{tenant_id}")
