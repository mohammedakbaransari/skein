"""
framework/multitenancy/context.py
====================================
Tenant routing for physically-isolated per-customer storage (Delta Lake
catalog/schema + storage container per tenant).

SCOPE: this module only ROUTES to already-provisioned tenant storage — it
does not create Databricks catalogs, ADLS containers, or databases. That
provisioning is an infrastructure/ops concern (Terraform, a Databricks
admin API script, etc.) outside this codebase. TenantRegistry answers
"given a tenant_id, which catalog/schema/container does its data live
in?" for the storage adapters in platform/databricks/adapter.py.

Physical isolation model:
    Each tenant gets a dedicated Unity Catalog catalog (and, if using
    external Delta tables, a dedicated ADLS Gen2 storage container) so
    that a bug or a leaked credential in one tenant's context cannot
    read/write another tenant's data — logical (shared-table,
    tenant_id-column) isolation was explicitly ruled out for this
    deployment.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Dict, Optional

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(name: str, kind: str) -> str:
    """Allowlist-validate a SQL identifier component (catalog/schema/table).

    Used instead of relying solely on engine-side identifier parameterization
    (e.g. Databricks' IDENTIFIER() clause), which this codebase has no live
    cluster to verify against — a strict allowlist is testable without one
    and fails closed on anything unexpected.
    """
    if not name or not _SAFE_IDENTIFIER.match(name):
        raise ValueError(
            f"Invalid {kind} name {name!r} — must match {_SAFE_IDENTIFIER.pattern}"
        )
    return name


@dataclass(frozen=True)
class TenantContext:
    """Physical storage routing for one tenant.

    catalog: dedicated Unity Catalog catalog for this tenant (physical
             isolation boundary — never shared across tenants).
    schema:  schema/database within that catalog.
    storage_container: optional ADLS Gen2 container URI backing external
             Delta tables for this tenant, if not using managed tables.
    """
    tenant_id:          str
    catalog:            str
    schema:             str = "skein"
    storage_container:  Optional[str] = None
    memory_table:       str = "institutional_memory"
    governance_table:   str = "governance_events"

    def __post_init__(self) -> None:
        validate_identifier(self.catalog, "catalog")
        validate_identifier(self.schema, "schema")
        validate_identifier(self.memory_table, "table")
        validate_identifier(self.governance_table, "table")

    def qualified_table(self, table: str) -> str:
        """Fully-qualified `catalog`.`schema`.`table` name for this tenant."""
        validate_identifier(table, "table")
        return f"{self.catalog}.{self.schema}.{table}"


class TenantRegistry:
    """Thread-safe in-process registry mapping tenant_id -> TenantContext.

    Populated at startup from already-provisioned tenant configuration
    (e.g. a config file or admin database) — provisioning itself is out
    of scope for this class.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tenants: Dict[str, TenantContext] = {}

    def register(self, context: TenantContext) -> None:
        with self._lock:
            self._tenants[context.tenant_id] = context

    def get(self, tenant_id: str) -> TenantContext:
        with self._lock:
            ctx = self._tenants.get(tenant_id)
        if ctx is None:
            raise KeyError(f"No tenant context registered for tenant_id={tenant_id!r}")
        return ctx

    def __contains__(self, tenant_id: str) -> bool:
        with self._lock:
            return tenant_id in self._tenants

    def __len__(self) -> int:
        with self._lock:
            return len(self._tenants)


_registry: Optional[TenantRegistry] = None
_registry_lock = threading.Lock()


def get_tenant_registry() -> TenantRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = TenantRegistry()
    return _registry


def reset_tenant_registry() -> None:
    """Reset the global registry. Tests only."""
    global _registry
    with _registry_lock:
        _registry = None
