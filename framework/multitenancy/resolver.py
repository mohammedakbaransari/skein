"""
framework/multitenancy/resolver.py
=====================================
Resolves per-tenant memory/governance stores for BaseAgent.run() to use —
the last piece of the multi-tenant foundation (see architecture assessment
§33/§34/§35): tenant_id already flows through every Task and is
authenticated on the task API, but until this module existed,
scripts/server.py still injected one global WorkingMemory/GovernanceLogger
into every agent regardless of which tenant a task belonged to.

Loads platform/databricks/adapter.py by file path rather than
`from platform.databricks.adapter import ...` — the top-level `platform/`
package name collides with the stdlib `platform` module, which is almost
always already cached in sys.modules by the time this runs in a real
process (see the P0-4 finding in the architecture assessment; the
project's own tests use this same workaround). Renaming the `platform/`
package is the proper long-term fix and remains out of scope here.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from framework.multitenancy.context import TenantContext, TenantRegistry

log = logging.getLogger(__name__)

_ADAPTER_PATH = Path(__file__).parent.parent.parent / "platform" / "databricks" / "adapter.py"
_adapter_module = None
_adapter_lock = threading.Lock()


def _load_databricks_adapter():
    global _adapter_module
    with _adapter_lock:
        if _adapter_module is None:
            spec = importlib.util.spec_from_file_location(
                "skein_databricks_adapter_runtime", _ADAPTER_PATH
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _adapter_module = module
        return _adapter_module


class TenantStoreResolver:
    """
    Lazily builds and caches one (memory, governance) store pair per
    tenant, backed by Delta tables routed to that tenant's dedicated
    catalog (physical isolation, per TenantContext — see §33).

    resolve(tenant_id) returns None (not a fallback pair) when tenant_id
    is missing or unregistered — BaseAgent.run() treats None as "leave
    this agent's current memory/governance untouched," so each agent's
    own configured default (e.g. InstitutionalMemoryAgent's persistent
    InstitutionalMemory store vs. every other agent's WorkingMemory) is
    preserved for single-tenant/dev tasks instead of being overridden by
    one resolver-wide default.

    Thread-safe: store construction is guarded by a lock; the resulting
    stores are cached and reused across calls for the same tenant_id.
    """

    def __init__(self, tenant_registry: TenantRegistry, spark: Optional[Any] = None) -> None:
        self._registry = tenant_registry
        self._spark    = spark
        self._lock  = threading.Lock()
        self._cache: Dict[str, Tuple[Any, Any]] = {}

    def resolve(self, tenant_id: Optional[str]) -> Optional[Tuple[Any, Any]]:
        if not tenant_id or tenant_id not in self._registry:
            return None

        with self._lock:
            cached = self._cache.get(tenant_id)
            if cached is not None:
                return cached

            context = self._registry.get(tenant_id)
            try:
                stores = self._build_stores(context)
            except Exception as exc:
                log.error(
                    "[multitenancy] Failed to build Delta stores for tenant_id=%s "
                    "— leaving the agent's default store in place: %s", tenant_id, exc,
                )
                return None

            self._cache[tenant_id] = stores
            return stores

    def _build_stores(self, context: TenantContext) -> Tuple[Any, Any]:
        adapter = _load_databricks_adapter()
        memory     = adapter.DeltaTableMemoryStore(tenant=context, spark=self._spark)
        governance = adapter.DeltaGovernanceStore(tenant=context, spark=self._spark)
        return memory, governance
