"""Tenant-tagged structured logging (R13 remainder).

Wraps a standard logger so every emitted record carries `tenant_id` in
its `extra` — makes cross-tenant leakage structurally visible/filterable
in any log aggregator without relying on developers remembering to pass
it on every call site.
"""

from __future__ import annotations

import logging
from typing import Any, MutableMapping, Tuple


class TenantLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: Any, kwargs: MutableMapping[str, Any]) -> Tuple[Any, MutableMapping[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("tenant_id", self.extra.get("tenant_id"))
        return msg, kwargs


def get_tenant_logger(name: str, tenant_id: str) -> TenantLoggerAdapter:
    return TenantLoggerAdapter(logging.getLogger(name), {"tenant_id": tenant_id})
