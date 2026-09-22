"""Per-tenant plugin agent extensibility without a core redeploy (R20).

Lets a platform vendor or large customer register a proprietary agent
class (given only a module path and class name) and enable/disable it
per tenant — the fixed, hardcoded `register_all_agents()` catalogue in
`scripts/server.py` remains the default set; this registry is additive.
"""

from __future__ import annotations

import importlib
import threading
from typing import Dict, Set, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from framework.agents.base import BaseAgent


class PluginAgentRegistry:
    """Thread-safe registry of dynamically-loaded agent classes plus
    per-tenant enablement, layered on top of the core `AgentRegistry`."""

    def __init__(self, agent_registry) -> None:
        self._agent_registry = agent_registry
        self._lock = threading.RLock()
        self._enabled_by_tenant: Dict[str, Set[str]] = {}

    def register_from_path(self, module_path: str, class_name: str) -> Type["BaseAgent"]:
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)
        self._agent_registry.register_class(agent_class)
        return agent_class

    def enable_for_tenant(self, tenant_id: str, agent_type: str) -> None:
        with self._lock:
            self._enabled_by_tenant.setdefault(tenant_id, set()).add(agent_type)

    def disable_for_tenant(self, tenant_id: str, agent_type: str) -> None:
        with self._lock:
            self._enabled_by_tenant.get(tenant_id, set()).discard(agent_type)

    def is_enabled(self, tenant_id: str, agent_type: str) -> bool:
        with self._lock:
            return agent_type in self._enabled_by_tenant.get(tenant_id, set())

    def enabled_agent_types(self, tenant_id: str) -> Set[str]:
        with self._lock:
            return set(self._enabled_by_tenant.get(tenant_id, set()))
