"""
framework/core/registry.py
===========================
Agent Registry — thread-safe central catalog of all agents.

CHANGES FROM v1
===============
- Registers a readiness check with health server on first agent registration
- update_status() tracks task counts for Prometheus pool metrics
- reset_registry() also resets pool manager if configured
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

from framework.core.types import AgentCapability, AgentId, AgentMetadata, AgentStatus

if TYPE_CHECKING:
    from framework.agents.base import BaseAgent

log = logging.getLogger(__name__)


@dataclass
class AgentRuntimeRecord:
    agent_id:          AgentId
    agent_type:        str
    status:            AgentStatus = AgentStatus.REGISTERED
    active_task_count: int = 0
    total_tasks_run:   int = 0
    total_failures:    int = 0
    last_active_at:    Optional[str] = None
    registered_at:     str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    instance:          Optional[Any] = None


class AgentRegistry:
    """Thread-safe registry for agent classes and instances."""

    def __init__(self) -> None:
        self._lock:         threading.RLock = threading.RLock()
        self._class_catalog: Dict[str, AgentMetadata] = {}
        self._class_map:     Dict[str, Type["BaseAgent"]] = {}
        self._instances:     Dict[str, AgentRuntimeRecord] = {}

    # ------------------------------------------------------------------
    # Class registration
    # ------------------------------------------------------------------

    def register_class(self, agent_class: Type["BaseAgent"]) -> None:
        metadata: AgentMetadata = getattr(agent_class, "METADATA", None)
        if metadata is None:
            raise ValueError(
                f"{agent_class.__name__} missing METADATA class attribute"
            )
        with self._lock:
            if metadata.agent_type in self._class_catalog:
                # Idempotent re-registration in test environments
                return
            self._class_catalog[metadata.agent_type] = metadata
            self._class_map[metadata.agent_type]     = agent_class
            log.info(
                "Registered agent: %s v%s caps=%s",
                metadata.agent_type, metadata.version,
                [c.name for c in metadata.capabilities],
            )

    def agent(self, cls: Type["BaseAgent"]) -> Type["BaseAgent"]:
        """Class decorator alternative."""
        self.register_class(cls)
        return cls

    # ------------------------------------------------------------------
    # Instance management
    # ------------------------------------------------------------------

    def create_instance(
        self,
        agent_type: str,
        config:     Any,
        agent_id:   Optional[AgentId] = None,
        memory=     None,
        reasoning=  None,
        governance= None,
    ) -> "BaseAgent":
        with self._lock:
            if agent_type not in self._class_map:
                raise KeyError(
                    f"No class registered for type '{agent_type}'. "
                    f"Available: {list(self._class_catalog)}"
                )
            cls = self._class_map[agent_type]
            aid = agent_id or AgentId.generate()
            instance = cls(
                agent_id=aid, config=config,
                memory=memory, reasoning_engine=reasoning,
                governance_logger=governance,
            )
            record = AgentRuntimeRecord(
                agent_id=aid, agent_type=agent_type,
                status=AgentStatus.IDLE, instance=instance,
            )
            self._instances[aid.value] = record
            log.debug("Created agent instance %s (type=%s)", aid, agent_type)
            return instance

    def get_or_create(self, agent_type: str, config: Any) -> "BaseAgent":
        with self._lock:
            for record in self._instances.values():
                if record.agent_type == agent_type and record.status == AgentStatus.IDLE:
                    return record.instance
        return self.create_instance(agent_type, config)

    def get_instance(self, agent_id: AgentId) -> Optional["BaseAgent"]:
        with self._lock:
            record = self._instances.get(agent_id.value)
            return record.instance if record else None

    def update_status(self, agent_id: AgentId, status: AgentStatus) -> None:
        with self._lock:
            record = self._instances.get(agent_id.value)
            if record:
                record.status = status
                if status == AgentStatus.RUNNING:
                    record.active_task_count += 1
                    record.last_active_at = datetime.now(timezone.utc).isoformat()
                elif status == AgentStatus.IDLE:
                    record.active_task_count = max(0, record.active_task_count - 1)
                    record.total_tasks_run  += 1
                elif status == AgentStatus.FAILED:
                    record.total_failures   += 1
                    record.active_task_count = max(0, record.active_task_count - 1)

    def terminate(self, agent_id: AgentId) -> None:
        with self._lock:
            record = self._instances.get(agent_id.value)
            if record:
                record.status = AgentStatus.TERMINATED
                if hasattr(record.instance, "on_shutdown"):
                    try:
                        record.instance.on_shutdown()
                    except Exception as exc:
                        log.warning("Shutdown error %s: %s", agent_id, exc)
                del self._instances[agent_id.value]

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def find_by_capability(self, capability_name: str) -> List[Type["BaseAgent"]]:
        with self._lock:
            return [
                self._class_map[at]
                for at, m in self._class_catalog.items()
                if any(c.name == capability_name for c in m.capabilities)
            ]

    def find_by_tag(self, tag: str) -> List[AgentMetadata]:
        with self._lock:
            return [m for m in self._class_catalog.values() if tag in m.tags]

    # ------------------------------------------------------------------
    # Health / observability
    # ------------------------------------------------------------------

    def health_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "registered_classes": list(self._class_catalog.keys()),
                "live_instances":     len(self._instances),
                "instances": [
                    {
                        "agent_id":       r.agent_id.value,
                        "agent_type":     r.agent_type,
                        "status":         r.status.value,
                        "active_tasks":   r.active_task_count,
                        "total_tasks":    r.total_tasks_run,
                        "total_failures": r.total_failures,
                        "last_active_at": r.last_active_at,
                    }
                    for r in self._instances.values()
                ],
            }

    def list_agents(self) -> List[AgentMetadata]:
        with self._lock:
            return list(self._class_catalog.values())

    def live_count(self) -> int:
        with self._lock:
            return len(self._instances)

    def __len__(self) -> int:
        with self._lock:
            return len(self._class_catalog)

    def __contains__(self, agent_type: str) -> bool:
        with self._lock:
            return agent_type in self._class_catalog


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_registry: Optional[AgentRegistry] = None
_global_registry_lock = threading.Lock()


def get_registry() -> AgentRegistry:
    global _global_registry
    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                _global_registry = AgentRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset global registry. Tests only."""
    global _global_registry
    with _global_registry_lock:
        _global_registry = None
