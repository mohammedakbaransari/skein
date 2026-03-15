"""
framework/core/registry.py
===========================
Agent Registry — the central catalog of all agents in the framework.

Responsibilities:
  1. Register agent classes at startup (once per process)
  2. Instantiate agents on demand (factory pattern)
  3. Track runtime state per agent instance
  4. Route tasks to capable agents (capability-based dispatch)
  5. Health monitoring — expose agent status for observability

Design decisions:
  - Separation of AgentMetadata (static, per class) from
    AgentRuntimeRecord (dynamic, per instance)
  - Thread-safe via RLock (registry is read-heavy, write-rare)
  - Supports multiple instances of the same agent type
    (e.g. multiple SupplierStressAgents for different regions)

Usage:
    registry = AgentRegistry()

    # Register a class once at startup
    registry.register_class(SupplierStressAgent)

    # Get or create an instance
    agent = registry.get_or_create("SupplierStressAgent", config)

    # Route by capability
    agents = registry.find_by_capability("supplier_stress_detection")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

from framework.core.types import (
    AgentCapability, AgentId, AgentMetadata, AgentStatus, SessionId
)

if TYPE_CHECKING:
    from framework.agents.base import BaseAgent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime record — one per live agent instance
# ---------------------------------------------------------------------------

@dataclass
class AgentRuntimeRecord:
    """
    Tracks the mutable runtime state of one agent instance.
    Separate from AgentMetadata (immutable class-level description).
    """
    agent_id:         AgentId
    agent_type:       str
    status:           AgentStatus = AgentStatus.REGISTERED
    active_task_count: int = 0
    total_tasks_run:  int = 0
    total_failures:   int = 0
    last_active_at:   Optional[str] = None
    registered_at:    str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    instance:         Optional[Any] = None  # the actual agent object


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class AgentRegistry:
    """
    Thread-safe central registry for agent classes and instances.

    Provides:
      - Class registration (decorator or explicit call)
      - Instance lifecycle management
      - Capability-based routing
      - Health snapshot for monitoring
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        # class-level catalog: agent_type → AgentMetadata
        self._class_catalog: Dict[str, AgentMetadata] = {}
        # class objects: agent_type → class
        self._class_map: Dict[str, Type["BaseAgent"]] = {}
        # instance-level records: agent_id.value → AgentRuntimeRecord
        self._instances: Dict[str, AgentRuntimeRecord] = {}

    # ------------------------------------------------------------------
    # Class registration
    # ------------------------------------------------------------------

    def register_class(self, agent_class: Type["BaseAgent"]) -> None:
        """
        Register an agent class in the catalog.
        Called once per class at framework startup.

        Reads AgentMetadata from agent_class.METADATA (required attribute).
        Raises ValueError if the class is already registered.
        """
        metadata: AgentMetadata = getattr(agent_class, "METADATA", None)
        if metadata is None:
            raise ValueError(
                f"{agent_class.__name__} is missing class attribute METADATA: AgentMetadata"
            )
        with self._lock:
            if metadata.agent_type in self._class_catalog:
                raise ValueError(
                    f"Agent type '{metadata.agent_type}' is already registered. "
                    f"Use a unique agent_type in METADATA."
                )
            self._class_catalog[metadata.agent_type] = metadata
            self._class_map[metadata.agent_type]     = agent_class
            log.info(
                "Registered agent class: %s v%s  capabilities=%s",
                metadata.agent_type, metadata.version,
                [c.name for c in metadata.capabilities],
            )

    def agent(self, agent_class: Type["BaseAgent"]) -> Type["BaseAgent"]:
        """
        Class decorator alternative to register_class().

        Usage:
            @registry.agent
            class SupplierStressAgent(ProcurementAgent):
                ...
        """
        self.register_class(agent_class)
        return agent_class

    # ------------------------------------------------------------------
    # Instance management
    # ------------------------------------------------------------------

    def create_instance(
        self,
        agent_type: str,
        config: Any,
        agent_id: Optional[AgentId] = None,
    ) -> "BaseAgent":
        """
        Instantiate a new agent of the given type.

        Args:
            agent_type: Must match an already-registered class.
            config:     AppConfig passed to agent constructor.
            agent_id:   Optional explicit ID; generated if not provided.

        Returns:
            Initialised agent instance, also stored in the registry.
        """
        with self._lock:
            if agent_type not in self._class_map:
                raise KeyError(
                    f"No agent class registered for type '{agent_type}'. "
                    f"Available: {list(self._class_catalog.keys())}"
                )
            cls = self._class_map[agent_type]
            aid = agent_id or AgentId.generate()
            instance = cls(agent_id=aid, config=config)
            record = AgentRuntimeRecord(
                agent_id=aid,
                agent_type=agent_type,
                status=AgentStatus.IDLE,
                instance=instance,
            )
            self._instances[aid.value] = record
            log.info("Created agent instance %s (type=%s)", aid, agent_type)
            return instance

    def get_or_create(
        self,
        agent_type: str,
        config: Any,
    ) -> "BaseAgent":
        """
        Return an existing IDLE instance of agent_type, or create one.
        Useful for agent pooling — avoid creating a new instance per task.
        """
        with self._lock:
            for record in self._instances.values():
                if record.agent_type == agent_type and record.status == AgentStatus.IDLE:
                    return record.instance
        return self.create_instance(agent_type, config)

    def get_instance(self, agent_id: AgentId) -> Optional["BaseAgent"]:
        """Retrieve an agent instance by ID."""
        with self._lock:
            record = self._instances.get(agent_id.value)
            return record.instance if record else None

    def update_status(self, agent_id: AgentId, status: AgentStatus) -> None:
        """Update the runtime status of an agent instance. Thread-safe."""
        with self._lock:
            record = self._instances.get(agent_id.value)
            if record:
                record.status = status
                if status == AgentStatus.RUNNING:
                    record.active_task_count += 1
                    record.last_active_at = datetime.now(timezone.utc).isoformat()
                elif status == AgentStatus.IDLE:
                    record.active_task_count = max(0, record.active_task_count - 1)
                    record.total_tasks_run += 1
                elif status == AgentStatus.FAILED:
                    record.total_failures += 1
                    record.active_task_count = max(0, record.active_task_count - 1)

    def terminate(self, agent_id: AgentId) -> None:
        """Gracefully terminate an agent and remove its instance record."""
        with self._lock:
            record = self._instances.get(agent_id.value)
            if record:
                record.status = AgentStatus.TERMINATED
                if hasattr(record.instance, "on_shutdown"):
                    try:
                        record.instance.on_shutdown()
                    except Exception as exc:
                        log.warning("Error during agent shutdown %s: %s", agent_id, exc)
                del self._instances[agent_id.value]
                log.info("Terminated agent %s", agent_id)

    # ------------------------------------------------------------------
    # Capability-based routing
    # ------------------------------------------------------------------

    def find_by_capability(self, capability_name: str) -> List[Type["BaseAgent"]]:
        """
        Return all registered agent classes that declare the given capability.
        Used by the orchestrator for task routing.
        """
        with self._lock:
            return [
                self._class_map[agent_type]
                for agent_type, metadata in self._class_catalog.items()
                if any(c.name == capability_name for c in metadata.capabilities)
            ]

    def find_by_tag(self, tag: str) -> List[AgentMetadata]:
        """Return metadata for all agents with the given tag."""
        with self._lock:
            return [m for m in self._class_catalog.values() if tag in m.tags]

    # ------------------------------------------------------------------
    # Health and observability
    # ------------------------------------------------------------------

    def health_snapshot(self) -> Dict[str, Any]:
        """
        Return a JSON-serialisable health snapshot.
        Called by monitoring endpoints.
        """
        with self._lock:
            return {
                "registered_classes": list(self._class_catalog.keys()),
                "live_instances":     len(self._instances),
                "instances": [
                    {
                        "agent_id":          r.agent_id.value,
                        "agent_type":        r.agent_type,
                        "status":            r.status.value,
                        "active_tasks":      r.active_task_count,
                        "total_tasks":       r.total_tasks_run,
                        "total_failures":    r.total_failures,
                        "last_active_at":    r.last_active_at,
                        "registered_at":     r.registered_at,
                    }
                    for r in self._instances.values()
                ],
            }

    def list_agents(self) -> List[AgentMetadata]:
        """Return metadata for all registered agent classes."""
        with self._lock:
            return list(self._class_catalog.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._class_catalog)

    def __contains__(self, agent_type: str) -> bool:
        with self._lock:
            return agent_type in self._class_catalog


# ---------------------------------------------------------------------------
# Global singleton registry (imported by agents and orchestrator)
# ---------------------------------------------------------------------------

_global_registry: Optional[AgentRegistry] = None
_global_registry_lock = threading.Lock()


def get_registry() -> AgentRegistry:
    """
    Return the process-level global registry (lazy singleton).
    Thread-safe initialisation via double-checked locking.
    """
    global _global_registry
    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                _global_registry = AgentRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry. Use in tests only."""
    global _global_registry
    with _global_registry_lock:
        _global_registry = None
