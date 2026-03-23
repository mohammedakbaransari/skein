"""
framework/resilience/pool.py
==============================
Agent Pool Manager with back-pressure and load management.

PROBLEM
=======
The original get_or_create() returns any idle agent regardless of system load.
Under high concurrency this leads to:
  - Unbounded agent creation consuming memory
  - No back-pressure signal to callers
  - No warm-standby agents for latency-sensitive workflows
  - All agents sharing the same LLM call budget

SOLUTION
========
AgentPool per agent_type:
  - min_size: warm standby agents (always ready, no cold-start)
  - max_size: hard cap on concurrent agents of this type
  - acquire_timeout_s: max wait for a free agent slot (back-pressure)
  - Semaphore enforces max_size limit

AgentPoolManager:
  - One pool per registered agent type
  - Exposes acquire/release API used by the orchestrator
  - Metrics: pool sizes reported to Prometheus

THREAD SAFETY
=============
AgentPool.acquire() / release() use threading.Semaphore (counter-based).
Pool creation uses pool-level Lock.
Pool registry uses RLock.

DATA SHARING SAFETY
===================
Each agent instance has its own memory store (injected at creation).
Session namespacing in WorkingMemory prevents cross-session data leakage.
AgentPool ensures each task gets an exclusive agent instance while running.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from framework.observability.metrics import get_metrics

if TYPE_CHECKING:
    from framework.agents.base import BaseAgent
    from framework.core.registry import AgentRegistry

log = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for one agent type's pool."""
    min_size:          int   = 1
    max_size:          int   = 10
    acquire_timeout_s: float = 30.0
    idle_timeout_s:    float = 300.0   # evict idle agents after this

    @classmethod
    def default(cls) -> "PoolConfig":
        return cls()

    @classmethod
    def high_throughput(cls) -> "PoolConfig":
        return cls(min_size=3, max_size=20, acquire_timeout_s=10.0)

    @classmethod
    def resource_constrained(cls) -> "PoolConfig":
        return cls(min_size=1, max_size=3, acquire_timeout_s=60.0)


class PoolExhaustedError(RuntimeError):
    """Raised when all pool slots are occupied and acquire times out."""

    def __init__(self, agent_type: str, timeout: float) -> None:
        self.agent_type = agent_type
        super().__init__(
            f"Agent pool for '{agent_type}' exhausted — "
            f"no slot available within {timeout:.1f}s"
        )


class AgentPool:
    """
    Thread-safe pool of agent instances for one agent_type.

    Lifecycle:
        pool.acquire()  → get an agent instance (blocks up to acquire_timeout_s)
        pool.release()  → return agent to pool after task
        pool.shutdown() → terminate all instances

    Isolation guarantee:
        An agent instance is held by exactly one task at a time.
        Memory store (if per-agent) is not shared between concurrent tasks.
    """

    def __init__(
        self,
        agent_type:  str,
        registry:    "AgentRegistry",
        config:      Any,     # AppConfig passed to agent constructor
        pool_config: PoolConfig,
    ) -> None:
        self._agent_type  = agent_type
        self._registry    = registry
        self._config      = config
        self._pool_config = pool_config
        self._lock        = threading.Lock()
        self._semaphore   = threading.Semaphore(pool_config.max_size)
        self._idle:       queue.Queue["BaseAgent"] = queue.Queue()
        self._all:        List["BaseAgent"] = []
        self._active:     int = 0

        # Warm up min_size agents
        self._warm_up()

    def _warm_up(self) -> None:
        """Pre-create min_size agents for warm-standby."""
        for _ in range(self._pool_config.min_size):
            try:
                agent = self._registry.create_instance(self._agent_type, self._config)
                self._idle.put(agent)
                with self._lock:
                    self._all.append(agent)
                log.debug("[pool:%s] Warm-up: created agent %s", self._agent_type, agent.agent_id)
            except Exception as exc:
                log.warning("[pool:%s] Warm-up failed: %s", self._agent_type, exc)

    def acquire(self) -> "BaseAgent":
        """
        Acquire an agent from the pool.

        Returns immediately if an idle agent is available.
        Creates a new agent if below max_size.
        Blocks up to acquire_timeout_s if at max_size.
        Raises PoolExhaustedError on timeout.
        """
        timeout = self._pool_config.acquire_timeout_s

        # Try semaphore (enforces max_size)
        acquired = self._semaphore.acquire(blocking=True, timeout=timeout)
        if not acquired:
            raise PoolExhaustedError(self._agent_type, timeout)

        # Get idle agent or create new one
        try:
            agent = self._idle.get_nowait()
        except queue.Empty:
            try:
                agent = self._registry.create_instance(self._agent_type, self._config)
                with self._lock:
                    self._all.append(agent)
            except Exception as exc:
                self._semaphore.release()
                raise RuntimeError(
                    f"Failed to create agent '{self._agent_type}': {exc}"
                ) from exc

        with self._lock:
            self._active += 1

        metrics = get_metrics()
        idle_count = self._idle.qsize()
        metrics.pool_size_updated(self._agent_type, idle_count, self._active)

        log.debug(
            "[pool:%s] Acquired agent %s (active=%d, idle=%d)",
            self._agent_type, agent.agent_id, self._active, idle_count,
        )
        return agent

    def release(self, agent: "BaseAgent") -> None:
        """
        Return an agent to the pool after task completion.
        Called by the orchestrator in a finally block.
        """
        self._idle.put(agent)
        with self._lock:
            self._active = max(0, self._active - 1)
        self._semaphore.release()

        metrics = get_metrics()
        metrics.pool_size_updated(self._agent_type, self._idle.qsize(), self._active)

        log.debug(
            "[pool:%s] Released agent %s (active=%d, idle=%d)",
            self._agent_type, agent.agent_id, self._active, self._idle.qsize(),
        )

    def stats(self) -> Dict[str, Any]:
        """Return current pool statistics."""
        with self._lock:
            return {
                "agent_type": self._agent_type,
                "total":      len(self._all),
                "idle":       self._idle.qsize(),
                "active":     self._active,
                "max_size":   self._pool_config.max_size,
                "min_size":   self._pool_config.min_size,
            }

    def shutdown(self) -> None:
        """Terminate all agents in the pool."""
        with self._lock:
            agents = list(self._all)
        for agent in agents:
            try:
                agent.on_shutdown()
            except Exception as exc:
                log.warning("[pool:%s] Error shutting down %s: %s",
                            self._agent_type, agent.agent_id, exc)
        log.info("[pool:%s] Shutdown complete (%d agents)", self._agent_type, len(agents))


class AgentPoolManager:
    """
    Manages pools for all registered agent types.

    Used by the orchestrator to acquire/release agents.
    Provides back-pressure through PoolExhaustedError.

    Thread-safe: RLock on pool registry.
    """

    def __init__(
        self,
        registry:     "AgentRegistry",
        config:       Any,
        default_pool: PoolConfig = PoolConfig(),
    ) -> None:
        self._registry    = registry
        self._config      = config
        self._default_cfg = default_pool
        self._lock        = threading.RLock()
        self._pools:      Dict[str, AgentPool] = {}
        self._type_configs: Dict[str, PoolConfig] = {}

    def configure(self, agent_type: str, pool_config: PoolConfig) -> None:
        """Set pool config for a specific agent type (before first use)."""
        with self._lock:
            self._type_configs[agent_type] = pool_config

    def _get_pool(self, agent_type: str) -> AgentPool:
        """Get or create the pool for agent_type."""
        with self._lock:
            if agent_type not in self._pools:
                cfg = self._type_configs.get(agent_type, self._default_cfg)
                self._pools[agent_type] = AgentPool(
                    agent_type=agent_type,
                    registry=self._registry,
                    config=self._config,
                    pool_config=cfg,
                )
            return self._pools[agent_type]

    def acquire(self, agent_type: str) -> "BaseAgent":
        """Acquire an agent from the appropriate pool."""
        return self._get_pool(agent_type).acquire()

    def release(self, agent_type: str, agent: "BaseAgent") -> None:
        """Return an agent to its pool."""
        pool = self._pools.get(agent_type)
        if pool:
            pool.release(agent)

    def all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return stats for all pools."""
        with self._lock:
            return {k: v.stats() for k, v in self._pools.items()}

    def shutdown_all(self) -> None:
        """Shutdown all pools."""
        with self._lock:
            pools = dict(self._pools)
        for pool in pools.values():
            pool.shutdown()
        log.info("[pool_manager] All pools shut down")
