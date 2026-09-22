"""
Chaos / failure-injection test suite (R26).

Induces real failure conditions — LLM timeouts, permanent LLM failures
(circuit breaker), pool exhaustion, and memory-store write failures —
and asserts the framework degrades predictably (typed errors, no crash,
no deadlock) rather than merely checking logical correctness under
ideal conditions.

Run separately from the standard suite:
  python -m unittest tests.chaos.test_failure_injection -v
"""

import json
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.agents.base import StructuralAgent
from framework.core.registry import AgentRegistry, reset_registry
from framework.core.types import AgentMetadata, RetryConfig, Severity, Task
from framework.memory.store import MemoryStore, WorkingMemory
from framework.orchestration.orchestrator import TaskOrchestrator
from framework.reasoning.engine import ReasoningEngine, ReasoningRequest
from framework.resilience.pool import AgentPool, PoolConfig, PoolExhaustedError
from framework.resilience.retry import CircuitOpenError, reset_circuit_registry


class _FlakyStrategy:
    """Reasoning strategy that always raises — simulates an LLM timeout/outage."""

    provider_name = "flaky"

    def __init__(self) -> None:
        self.calls = 0

    def reason(self, request: ReasoningRequest):
        self.calls += 1
        raise TimeoutError("simulated LLM timeout")


class _SlowAgent(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_SlowAgent", display_name="Slow", description="Chaos test",
        version="1.0", capabilities=(),
    )

    def observe(self, task: Task) -> Dict[str, Any]:
        return {}

    def reason(self, observations, task):
        return self.reasoning.reason(ReasoningRequest(
            system_prompt="s", user_prompt="u", observations=observations,
        )).content

    def parse_findings(self, observations, reasoning, task):
        return [self._make_finding("chaos", Severity.INFO, "ok")]


class _WriteFailingMemory(MemoryStore):
    """Memory store whose set() always raises — simulates a Delta write outage."""

    def set(self, key, value, session_id=None, agent_id=None, ttl_seconds=None) -> None:
        raise IOError("simulated memory backend outage")

    def get(self, key, session_id=None):
        return None

    def delete(self, key, session_id=None) -> None:
        pass

    def keys(self, session_id=None):
        return []


class TestLLMFailureInjection(unittest.TestCase):
    def setUp(self):
        reset_registry()
        reset_circuit_registry()

    def test_permanent_llm_failure_opens_circuit_and_agent_still_returns_result(self):
        engine = ReasoningEngine(
            _FlakyStrategy(),
            retry_config=RetryConfig(max_attempts=2, initial_delay_s=0.01, max_delay_s=0.02),
            circuit_failure_threshold=2, circuit_recovery_s=60.0,
        )
        agent = _SlowAgent(reasoning_engine=engine)
        task = Task.create("_SlowAgent", {})

        result = agent.run(task)

        self.assertFalse(result.succeeded)
        self.assertIsNotNone(result.error)

    def test_circuit_opens_after_threshold_and_fails_fast(self):
        engine = ReasoningEngine(
            _FlakyStrategy(),
            retry_config=RetryConfig(max_attempts=1),
            circuit_failure_threshold=1, circuit_recovery_s=60.0,
        )
        agent = _SlowAgent(reasoning_engine=engine)

        agent.run(Task.create("_SlowAgent", {}))
        t0 = time.monotonic()
        result = agent.run(Task.create("_SlowAgent", {}))
        elapsed = time.monotonic() - t0

        self.assertFalse(result.succeeded)
        self.assertLess(elapsed, 0.5, "circuit-open call should fail fast, not retry/backoff")


class TestPoolExhaustionInjection(unittest.TestCase):
    def test_acquire_beyond_capacity_raises_typed_error_not_deadlock(self):
        reset_registry()
        registry = AgentRegistry()
        registry.register_class(_SlowAgent)
        pool = AgentPool(
            "_SlowAgent", registry, config=None,
            pool_config=PoolConfig(min_size=0, max_size=1, acquire_timeout_s=0.2),
        )
        held = pool.acquire()
        try:
            with self.assertRaises(PoolExhaustedError):
                pool.acquire()
        finally:
            pool.release(held)


class TestMemoryBackendFailureInjection(unittest.TestCase):
    def test_memory_write_failure_does_not_crash_agent_run(self):
        reset_registry()
        reset_circuit_registry()

        class _EchoAgent(StructuralAgent):
            METADATA = AgentMetadata(
                agent_type="_ChaosEchoAgent", display_name="Echo", description="Chaos test",
                version="1.0", capabilities=(),
            )

            def observe(self, task):
                return {"msg": task.payload.get("msg", "")}

            def reason(self, observations, task):
                return json.dumps(observations)

            def parse_findings(self, observations, reasoning, task):
                return [self._make_finding("chaos", Severity.INFO, "ok")]

        agent = _EchoAgent(memory=_WriteFailingMemory())
        result = agent.run(Task.create("_ChaosEchoAgent", {"msg": "hi"}))

        # observe/reason/parse_findings succeeded; only the memory.remember()
        # write should have been affected — confirm it degraded rather than
        # propagating an unhandled backend exception out of run().
        self.assertIsInstance(result.succeeded, bool)


class TestOrchestratorRetryUnderInducedFailures(unittest.TestCase):
    def test_task_orchestrator_retries_then_reports_failure_cleanly(self):
        reset_registry()
        reset_circuit_registry()
        registry = AgentRegistry()
        registry.register_class(_SlowAgent)
        engine = ReasoningEngine(
            _FlakyStrategy(), retry_config=RetryConfig(max_attempts=1),
            circuit_failure_threshold=10, circuit_recovery_s=60.0,
        )

        orig_create = registry.create_instance
        def factory(agent_type, config, **kwargs):
            inst = orig_create(agent_type, config, **kwargs)
            inst.reasoning = engine
            return inst
        registry.create_instance = factory

        orchestrator = TaskOrchestrator(registry, config=None)
        task = Task.create("_SlowAgent", {}, retry_config=RetryConfig(max_attempts=2, initial_delay_s=0.01, max_delay_s=0.02))

        result = orchestrator.run_task(task)

        self.assertFalse(result.succeeded)
        self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main()
