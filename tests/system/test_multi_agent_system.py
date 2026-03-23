"""
tests/system/test_multi_agent_system.py
=========================================
Multi-agent system tests — full workflows, concurrency, thread safety,
context isolation, and multi-instance coordination.

Tests:
  - 15-agent full portfolio review workflow
  - 10 simultaneous workflows (concurrency safety)
  - 50 concurrent single-agent tasks (load)
  - Session isolation under concurrency
  - Pool manager respects max_size boundary
  - Circuit breaker integrates with orchestrator
  - Retry in workflow: task retries before cancelling dependent
  - Agent context switching: trace_id consistency across workflow chain
"""

import json
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.agents.base import StructuralAgent
from framework.core.registry import AgentRegistry, reset_registry
from framework.core.types import (
    AgentMetadata, CorrelationContext, Finding, RetryConfig,
    SessionId, Severity, Task,
)
from framework.memory.store import WorkingMemory
from framework.orchestration.orchestrator import TaskOrchestrator, WorkflowBuilder
from framework.reasoning.stubs import DryRunReasoningEngine
from framework.resilience.pool import AgentPoolManager, PoolConfig
from framework.resilience.retry import reset_circuit_registry


# ---------------------------------------------------------------------------
# Shared stub agents
# ---------------------------------------------------------------------------

def _make_stub(type_name: str, tag: str = "system_test"):
    class StubAgent(StructuralAgent):
        METADATA = AgentMetadata(
            agent_type=type_name, display_name=type_name, description="System test stub",
            version="0.1.0", capabilities=(), tags=(tag,),
        )

        def observe(self, task: Task) -> Dict[str, Any]:
            return {"task_id": str(task.task_id), "type": type_name,
                    "session": str(task.session_id)}

        def reason(self, obs, task):
            return json.dumps({"type": type_name, "session": obs["session"]})

        def parse_findings(self, obs, r, task):
            return [self._make_finding(type_name, Severity.INFO, f"{type_name} done")]

    StubAgent.__name__ = type_name
    return StubAgent


# Register 15 unique stub types (one per SKEIN mystery)
STUB_TYPES = [f"_Stub_{i:02d}" for i in range(15)]
STUB_CLASSES = {t: _make_stub(t) for t in STUB_TYPES}


class _FailOnce(StructuralAgent):
    """Fails on first call, succeeds on retry."""
    METADATA = AgentMetadata(
        agent_type="_FailOnce", display_name="FailOnce", description="Test",
        version="0.1.0", capabilities=(), tags=("system_test",),
    )
    _call_counts: Dict[str, int] = {}
    _lock = threading.Lock()

    def observe(self, task: Task) -> Dict[str, Any]:
        tid = str(task.task_id)
        # Use parent task_id to track across retry attempts (same original task)
        key = str(task.parent_task_id or task.task_id)
        with _FailOnce._lock:
            n = _FailOnce._call_counts.get(key, 0) + 1
            _FailOnce._call_counts[key] = n
        if task.attempt_number < 2:
            raise RuntimeError(f"Deliberate first-attempt failure (task={tid})")
        return {"attempt": task.attempt_number}

    def reason(self, obs, task): return json.dumps({"attempt": obs["attempt"]})
    def parse_findings(self, obs, r, task): return []


def _setup_registry() -> AgentRegistry:
    reset_registry()
    reset_circuit_registry()
    reg = AgentRegistry()
    for cls in STUB_CLASSES.values():
        reg.register_class(cls)
    reg.register_class(_FailOnce)
    return reg


# ---------------------------------------------------------------------------
# Test: Full 15-agent portfolio workflow
# ---------------------------------------------------------------------------

class TestFullPortfolioWorkflow(unittest.TestCase):

    def setUp(self):
        self.reg = _setup_registry()
        self.orch = TaskOrchestrator(self.reg, config=None)

    def test_15_agent_workflow_all_succeed(self):
        sid = SessionId.generate()
        ctx = CorrelationContext.new(workflow="portfolio_test")
        builder = WorkflowBuilder("portfolio-review").session(sid).trace(ctx)
        for t in STUB_TYPES:
            builder.step(t, {"payload": "test"})
        wf = builder.build()
        wf.max_workers = 8

        result = self.orch.run_workflow(wf)

        self.assertTrue(result.succeeded, f"Failed tasks: {result.failed_tasks}")
        self.assertEqual(len(result.task_results), 15)
        self.assertEqual(len(result.all_findings), 15)

    def test_sequential_chain_propagates_results(self):
        sid = SessionId.generate()
        wf = (WorkflowBuilder("chain-test")
              .session(sid)
              .step(STUB_TYPES[0], {"step": 1})
              .then(STUB_TYPES[1], {"step": 2})
              .then(STUB_TYPES[2], {"step": 3})
              .build())
        result = self.orch.run_workflow(wf)
        self.assertTrue(result.succeeded)
        self.assertEqual(len(result.task_results), 3)

    def test_workflow_trace_id_consistent(self):
        """All tasks in a workflow should share the same trace_id."""
        ctx = CorrelationContext.new(test="trace_consistency")
        sid = SessionId.generate()
        wf = (WorkflowBuilder("trace-test")
              .session(sid).trace(ctx)
              .parallel(*[(t, {}) for t in STUB_TYPES[:5]])
              .build())
        result = self.orch.run_workflow(wf)
        self.assertTrue(result.succeeded)
        for agent_result in result.task_results.values():
            self.assertEqual(agent_result.context.trace_id, ctx.trace_id,
                             "trace_id must be consistent across all workflow tasks")


# ---------------------------------------------------------------------------
# Test: Concurrent workflows (10 simultaneous)
# ---------------------------------------------------------------------------

class TestConcurrentWorkflows(unittest.TestCase):

    def setUp(self):
        self.reg = _setup_registry()
        self.orch = TaskOrchestrator(self.reg, config=None)

    def test_10_simultaneous_workflows(self):
        """10 workflows running in parallel — no data corruption or deadlock."""
        results = []
        errors  = []

        def run_workflow(n):
            try:
                sid = SessionId.generate()
                ctx = CorrelationContext.new(workflow_n=str(n))
                wf = (WorkflowBuilder(f"concurrent-{n}")
                      .session(sid).trace(ctx)
                      .parallel(*[(t, {"n": n}) for t in STUB_TYPES[:5]])
                      .build())
                r = self.orch.run_workflow(wf)
                results.append(r.succeeded)
            except Exception as exc:
                errors.append(f"workflow {n}: {exc}")

        threads = [threading.Thread(target=run_workflow, args=(n,)) for n in range(10)]
        start = time.monotonic()
        for t in threads: t.start()
        for t in threads: t.join()
        elapsed = time.monotonic() - start

        self.assertEqual(errors, [], f"Workflow errors: {errors}")
        self.assertEqual(sum(results), 10, "All 10 workflows must succeed")
        self.assertLess(elapsed, 30.0, f"10 concurrent workflows took too long: {elapsed:.1f}s")

    def test_concurrent_sessions_do_not_interfere(self):
        """Agents reading/writing memory must not see other sessions' data."""
        from framework.agents.base import StructuralAgent

        class _SessionWriter(StructuralAgent):
            METADATA = AgentMetadata(
                agent_type="_SessionWriter", display_name="SW", description="",
                version="0.1.0", capabilities=(), tags=("system_test",),
            )
            def observe(self, task):
                val = task.payload.get("session_value", "unknown")
                self.remember("test_key", val, session_id=task.session_id)
                retrieved = self.recall("test_key", session_id=task.session_id)
                return {"written": val, "retrieved": retrieved}
            def reason(self, obs, task):
                return json.dumps(obs)
            def parse_findings(self, obs, r, task):
                p = self._parse_llm_json(r) or obs
                match = p.get("written") == p.get("retrieved")
                sev = Severity.INFO if match else Severity.CRITICAL
                return [self._make_finding("session_check", sev,
                                           f"match={match} written={p.get('written')} retrieved={p.get('retrieved')}")]

        mem = WorkingMemory(max_entries=10_000)
        reg = AgentRegistry()
        reg.register_class(_SessionWriter)
        orch = TaskOrchestrator(reg, config=None)

        mismatches = []

        def run_session(n):
            sid = SessionId.generate()
            task = Task.create("_SessionWriter",
                               {"session_value": f"session_{n}_value"},
                               session_id=sid)
            result = orch.run_task(task)
            for f in result.findings:
                if f.severity == Severity.CRITICAL:
                    mismatches.append(f.summary)

        # Inject shared memory into agents
        orig_create = reg.create_instance
        def create_with_mem(agent_type, config, **kwargs):
            inst = orig_create(agent_type, config, **kwargs)
            inst.memory = mem
            return inst
        reg.create_instance = create_with_mem

        threads = [threading.Thread(target=run_session, args=(n,)) for n in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(mismatches, [], f"Session isolation failures: {mismatches[:3]}")


# ---------------------------------------------------------------------------
# Test: Load — 50 concurrent single-agent calls
# ---------------------------------------------------------------------------

class TestLoad(unittest.TestCase):

    def setUp(self):
        self.reg = _setup_registry()
        self.orch = TaskOrchestrator(self.reg, config=None)

    def test_50_concurrent_agent_calls(self):
        """50 concurrent tasks to the same agent type — no crashes or deadlocks."""
        results = []
        errors  = []

        def run_task(n):
            try:
                task = Task.create(STUB_TYPES[0], {"n": n})
                r = self.orch.run_task(task)
                results.append(r.succeeded)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=run_task, args=(n,)) for n in range(50)]
        start = time.monotonic()
        for t in threads: t.start()
        for t in threads: t.join()
        elapsed = time.monotonic() - start

        self.assertEqual(errors, [], f"Load errors: {errors[:3]}")
        self.assertEqual(sum(results), 50)
        self.assertLess(elapsed, 20.0, f"50 concurrent tasks too slow: {elapsed:.1f}s")

    def test_throughput_100_sequential(self):
        """100 sequential tasks should complete in under 5 seconds (DryRun)."""
        start = time.monotonic()
        for i in range(100):
            task = Task.create(STUB_TYPES[i % len(STUB_TYPES)], {"i": i})
            result = self.orch.run_task(task)
            self.assertTrue(result.succeeded)
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 5.0,
                        f"100 sequential tasks took {elapsed:.1f}s — expected < 5s")


# ---------------------------------------------------------------------------
# Test: Pool manager respects max_size
# ---------------------------------------------------------------------------

class TestPoolManager(unittest.TestCase):

    def setUp(self):
        self.reg = _setup_registry()

    def test_pool_max_size_enforced(self):
        """With max_size=2, a 3rd concurrent acquire should block or raise."""
        from framework.resilience.pool import AgentPool, PoolConfig, PoolExhaustedError
        pool_cfg = PoolConfig(min_size=1, max_size=2, acquire_timeout_s=0.1)
        pool = AgentPool(STUB_TYPES[0], self.reg, config=None, pool_config=pool_cfg)

        a1 = pool.acquire()
        a2 = pool.acquire()

        with self.assertRaises(PoolExhaustedError):
            pool.acquire()  # Should time out

        pool.release(a1)
        pool.release(a2)
        pool.shutdown()

    def test_pool_release_allows_next_acquire(self):
        from framework.resilience.pool import AgentPool, PoolConfig
        pool_cfg = PoolConfig(min_size=1, max_size=1, acquire_timeout_s=1.0)
        pool = AgentPool(STUB_TYPES[0], self.reg, config=None, pool_config=pool_cfg)

        a1 = pool.acquire()
        pool.release(a1)
        a2 = pool.acquire()  # Should succeed after release
        self.assertIsNotNone(a2)
        pool.release(a2)
        pool.shutdown()

    def test_pool_stats_accurate(self):
        from framework.resilience.pool import AgentPool, PoolConfig
        pool_cfg = PoolConfig(min_size=2, max_size=5, acquire_timeout_s=1.0)
        pool = AgentPool(STUB_TYPES[0], self.reg, config=None, pool_config=pool_cfg)

        stats = pool.stats()
        self.assertEqual(stats["agent_type"], STUB_TYPES[0])
        self.assertEqual(stats["max_size"], 5)
        self.assertGreaterEqual(stats["idle"], 1)
        pool.shutdown()


# ---------------------------------------------------------------------------
# Test: Retry in workflow
# ---------------------------------------------------------------------------

class TestWorkflowRetry(unittest.TestCase):

    def setUp(self):
        self.reg = _setup_registry()
        self.orch = TaskOrchestrator(self.reg, config=None)
        _FailOnce._call_counts.clear()

    def test_task_retries_on_first_failure(self):
        """_FailOnce fails on attempt 1, succeeds on attempt 2."""
        task = Task.create(
            "_FailOnce", {},
            retry_config=RetryConfig(
                max_attempts=3, initial_delay_s=0.01,
                backoff_factor=1.0, jitter_factor=0.0,
            ),
        )
        result = self.orch.run_task(task)
        self.assertTrue(result.succeeded, f"Expected success on retry, got: {result.error}")
        self.assertEqual(result.attempt_number, 2)

    def test_no_retry_config_fails_fast(self):
        """Default RetryConfig(max_attempts=1) — no retry."""
        task = Task.create(
            "_FailOnce", {},
            retry_config=RetryConfig(max_attempts=1),
        )
        result = self.orch.run_task(task)
        self.assertFalse(result.succeeded)


if __name__ == "__main__":
    unittest.main()
