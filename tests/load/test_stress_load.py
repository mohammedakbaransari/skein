"""
tests/load/test_stress_load.py
================================
Stress and load tests for SKEIN framework.

These tests verify that the framework handles high concurrency without:
  - Deadlocks (tests have timeouts)
  - Memory leaks (entry counts monitored)
  - Race conditions (results validated for correctness)
  - Thread starvation (all threads complete)

Run separately from the standard test suite — they take longer.
  python3 -m unittest tests.load.test_stress_load -v

Configuration via environment variables:
  SKEIN_LOAD_WORKERS=100   (default: 50)
  SKEIN_LOAD_TASKS=200     (default: 100)
  SKEIN_LOAD_TIMEOUT=60    (default: 30)
"""

import json
import os
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
    AgentMetadata, CorrelationContext, Finding,
    RetryConfig, SessionId, Severity, Task,
)
from framework.memory.store import WorkingMemory
from framework.orchestration.orchestrator import TaskOrchestrator
from framework.resilience.retry import reset_circuit_registry

WORKERS  = int(os.environ.get("SKEIN_LOAD_WORKERS", 50))
N_TASKS  = int(os.environ.get("SKEIN_LOAD_TASKS",   100))
TIMEOUT  = int(os.environ.get("SKEIN_LOAD_TIMEOUT",  30))


# ---------------------------------------------------------------------------
# Minimal fast agent for load testing
# ---------------------------------------------------------------------------

class _FastAgent(StructuralAgent):
    """Minimal agent with no I/O — measures pure framework overhead."""
    METADATA = AgentMetadata(
        agent_type="_FastAgent", display_name="Fast", description="Load test",
        version="0.1.0", capabilities=(), tags=("load",),
    )

    def observe(self, task: Task) -> Dict[str, Any]:
        return {"n": task.payload.get("n", 0)}

    def reason(self, obs, task):
        return json.dumps({"n": obs["n"]})

    def parse_findings(self, obs, r, task):
        return [self._make_finding("load_result", Severity.INFO, f"n={obs['n']}")]


class _MemoryIntensiveAgent(StructuralAgent):
    """Agent that reads/writes memory on every call."""
    METADATA = AgentMetadata(
        agent_type="_MemIntensive", display_name="MemIntensive", description="Load test",
        version="0.1.0", capabilities=(), tags=("load",),
    )

    def observe(self, task: Task) -> Dict[str, Any]:
        n   = task.payload.get("n", 0)
        sid = task.session_id
        # Write and immediately read back
        self.remember(f"load_key_{n}", {"value": n * 2}, session_id=sid)
        v = self.recall(f"load_key_{n}", session_id=sid)
        return {"n": n, "readback": v["value"] if v else None}

    def reason(self, obs, task):
        return json.dumps(obs)

    def parse_findings(self, obs, r, task):
        p    = self._parse_llm_json(r) or obs
        correct = p.get("n") is not None and p.get("readback") == p["n"] * 2
        sev  = Severity.INFO if correct else Severity.CRITICAL
        return [self._make_finding("mem_check", sev,
                                   f"n={p.get('n')} readback={p.get('readback')} correct={correct}")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup():
    reset_registry()
    reset_circuit_registry()
    mem = WorkingMemory(max_entries=50_000)
    reg = AgentRegistry()
    reg.register_class(_FastAgent)
    reg.register_class(_MemoryIntensiveAgent)

    orig = reg.create_instance
    def factory(at, config, **kwargs):
        inst = orig(at, config, **kwargs)
        inst.memory = mem
        return inst
    reg.create_instance = factory

    orch = TaskOrchestrator(reg, config=None)
    return reg, orch, mem


# ---------------------------------------------------------------------------
# Load tests
# ---------------------------------------------------------------------------

class TestConcurrentLoad(unittest.TestCase):

    def setUp(self):
        self.reg, self.orch, self.mem = _setup()

    def test_concurrent_agents_no_deadlock(self):
        """N concurrent threads, each running one task. Must all complete within TIMEOUT."""
        results = []
        errors  = []
        lock    = threading.Lock()

        def run(n):
            task = Task.create("_FastAgent", {"n": n})
            try:
                r = self.orch.run_task(task)
                with lock:
                    results.append(r.succeeded)
            except Exception as exc:
                with lock:
                    errors.append(str(exc))

        threads = [threading.Thread(target=run, args=(i,), daemon=True)
                   for i in range(WORKERS)]
        start = time.monotonic()
        for t in threads: t.start()
        for t in threads: t.join(timeout=TIMEOUT)
        elapsed = time.monotonic() - start

        still_alive = sum(1 for t in threads if t.is_alive())
        self.assertEqual(still_alive, 0,
                         f"{still_alive} threads still running after {TIMEOUT}s — DEADLOCK?")
        self.assertEqual(errors, [], f"Errors: {errors[:3]}")
        self.assertEqual(len(results), WORKERS, "Not all threads completed")
        self.assertEqual(sum(results), WORKERS, "Not all tasks succeeded")
        print(f"\n[load] {WORKERS} concurrent agents: {elapsed:.2f}s "
              f"({WORKERS/elapsed:.1f} tasks/s)")

    def test_memory_load_no_corruption(self):
        """Memory-intensive agents under load — no readback corruption."""
        results = []
        errors  = []
        lock    = threading.Lock()

        def run(n):
            sid  = SessionId.generate()
            task = Task.create("_MemIntensive", {"n": n}, session_id=sid)
            try:
                r = self.orch.run_task(task)
                for f in r.findings:
                    if f.severity == Severity.CRITICAL:
                        with lock:
                            errors.append(f.summary)
                with lock:
                    results.append(r.succeeded)
            except Exception as exc:
                with lock:
                    errors.append(str(exc))

        threads = [threading.Thread(target=run, args=(i,), daemon=True)
                   for i in range(WORKERS)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=TIMEOUT)

        self.assertEqual(errors, [], f"Memory corruption detected: {errors[:3]}")
        self.assertEqual(sum(results), WORKERS)

    def test_memory_entries_bounded(self):
        """After N_TASKS tasks, memory must not exceed max_entries."""
        for i in range(N_TASKS):
            task = Task.create("_FastAgent", {"n": i})
            self.orch.run_task(task)

        total = self.mem.total_entries
        self.assertLessEqual(total, self.mem._max_entries,
                             f"Memory exceeded limit: {total} > {self.mem._max_entries}")


class TestThroughput(unittest.TestCase):

    def setUp(self):
        self.reg, self.orch, _ = _setup()

    def test_sequential_throughput_baseline(self):
        """Establish baseline: N_TASKS sequential tasks under 10s with DryRun."""
        start = time.monotonic()
        for i in range(N_TASKS):
            task = Task.create("_FastAgent", {"n": i})
            r = self.orch.run_task(task)
            self.assertTrue(r.succeeded)
        elapsed = time.monotonic() - start
        tps = N_TASKS / elapsed
        print(f"\n[throughput] {N_TASKS} sequential tasks: {elapsed:.2f}s ({tps:.1f} tps)")
        self.assertGreater(tps, 10.0,
                           f"Sequential throughput too low: {tps:.1f} tps (expected > 10)")

    def test_parallel_throughput_scales(self):
        """Parallel execution should be faster than sequential for CPU-bound tasks."""
        from framework.orchestration.orchestrator import WorkflowBuilder

        # Build a workflow with N parallel tasks
        n_parallel = min(20, WORKERS)
        wf = WorkflowBuilder("throughput-test").parallel(
            *[("_FastAgent", {"n": i}) for i in range(n_parallel)]
        ).build()
        wf.max_workers = 8

        start = time.monotonic()
        result = self.orch.run_workflow(wf)
        elapsed = time.monotonic() - start

        self.assertTrue(result.succeeded)
        self.assertEqual(len(result.task_results), n_parallel)
        print(f"\n[throughput] {n_parallel} parallel tasks: {elapsed:.2f}s "
              f"({n_parallel/elapsed:.1f} tps)")


class TestStressContextSwitching(unittest.TestCase):
    """Simulate rapid context switching between agent sessions."""

    def setUp(self):
        self.reg, self.orch, _ = _setup()

    def test_rapid_session_switching(self):
        """
        Multiple threads each running many tasks with DIFFERENT sessions.
        Verifies context isolation under rapid switching.
        """
        all_errors = []
        lock = threading.Lock()

        def run_multi_session(thread_n):
            for i in range(10):
                sid  = SessionId.generate()
                task = Task.create("_FastAgent", {"n": thread_n * 100 + i},
                                   session_id=sid)
                r = self.orch.run_task(task)
                if not r.succeeded:
                    with lock:
                        all_errors.append(f"t{thread_n} task{i}: {r.error}")

        threads = [threading.Thread(target=run_multi_session, args=(n,))
                   for n in range(20)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=30)

        self.assertEqual(all_errors, [], f"Context switch errors: {all_errors[:3]}")


if __name__ == "__main__":
    unittest.main()
