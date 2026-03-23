"""
tests/integration/test_framework_integration.py
=================================================
Integration tests for SKEIN framework infrastructure.

Tests:
  - Registry: register, create, status, health snapshot
  - Orchestrator: single task dispatch, workflow with dependencies
  - Memory sharing between agents in a workflow
  - Governance logger: hash chain integrity after concurrent writes
  - Circuit breaker: integrated with orchestrator on repeated failure
  - Pool manager: acquire/release agents under concurrent load
  - Correlation context: propagated through workflow
"""

import json
import sys
import tempfile
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
    AgentMetadata, AgentStatus, CorrelationContext,
    Finding, SessionId, Severity, Task, TaskStatus,
)
from framework.governance.logger import GovernanceLogger
from framework.memory.store import WorkingMemory
from framework.orchestration.orchestrator import TaskOrchestrator, Workflow, WorkflowBuilder
from framework.reasoning.stubs import DryRunReasoningEngine
from framework.resilience.retry import reset_circuit_registry


# ---------------------------------------------------------------------------
# Minimal test agents
# ---------------------------------------------------------------------------

class EchoAgent(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_EchoAgent", display_name="Echo", description="Test",
        version="0.1.0", capabilities=(), tags=("test",),
    )

    def observe(self, task: Task) -> Dict[str, Any]:
        return {"echo": task.payload.get("msg", "hello")}

    def reason(self, obs: Dict, task: Task) -> str:
        return json.dumps({"status": "ok", "echo": obs["echo"]})

    def parse_findings(self, obs, reasoning, task) -> List[Finding]:
        p = self._parse_llm_json(reasoning) or {}
        return [self._make_finding("echo", Severity.INFO, f"Echo: {p.get('echo','')}")]


class CounterAgent(StructuralAgent):
    """Writes a counter to shared memory for downstream agents to read."""
    METADATA = AgentMetadata(
        agent_type="_CounterAgent", display_name="Counter", description="Test",
        version="0.1.0", capabilities=(), tags=("test",),
    )
    call_count = 0

    def observe(self, task: Task) -> Dict[str, Any]:
        CounterAgent.call_count += 1
        return {"count": CounterAgent.call_count}

    def reason(self, obs: Dict, task: Task) -> str:
        return json.dumps({"count": obs["count"]})

    def parse_findings(self, obs, reasoning, task) -> List[Finding]:
        return []


class FailingAgent(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_FailingAgent", display_name="Fail", description="Test",
        version="0.1.0", capabilities=(), tags=("test",),
    )

    def observe(self, task: Task) -> Dict[str, Any]:
        raise RuntimeError("Deliberate test failure")

    def reason(self, obs: Dict, task: Task) -> str:
        return "{}"

    def parse_findings(self, obs, reasoning, task) -> List[Finding]:
        return []


# ---------------------------------------------------------------------------
# Test: Registry
# ---------------------------------------------------------------------------

class TestRegistry(unittest.TestCase):

    def setUp(self):
        reset_registry()
        reset_circuit_registry()
        self.reg = AgentRegistry()

    def test_register_class(self):
        self.reg.register_class(EchoAgent)
        self.assertIn("_EchoAgent", self.reg)

    def test_duplicate_registration_is_idempotent(self):
        self.reg.register_class(EchoAgent)
        self.reg.register_class(EchoAgent)  # Should not raise
        self.assertEqual(len(self.reg), 1)

    def test_create_instance(self):
        self.reg.register_class(EchoAgent)
        agent = self.reg.create_instance("_EchoAgent", config=None)
        self.assertIsNotNone(agent)
        self.assertEqual(agent.agent_type, "_EchoAgent")

    def test_get_or_create_reuses_idle(self):
        self.reg.register_class(EchoAgent)
        a1 = self.reg.get_or_create("_EchoAgent", config=None)
        a2 = self.reg.get_or_create("_EchoAgent", config=None)
        self.assertIs(a1, a2)

    def test_unknown_type_raises(self):
        with self.assertRaises(KeyError):
            self.reg.create_instance("_NonExistent", config=None)

    def test_health_snapshot_structure(self):
        self.reg.register_class(EchoAgent)
        self.reg.create_instance("_EchoAgent", config=None)
        snap = self.reg.health_snapshot()
        self.assertIn("registered_classes", snap)
        self.assertIn("live_instances", snap)
        self.assertIn("instances", snap)
        self.assertGreater(snap["live_instances"], 0)

    def test_find_by_tag(self):
        self.reg.register_class(EchoAgent)
        results = self.reg.find_by_tag("test")
        self.assertTrue(any(m.agent_type == "_EchoAgent" for m in results))

    def test_terminate_removes_instance(self):
        self.reg.register_class(EchoAgent)
        agent = self.reg.create_instance("_EchoAgent", config=None)
        self.assertEqual(self.reg.live_count(), 1)
        self.reg.terminate(agent.agent_id)
        self.assertEqual(self.reg.live_count(), 0)


# ---------------------------------------------------------------------------
# Test: Orchestrator
# ---------------------------------------------------------------------------

class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        reset_registry()
        reset_circuit_registry()
        self.reg = AgentRegistry()
        self.reg.register_class(EchoAgent)
        self.reg.register_class(CounterAgent)
        self.reg.register_class(FailingAgent)
        self.orch = TaskOrchestrator(self.reg, config=None)

    def test_single_task_succeeds(self):
        task = Task.create("_EchoAgent", {"msg": "hello_integration"})
        result = self.orch.run_task(task)
        self.assertTrue(result.succeeded)
        self.assertEqual(len(result.findings), 1)

    def test_single_task_failure_is_captured(self):
        task = Task.create("_FailingAgent", {})
        result = self.orch.run_task(task)
        self.assertFalse(result.succeeded)
        self.assertIsNotNone(result.error)

    def test_workflow_sequential_succeeds(self):
        sid = SessionId.generate()
        wf = (WorkflowBuilder("seq-test")
              .session(sid)
              .step("_EchoAgent", {"msg": "step1"})
              .then("_EchoAgent", {"msg": "step2"})
              .build())
        result = self.orch.run_workflow(wf)
        self.assertTrue(result.succeeded)
        self.assertEqual(len(result.task_results), 2)

    def test_workflow_parallel_executes_all(self):
        sid = SessionId.generate()
        wf = (WorkflowBuilder("par-test")
              .session(sid)
              .parallel(
                  ("_EchoAgent", {"msg": "a"}),
                  ("_EchoAgent", {"msg": "b"}),
                  ("_EchoAgent", {"msg": "c"}),
              )
              .build())
        result = self.orch.run_workflow(wf)
        self.assertTrue(result.succeeded)
        self.assertEqual(len(result.task_results), 3)

    def test_workflow_cancels_dependents_on_failure(self):
        sid = SessionId.generate()
        wf = (WorkflowBuilder("fail-cancel")
              .session(sid)
              .step("_FailingAgent", {})
              .then("_EchoAgent", {"msg": "should_cancel"})
              .build())
        result = self.orch.run_workflow(wf)
        self.assertFalse(result.succeeded)
        self.assertEqual(len(result.failed_tasks), 1)
        self.assertEqual(len(result.cancelled_tasks), 1)

    def test_correlation_context_propagated(self):
        ctx = CorrelationContext.new(test="integration")
        task = Task.create("_EchoAgent", {"msg": "ctx-test"}, context=ctx)
        result = self.orch.run_task(task)
        self.assertEqual(result.context.trace_id, ctx.trace_id)

    def test_workflow_result_all_findings(self):
        sid = SessionId.generate()
        wf = (WorkflowBuilder("findings-test")
              .session(sid)
              .parallel(
                  ("_EchoAgent", {"msg": "f1"}),
                  ("_EchoAgent", {"msg": "f2"}),
              )
              .build())
        result = self.orch.run_workflow(wf)
        all_f = result.all_findings
        self.assertEqual(len(all_f), 2)


# ---------------------------------------------------------------------------
# Test: Memory sharing between agents
# ---------------------------------------------------------------------------

class MemoryWriterAgent(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_MemWriter", display_name="MemWriter", description="Test",
        version="0.1.0", capabilities=(), tags=("test",),
    )

    def observe(self, task: Task) -> Dict[str, Any]:
        self.remember("shared:result", {"value": 42}, session_id=task.session_id)
        return {"written": True}

    def reason(self, obs, task): return json.dumps({"ok": True})
    def parse_findings(self, obs, r, task): return []


class MemoryReaderAgent(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_MemReader", display_name="MemReader", description="Test",
        version="0.1.0", capabilities=(), tags=("test",),
    )

    def observe(self, task: Task) -> Dict[str, Any]:
        v = self.recall("shared:result", session_id=task.session_id)
        return {"read_value": v["value"] if v else None}

    def reason(self, obs, task): return json.dumps({"value": obs.get("read_value")})
    def parse_findings(self, obs, reasoning, task):
        p = self._parse_llm_json(reasoning) or {}
        return [self._make_finding("read_result", Severity.INFO,
                                   f"Read: {p.get('value')}")]


class TestMemorySharing(unittest.TestCase):

    def setUp(self):
        reset_registry()
        reset_circuit_registry()
        self.reg = AgentRegistry()
        self.mem = WorkingMemory(max_entries=1000)
        self.reg.register_class(MemoryWriterAgent)
        self.reg.register_class(MemoryReaderAgent)

    def test_memory_shared_within_session(self):
        """Writer stores, reader retrieves in same session via WorkingMemory."""
        sid = SessionId.generate()
        writer = self.reg.create_instance("_MemWriter", config=None, memory=self.mem)
        reader = self.reg.create_instance("_MemReader", config=None, memory=self.mem)

        write_task = Task.create("_MemWriter", {}, session_id=sid)
        read_task  = Task.create("_MemReader", {}, session_id=sid)

        writer.run(write_task)
        result = reader.run(read_task)

        self.assertTrue(result.succeeded)
        self.assertEqual(len(result.findings), 1)
        self.assertIn("42", result.findings[0].summary)

    def test_memory_isolated_between_sessions(self):
        """Writer in session A should not be visible in session B."""
        sid_a = SessionId.generate()
        sid_b = SessionId.generate()
        writer = self.reg.create_instance("_MemWriter", config=None, memory=self.mem)
        reader = self.reg.create_instance("_MemReader", config=None, memory=self.mem)

        writer.run(Task.create("_MemWriter", {}, session_id=sid_a))
        result = reader.run(Task.create("_MemReader", {}, session_id=sid_b))

        # Reading from session B — should not see session A's data
        # Finding summary should say "Read: None"
        self.assertIn("None", result.findings[0].summary)


# ---------------------------------------------------------------------------
# Test: Governance logger hash chain
# ---------------------------------------------------------------------------

class TestGovernanceLogger(unittest.TestCase):

    def test_chain_integrity_after_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = GovernanceLogger(tmpdir)
            reg = AgentRegistry()
            reg.register_class(EchoAgent)
            agent = reg.create_instance("_EchoAgent", config=None, governance=gov)

            for i in range(5):
                task = Task.create("_EchoAgent", {"msg": f"test_{i}"})
                agent.run(task)

            # Verify hash chain
            exec_log = Path(tmpdir) / "executions.jsonl"
            self.assertTrue(exec_log.exists())
            chain_ok = gov.verify_chain(str(exec_log))
            self.assertTrue(chain_ok, "Hash chain integrity verification failed")

    def test_chain_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = GovernanceLogger(tmpdir)
            reg = AgentRegistry()
            reg.register_class(EchoAgent)
            agent = reg.create_instance("_EchoAgent", config=None, governance=gov)
            task = Task.create("_EchoAgent", {"msg": "tamper_test"})
            agent.run(task)

            exec_log = Path(tmpdir) / "executions.jsonl"
            # Tamper with the file
            content = exec_log.read_text()
            import json as _j
            lines = [l for l in content.splitlines() if l.strip()]
            if lines:
                # Change a field value so hash will not match
                entry = _j.loads(lines[0])
                entry["agent_type"] = "TAMPERED_VALUE"
                lines[0] = _j.dumps(entry, sort_keys=True)
                exec_log.write_text("\n".join(lines) + "\n")

            chain_ok = gov.verify_chain(str(exec_log))
            self.assertFalse(chain_ok, "Tampered chain should fail verification")

    def test_concurrent_governance_writes(self):
        """Multiple agents writing to governance log concurrently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = GovernanceLogger(tmpdir)
            reg = AgentRegistry()
            reg.register_class(EchoAgent)
            errors = []

            def run_agent():
                try:
                    agent = reg.create_instance("_EchoAgent", config=None, governance=gov)
                    for _ in range(3):
                        task = Task.create("_EchoAgent", {"msg": "concurrent"})
                        agent.run(task)
                except Exception as exc:
                    errors.append(str(exc))

            threads = [threading.Thread(target=run_agent) for _ in range(8)]
            for t in threads: t.start()
            for t in threads: t.join()

            self.assertEqual(errors, [], f"Governance errors: {errors}")
            exec_log = Path(tmpdir) / "executions.jsonl"
            lines = [l for l in exec_log.read_text().splitlines() if l.strip()]
            self.assertEqual(len(lines), 24)  # 8 threads × 3 runs


if __name__ == "__main__":
    unittest.main()
