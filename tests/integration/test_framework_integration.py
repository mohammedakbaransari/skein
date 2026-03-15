"""
tests/integration/test_framework_integration.py
=================================================
Integration tests for the SKEIN framework infrastructure.

Tests:
  - AgentRegistry lifecycle (register, create, status, terminate)
  - Orchestrator single-task dispatch
  - Multi-agent workflow with dependencies
  - Memory sharing between agents in a workflow
  - Governance logger output
  - Registry health snapshot
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.agents.base import ProcurementAgent
from framework.agents.catalogue import SUPPLIER_STRESS_METADATA, DECISION_AUDIT_METADATA
from framework.core.registry import AgentRegistry, reset_registry
from framework.core.types import (
    AgentMetadata, AgentStatus, Finding, SessionId, Severity, Task, TaskStatus,
)
from framework.governance.logger import GovernanceLogger
from framework.memory.store import WorkingMemory
from framework.orchestration.orchestrator import TaskOrchestrator, Workflow, WorkflowBuilder


# ---------------------------------------------------------------------------
# Minimal test agent (no real LLM)
# ---------------------------------------------------------------------------

class _EchoAgent(ProcurementAgent):
    """Stub agent that echoes payload fields as findings."""

    METADATA = AgentMetadata(
        agent_type="_EchoAgent",
        display_name="Echo Test Agent",
        description="Testing stub",
        version="0.1.0",
        capabilities=(),
        tags=("test",),
    )

    def observe(self, task: Task) -> Dict[str, Any]:
        return {"echo": task.payload.get("message", "hello"), "task_id": str(task.task_id)}

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        return json.dumps({"status": "ok", "echo": observations.get("echo", "")})

    def parse_findings(self, observations, reasoning, task) -> List[Finding]:
        parsed = self._parse_llm_json(reasoning) or {}
        return [self._make_finding(
            finding_type="echo",
            severity=Severity.INFO,
            summary=f"Echo: {parsed.get('echo', '')}",
        )]


class _FailingAgent(ProcurementAgent):
    """Stub agent that always fails."""

    METADATA = AgentMetadata(
        agent_type="_FailingAgent",
        display_name="Failing Test Agent",
        description="Always fails for testing",
        version="0.1.0",
        capabilities=(),
        tags=("test",),
    )

    def observe(self, task: Task) -> Dict[str, Any]:
        raise RuntimeError("Intentional failure for testing")

    def reason(self, observations, task) -> str:
        return ""

    def parse_findings(self, observations, reasoning, task):
        return []


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestAgentRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = AgentRegistry()

    def test_register_class_and_retrieve_metadata(self):
        self.registry.register_class(_EchoAgent)
        self.assertIn("_EchoAgent", self.registry)
        metadata = self.registry.list_agents()
        types = [m.agent_type for m in metadata]
        self.assertIn("_EchoAgent", types)

    def test_duplicate_registration_raises(self):
        self.registry.register_class(_EchoAgent)
        with self.assertRaises(ValueError):
            self.registry.register_class(_EchoAgent)

    def test_class_without_metadata_raises(self):
        class BadAgent:
            pass
        with self.assertRaises((ValueError, AttributeError)):
            self.registry.register_class(BadAgent)  # type: ignore

    def test_create_instance_returns_agent(self):
        self.registry.register_class(_EchoAgent)
        agent = self.registry.create_instance("_EchoAgent", config=None)
        self.assertIsInstance(agent, _EchoAgent)

    def test_create_instance_unknown_type_raises(self):
        with self.assertRaises(KeyError):
            self.registry.create_instance("NonExistentAgent", config=None)

    def test_get_or_create_reuses_idle_instance(self):
        self.registry.register_class(_EchoAgent)
        a1 = self.registry.get_or_create("_EchoAgent", config=None)
        a2 = self.registry.get_or_create("_EchoAgent", config=None)
        self.assertIs(a1, a2)  # same object reused

    def test_update_status_running_increments_counter(self):
        self.registry.register_class(_EchoAgent)
        agent = self.registry.create_instance("_EchoAgent", config=None)
        self.registry.update_status(agent.agent_id, AgentStatus.RUNNING)
        snapshot = self.registry.health_snapshot()
        rec = next(
            r for r in snapshot["instances"]
            if r["agent_id"] == agent.agent_id.value
        )
        self.assertEqual(rec["active_tasks"], 1)

    def test_terminate_removes_instance(self):
        self.registry.register_class(_EchoAgent)
        agent = self.registry.create_instance("_EchoAgent", config=None)
        aid   = agent.agent_id
        self.registry.terminate(aid)
        self.assertIsNone(self.registry.get_instance(aid))

    def test_health_snapshot_is_serialisable(self):
        self.registry.register_class(_EchoAgent)
        snapshot = self.registry.health_snapshot()
        # Must not raise
        json.dumps(snapshot)

    def test_find_by_tag_works(self):
        self.registry.register_class(_EchoAgent)
        matches = self.registry.find_by_tag("test")
        types = [m.agent_type for m in matches]
        self.assertIn("_EchoAgent", types)

    def test_thread_safe_concurrent_registration(self):
        """Multiple threads registering different agent classes concurrently."""
        registry = AgentRegistry()
        errors   = []

        def register_echo():
            try:
                # Each thread creates a unique subclass
                new_cls = type(
                    f"Echo_{threading.current_thread().name}",
                    (_EchoAgent,),
                    {"METADATA": AgentMetadata(
                        agent_type=f"Echo_{threading.current_thread().name}",
                        display_name="Thread Echo",
                        description="Thread test",
                        version="0.1.0",
                        capabilities=(),
                    )},
                )
                registry.register_class(new_cls)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=register_echo, name=f"T{i}") for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        self.assertEqual(len(registry), 10)


# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------

class TestWorkingMemory(unittest.TestCase):

    def test_set_and_get(self):
        mem = WorkingMemory()
        sid = SessionId.generate()
        mem.set("key1", {"data": 42}, session_id=sid)
        self.assertEqual(mem.get("key1", session_id=sid), {"data": 42})

    def test_session_isolation(self):
        mem  = WorkingMemory()
        sid1 = SessionId.generate()
        sid2 = SessionId.generate()
        mem.set("shared_key", "session1_value", session_id=sid1)
        mem.set("shared_key", "session2_value", session_id=sid2)
        self.assertEqual(mem.get("shared_key", session_id=sid1), "session1_value")
        self.assertEqual(mem.get("shared_key", session_id=sid2), "session2_value")

    def test_missing_key_returns_none(self):
        mem = WorkingMemory()
        self.assertIsNone(mem.get("nonexistent"))

    def test_ttl_expiry(self):
        import time
        mem = WorkingMemory()
        mem.set("expiring", "value", ttl_seconds=0.01)
        time.sleep(0.05)
        self.assertIsNone(mem.get("expiring"))  # expired

    def test_delete_removes_key(self):
        mem = WorkingMemory()
        mem.set("to_delete", "value")
        mem.delete("to_delete")
        self.assertIsNone(mem.get("to_delete"))

    def test_concurrent_writes_no_corruption(self):
        """100 threads writing different keys concurrently."""
        mem  = WorkingMemory(max_entries=1000)
        lock = threading.Lock()
        errors = []

        def write(i):
            try:
                mem.set(f"key_{i}", i)
            except Exception as exc:
                with lock:
                    errors.append(str(exc))

        threads = [threading.Thread(target=write, args=(i,)) for i in range(100)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(len(errors), 0)
        self.assertLessEqual(mem.total_entries, 100)

    def test_get_or_default(self):
        mem = WorkingMemory()
        result = mem.get_or_default("missing", default="fallback")
        self.assertEqual(result, "fallback")


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

class TestTaskOrchestrator(unittest.TestCase):

    def setUp(self):
        self.registry = AgentRegistry()
        self.registry.register_class(_EchoAgent)
        self.registry.register_class(_FailingAgent)
        self.orchestrator = TaskOrchestrator(self.registry, config=None)

    def test_single_task_succeeds(self):
        task   = Task.create("_EchoAgent", {"message": "hello"})
        result = self.orchestrator.run_task(task)
        self.assertTrue(result.succeeded)
        self.assertGreater(len(result.findings), 0)
        self.assertEqual(result.findings[0].finding_type, "echo")

    def test_single_task_with_failing_agent(self):
        task   = Task.create("_FailingAgent", {})
        result = self.orchestrator.run_task(task)
        self.assertFalse(result.succeeded)
        self.assertIsNotNone(result.error)

    def test_workflow_with_two_sequential_tasks(self):
        t1 = Task.create("_EchoAgent", {"message": "step1"}, session_id=SessionId.generate())
        t2 = Task.create("_EchoAgent", {"message": "step2"}, session_id=t1.session_id,
                          depends_on=[t1.task_id])

        workflow = Workflow(
            workflow_id="wf-test-1",
            name="Sequential Test",
            session_id=t1.session_id,
            tasks=[t1, t2],
        )
        wf_result = self.orchestrator.run_workflow(workflow)
        self.assertTrue(wf_result.succeeded)
        self.assertEqual(len(wf_result.task_results), 2)

    def test_workflow_cancels_dependent_on_failure(self):
        failing = Task.create("_FailingAgent", {})
        dependent = Task.create(
            "_EchoAgent", {"message": "should not run"},
            depends_on=[failing.task_id]
        )
        workflow = Workflow(
            workflow_id="wf-test-fail",
            name="Failure Cascade Test",
            session_id=SessionId.generate(),
            tasks=[failing, dependent],
        )
        wf_result = self.orchestrator.run_workflow(workflow)
        self.assertFalse(wf_result.succeeded)
        self.assertIn(failing.task_id.value, wf_result.failed_tasks)
        self.assertIn(dependent.task_id.value, wf_result.cancelled_tasks)

    def test_dag_cycle_detection(self):
        t1 = Task.create("_EchoAgent", {})
        t2 = Task.create("_EchoAgent", {}, depends_on=[t1.task_id])
        # Create cycle: t1 depends on t2, t2 depends on t1
        t1.depends_on.append(t2.task_id)
        workflow = Workflow("wf-cycle", "Cycle Test", SessionId.generate(), [t1, t2])
        with self.assertRaises(ValueError):
            workflow.validate_dag()

    def test_parallel_tasks_run_concurrently(self):
        """Three independent tasks should all succeed in a workflow."""
        tasks = [
            Task.create("_EchoAgent", {"message": f"parallel_{i}"},
                        session_id=SessionId.generate())
            for i in range(3)
        ]
        workflow = Workflow("wf-parallel", "Parallel Test", tasks[0].session_id, tasks)
        wf_result = self.orchestrator.run_workflow(workflow)
        self.assertTrue(wf_result.succeeded)
        self.assertEqual(len(wf_result.task_results), 3)

    def test_workflow_builder_fluent_api(self):
        sid = SessionId.generate()
        workflow = (
            WorkflowBuilder("fluent-test")
            .session(sid)
            .step("_EchoAgent", {"message": "step0"})
            .then("_EchoAgent", {"message": "step1"})
            .build()
        )
        self.assertEqual(workflow.name, "fluent-test")
        self.assertEqual(len(workflow.tasks), 2)
        # step1 should depend on step0
        step1 = workflow.tasks[1]
        self.assertEqual(len(step1.depends_on), 1)

    def test_event_callback_called_on_task_completion(self):
        events = []
        def capture_event(event_type, data):
            events.append(event_type)

        orchestrator = TaskOrchestrator(self.registry, config=None,
                                         event_callback=capture_event)
        task = Task.create("_EchoAgent", {"message": "event_test"})
        orchestrator.run_task(task)
        self.assertIn("task_started", events)
        self.assertIn("task_completed", events)


# ---------------------------------------------------------------------------
# Governance logger tests
# ---------------------------------------------------------------------------

class TestGovernanceLogger(unittest.TestCase):

    def test_execution_record_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = GovernanceLogger(tmpdir)
            registry = AgentRegistry()
            registry.register_class(_EchoAgent)

            agent = registry.create_instance("_EchoAgent", config=None)
            agent.governance = logger
            task   = Task.create("_EchoAgent", {"message": "gov_test"})
            result = agent.run(task)

            exec_log = Path(tmpdir) / "executions.jsonl"
            self.assertTrue(exec_log.exists())
            lines = exec_log.read_text().strip().split("\n")
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["record_type"], "execution")
            self.assertTrue(entry["succeeded"])

    def test_hash_chain_integrity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = GovernanceLogger(tmpdir)
            registry = AgentRegistry()
            registry.register_class(_EchoAgent)

            agent = registry.create_instance("_EchoAgent", config=None)
            agent.governance = logger

            for i in range(5):
                task = Task.create("_EchoAgent", {"message": f"msg_{i}"})
                agent.run(task)

            exec_log = str(Path(tmpdir) / "executions.jsonl")
            self.assertTrue(logger.verify_chain(exec_log))

    def test_audit_log_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = GovernanceLogger(tmpdir)
            logger.audit("framework_startup", {"version": "1.0.0"})
            audit_log = Path(tmpdir) / "audit.jsonl"
            self.assertTrue(audit_log.exists())
            entry = json.loads(audit_log.read_text().strip())
            self.assertEqual(entry["event_type"], "framework_startup")


if __name__ == "__main__":
    unittest.main(verbosity=2)
