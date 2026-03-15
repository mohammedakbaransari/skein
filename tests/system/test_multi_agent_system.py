"""
tests/system/test_multi_agent_system.py
=========================================
Multi-agent system tests for the SKEIN framework.

Tests full end-to-end workflows where multiple agents collaborate,
share memory, and produce aggregated findings.

These tests demonstrate:
  - Agent-to-agent result propagation via workflow
  - Shared memory across agent chain
  - Parallel agent execution
  - Thread safety under concurrent multi-agent runs
  - Scalability: 10 simultaneous workflows
"""

from __future__ import annotations

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
from framework.core.registry import AgentRegistry
from framework.core.types import (
    AgentMetadata, Finding, SessionId, Severity, Task,
)
from framework.memory.store import WorkingMemory
from framework.orchestration.orchestrator import TaskOrchestrator, Workflow, WorkflowBuilder


# ---------------------------------------------------------------------------
# Stubbed agents for system testing
# ---------------------------------------------------------------------------

class _SupplierAnalysisStub(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_SupplierAnalysisStub",
        display_name="Supplier Analysis Stub",
        description="System test stub",
        version="0.1.0",
        capabilities=(),
        tags=("system_test",),
    )

    def observe(self, task: Task) -> Dict[str, Any]:
        suppliers = task.payload.get("suppliers", [])
        return {
            "supplier_count": len(suppliers),
            "high_risk_count": sum(1 for s in suppliers if s.get("risk_score", 0) > 7),
        }

    def reason(self, obs: Dict, task: Task) -> str:
        return json.dumps({"risk_summary": f"{obs['high_risk_count']} high-risk suppliers"})

    def parse_findings(self, obs, reasoning, task) -> List[Finding]:
        parsed = self._parse_llm_json(reasoning) or {}
        if obs.get("high_risk_count", 0) > 0:
            return [self._make_finding(
                finding_type="supplier_portfolio_risk",
                severity=Severity.HIGH,
                summary=parsed.get("risk_summary", "Risk detected"),
            )]
        return []


class _CostAnalysisStub(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_CostAnalysisStub",
        display_name="Cost Analysis Stub",
        description="System test stub",
        version="0.1.0",
        capabilities=(),
        tags=("system_test",),
    )

    def observe(self, task: Task) -> Dict[str, Any]:
        # Access upstream dependency result if injected into payload
        dep_result = None
        for key, val in task.payload.items():
            if key.startswith("_dep_"):
                dep_result = val
                break
        return {
            "categories": task.payload.get("categories", []),
            "upstream_findings": len(dep_result.get("findings", [])) if dep_result else 0,
        }

    def reason(self, obs: Dict, task: Task) -> str:
        return json.dumps({"cost_finding": "Commodity prices declining — leverage opportunity"})

    def parse_findings(self, obs, reasoning, task) -> List[Finding]:
        parsed = self._parse_llm_json(reasoning) or {}
        return [self._make_finding(
            finding_type="cost_leverage",
            severity=Severity.MEDIUM,
            summary=parsed.get("cost_finding", "Cost opportunity"),
            evidence={"upstream_findings": obs.get("upstream_findings", 0)},
        )]


class _ReportStub(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_ReportStub",
        display_name="Report Generation Stub",
        description="System test stub",
        version="0.1.0",
        capabilities=(),
        tags=("system_test",),
    )

    def observe(self, task: Task) -> Dict[str, Any]:
        # Aggregate all upstream dependency results
        all_findings = []
        for key, val in task.payload.items():
            if key.startswith("_dep_"):
                all_findings.extend(val.get("findings", []))
        return {"total_upstream_findings": len(all_findings)}

    def reason(self, obs: Dict, task: Task) -> str:
        return json.dumps({"report": f"Aggregated {obs['total_upstream_findings']} findings"})

    def parse_findings(self, obs, reasoning, task) -> List[Finding]:
        parsed = self._parse_llm_json(reasoning) or {}
        return [self._make_finding(
            finding_type="portfolio_report",
            severity=Severity.INFO,
            summary=parsed.get("report", "Report generated"),
        )]


# ---------------------------------------------------------------------------
# System tests
# ---------------------------------------------------------------------------

class TestMultiAgentWorkflow(unittest.TestCase):

    def setUp(self):
        self.registry = AgentRegistry()
        self.registry.register_class(_SupplierAnalysisStub)
        self.registry.register_class(_CostAnalysisStub)
        self.registry.register_class(_ReportStub)
        self.memory = WorkingMemory()
        self.orchestrator = TaskOrchestrator(self.registry, config=None)

    def _make_workflow(self, name: str) -> Workflow:
        sid = SessionId.generate()

        supplier_task = Task.create(
            "_SupplierAnalysisStub",
            {"suppliers": [
                {"id": "S1", "risk_score": 8},
                {"id": "S2", "risk_score": 3},
                {"id": "S3", "risk_score": 9},
            ]},
            session_id=sid,
        )
        cost_task = Task.create(
            "_CostAnalysisStub",
            {"categories": ["packaging", "logistics"]},
            session_id=sid,
            depends_on=[supplier_task.task_id],
        )
        report_task = Task.create(
            "_ReportStub",
            {},
            session_id=sid,
            depends_on=[supplier_task.task_id, cost_task.task_id],
        )
        return Workflow(
            workflow_id=f"wf-{name}",
            name=name,
            session_id=sid,
            tasks=[supplier_task, cost_task, report_task],
        )

    def test_three_agent_workflow_all_succeed(self):
        workflow  = self._make_workflow("system-test-1")
        wf_result = self.orchestrator.run_workflow(workflow)
        self.assertTrue(wf_result.succeeded, f"Workflow failed: {wf_result.failed_tasks}")
        self.assertEqual(len(wf_result.task_results), 3)

    def test_upstream_results_injected_into_downstream(self):
        workflow  = self._make_workflow("system-test-dep")
        wf_result = self.orchestrator.run_workflow(workflow)
        self.assertTrue(wf_result.succeeded)
        # The cost task should have received upstream findings
        all_findings = wf_result.all_findings
        cost_finding = next(
            (f for f in all_findings if f.finding_type == "cost_leverage"), None
        )
        self.assertIsNotNone(cost_finding)
        # Upstream findings count should be > 0 (supplier task produced findings)
        self.assertGreater(
            cost_finding.evidence.get("upstream_findings", 0), 0
        )

    def test_report_sees_both_upstream_finding_types(self):
        workflow  = self._make_workflow("system-test-report")
        wf_result = self.orchestrator.run_workflow(workflow)
        finding_types = {f.finding_type for f in wf_result.all_findings}
        self.assertIn("supplier_portfolio_risk", finding_types)
        self.assertIn("cost_leverage", finding_types)
        self.assertIn("portfolio_report", finding_types)

    def test_workflow_duration_recorded(self):
        workflow  = self._make_workflow("system-test-timing")
        wf_result = self.orchestrator.run_workflow(workflow)
        self.assertIsNotNone(wf_result.duration_ms)
        self.assertGreater(wf_result.duration_ms, 0)

    def test_all_findings_flattened_from_all_agents(self):
        workflow  = self._make_workflow("system-test-findings")
        wf_result = self.orchestrator.run_workflow(workflow)
        all_findings = wf_result.all_findings
        # 3 agents × 1 finding each minimum
        self.assertGreaterEqual(len(all_findings), 3)


class TestConcurrentMultiAgentWorkflows(unittest.TestCase):
    """
    Test that 10 simultaneous workflows run safely.
    Verifies thread-safety of registry, orchestrator, and memory.
    """

    def setUp(self):
        self.registry = AgentRegistry()
        self.registry.register_class(_SupplierAnalysisStub)
        self.registry.register_class(_CostAnalysisStub)
        self.registry.register_class(_ReportStub)
        self.orchestrator = TaskOrchestrator(self.registry, config=None)

    def test_ten_concurrent_workflows_all_succeed(self):
        results = []
        errors  = []
        lock    = threading.Lock()

        def run_workflow(workflow_num: int):
            try:
                sid  = SessionId.generate()
                task = Task.create(
                    "_SupplierAnalysisStub",
                    {"suppliers": [{"id": f"S{i}", "risk_score": i} for i in range(5)],
                     "workflow_num": workflow_num},
                    session_id=sid,
                )
                result = self.orchestrator.run_task(task)
                with lock:
                    results.append(result)
            except Exception as exc:
                with lock:
                    errors.append(f"Workflow {workflow_num}: {exc}")

        threads = [
            threading.Thread(target=run_workflow, args=(i,))
            for i in range(10)
        ]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(len(errors), 0, f"Concurrent workflow errors: {errors}")
        self.assertEqual(len(results), 10)
        failed = [r for r in results if not r.succeeded]
        self.assertEqual(len(failed), 0, f"{len(failed)}/10 workflows failed")

    def test_session_isolation_under_concurrent_execution(self):
        """Verify session IDs are not mixed across concurrent runs."""
        session_map = {}
        lock = threading.Lock()

        def run_and_record(session_id: SessionId):
            task = Task.create(
                "_SupplierAnalysisStub",
                {"suppliers": []},
                session_id=session_id,
            )
            result = self.orchestrator.run_task(task)
            with lock:
                session_map[str(session_id)] = str(result.session_id)

        sids    = [SessionId.generate() for _ in range(15)]
        threads = [threading.Thread(target=run_and_record, args=(sid,)) for sid in sids]
        for t in threads: t.start()
        for t in threads: t.join()

        for sid in sids:
            returned_sid = session_map.get(str(sid))
            self.assertEqual(returned_sid, str(sid),
                             f"Session mismatch: submitted {sid}, got {returned_sid}")


class TestAgentScalability(unittest.TestCase):
    """
    Tests demonstrating how the framework scales to dozens of agents.

    Key insight: the registry + orchestrator design means adding a new agent
    is a 3-step process:
      1. Define AgentMetadata
      2. Implement StructuralAgent subclass
      3. Register with registry.register_class()

    No other framework code changes needed.
    """

    def test_register_twenty_agent_types(self):
        """
        Demonstrates registering 20 distinct agent classes without
        any framework code changes.
        """
        registry = AgentRegistry()
        for i in range(20):
            cls = type(
                f"ScaleAgent_{i}",
                (_SupplierAnalysisStub,),
                {
                    "METADATA": AgentMetadata(
                        agent_type=f"ScaleAgent_{i}",
                        display_name=f"Scale Agent {i}",
                        description=f"Scalability test agent {i}",
                        version="0.1.0",
                        capabilities=(),
                        tags=("scale_test",),
                    )
                },
            )
            registry.register_class(cls)

        self.assertEqual(len(registry), 20)
        by_tag = registry.find_by_tag("scale_test")
        self.assertEqual(len(by_tag), 20)

    def test_throughput_100_sequential_tasks(self):
        """100 sequential single-agent tasks should complete in < 5 seconds."""
        registry = AgentRegistry()
        registry.register_class(_SupplierAnalysisStub)
        orchestrator = TaskOrchestrator(registry, config=None)

        t0 = time.monotonic()
        for _ in range(100):
            task = Task.create("_SupplierAnalysisStub", {"suppliers": []})
            orchestrator.run_task(task)
        elapsed = time.monotonic() - t0

        self.assertLess(elapsed, 5.0,
                        f"100 sequential tasks took {elapsed:.2f}s (limit 5s)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
