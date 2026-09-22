import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.findings.store import FindingRecord, FindingsStore
from framework.findings.review import ReviewState, ReviewWorkflow
from framework.orchestration.dead_letter import DeadLetterQueue
from framework.multitenancy.config_service import TenantConfigService
from framework.multitenancy.policy import TenantPolicy
from framework.resilience.pool import PoolConfig
from framework.core.types import AgentMetadata, Severity, Task, TenantId
from framework.agents.base import StructuralAgent
from framework.orchestration.orchestrator import TaskOrchestrator
from framework.core.registry import AgentRegistry, reset_registry
from framework.resilience.retry import reset_circuit_registry
from framework.governance.logger import GovernanceLogger
from framework.adapters.audit import InMemoryAuditSink


class _AlwaysFailsAgent(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_AlwaysFailsAgent", display_name="Fails", description="Test",
        version="1.0", capabilities=(),
    )

    def observe(self, task):
        return {}

    def reason(self, observations, task):
        return "{}"

    def parse_findings(self, observations, reasoning, task):
        raise RuntimeError("deliberate failure")


class _FindingAgent(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_FindingAgent", display_name="Finding", description="Test",
        version="1.0", capabilities=(),
    )

    def observe(self, task):
        return {}

    def reason(self, observations, task):
        return "{}"

    def parse_findings(self, observations, reasoning, task):
        return [self._make_finding("signal", Severity.HIGH, "issue detected")]


class TestFindingsStore(unittest.TestCase):
    def test_query_filters_by_tenant_and_severity(self):
        store = FindingsStore()
        store.add(FindingRecord(
            finding_id="f1", tenant_id="acme", task_id="t1", agent_type="A",
            severity="high", finding_type="x", summary="s", entity_id=None,
            confidence_score=0.5,
        ))
        store.add(FindingRecord(
            finding_id="f2", tenant_id="globex", task_id="t2", agent_type="A",
            severity="low", finding_type="x", summary="s", entity_id=None,
            confidence_score=0.5,
        ))
        results = store.query(tenant_id="acme")
        self.assertEqual([r.finding_id for r in results], ["f1"])
        self.assertEqual(store.query(severity="low")[0].finding_id, "f2")

    def test_add_result_persists_agent_findings(self):
        reset_registry()
        reset_circuit_registry()
        agent = _FindingAgent()
        result = agent.run(Task.create("_FindingAgent", {}, tenant_id=TenantId("acme")))
        store = FindingsStore()
        store.add_result("acme", "task-1", "_FindingAgent", result)
        self.assertEqual(len(store), 1)
        self.assertEqual(store.query(tenant_id="acme")[0].severity, "high")


class TestReviewWorkflow(unittest.TestCase):
    def test_valid_transition_chain(self):
        workflow = ReviewWorkflow()
        workflow.transition("f1", ReviewState.IN_REVIEW, assignee="alice")
        review = workflow.transition("f1", ReviewState.ACTIONED, comment="done")
        self.assertEqual(review.state, ReviewState.ACTIONED)
        self.assertEqual(review.assignee, "alice")

    def test_invalid_transition_is_rejected(self):
        workflow = ReviewWorkflow()
        workflow.transition("f1", ReviewState.IN_REVIEW)
        workflow.transition("f1", ReviewState.ACTIONED)
        with self.assertRaises(ValueError):
            workflow.transition("f1", ReviewState.IN_REVIEW)


class TestDeadLetterQueue(unittest.TestCase):
    def test_exhausted_retries_are_captured_and_queryable(self):
        reset_registry()
        reset_circuit_registry()
        registry = AgentRegistry()
        registry.register_class(_AlwaysFailsAgent)
        dlq = DeadLetterQueue()
        orch = TaskOrchestrator(registry, config=None, dead_letter_queue=dlq)
        from framework.core.types import RetryConfig
        task = Task.create("_AlwaysFailsAgent", {}, tenant_id=TenantId("acme"),
                            retry_config=RetryConfig(max_attempts=1))
        result = orch.run_task(task)
        self.assertFalse(result.succeeded)
        entries = dlq.list_entries(tenant_id="acme")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].agent_type, "_AlwaysFailsAgent")


class TestTenantConfigService(unittest.TestCase):
    def test_override_falls_back_to_defaults_and_records_history(self):
        service = TenantConfigService(defaults={"threshold": 5})
        self.assertEqual(service.get("acme", "threshold"), 5)
        service.set("acme", "threshold", 9, changed_by="ops-1")
        self.assertEqual(service.get("acme", "threshold"), 9)
        self.assertEqual(service.get("globex", "threshold"), 5)
        history = service.history("acme")
        self.assertEqual(history[0].old_value, 5)
        self.assertEqual(history[0].new_value, 9)

    def test_changed_by_is_required(self):
        service = TenantConfigService()
        with self.assertRaises(ValueError):
            service.set("acme", "k", "v", changed_by="")


class TestTenantPolicyPoolSizing(unittest.TestCase):
    def test_pool_config_derives_from_workflow_limits(self):
        policy = TenantPolicy(workflow_limits={"max_size": 25, "min_size": 3})
        cfg = PoolConfig.from_tenant_policy(policy)
        self.assertEqual(cfg.max_size, 25)
        self.assertEqual(cfg.min_size, 3)

    def test_missing_limits_use_defaults(self):
        cfg = PoolConfig.from_tenant_policy(TenantPolicy())
        self.assertEqual(cfg.max_size, PoolConfig().max_size)


class TestGovernanceAuditSinkWiring(unittest.TestCase):
    def test_execution_emits_audit_event_with_principal(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            sink = InMemoryAuditSink()
            governance = GovernanceLogger(directory, audit_sink=sink)
            reset_registry()
            reset_circuit_registry()
            agent = _FindingAgent()
            agent.governance = governance
            task = Task.create("_FindingAgent", {}, tenant_id=TenantId("acme"))
            task.principal_id = "user-1"
            agent.run(task)
            events = sink.events()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].principal_id, "user-1")
            self.assertEqual(events[0].tenant_id, "acme")


if __name__ == "__main__":
    unittest.main()
