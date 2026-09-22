import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.api.server import start_task_api_server, stop_task_api_server
from framework.billing.ledger import TokenQuotaEnforcer, UsageLedger
from framework.core.registry import AgentRegistry, reset_registry
from framework.core.types import AgentMetadata, Severity, Task
from framework.agents.base import StructuralAgent
from framework.multitenancy.context import TenantContext, TenantRegistry
from framework.orchestration.orchestrator import TaskOrchestrator
from framework.resilience.retry import reset_circuit_registry


class _QuotaAgent(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_QuotaAgent", display_name="Quota", description="Test",
        version="1.0", capabilities=(),
    )

    def observe(self, task: Task):
        return {"ok": True}

    def reason(self, observations, task):
        return "{}"

    def parse_findings(self, observations, reasoning, task):
        return [self._make_finding("quota", Severity.INFO, "ok")]


def _post(port, body):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/tasks", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class TestUsageQuotaAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reset_registry()
        reset_circuit_registry()
        cls.registry = AgentRegistry()
        cls.registry.register_class(_QuotaAgent)
        cls.orchestrator = TaskOrchestrator(cls.registry, config=None)
        cls.tenants = TenantRegistry()
        cls.tenants.register(TenantContext(tenant_id="acme", catalog="tenant_acme"))
        cls.ledger = UsageLedger()
        cls.quota = TokenQuotaEnforcer(cls.ledger, {"acme": 10})
        cls.port = start_task_api_server(
            cls.orchestrator, cls.tenants, port=0, agent_registry=cls.registry,
            usage_ledger=cls.ledger, quota_enforcer=cls.quota,
        )

    def setUp(self):
        self.ledger._entries.clear()

    @classmethod
    def tearDownClass(cls):
        stop_task_api_server()

    def test_task_records_usage_for_tenant(self):
        status, body = _post(self.port, {
            "agent_type": "_QuotaAgent", "payload": {"x": "small"},
            "tenant_id": "acme", "estimated_tokens": 4,
        })
        self.assertEqual(status, 200)
        self.assertEqual(self.ledger.total_tokens("acme"), 4)

    def test_projected_usage_over_quota_returns_429(self):
        self.ledger.record("acme", 4, "prior-task")
        status, body = _post(self.port, {
            "agent_type": "_QuotaAgent", "payload": {"x": "small"},
            "tenant_id": "acme", "estimated_tokens": 7,
        })
        self.assertEqual(status, 429)
        self.assertIn("quota", body["error"])


if __name__ == "__main__":
    unittest.main()
