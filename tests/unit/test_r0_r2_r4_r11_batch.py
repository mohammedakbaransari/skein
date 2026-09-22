import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.agents.base import StructuralAgent
from framework.api.server import start_task_api_server, stop_task_api_server
from framework.core.registry import AgentRegistry, reset_registry
from framework.core.types import AgentMetadata, Severity, Task
from framework.multitenancy.context import TenantContext, TenantRegistry
from framework.orchestration.orchestrator import TaskOrchestrator
from framework.resilience.retry import reset_circuit_registry
from framework.security.authorization import AuthorizationPolicy, RoleAuthorizationError
from framework.adapters.identity import Principal
from datetime import datetime, timezone


class _EchoAgent(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_EchoAgent2", display_name="Echo2", description="Test",
        version="1.0", capabilities=(),
    )

    def observe(self, task: Task):
        return {"echo": task.payload.get("msg", "")}

    def reason(self, obs, task):
        return json.dumps(obs)

    def parse_findings(self, obs, reasoning, task):
        return [self._make_finding("echo", Severity.INFO, f"echo:{obs['echo']}")]


def _post(port, path, body, headers=None):
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json", **(headers or {})}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class TestWorkflowEngineWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reset_registry()
        reset_circuit_registry()
        cls.registry = AgentRegistry()
        cls.registry.register_class(_EchoAgent)
        cls.orchestrator = TaskOrchestrator(cls.registry, config=None)
        cls.tenants = TenantRegistry()
        cls.tenants.register(TenantContext(tenant_id="acme", catalog="tenant_acme"))
        cls.port = start_task_api_server(
            cls.orchestrator, cls.tenants, port=0, agent_registry=cls.registry,
        )

    @classmethod
    def tearDownClass(cls):
        stop_task_api_server()

    def test_sync_workflow_executes_through_engine(self):
        status, body = _post(self.port, "/v1/workflows", {
            "name": "wf-engine", "tenant_id": "acme",
            "steps": [{"agent_type": "_EchoAgent2", "payload": {"msg": "hi"}}],
        })
        self.assertEqual(status, 200)
        self.assertTrue(body["succeeded"])


class TestReviewAuthorizationEnforcement(unittest.TestCase):
    def test_role_required_when_principal_present(self):
        with self.assertRaises(RoleAuthorizationError):
            AuthorizationPolicy().require_any_role(
                Principal(
                    subject_id="u1", tenant_id="acme", issuer="https://issuer.example",
                    authentication_method="oidc", token_expiry=datetime.now(timezone.utc), roles=("viewer",),
                ),
                {"reviewer", "admin"}, "finding review transition",
            )


class TestSupplierStressAlternativePayloadMode(unittest.TestCase):
    def test_single_object_transaction_data_is_normalized(self):
        """A single-object payload must not crash observe() the way an
        un-normalized dict would (iterating a dict yields string keys,
        which have no '.get()') — the agent's >=4-months business rule
        still legitimately yields zero profiles for one record."""
        from agents.supply_risk.supplier_stress import SupplierStressAgent
        agent = SupplierStressAgent()
        observations = agent.observe(Task.create("SupplierStressAgent", {
            "transaction_data": {
                "supplier_id": "S001", "supplier_name": "TestCo", "month": "2024-01",
                "po_ack_days": 2.0, "otd_pct": 97.0, "quality_hold_pct": 0.8,
                "invoice_disputes": 1, "unsolicited_discounts": 0,
                "sales_response_hours": 4.0,
            },
        }))
        self.assertEqual(observations["supplier_count"], 0)


class TestSeverityClaimGrounding(unittest.TestCase):
    def test_summary_claiming_wrong_severity_is_flagged(self):
        agent = _EchoAgent()
        finding = agent._make_finding("signal", Severity.LOW, "This is a critical issue")
        warnings = agent._validate_grounded_findings([finding], {})
        self.assertTrue(any("ungrounded severity claim" in warning for warning in warnings))

    def test_matching_severity_claim_is_not_flagged(self):
        agent = _EchoAgent()
        finding = agent._make_finding("signal", Severity.CRITICAL, "This is a critical issue")
        warnings = agent._validate_grounded_findings([finding], {})
        self.assertFalse(any("ungrounded severity claim" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
