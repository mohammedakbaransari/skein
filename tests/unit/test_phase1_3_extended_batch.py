import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.billing.ledger import QuotaExceededError, TokenQuotaEnforcer, UsageLedger
from framework.agents.plugins import PluginAgentRegistry
from framework.findings.feedback import FeedbackStore
from framework.multitenancy.isolation import (
    CrossTenantAccessError, LogicalIsolationEnforcer, assert_scoped_to_tenant, scoped_key,
)
from framework.api.openapi import ROUTES, build_openapi_spec
from framework.core.registry import AgentRegistry, reset_registry
from framework.core.types import AgentMetadata, Task


class TestUsageLedgerAndQuota(unittest.TestCase):
    def test_records_tokens_and_computes_cost(self):
        ledger = UsageLedger(usd_per_1k_tokens=0.02)
        ledger.record("acme", 1000, "task-1")
        self.assertEqual(ledger.total_tokens("acme"), 1000)
        self.assertEqual(ledger.total_cost_usd("acme"), 0.02)

    def test_quota_enforcer_blocks_projected_overage(self):
        ledger = UsageLedger()
        enforcer = TokenQuotaEnforcer(ledger, quotas={"acme": 1000})
        ledger.record("acme", 900, "task-1")
        with self.assertRaises(QuotaExceededError):
            enforcer.check("acme", 200)

    def test_quota_not_enforced_when_unset(self):
        ledger = UsageLedger()
        enforcer = TokenQuotaEnforcer(ledger)
        enforcer.check("globex", 1_000_000)


class TestPluginAgentRegistry(unittest.TestCase):
    def test_register_from_path_and_per_tenant_enablement(self):
        reset_registry()
        registry = AgentRegistry()
        plugins = PluginAgentRegistry(registry)
        agent_class = plugins.register_from_path(
            "tests.unit.test_phase1_2_3_foundations", "_FindingAgent"
        )
        self.assertIn("_FindingAgent", registry)
        plugins.enable_for_tenant("acme", agent_class.METADATA.agent_type)
        self.assertTrue(plugins.is_enabled("acme", agent_class.METADATA.agent_type))
        self.assertFalse(plugins.is_enabled("globex", agent_class.METADATA.agent_type))


class TestFeedbackStore(unittest.TestCase):
    def test_feedback_converts_to_confidence_samples(self):
        store = FeedbackStore()
        store.record("f1", "SupplierStressAgent", 0.8, True, reviewer="alice")
        store.record("f2", "SupplierStressAgent", 0.4, False, reviewer="bob")
        samples = store.to_confidence_samples("SupplierStressAgent")
        self.assertEqual(len(samples), 2)
        self.assertEqual(len(store), 2)

    def test_reviewer_is_required(self):
        store = FeedbackStore()
        with self.assertRaises(ValueError):
            store.record("f1", "Agent", 0.5, True, reviewer="")


class TestLogicalIsolation(unittest.TestCase):
    def test_scoped_key_round_trips_and_rejects_cross_tenant_access(self):
        key = scoped_key("acme", "session:123")
        assert_scoped_to_tenant(key, "acme")
        with self.assertRaises(CrossTenantAccessError):
            assert_scoped_to_tenant(key, "globex")

    def test_enforcer_isolates_backing_store_by_tenant(self):
        backing = {}
        enforcer = LogicalIsolationEnforcer(backing)
        enforcer.put("acme", "k", "acme-value")
        enforcer.put("globex", "k", "globex-value")
        self.assertEqual(enforcer.get("acme", "k"), "acme-value")
        self.assertEqual(enforcer.get("globex", "k"), "globex-value")
        self.assertEqual(len(enforcer.keys_for_tenant("acme")), 1)


class TestOpenApiContract(unittest.TestCase):
    def test_spec_declares_every_route(self):
        spec = build_openapi_spec()
        for (method, path) in ROUTES:
            self.assertIn(path, spec["paths"])
            self.assertIn(method.lower(), spec["paths"][path])


if __name__ == "__main__":
    unittest.main()
