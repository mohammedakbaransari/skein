"""
tests/unit/test_tenant_store_resolver.py
===========================================
Tests for framework/multitenancy/resolver.py::TenantStoreResolver.
Uses a fake Spark session (no live Databricks/PySpark required).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.multitenancy.context import TenantContext, TenantRegistry
from framework.multitenancy.resolver import TenantStoreResolver


class _FakeCollectResult:
    def __init__(self, rows):
        self._rows = rows

    def collect(self):
        return self._rows


class _FakeSpark:
    def sql(self, query, args=None):
        return _FakeCollectResult([])


class TestTenantStoreResolver(unittest.TestCase):

    def setUp(self):
        self.registry = TenantRegistry()
        self.registry.register(TenantContext(tenant_id="acme", catalog="tenant_acme"))
        self.resolver = TenantStoreResolver(self.registry, spark=_FakeSpark())

    def test_missing_tenant_id_returns_none(self):
        self.assertIsNone(self.resolver.resolve(None))
        self.assertIsNone(self.resolver.resolve(""))

    def test_unregistered_tenant_returns_none(self):
        self.assertIsNone(self.resolver.resolve("does-not-exist"))

    def test_registered_tenant_returns_delta_backed_stores(self):
        result = self.resolver.resolve("acme")
        self.assertIsNotNone(result)
        memory, governance = result
        self.assertEqual(memory._table, "tenant_acme.skein.institutional_memory")
        self.assertEqual(governance._table, "tenant_acme.skein.governance_events")

    def test_stores_are_cached_across_calls(self):
        first = self.resolver.resolve("acme")
        second = self.resolver.resolve("acme")
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

    def test_two_tenants_get_independent_stores(self):
        self.registry.register(TenantContext(tenant_id="globex", catalog="tenant_globex"))
        acme_memory, acme_gov = self.resolver.resolve("acme")
        globex_memory, globex_gov = self.resolver.resolve("globex")
        self.assertNotEqual(acme_memory._table, globex_memory._table)
        self.assertNotEqual(acme_gov._table, globex_gov._table)

    def test_stores_still_construct_when_spark_queries_fail(self):
        # DeltaTableMemoryStore/DeltaGovernanceStore catch Spark errors
        # internally and fall back to an in-memory cache rather than
        # raising (by design, per P0-4/§14) — construction still succeeds.
        class _BrokenSpark:
            def sql(self, *a, **kw):
                raise RuntimeError("cluster unreachable")

        registry = TenantRegistry()
        registry.register(TenantContext(tenant_id="brokentenant", catalog="tenant_broken"))
        resolver = TenantStoreResolver(registry, spark=_BrokenSpark())
        result = resolver.resolve("brokentenant")
        self.assertIsNotNone(result)

    def test_resolve_falls_back_to_none_if_store_construction_itself_raises(self):
        # Directly exercises resolve()'s except-branch by forcing
        # _build_stores to raise, independent of how defensive the Delta
        # store classes themselves happen to be.
        def _broken_build_stores(context):
            raise RuntimeError("simulated unrecoverable construction failure")

        self.resolver._build_stores = _broken_build_stores
        self.assertIsNone(self.resolver.resolve("acme"))


if __name__ == "__main__":
    unittest.main()
