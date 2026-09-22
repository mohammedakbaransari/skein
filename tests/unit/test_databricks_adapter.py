"""
tests/unit/test_databricks_adapter.py
=======================================
Regression tests for DeltaTableMemoryStore SQL parameterization (P0-4 fix).
Uses a fake Spark session — no live Databricks/PySpark required.
"""

import sys
import unittest
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# Loaded by file path, not via `import platform.databricks.adapter`: the
# top-level `platform/` package name collides with the stdlib `platform`
# module, which is almost always already cached in sys.modules by the time
# this test runs — a normal package import would fail with
# "ModuleNotFoundError: No module named 'platform.databricks'".
_ADAPTER_PATH = ROOT / "platform" / "databricks" / "adapter.py"
_spec = importlib.util.spec_from_file_location("skein_databricks_adapter", _ADAPTER_PATH)
_adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adapter)
DeltaTableMemoryStore = _adapter.DeltaTableMemoryStore
DeltaGovernanceStore  = _adapter.DeltaGovernanceStore

from framework.multitenancy.context import TenantContext


class _FakeCollectResult:
    def __init__(self, rows):
        self._rows = rows

    def collect(self):
        return self._rows


class _FakeSpark:
    """Records every sql() call so tests can assert on query text and args."""

    def __init__(self):
        self.calls = []

    def sql(self, query, args=None):
        self.calls.append((query, args or {}))
        return _FakeCollectResult([])


INJECTION_KEY = "pattern:'; DROP TABLE skein.institutional_memory; --"
INJECTION_VALUE = {"note": "'; DELETE FROM skein.institutional_memory; --"}


class TestDeltaTableMemoryStoreParameterization(unittest.TestCase):

    def setUp(self):
        self.spark = _FakeSpark()
        self.store = DeltaTableMemoryStore(spark=self.spark)

    def test_set_uses_parameterized_args_not_string_interpolation(self):
        self.store.set(INJECTION_KEY, INJECTION_VALUE)
        query, args = self.spark.calls[-1]
        self.assertNotIn(INJECTION_KEY, query, "Key must not be interpolated into SQL text")
        self.assertNotIn("DROP TABLE", query, "Injected SQL must not appear in query text")
        self.assertEqual(args["key"], INJECTION_KEY)
        self.assertIn("value_json", args)

    def test_get_uses_parameterized_args_not_string_interpolation(self):
        self.store.get(INJECTION_KEY)
        query, args = self.spark.calls[-1]
        self.assertNotIn(INJECTION_KEY, query)
        self.assertNotIn("DROP TABLE", query)
        self.assertEqual(args["key"], INJECTION_KEY)

    def test_delete_uses_parameterized_args_not_string_interpolation(self):
        self.store.delete(INJECTION_KEY)
        query, args = self.spark.calls[-1]
        self.assertNotIn(INJECTION_KEY, query)
        self.assertNotIn("DROP TABLE", query)
        self.assertEqual(args["key"], INJECTION_KEY)


class TestDeltaTableMemoryStoreTenantRouting(unittest.TestCase):
    """Physical multi-tenant isolation: each tenant's data routes to its
    own dedicated catalog, never a shared table."""

    def test_tenant_context_routes_to_dedicated_catalog(self):
        spark = _FakeSpark()
        tenant = TenantContext(tenant_id="acme", catalog="tenant_acme")
        store = DeltaTableMemoryStore(spark=spark, tenant=tenant)
        self.assertEqual(store._table, "tenant_acme.skein.institutional_memory")

    def test_two_tenants_never_share_a_table(self):
        spark = _FakeSpark()
        acme = DeltaTableMemoryStore(spark=spark, tenant=TenantContext(tenant_id="acme", catalog="tenant_acme"))
        globex = DeltaTableMemoryStore(spark=spark, tenant=TenantContext(tenant_id="globex", catalog="tenant_globex"))
        self.assertNotEqual(acme._table, globex._table)

    def test_no_tenant_falls_back_to_legacy_single_tenant_table(self):
        # Backward compatibility: existing single-tenant callers are unaffected.
        spark = _FakeSpark()
        store = DeltaTableMemoryStore(spark=spark)
        self.assertEqual(store._table, "skein.institutional_memory")


class TestDeltaGovernanceStore(unittest.TestCase):
    """Tenant-partitioned, hash-chained governance log (Delta-backed).

    Each test uses a unique catalog name (not shared with other tests) so
    that DeltaGovernanceStore's class-level chain-state dicts (keyed by
    qualified table name, mirroring HashChainedWriter's fix in §28) cannot
    leak state between test methods regardless of execution order.
    """

    def _new_store(self, catalog_suffix: str):
        spark = _FakeSpark()
        tenant = TenantContext(tenant_id=catalog_suffix, catalog=f"tenant_{catalog_suffix}")
        return DeltaGovernanceStore(tenant=tenant, spark=spark), spark

    def test_table_is_tenant_qualified(self):
        store, _ = self._new_store("gov_a")
        self.assertEqual(store._table, "tenant_gov_a.skein.governance_events")

    def test_record_execution_uses_parameterized_insert(self):
        store, spark = self._new_store("gov_b")

        class _Result:
            succeeded = True
            duration_ms = 12.5
            findings = []

        class _Task:
            task_id = "task-1"
            session_id = "session-1"

        store.record_execution("agent-1", "SomeAgent", _Task(), _Result())
        query, args = spark.calls[-1]
        self.assertIn("INSERT INTO", query)
        self.assertIn("tenant_gov_b.skein.governance_events", query)
        self.assertEqual(args["event_type"], "execution")
        self.assertIn("prev_hash", args)
        self.assertIn("hash", args)

    def test_chain_is_valid_after_several_appends_via_fallback_buffer(self):
        # Force the in-memory fallback path (simulates PySpark unavailable)
        # so verify_chain() can be exercised without a real collect() shape.
        store, _ = self._new_store("gov_c")
        store._available = False
        for i in range(5):
            store.audit("test_event", {"i": i})
        self.assertTrue(store.verify_chain())

    def test_two_tenants_get_independent_chains(self):
        store_a, _ = self._new_store("gov_d1")
        store_b, _ = self._new_store("gov_d2")
        self.assertNotEqual(store_a._table, store_b._table)


if __name__ == "__main__":
    unittest.main()
