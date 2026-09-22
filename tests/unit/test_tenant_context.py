"""
tests/unit/test_tenant_context.py
====================================
Tests for framework/multitenancy/context.py: TenantContext identifier
validation and TenantRegistry routing (P2-3/P2-5 multi-tenant foundation).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.multitenancy.context import (
    TenantContext, TenantRegistry, get_tenant_registry,
    reset_tenant_registry, validate_identifier,
)


class TestValidateIdentifier(unittest.TestCase):

    def test_valid_identifier_passes(self):
        self.assertEqual(validate_identifier("tenant_acme", "catalog"), "tenant_acme")

    def test_rejects_sql_injection_attempt(self):
        with self.assertRaises(ValueError):
            validate_identifier("acme; DROP TABLE x; --", "catalog")

    def test_rejects_empty_string(self):
        with self.assertRaises(ValueError):
            validate_identifier("", "catalog")

    def test_rejects_leading_digit(self):
        with self.assertRaises(ValueError):
            validate_identifier("1acme", "catalog")

    def test_rejects_dotted_name(self):
        # dots must not be accepted here — qualified_table() joins
        # components explicitly, this guards against a caller trying to
        # smuggle extra identifier parts through one field.
        with self.assertRaises(ValueError):
            validate_identifier("acme.other_catalog", "catalog")


class TestTenantContext(unittest.TestCase):

    def test_qualified_table_name(self):
        ctx = TenantContext(tenant_id="acme", catalog="tenant_acme")
        self.assertEqual(ctx.qualified_table("institutional_memory"),
                          "tenant_acme.skein.institutional_memory")

    def test_construction_rejects_invalid_catalog(self):
        with self.assertRaises(ValueError):
            TenantContext(tenant_id="acme", catalog="acme; DROP TABLE x")

    def test_qualified_table_rejects_invalid_table_argument(self):
        ctx = TenantContext(tenant_id="acme", catalog="tenant_acme")
        with self.assertRaises(ValueError):
            ctx.qualified_table("x; DROP TABLE y")


class TestTenantRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = TenantRegistry()

    def test_register_and_get(self):
        ctx = TenantContext(tenant_id="acme", catalog="tenant_acme")
        self.registry.register(ctx)
        self.assertIs(self.registry.get("acme"), ctx)

    def test_unknown_tenant_raises(self):
        with self.assertRaises(KeyError):
            self.registry.get("does-not-exist")

    def test_contains_and_len(self):
        self.registry.register(TenantContext(tenant_id="acme", catalog="tenant_acme"))
        self.registry.register(TenantContext(tenant_id="globex", catalog="tenant_globex"))
        self.assertIn("acme", self.registry)
        self.assertNotIn("initech", self.registry)
        self.assertEqual(len(self.registry), 2)

    def test_physical_isolation_two_tenants_get_distinct_catalogs(self):
        acme = TenantContext(tenant_id="acme", catalog="tenant_acme")
        globex = TenantContext(tenant_id="globex", catalog="tenant_globex")
        self.registry.register(acme)
        self.registry.register(globex)
        self.assertNotEqual(
            self.registry.get("acme").qualified_table("institutional_memory"),
            self.registry.get("globex").qualified_table("institutional_memory"),
        )


class TestGlobalTenantRegistry(unittest.TestCase):

    def tearDown(self):
        reset_tenant_registry()

    def test_singleton_returns_same_instance(self):
        reset_tenant_registry()
        self.assertIs(get_tenant_registry(), get_tenant_registry())

    def test_reset_creates_a_fresh_registry(self):
        get_tenant_registry().register(TenantContext(tenant_id="acme", catalog="tenant_acme"))
        reset_tenant_registry()
        self.assertNotIn("acme", get_tenant_registry())


if __name__ == "__main__":
    unittest.main()
