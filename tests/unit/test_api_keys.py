"""
tests/unit/test_api_keys.py
==============================
Tests for framework/auth/api_keys.py: ApiKeyStore authentication and
tenant-match authorization (authN/authZ for the task-submission API).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.auth.api_keys import (
    ApiKeyStore, AuthContext, AuthenticationError, AuthorizationError,
    authorize_tenant_match, generate_api_key, hash_api_key,
)


class TestApiKeyStore(unittest.TestCase):

    def setUp(self):
        self.store = ApiKeyStore()
        self.raw_key = generate_api_key()
        self.store.register("acme", self.raw_key)

    def test_authenticate_valid_key(self):
        ctx = self.store.authenticate(self.raw_key)
        self.assertEqual(ctx.tenant_id, "acme")

    def test_authenticate_rejects_unknown_key(self):
        with self.assertRaises(AuthenticationError):
            self.store.authenticate("skein_not-a-real-key")

    def test_authenticate_rejects_missing_key(self):
        with self.assertRaises(AuthenticationError):
            self.store.authenticate(None)
        with self.assertRaises(AuthenticationError):
            self.store.authenticate("")

    def test_plaintext_key_is_never_stored(self):
        # Only the hash should ever appear in the store's internal state.
        for stored_hash in self.store._keys.keys():
            self.assertNotEqual(stored_hash, self.raw_key)
            self.assertEqual(stored_hash, hash_api_key(self.raw_key))

    def test_revoke_removes_key(self):
        self.store.revoke(self.raw_key)
        with self.assertRaises(AuthenticationError):
            self.store.authenticate(self.raw_key)

    def test_two_tenants_have_independent_keys(self):
        other_key = generate_api_key()
        self.store.register("globex", other_key)
        self.assertEqual(self.store.authenticate(self.raw_key).tenant_id, "acme")
        self.assertEqual(self.store.authenticate(other_key).tenant_id, "globex")

    def test_len_reflects_registered_key_count(self):
        self.assertEqual(len(self.store), 1)
        self.store.register("globex", generate_api_key())
        self.assertEqual(len(self.store), 2)

    def test_generate_api_key_is_high_entropy_and_unique(self):
        keys = {generate_api_key() for _ in range(100)}
        self.assertEqual(len(keys), 100)


class TestAuthorizeTenantMatch(unittest.TestCase):

    def test_matching_tenant_is_allowed(self):
        auth = AuthContext(tenant_id="acme", key_id="acme-1")
        self.assertEqual(authorize_tenant_match(auth, "acme"), "acme")

    def test_no_body_tenant_defaults_to_authenticated_tenant(self):
        auth = AuthContext(tenant_id="acme", key_id="acme-1")
        self.assertEqual(authorize_tenant_match(auth, None), "acme")

    def test_mismatched_tenant_is_rejected(self):
        auth = AuthContext(tenant_id="acme", key_id="acme-1")
        with self.assertRaises(AuthorizationError):
            authorize_tenant_match(auth, "globex")


if __name__ == "__main__":
    unittest.main()
