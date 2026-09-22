import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.adapters.secrets.vault import VaultRequestError, VaultSecretsProvider
from framework.adapters.secrets.factory import UnknownSecretsProviderError, build_secrets_provider
from framework.adapters.secrets import EnvironmentSecretsProvider, FileSecretsProvider
from framework.adapters.identity.jwks import JWKSVerifier, JWTVerificationError
from framework.security.authorization import AuthorizationPolicy, RoleAuthorizationError
from framework.billing.ledger import UsageLedger
from framework.adapters.identity import Principal
from datetime import datetime, timezone


class _VaultKVHandler(BaseHTTPRequestHandler):
    store = {"API_KEY": {"data": {"value": "vault-secret"}, "metadata": {"version": "1"}}}

    def do_GET(self):
        name = self.path.rsplit("/", 1)[-1]
        record = self.store.get(name)
        if record is None:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps({"data": record}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        name = self.path.rsplit("/", 1)[-1]
        self.store[name] = {"data": payload["data"], "metadata": {"version": "2"}}
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


class TestVaultSecretsProvider(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _VaultKVHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_reads_existing_secret(self):
        provider = VaultSecretsProvider(f"http://127.0.0.1:{self.port}", token="dev-token")
        self.assertEqual(provider.get_secret("API_KEY"), "vault-secret")

    def test_missing_secret_raises_key_error(self):
        provider = VaultSecretsProvider(f"http://127.0.0.1:{self.port}", token="dev-token")
        with self.assertRaises((KeyError, VaultRequestError)):
            provider.get_secret("NOPE")

    def test_rotate_secret_writes_new_version(self):
        provider = VaultSecretsProvider(f"http://127.0.0.1:{self.port}", token="dev-token")
        provider.rotate_secret("API_KEY", "rotated-value")
        self.assertEqual(provider.get_secret("API_KEY"), "rotated-value")
        self.assertEqual(provider.get_secret_metadata("API_KEY").version, "2")


class TestSecretsProviderFactory(unittest.TestCase):
    def test_environment_is_default(self):
        self.assertIsInstance(build_secrets_provider({}), EnvironmentSecretsProvider)

    def test_file_provider_requires_path(self):
        with self.assertRaises(ValueError):
            build_secrets_provider({"provider": "file"})

    def test_file_provider_selected_with_path(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = build_secrets_provider({"provider": "file", "file_path": str(Path(directory) / "s.json")})
            self.assertIsInstance(provider, FileSecretsProvider)

    def test_unknown_provider_rejected(self):
        with self.assertRaises(UnknownSecretsProviderError):
            build_secrets_provider({"provider": "not-a-real-provider"})


class TestJWKSVerifier(unittest.TestCase):
    def test_valid_signature_returns_claims(self):
        from cryptography.hazmat.primitives.asymmetric import rsa
        import base64

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_numbers = private_key.public_key().public_numbers()

        def b64url(value: int) -> str:
            length = (value.bit_length() + 7) // 8
            return base64.urlsafe_b64encode(value.to_bytes(length, "big")).decode().rstrip("=")

        jwks = {"keys": [{"kid": "key-1", "n": b64url(public_numbers.n), "e": b64url(public_numbers.e)}]}
        header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "kid": "key-1"}).encode()).decode().rstrip("=")
        payload = base64.urlsafe_b64encode(json.dumps({"sub": "user-1", "tid": "acme"}).encode()).decode().rstrip("=")
        signing_input = f"{header}.{payload}".encode()

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        signature = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
        token = f"{header}.{payload}.{base64.urlsafe_b64encode(signature).decode().rstrip('=')}"

        claims = JWKSVerifier(jwks).verify(token)
        self.assertEqual(claims["sub"], "user-1")

    def test_tampered_payload_is_rejected(self):
        from cryptography.hazmat.primitives.asymmetric import rsa
        import base64

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_numbers = private_key.public_key().public_numbers()

        def b64url(value: int) -> str:
            length = (value.bit_length() + 7) // 8
            return base64.urlsafe_b64encode(value.to_bytes(length, "big")).decode().rstrip("=")

        jwks = {"keys": [{"kid": "key-1", "n": b64url(public_numbers.n), "e": b64url(public_numbers.e)}]}
        header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "kid": "key-1"}).encode()).decode().rstrip("=")
        payload = base64.urlsafe_b64encode(json.dumps({"sub": "user-1"}).encode()).decode().rstrip("=")
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        signature = private_key.sign(f"{header}.{payload}".encode(), padding.PKCS1v15(), hashes.SHA256())
        tampered_payload = base64.urlsafe_b64encode(json.dumps({"sub": "attacker"}).encode()).decode().rstrip("=")
        token = f"{header}.{tampered_payload}.{base64.urlsafe_b64encode(signature).decode().rstrip('=')}"

        with self.assertRaises(JWTVerificationError):
            JWKSVerifier(jwks).verify(token)


class TestAuthorizationPolicy(unittest.TestCase):
    def _principal(self, roles):
        return Principal(
            subject_id="user-1", tenant_id="acme", issuer="https://issuer.example",
            authentication_method="oidc", token_expiry=datetime.now(timezone.utc), roles=tuple(roles),
        )

    def test_no_principal_is_permissive(self):
        AuthorizationPolicy().require_any_role(None, {"reviewer"}, "action")

    def test_principal_with_required_role_passes(self):
        AuthorizationPolicy().require_any_role(self._principal(["reviewer"]), {"reviewer", "admin"}, "action")

    def test_principal_without_required_role_rejected(self):
        with self.assertRaises(RoleAuthorizationError):
            AuthorizationPolicy().require_any_role(self._principal(["viewer"]), {"reviewer", "admin"}, "action")


class TestUsageLedgerExportAndReconciliation(unittest.TestCase):
    def test_export_csv_and_reconciliation_report(self):
        ledger = UsageLedger(usd_per_1k_tokens=0.02)
        ledger.record("acme", 1000, "t1")
        ledger.record("acme", 500, "t2")
        csv_text = ledger.export_csv("acme")
        self.assertIn("t1", csv_text)
        self.assertIn("t2", csv_text)
        report = ledger.reconciliation_report("acme")
        self.assertTrue(report["reconciled"])
        self.assertEqual(report["entry_count"], 2)


if __name__ == "__main__":
    unittest.main()
