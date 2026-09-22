import tempfile
import unittest
from pathlib import Path

from framework.adapters.secrets import FileSecretsProvider
from framework.api.openapi import build_openapi_spec


class TestFileSecretsProvider(unittest.TestCase):
    def test_rotation_and_metadata_never_expose_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = FileSecretsProvider(Path(directory) / "secrets.json")
            provider.rotate_secret("API_KEY", "first-value")
            self.assertEqual(provider.get_secret("API_KEY"), "first-value")
            metadata = provider.get_secret_metadata("API_KEY")
            self.assertEqual(metadata.version, "1")
            self.assertNotIn("first-value", repr(metadata))
            provider.rotate_secret("API_KEY", "second-value")
            self.assertEqual(provider.get_secret("API_KEY"), "second-value")
            self.assertEqual(provider.get_secret_metadata("API_KEY").version, "2")

    def test_missing_secret_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyError):
                FileSecretsProvider(Path(directory) / "missing.json").get_secret("NOPE")

    def test_rotation_requires_replacement_value(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                FileSecretsProvider(Path(directory) / "secrets.json").rotate_secret("API_KEY")


class TestWorkflowContractRoutes(unittest.TestCase):
    def test_workflow_routes_are_declared(self):
        spec = build_openapi_spec()
        self.assertIn("/v1/workflows", spec["paths"])
        self.assertIn("/v1/workflows/async", spec["paths"])


if __name__ == "__main__":
    unittest.main()
