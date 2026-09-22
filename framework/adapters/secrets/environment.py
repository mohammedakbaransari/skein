"""Environment-backed development secrets provider."""

from __future__ import annotations

import os

from framework.adapters.secrets.interfaces import SecretMetadata


class EnvironmentSecretsProvider:
    """Development-only provider; production deployments must use an adapter."""

    def get_secret(self, name: str) -> str:
        value = os.environ.get(name)
        if value is None or value == "":
            raise KeyError(f"required secret {name!r} is not configured")
        return value

    def get_secret_metadata(self, name: str) -> SecretMetadata:
        self.get_secret(name)
        return SecretMetadata(name=name, version="environment")

    def rotate_secret(self, name: str) -> None:
        raise NotImplementedError(
            "environment secrets cannot be rotated by SKEIN"
        )