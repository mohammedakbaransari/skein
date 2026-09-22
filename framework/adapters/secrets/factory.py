"""Startup secrets-provider selection (R5) — config-driven, not hardcoded.

Selecting the concrete adapter is a deployment decision, never a code
change: `secrets.provider` in config.yaml (or `SKEIN_SECRETS_PROVIDER`
env var) picks `environment` (dev default), `file`, or `vault`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from framework.adapters.secrets.environment import EnvironmentSecretsProvider
from framework.adapters.secrets.file import FileSecretsProvider
from framework.adapters.secrets.vault import VaultSecretsProvider


class UnknownSecretsProviderError(ValueError):
    """Raised when an unrecognized secrets provider is configured."""


def build_secrets_provider(config: Optional[Dict[str, Any]] = None):
    config = config or {}
    provider = os.environ.get("SKEIN_SECRETS_PROVIDER", config.get("provider", "environment")).lower()

    if provider == "environment":
        return EnvironmentSecretsProvider()

    if provider == "file":
        path = os.environ.get("SKEIN_SECRETS_FILE_PATH", config.get("file_path"))
        if not path:
            raise ValueError("file secrets provider requires 'file_path' (or SKEIN_SECRETS_FILE_PATH)")
        return FileSecretsProvider(path)

    if provider == "vault":
        addr = os.environ.get("VAULT_ADDR", config.get("vault_addr"))
        token = os.environ.get("VAULT_TOKEN", config.get("vault_token"))
        if not addr or not token:
            raise ValueError("vault secrets provider requires 'vault_addr' and 'vault_token'")
        return VaultSecretsProvider(addr, token, mount=config.get("vault_mount", "secret"))

    raise UnknownSecretsProviderError(f"unknown secrets provider: {provider!r}")
