"""Provider-neutral secrets contracts and development adapter."""

from framework.adapters.secrets.environment import EnvironmentSecretsProvider
from framework.adapters.secrets.factory import UnknownSecretsProviderError, build_secrets_provider
from framework.adapters.secrets.file import FileSecretsProvider
from framework.adapters.secrets.interfaces import SecretMetadata, SecretsProvider
from framework.adapters.secrets.vault import VaultSecretsProvider

__all__ = [
    "EnvironmentSecretsProvider", "FileSecretsProvider", "VaultSecretsProvider",
    "SecretMetadata", "SecretsProvider", "build_secrets_provider", "UnknownSecretsProviderError",
]