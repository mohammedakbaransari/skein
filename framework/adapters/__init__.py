"""Provider-neutral contracts and infrastructure adapters for SKEIN."""

from framework.adapters.audit import AuditEvent, AuditSink, InMemoryAuditSink
from framework.adapters.identity import IdentityAdapter, Principal
from framework.adapters.secrets import (
    EnvironmentSecretsProvider,
    FileSecretsProvider,
    SecretMetadata,
    SecretsProvider,
    VaultSecretsProvider,
    build_secrets_provider,
)
from framework.adapters.workflow import (
    InMemoryWorkflowEngine,
    WorkflowDefinition,
    WorkflowEngine,
    WorkflowRun,
    WorkflowState,
)

__all__ = [
    "AuditEvent",
    "AuditSink",
    "EnvironmentSecretsProvider",
    "FileSecretsProvider",
    "VaultSecretsProvider",
    "build_secrets_provider",
    "IdentityAdapter",
    "InMemoryAuditSink",
    "InMemoryWorkflowEngine",
    "Principal",
    "SecretMetadata",
    "SecretsProvider",
    "WorkflowDefinition",
    "WorkflowEngine",
    "WorkflowRun",
    "WorkflowState",
]