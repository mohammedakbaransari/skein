"""Normalized principal contract for OIDC, SAML, and other adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Protocol, Tuple


@dataclass(frozen=True)
class Principal:
    subject_id: str
    tenant_id: str
    issuer: str
    authentication_method: str
    token_expiry: datetime
    roles: Tuple[str, ...] = ()
    groups: Tuple[str, ...] = ()
    permissions: Tuple[str, ...] = ()
    claims: Mapping[str, Any] = field(default_factory=dict)


class IdentityAdapter(Protocol):
    """Translate an external identity into SKEIN's stable principal model."""

    def normalize(self, external_token: Mapping[str, Any]) -> Principal:
        ...