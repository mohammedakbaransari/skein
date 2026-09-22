"""
framework/auth/api_keys.py
=============================
Minimal API-key authentication/authorization for the task-submission API
(framework/api/server.py) — closes the "no authN/authZ" gap flagged
throughout the architecture assessment (§12/§15/§17/§33/§34): a tenant
model with nothing authenticating it only prevents *accidental*
cross-tenant mixing in storage, not a caller simply claiming to be any
tenant_id they like in a request body.

SCOPE: this is API-key auth for a service-to-service API, not a full
identity provider. It answers two questions:
  1. AuthN — is this a key we issued? (constant-time hash comparison)
  2. AuthZ — is the tenant this key belongs to the same tenant_id the
     request is trying to act as?

It deliberately does NOT implement OAuth2/OIDC/JWT, user accounts, roles,
or scopes — real future work if this API is ever exposed directly to end
users rather than sitting behind a gateway. Key provisioning/rotation/
storage-at-rest (e.g. in a real secrets manager) is also out of scope,
mirroring the same "routing, not provisioning" boundary drawn for tenant
storage in framework/multitenancy.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from dataclasses import dataclass
from typing import Dict, Optional


class AuthenticationError(Exception):
    """No valid API key presented (maps to HTTP 401)."""


class AuthorizationError(Exception):
    """A valid key was presented, but not for the tenant being acted as (HTTP 403)."""


@dataclass(frozen=True)
class AuthContext:
    """The authenticated identity for one request."""
    tenant_id: str
    key_id:    str


@dataclass(frozen=True)
class _KeyRecord:
    tenant_id: str
    key_id:    str


def generate_api_key() -> str:
    """Generate a new, high-entropy raw API key.

    Give this to the tenant exactly once at provisioning time — only its
    hash is ever stored server-side, so it cannot be recovered later.
    """
    return f"skein_{secrets.token_urlsafe(32)}"


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


class ApiKeyStore:
    """Thread-safe store of hashed API keys -> tenant identity.

    Never stores plaintext keys. authenticate() uses hmac.compare_digest
    for constant-time comparison per candidate key, to resist timing
    attacks; it is O(n) in the number of registered keys, acceptable at
    the scale this targets (hundreds of tenants, a handful of keys each).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._keys: Dict[str, _KeyRecord] = {}  # key_hash -> record

    def register(self, tenant_id: str, raw_key: str, key_id: Optional[str] = None) -> None:
        key_id = key_id or f"{tenant_id}-{secrets.token_hex(4)}"
        with self._lock:
            self._keys[hash_api_key(raw_key)] = _KeyRecord(tenant_id=tenant_id, key_id=key_id)

    def revoke(self, raw_key: str) -> None:
        with self._lock:
            self._keys.pop(hash_api_key(raw_key), None)

    def authenticate(self, raw_key: Optional[str]) -> AuthContext:
        if not raw_key:
            raise AuthenticationError("no API key presented")
        presented_hash = hash_api_key(raw_key)
        with self._lock:
            for stored_hash, record in self._keys.items():
                if hmac.compare_digest(stored_hash, presented_hash):
                    return AuthContext(tenant_id=record.tenant_id, key_id=record.key_id)
        raise AuthenticationError("API key not recognised")

    def __len__(self) -> int:
        with self._lock:
            return len(self._keys)


def authorize_tenant_match(auth: AuthContext, requested_tenant_id: Optional[str]) -> str:
    """Return the tenant_id this authenticated request is allowed to act as.

    If the request body also specifies a tenant_id, it must match the
    authenticated key's tenant — this is precisely the check that stops
    one tenant's credentials from being used to act as another tenant.
    """
    if requested_tenant_id and requested_tenant_id != auth.tenant_id:
        raise AuthorizationError(
            f"API key is provisioned for tenant_id={auth.tenant_id!r}, "
            f"not {requested_tenant_id!r}"
        )
    return auth.tenant_id
