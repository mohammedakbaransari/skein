"""OIDC claims-to-Principal identity adapter (R12).

Scope: this adapter maps ALREADY-VALIDATED OIDC claims (as decoded by an
upstream gateway/JWKS-verifying library) into SKEIN's stable `Principal`
model. It intentionally does not perform JWKS fetch/signature
verification itself — that requires a live IdP and is out of scope for a
library-level adapter validated without network access. Trusted issuers/
audiences are enforced here; cryptographic verification is the deployment
gateway's responsibility (documented explicitly, not silently assumed).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Tuple

from framework.adapters.identity.interfaces import Principal


class OIDCClaimsError(ValueError):
    """Raised when OIDC claims fail issuer/audience/expiry validation."""


class OIDCIdentityAdapter:
    """Maps decoded OIDC claims to a `Principal`, no vendor claim names
    beyond this adapter boundary (see roadmap R12)."""

    def __init__(
        self, trusted_issuers: Iterable[str], accepted_audiences: Iterable[str],
        tenant_claim: str = "tid", roles_claim: str = "roles",
    ) -> None:
        self._trusted_issuers = set(trusted_issuers)
        self._accepted_audiences = set(accepted_audiences)
        self._tenant_claim = tenant_claim
        self._roles_claim = roles_claim

    def normalize(self, external_token: Mapping[str, Any]) -> Principal:
        issuer = external_token.get("iss", "")
        if issuer not in self._trusted_issuers:
            raise OIDCClaimsError(f"untrusted issuer: {issuer!r}")

        audience = external_token.get("aud")
        audiences: Tuple[str, ...] = (audience,) if isinstance(audience, str) else tuple(audience or ())
        if not self._accepted_audiences.intersection(audiences):
            raise OIDCClaimsError(f"audience not accepted: {audiences!r}")

        exp = external_token.get("exp")
        if exp is None:
            raise OIDCClaimsError("token is missing 'exp'")
        token_expiry = datetime.fromtimestamp(float(exp), tz=timezone.utc)
        if token_expiry <= datetime.now(timezone.utc):
            raise OIDCClaimsError("token has expired")

        subject_id = external_token.get("sub")
        if not subject_id:
            raise OIDCClaimsError("token is missing 'sub'")

        tenant_id = external_token.get(self._tenant_claim, "")
        if not tenant_id:
            raise OIDCClaimsError(f"token is missing tenant claim {self._tenant_claim!r}")

        roles = external_token.get(self._roles_claim, ())
        if isinstance(roles, str):
            roles = (roles,)

        return Principal(
            subject_id=str(subject_id), tenant_id=str(tenant_id), issuer=issuer,
            authentication_method="oidc", token_expiry=token_expiry,
            roles=tuple(roles), claims=dict(external_token),
        )
