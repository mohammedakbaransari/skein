"""JWKS-based RS256 JWT signature verification (R12).

Real cryptographic verification of a compact JWT against a JSON Web Key
Set, using the `cryptography` library (already a project dependency).
Fetching the JWKS from a live IdP is a network operation left to the
caller (`fetch_jwks`); this module's core `verify` never makes a network
call, so it is fully testable with a locally-generated RSA keypair.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from typing import Any, Dict

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class JWTVerificationError(ValueError):
    """Raised when a JWT fails structural, signature, or claims checks."""


def _b64url_decode(segment: str) -> bytes:
    padding_needed = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding_needed)


def _rsa_public_key_from_jwk(jwk: Dict[str, Any]) -> rsa.RSAPublicKey:
    n = int.from_bytes(_b64url_decode(jwk["n"]), "big")
    e = int.from_bytes(_b64url_decode(jwk["e"]), "big")
    return rsa.RSAPublicNumbers(e, n).public_key()


def fetch_jwks(jwks_url: str, timeout_seconds: float = 5.0) -> Dict[str, Any]:
    """Retrieve a JWKS document over HTTP. The only network call in this module."""
    with urllib.request.urlopen(jwks_url, timeout=timeout_seconds) as response:
        return json.loads(response.read())


class JWKSVerifier:
    """Verifies RS256-signed compact JWTs against a JSON Web Key Set."""

    def __init__(self, jwks: Dict[str, Any]) -> None:
        self._keys = {key["kid"]: key for key in jwks.get("keys", []) if "kid" in key}

    def verify(self, compact_jwt: str) -> Dict[str, Any]:
        try:
            header_segment, payload_segment, signature_segment = compact_jwt.split(".")
        except ValueError as exc:
            raise JWTVerificationError("token is not in compact JWS format") from exc

        header = json.loads(_b64url_decode(header_segment))
        if header.get("alg") != "RS256":
            raise JWTVerificationError(f"unsupported algorithm: {header.get('alg')!r}")

        jwk = self._keys.get(header.get("kid"))
        if jwk is None:
            raise JWTVerificationError(f"no matching key for kid={header.get('kid')!r}")

        public_key = _rsa_public_key_from_jwk(jwk)
        signing_input = f"{header_segment}.{payload_segment}".encode()
        signature = _b64url_decode(signature_segment)
        try:
            public_key.verify(signature, signing_input, padding.PKCS1v15(), hashes.SHA256())
        except InvalidSignature as exc:
            raise JWTVerificationError("signature verification failed") from exc

        return json.loads(_b64url_decode(payload_segment))
