"""HashiCorp Vault KV v2 secrets adapter (R5).

Implements Vault's documented KV v2 HTTP API (read/write) using only the
stdlib `urllib` — no `hvac` dependency, consistent with this project's
minimal-core philosophy. This adapter has NOT been validated against a
live Vault cluster; it is a protocol-conformant reference implementation
only. Do not claim production Vault integration works until it has been
exercised against a real Vault instance.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime
from typing import Optional

from framework.adapters.secrets.interfaces import SecretMetadata


class VaultRequestError(RuntimeError):
    """Raised when Vault's HTTP API returns an error response."""


class VaultSecretsProvider:
    """Vault KV v2 adapter: `GET/POST <addr>/v1/<mount>/data/<path>`."""

    def __init__(self, addr: str, token: str, mount: str = "secret", timeout_seconds: float = 5.0) -> None:
        self._addr = addr.rstrip("/")
        self._token = token
        self._mount = mount
        self._timeout_seconds = timeout_seconds

    def get_secret(self, name: str) -> str:
        payload = self._read(name)
        value = payload.get("value")
        if value is None:
            raise KeyError(f"required secret {name!r} has no 'value' field in Vault")
        return str(value)

    def get_secret_metadata(self, name: str) -> SecretMetadata:
        response = self._request("GET", f"/v1/{self._mount}/data/{name}")
        metadata = response.get("data", {}).get("metadata", {})
        return SecretMetadata(
            name=name,
            version=str(metadata.get("version", "1")),
            expires_at=(datetime.fromisoformat(metadata["expires_at"]) if metadata.get("expires_at") else None),
        )

    def rotate_secret(self, name: str, value: Optional[str] = None) -> None:
        if not value:
            raise ValueError("a replacement secret value is required")
        self._request("POST", f"/v1/{self._mount}/data/{name}", body={"data": {"value": value}})

    def _read(self, name: str) -> dict:
        response = self._request("GET", f"/v1/{self._mount}/data/{name}")
        data = response.get("data", {}).get("data")
        if data is None:
            raise KeyError(f"required secret {name!r} is not configured in Vault")
        return data

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        request = urllib.request.Request(
            f"{self._addr}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            headers={"X-Vault-Token": self._token, "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            raise VaultRequestError(f"Vault request failed: {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise VaultRequestError(f"Vault request failed: {exc}") from exc
