"""Provider-neutral file-backed secrets adapter for dev/staging."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from framework.adapters.secrets.interfaces import SecretMetadata


class FileSecretsProvider:
    """Atomic JSON secret store for controlled non-production deployments.

    Production deployments should replace this adapter with an approved
    KMS/Vault implementation; plaintext values are never included in
    metadata or exception text.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()

    def get_secret(self, name: str) -> str:
        with self._lock:
            value = self._read().get(name)
        if not value:
            raise KeyError(f"required secret {name!r} is not configured")
        return str(value["value"] if isinstance(value, dict) else value)

    def get_secret_metadata(self, name: str) -> SecretMetadata:
        with self._lock:
            value = self._read().get(name)
        if value is None:
            raise KeyError(f"required secret {name!r} is not configured")
        return SecretMetadata(
            name=name,
            version=str(value.get("version", "1")) if isinstance(value, dict) else "1",
            expires_at=(datetime.fromisoformat(value["expires_at"]) if isinstance(value, dict) and value.get("expires_at") else None),
        )

    def rotate_secret(self, name: str, value: str | None = None) -> None:
        if not value:
            raise ValueError("a replacement secret value is required")
        with self._lock:
            data = self._read()
            current = data.get(name, {})
            version = int(current.get("version", 0)) + 1 if isinstance(current, dict) else 2
            data[name] = {"value": value, "version": str(version), "rotated_at": datetime.now(timezone.utc).isoformat()}
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(self._path.suffix + ".tmp")
            temporary.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
            os.replace(temporary, self._path)

    def _read(self) -> dict:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))