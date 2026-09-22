"""Portable secrets-provider contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class SecretMetadata:
    name: str
    version: str = ""
    expires_at: datetime | None = None


class SecretsProvider(Protocol):
    def get_secret(self, name: str) -> str:
        ...

    def get_secret_metadata(self, name: str) -> SecretMetadata:
        ...

    def rotate_secret(self, name: str) -> None:
        ...