"""Durable usage ledger adapter using provider-neutral JSONL storage."""

from __future__ import annotations

from pathlib import Path

from framework.adapters.storage.jsonl import JsonlStore
from framework.billing.ledger import UsageEntry, UsageLedger


class JsonlUsageLedger(UsageLedger):
    def __init__(self, path: str | Path, usd_per_1k_tokens: float = 0.01) -> None:
        super().__init__(usd_per_1k_tokens=usd_per_1k_tokens)
        self._persistent = JsonlStore(path)
        for payload in self._persistent.read():
            self._entries.append(UsageEntry(**payload))

    def record(self, tenant_id: str, tokens: int, task_id: str) -> UsageEntry:
        entry = super().record(tenant_id, tokens, task_id)
        self._persistent.append(entry)
        return entry
