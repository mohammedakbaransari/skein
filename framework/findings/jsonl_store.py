"""Provider-neutral durable findings store adapter."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

from framework.adapters.storage.jsonl import JsonlStore
from framework.findings.store import FindingRecord, FindingsStore


class JsonlFindingsStore(FindingsStore):
    """FindingsStore with append-only JSONL persistence and tenant queries."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._persistent = JsonlStore(path)
        for payload in self._persistent.read():
            self._records.append(FindingRecord(**payload))

    def add(self, record: FindingRecord) -> None:
        super().add(record)
        self._persistent.append(record)
