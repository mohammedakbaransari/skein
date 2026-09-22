"""Durable feedback store adapter using provider-neutral JSONL storage."""

from __future__ import annotations

from pathlib import Path

from framework.adapters.storage.jsonl import JsonlStore
from framework.findings.feedback import FeedbackRecord, FeedbackStore


class JsonlFeedbackStore(FeedbackStore):
    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._persistent = JsonlStore(path)
        for payload in self._persistent.read():
            self._records.append(FeedbackRecord(**payload))

    def record(self, finding_id: str, agent_type: str, raw_confidence: float,
               correct: bool, reviewer: str, notes: str = "") -> FeedbackRecord:
        entry = super().record(finding_id, agent_type, raw_confidence, correct, reviewer, notes)
        self._persistent.append(entry)
        return entry
