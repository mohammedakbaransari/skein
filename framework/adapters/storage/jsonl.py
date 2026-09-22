"""Append-only JSONL persistence adapter for provider-neutral stores."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional


class JsonlStore:
    """Thread-safe append-only JSONL store with injectable decoder."""

    def __init__(self, path: str | Path, decoder: Optional[Callable[[Dict[str, Any]], Any]] = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._decoder = decoder or (lambda value: value)
        self._lock = threading.RLock()

    def append(self, value: Any) -> None:
        payload = asdict(value) if is_dataclass(value) else value
        with self._lock, self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, default=str, sort_keys=True) + "\n")

    def read(self) -> list[Any]:
        if not self._path.exists():
            return []
        with self._lock, self._path.open(encoding="utf-8") as stream:
            return [self._decoder(json.loads(line)) for line in stream if line.strip()]

    def __len__(self) -> int:
        return len(self.read())
