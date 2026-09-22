"""Evidence-based confidence scoring and offline calibration helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Protocol


@dataclass(frozen=True)
class ConfidenceSample:
    raw_score: float
    correct: bool


@dataclass(frozen=True)
class CalibrationProfile:
    agent_type: str
    empirical_accuracy: float
    sample_count: int
    dataset_version: str
    calibrated_at: str
    approved: bool = False
    approved_by: str = ""
    approved_at: str = ""


class CalibrationStore(Protocol):
    def get(self, agent_type: str) -> Optional[float]: ...
    def put(self, agent_type: str, empirical_accuracy: float) -> None: ...


class JsonCalibrationStore:
    """Portable local calibration store; deployment adapters can replace it."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)

    def get(self, agent_type: str) -> Optional[float]:
        data = self._read()
        value = data.get(agent_type)
        if isinstance(value, dict):
            if not value.get("approved", False):
                return None
            value = value.get("empirical_accuracy")
        return float(value) if value is not None else None

    def get_profile(self, agent_type: str) -> Optional[CalibrationProfile]:
        value = self._read().get(agent_type)
        if not isinstance(value, dict):
            return None
        return CalibrationProfile(**value)

    def put(self, agent_type: str, empirical_accuracy: float) -> None:
        if not 0.0 <= empirical_accuracy <= 1.0:
            raise ValueError("empirical_accuracy must be between 0 and 1")
        data = self._read()
        data[agent_type] = round(empirical_accuracy, 3)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self._path)

    def put_profile(self, profile: CalibrationProfile) -> None:
        if profile.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if not 0.0 <= profile.empirical_accuracy <= 1.0:
            raise ValueError("empirical_accuracy must be between 0 and 1")
        data = self._read()
        data[profile.agent_type] = asdict(profile)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self._path)

    def approve(self, agent_type: str, approved_by: str) -> CalibrationProfile:
        profile = self.get_profile(agent_type)
        if profile is None:
            raise KeyError(f"no calibration profile for {agent_type!r}")
        if not approved_by.strip():
            raise ValueError("approved_by is required")
        approved = CalibrationProfile(
            **{**asdict(profile), "approved": True, "approved_by": approved_by,
               "approved_at": datetime.now(timezone.utc).isoformat()}
        )
        self.put_profile(approved)
        return approved

    def _read(self) -> Dict[str, float]:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))


class ConfidenceScorer:
    """Deterministic scorer suitable for per-agent offline calibration."""

    @staticmethod
    def score(evidence: dict[str, Any], detail: str = "", entity_id: Optional[str] = None,
              entity_name: Optional[str] = None) -> float:
        score = 0.45
        score += min(0.25, 0.08 * sum(value not in (None, "", [], {}) for value in evidence.values()))
        score += 0.12 if entity_id else 0.0
        score += 0.08 if entity_name else 0.0
        score += 0.10 if len(detail.strip()) >= 40 else 0.0
        return round(min(0.95, score), 3)

    @staticmethod
    def calibrate(samples: Iterable[ConfidenceSample]) -> float:
        values = list(samples)
        if not values:
            raise ValueError("at least one confidence sample is required")
        return round(sum(sample.correct for sample in values) / len(values), 3)

    @staticmethod
    def build_profile(agent_type: str, samples: Iterable[ConfidenceSample], dataset_version: str) -> CalibrationProfile:
        values = list(samples)
        return CalibrationProfile(
            agent_type=agent_type,
            empirical_accuracy=ConfidenceScorer.calibrate(values),
            sample_count=len(values),
            dataset_version=dataset_version,
            calibrated_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def apply_calibration(raw_score: float, empirical_accuracy: float) -> float:
        if not 0.0 <= empirical_accuracy <= 1.0:
            raise ValueError("empirical_accuracy must be between 0 and 1")
        return round(max(0.0, min(1.0, (raw_score + empirical_accuracy) / 2)), 3)