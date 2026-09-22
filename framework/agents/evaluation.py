"""Offline prompt/confidence evaluation harness (R22 remainder).

Runs an agent's parse_findings against fixed golden observations/
reasoning text (no live LLM call required) and asserts expected
findings/severities — a regression gate for prompt or parsing changes,
independent of any specific model provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class GoldenCase:
    name: str
    observations: Dict[str, Any]
    reasoning_text: str
    expected_min_findings: int = 1
    expected_severities: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class EvaluationResult:
    case_name: str
    passed: bool
    reason: str = ""


class EvaluationHarness:
    """Runs an agent's `parse_findings` against golden cases, no LLM call."""

    def __init__(self, agent, task_factory) -> None:
        self._agent = agent
        self._task_factory = task_factory

    def run(self, cases: List[GoldenCase]) -> List[EvaluationResult]:
        results = []
        for case in cases:
            results.append(self._run_one(case))
        return results

    def _run_one(self, case: GoldenCase) -> EvaluationResult:
        task = self._task_factory()
        try:
            findings = self._agent.parse_findings(case.observations, case.reasoning_text, task)
        except Exception as exc:
            return EvaluationResult(case.name, False, f"parse_findings raised: {exc}")

        if len(findings) < case.expected_min_findings:
            return EvaluationResult(
                case.name, False,
                f"expected >= {case.expected_min_findings} findings, got {len(findings)}",
            )
        if case.expected_severities:
            actual = {f.severity.value for f in findings}
            expected = set(case.expected_severities)
            if not expected.issubset(actual):
                return EvaluationResult(
                    case.name, False, f"expected severities {expected} not all in {actual}",
                )
        return EvaluationResult(case.name, True)

    @staticmethod
    def all_passed(results: List[EvaluationResult]) -> bool:
        return all(r.passed for r in results)
