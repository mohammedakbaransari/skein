"""
framework/reasoning/stubs.py
==============================
Stub reasoning components for testing.

DryRunReasoningEngine: returns a configurable synthetic response
without any LLM call. Used in all tests that do not require a
live LLM provider.

Usage:
    from framework.reasoning.stubs import DryRunReasoningEngine

    engine = DryRunReasoningEngine(synthetic_json={"status": "ok"})
    agent  = SupplierStressAgent(reasoning_engine=engine)
    result = agent.run(task)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from framework.reasoning.engine import (
    ReasoningEngine, ReasoningRequest, ReasoningResponse,
)
from framework.core.types import ReasoningStrategy

log = logging.getLogger(__name__)


class DryRunReasoningStrategy:
    """
    Deterministic reasoning stub.
    Returns a fixed synthetic response without any network call.
    Thread-safe: no mutable state.
    """

    def __init__(self, synthetic_response: str = '{"status": "DRY_RUN"}') -> None:
        self._response = synthetic_response

    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        log.debug("[DRY_RUN] LLM call suppressed for session=%s", request.session_id)
        parsed = None
        try:
            parsed = json.loads(self._response)
            if not isinstance(parsed, dict):
                parsed = {"_value": parsed}
        except Exception:
            pass
        return ReasoningResponse(
            content=self._response,
            strategy_used=request.strategy or ReasoningStrategy.CHAIN_OF_THOUGHT,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            model_used="dry_run",
            parsed_output=parsed,
        )


def DryRunReasoningEngine(
    synthetic_json: Optional[Dict[str, Any]] = None,
    synthetic_text: Optional[str] = None,
) -> ReasoningEngine:
    """
    Factory: return a ReasoningEngine backed by DryRunReasoningStrategy.

    Args:
        synthetic_json: Dict to serialise as the synthetic response.
        synthetic_text: Raw string response (overrides synthetic_json).

    Returns:
        A ReasoningEngine instance suitable for use in tests.

    Usage:
        engine = DryRunReasoningEngine({"executive_summary": "test"})
        agent  = SupplierStressAgent(reasoning_engine=engine)
    """
    if synthetic_text is not None:
        response_str = synthetic_text
    elif synthetic_json is not None:
        response_str = json.dumps(synthetic_json)
    else:
        response_str = '{"status": "DRY_RUN"}'

    return ReasoningEngine(
        primary_strategy=DryRunReasoningStrategy(response_str),
    )
