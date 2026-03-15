"""
framework/reasoning/engine.py
==============================
ReasoningEngine — pluggable reasoning strategy layer.

WHY A SEPARATE REASONING LAYER:
  The original POC called the LLM gateway directly from agents. This couples
  agents to one specific calling pattern (system+user prompt, JSON output).
  A real framework needs:
    - Multiple reasoning strategies (CoT, ReAct, Plan-Execute, Reflexion)
    - Framework integrations (LangChain, LangGraph, CrewAI)
    - Structured output enforcement
    - Token counting and budget management
    - Retry with strategy escalation

STRATEGY PATTERN:
  ReasoningEngine holds a strategy (ReasoningStrategy).
  Each strategy encapsulates one approach to LLM calling.
  Agents pick a strategy via config; framework can swap at runtime.

INTEGRATIONS:
  - NativeStrategy:     direct LLM gateway call (default, no extra deps)
  - LangChainStrategy:  uses LangChain LCEL chains
  - LangGraphStrategy:  uses LangGraph StateGraph for multi-step reasoning
  - CrewAIStrategy:     delegates to a CrewAI crew
"""

from __future__ import annotations

import abc
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from framework.core.types import ReasoningStrategy

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reasoning request / response
# ---------------------------------------------------------------------------

@dataclass
class ReasoningRequest:
    """Input to a reasoning strategy."""
    system_prompt:    str
    user_prompt:      str
    observations:     Dict[str, Any]
    strategy:         ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT
    output_schema:    Optional[Dict] = None   # JSON schema for structured output
    max_tokens:       int = 2048
    temperature:      float = 0.1
    session_id:       str = "default"


@dataclass
class ReasoningResponse:
    """Output from a reasoning strategy."""
    content:        str
    strategy_used:  ReasoningStrategy
    input_tokens:   Optional[int] = None
    output_tokens:  Optional[int] = None
    latency_ms:     Optional[float] = None
    model_used:     str = ""
    raw_response:   Optional[Any] = None   # provider-specific raw response
    parsed_output:  Optional[Dict] = None  # if output_schema was provided


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------

class ReasoningStrategyProtocol(Protocol):
    """Protocol all reasoning strategies must satisfy."""

    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        """Execute reasoning and return a structured response."""
        ...


# ---------------------------------------------------------------------------
# Native strategy — direct LLM gateway (default, no extra dependencies)
# ---------------------------------------------------------------------------

class NativeReasoningStrategy:
    """
    Direct LLM gateway call.
    Works with Ollama, Anthropic, OpenAI, Azure out of the box.
    No LangChain dependency.
    """

    def __init__(self, gateway) -> None:
        self._gateway = gateway

    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        import time
        t0 = time.monotonic()
        resp = self._gateway.complete(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            session_id=request.session_id,
        )
        elapsed = (time.monotonic() - t0) * 1000

        # If output schema requested, try to parse
        parsed = None
        if request.output_schema:
            parsed = _try_parse_json(resp.content)

        return ReasoningResponse(
            content=resp.content,
            strategy_used=request.strategy,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            latency_ms=round(elapsed, 1),
            model_used=resp.model,
            parsed_output=parsed,
        )


# ---------------------------------------------------------------------------
# LangChain strategy — uses LCEL runnable chains
# ---------------------------------------------------------------------------

class LangChainReasoningStrategy:
    """
    LangChain LCEL-based reasoning strategy.

    Requires: pip install langchain langchain-core

    Creates a chain: prompt | llm | output_parser
    Supports structured output via LangChain's JsonOutputParser.
    """

    def __init__(self, llm_config) -> None:
        self._llm_config = llm_config
        self._chain = None
        self._build_chain()

    def _build_chain(self) -> None:
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough

            llm = self._get_langchain_llm()
            self._prompt_template = ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}"),
                ("human",  "{user_prompt}"),
            ])
            self._chain = self._prompt_template | llm | StrOutputParser()
            log.info("LangChain LCEL chain built successfully")
        except ImportError:
            log.warning(
                "LangChain not installed — NativeReasoningStrategy will be used as fallback. "
                "Install with: pip install langchain langchain-core"
            )
            self._chain = None

    def _get_langchain_llm(self):
        """Build the LangChain LLM object from our config."""
        provider = self._llm_config.provider

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self._llm_config.model,
                api_key=self._llm_config.api_key,
                temperature=self._llm_config.temperature,
                max_tokens=self._llm_config.max_tokens,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self._llm_config.model,
                api_key=self._llm_config.api_key,
                temperature=self._llm_config.temperature,
                max_tokens=self._llm_config.max_tokens,
            )
        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=self._llm_config.model,
                base_url=self._llm_config.base_url,
                temperature=self._llm_config.temperature,
            )
        else:
            raise ValueError(f"No LangChain adapter for provider '{provider}'")

    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        if self._chain is None:
            # Graceful degradation: warn and return empty
            log.warning("LangChain chain unavailable — returning empty response")
            return ReasoningResponse(
                content="",
                strategy_used=request.strategy,
            )
        import time
        t0 = time.monotonic()
        content = self._chain.invoke({
            "system_prompt": request.system_prompt,
            "user_prompt":   request.user_prompt,
        })
        elapsed = (time.monotonic() - t0) * 1000
        return ReasoningResponse(
            content=content,
            strategy_used=request.strategy,
            latency_ms=round(elapsed, 1),
            parsed_output=_try_parse_json(content) if request.output_schema else None,
        )


# ---------------------------------------------------------------------------
# LangGraph strategy — multi-step stateful reasoning
# ---------------------------------------------------------------------------

class LangGraphReasoningStrategy:
    """
    LangGraph StateGraph-based reasoning for multi-step procurement analysis.

    Requires: pip install langgraph langchain-core

    Implements a 3-node graph:
      analyse → critique → refine
    Useful for complex procurement decisions requiring self-correction.
    """

    def __init__(self, llm_config) -> None:
        self._llm_config = llm_config
        self._graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph, END
            from langchain_core.messages import HumanMessage, SystemMessage
            from typing import TypedDict, Annotated
            import operator

            class GraphState(TypedDict):
                system_prompt: str
                user_prompt:   str
                analysis:      str
                critique:      str
                final_output:  str
                iteration:     int

            llm = LangChainReasoningStrategy(self._llm_config)._get_langchain_llm()

            def analyse_node(state: GraphState) -> dict:
                resp = llm.invoke([
                    SystemMessage(content=state["system_prompt"]),
                    HumanMessage(content=state["user_prompt"]),
                ])
                return {"analysis": resp.content, "iteration": 1}

            def critique_node(state: GraphState) -> dict:
                critique_prompt = (
                    f"Review this procurement analysis for completeness and accuracy:\n\n"
                    f"{state['analysis']}\n\n"
                    f"Identify any gaps, missing evidence, or logical errors."
                )
                resp = llm.invoke([HumanMessage(content=critique_prompt)])
                return {"critique": resp.content}

            def refine_node(state: GraphState) -> dict:
                refine_prompt = (
                    f"Original analysis:\n{state['analysis']}\n\n"
                    f"Critique:\n{state['critique']}\n\n"
                    f"Produce an improved final analysis addressing the critique. "
                    f"Output valid JSON only."
                )
                resp = llm.invoke([HumanMessage(content=refine_prompt)])
                return {"final_output": resp.content}

            def should_refine(state: GraphState) -> str:
                # Only refine once — prevent infinite loops
                return "refine" if state.get("iteration", 0) < 1 else END

            graph = StateGraph(GraphState)
            graph.add_node("analyse", analyse_node)
            graph.add_node("critique", critique_node)
            graph.add_node("refine", refine_node)
            graph.set_entry_point("analyse")
            graph.add_edge("analyse", "critique")
            graph.add_conditional_edges("critique", should_refine)
            graph.add_edge("refine", END)
            return graph.compile()

        except ImportError:
            log.warning(
                "LangGraph not installed — NativeReasoningStrategy fallback active. "
                "Install with: pip install langgraph"
            )
            return None

    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        if self._graph is None:
            return ReasoningResponse(content="", strategy_used=request.strategy)
        import time
        t0 = time.monotonic()
        final_state = self._graph.invoke({
            "system_prompt": request.system_prompt,
            "user_prompt":   request.user_prompt,
            "analysis":      "",
            "critique":      "",
            "final_output":  "",
            "iteration":     0,
        })
        elapsed = (time.monotonic() - t0) * 1000
        content = final_state.get("final_output") or final_state.get("analysis", "")
        return ReasoningResponse(
            content=content,
            strategy_used=request.strategy,
            latency_ms=round(elapsed, 1),
            parsed_output=_try_parse_json(content),
        )


# ---------------------------------------------------------------------------
# CrewAI strategy — multi-agent crew delegation
# ---------------------------------------------------------------------------

class CrewAIReasoningStrategy:
    """
    CrewAI-based reasoning: delegates to a specialist crew.

    Requires: pip install crewai

    Useful when a procurement task benefits from multiple specialised
    AI roles (analyst + critic + writer) working together.
    Each SKEIN agent can define its own CrewAI crew composition.
    """

    def __init__(self, crew_factory) -> None:
        """
        Args:
            crew_factory: Callable that returns a configured crewai.Crew instance.
        """
        self._crew_factory = crew_factory

    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        try:
            crew = self._crew_factory(request)
            import time
            t0     = time.monotonic()
            result = crew.kickoff()
            elapsed = (time.monotonic() - t0) * 1000
            content = str(result)
            return ReasoningResponse(
                content=content,
                strategy_used=request.strategy,
                latency_ms=round(elapsed, 1),
                parsed_output=_try_parse_json(content),
            )
        except ImportError:
            log.warning("CrewAI not installed. pip install crewai")
            return ReasoningResponse(content="", strategy_used=request.strategy)
        except Exception as exc:
            log.error("CrewAI reasoning failed: %s", exc, exc_info=True)
            return ReasoningResponse(
                content="",
                strategy_used=request.strategy,
            )


# ---------------------------------------------------------------------------
# ReasoningEngine — facade wiring strategies together
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """
    Facade for reasoning strategies.

    Agents interact with ReasoningEngine, not individual strategies.
    Strategy can be swapped at runtime without changing agent code.

    Fallback chain: primary strategy → NativeReasoningStrategy
    """

    def __init__(
        self,
        primary_strategy: ReasoningStrategyProtocol,
        fallback_gateway=None,
    ) -> None:
        self._primary = primary_strategy
        self._fallback = (
            NativeReasoningStrategy(fallback_gateway) if fallback_gateway else None
        )

    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        """
        Execute reasoning with primary strategy.
        Falls back to NativeReasoningStrategy on error.
        """
        try:
            response = self._primary.reason(request)
            if response.content:
                return response
            raise ValueError("Primary strategy returned empty content")
        except Exception as exc:
            if self._fallback:
                log.warning(
                    "Primary reasoning strategy failed (%s) — using fallback", exc
                )
                return self._fallback.reason(request)
            raise

    @classmethod
    def native(cls, gateway) -> "ReasoningEngine":
        """Factory: native strategy with no fallback."""
        return cls(NativeReasoningStrategy(gateway))

    @classmethod
    def langchain(cls, llm_config, fallback_gateway=None) -> "ReasoningEngine":
        """Factory: LangChain strategy with native fallback."""
        return cls(
            LangChainReasoningStrategy(llm_config),
            fallback_gateway=fallback_gateway,
        )

    @classmethod
    def langgraph(cls, llm_config, fallback_gateway=None) -> "ReasoningEngine":
        """Factory: LangGraph strategy with native fallback."""
        return cls(
            LangGraphReasoningStrategy(llm_config),
            fallback_gateway=fallback_gateway,
        )

    @classmethod
    def crewai(cls, crew_factory, fallback_gateway=None) -> "ReasoningEngine":
        """Factory: CrewAI strategy with native fallback."""
        return cls(
            CrewAIReasoningStrategy(crew_factory),
            fallback_gateway=fallback_gateway,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _try_parse_json(text: str) -> Optional[Dict]:
    """Try to parse JSON from LLM text. Returns None on failure."""
    import re
    clean = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", clean, re.DOTALL)
    if fence:
        clean = fence.group(1)
    try:
        result = json.loads(clean)
        return result if isinstance(result, dict) else {"_value": result}
    except json.JSONDecodeError:
        return None
