# SKEIN — Structural Knowledge and Enterprise Intelligence Network

**A domain-agnostic framework for deploying structural intelligence agents in complex enterprise systems**

*Research companion to: "The 15 Structural Mysteries of Procurement AI" — Mohammed Akbar Ansari, March 2026*

[![Tests](https://img.shields.io/badge/tests-100%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Research](https://img.shields.io/badge/paper-SSRN-orange)](https://papers.ssrn.com)
[![arXiv](https://img.shields.io/badge/preprint-arXiv-red)](https://arxiv.org)

---

**Author:** Mohammed Akbar Ansari
Senior Cloud Solutions Architect · Enterprise AI · Multi-Cloud Architecture
18+ years in enterprise architecture, procurement technology, and agentic AI systems

[GitHub](https://github.com/mohammedakbaransari) · [LinkedIn](https://linkedin.com/in/akbaransari)

*Personal research and open-source work. Views, findings, and all implementations are entirely the author's own.
Not affiliated with, endorsed by, or derived from any employer, vendor, or customer engagement.*

---

## What SKEIN Is

SKEIN is a modular, extensible AI agent framework for structural intelligence — the class of intelligence that operates on the hidden structural problems in complex enterprise functions rather than the visible transactional layer.

The name comes from the textile term: a skein is the tangled but coherent bundle of interconnected threads that, once resolved, allows the whole fabric to function properly. The fifteen mysteries of enterprise AI are exactly that — interconnected structural threads invisible at the surface but determinative of outcomes.

SKEIN is designed to be domain-agnostic. The first reference implementation addresses procurement. The architecture applies equally to healthcare, finance, legal operations, manufacturing, and any complex enterprise domain where important patterns are hidden beneath the visible surface of transactions and dashboards.

---

## The Problem SKEIN Addresses

49% of enterprise teams ran AI pilots in 2024. Only 4% reached meaningful production deployment.

That gap is not a technology failure. It is a research failure — the field is solving visible, demonstrable problems while leaving structural problems almost entirely unaddressed.

The structural problems that determine whether an enterprise function actually performs:
- They resist the architectural assumptions of current-generation platforms
- They require operating across boundaries enterprise systems have never bridged  
- They demand intelligence that reasons across trade-offs, not dashboards that display data
- They compound over time when left unsolved, and when AI is deployed without addressing them, it amplifies rather than resolves them

The research paper identifies fifteen such structural mysteries in enterprise procurement and proposes SKEIN as the architectural response. The framework is released open-source so the research community can extend it, critique it, and adapt it to other domains.

---

## Framework Architecture

```
SKEIN
├── framework/                        Domain-agnostic framework core
│   ├── core/
│   │   ├── types.py                  Canonical types: Task, Finding, AgentResult
│   │   └── registry.py               Agent registry with lifecycle management
│   ├── agents/
│   │   ├── base.py                   BaseAgent → StructuralAgent → ToolAgent → DecisionAgent
│   │   └── catalogue.py              15-agent metadata catalogue
│   ├── reasoning/
│   │   ├── engine.py                 Pluggable: native · LangChain · LangGraph · CrewAI
│   │   └── stubs.py                  DryRunReasoningEngine for testing without LLM
│   ├── memory/
│   │   └── store.py                  Working · Context · Institutional (3-tier)
│   ├── orchestration/
│   │   └── orchestrator.py           DAG workflows · parallel execution · WorkflowBuilder DSL
│   ├── governance/
│   │   └── logger.py                 Hash-chained audit logs · decision accountability
│   └── tools/
│       └── base.py                   ToolRegistry · ERP connector interface
│
├── agents/                           Structural intelligence agents (procurement domain)
│   ├── supply_risk/                  Mystery 02 — Supplier Stress Signal
│   ├── cost_intelligence/            Mystery 06 — Should-Cost · Mystery 14 — Total Cost
│   ├── contract_analysis/            Mystery 11 — Value Realisation
│   ├── decision_audit/               Mystery 13 — Decision Accountability
│   ├── bias_detection/               Mystery 15 — Incumbent Advantage Bias
│   ├── compliance/                   Mystery 09 — Compliance Verification
│   └── market_intelligence/          Mysteries 01, 03, 04, 05, 07, 08, 10, 12
│
├── config/config.yaml                LLM provider · reasoning strategy · governance
├── data/synthetic/                   Realistic synthetic datasets for development
└── tests/
    ├── unit/                         55 tests · pure domain logic
    ├── integration/                  29 tests · framework infrastructure
    └── system/                       16 tests · multi-agent workflows · concurrency
```

---

## The Agent Hierarchy

```
BaseAgent
├── StructuralAgent          observe → reason → parse_findings pipeline
│   ├── SupplierStressAgent               (Mystery 02)
│   ├── ShouldCostAgent                   (Mystery 06)
│   ├── ValueRealisationAgent             (Mystery 11)
│   ├── ComplianceVerificationAgent       (Mystery 09)
│   ├── TotalCostIntelligenceAgent        (Mystery 14)
│   ├── ProcurementBiasDetectorAgent      (Mystery 15)
│   ├── NegotiationIntelligenceAgent      (Mystery 03)
│   ├── SpecificationInflationAgent       (Mystery 04)
│   ├── DemandIntelligenceAgent           (Mystery 07)
│   ├── SupplierInnovationAgent           (Mystery 08)
│   ├── DecisionCopilotAgent              (Mystery 10)
│   └── TradeScenarioAgent                (Mystery 12)
├── ToolAgent                ERP connectors · market data APIs · document parsers
└── DecisionAgent            StructuralAgent + formal authority + escalation hooks
    ├── InstitutionalMemoryAgent           (Mystery 01)
    ├── WorkingCapitalOptimiserAgent       (Mystery 05)
    └── DecisionAuditAgent                (Mystery 13)
```

Every `StructuralAgent` follows a strict three-step pipeline. `observe()` is a pure function — no LLM, no side effects, fully unit-testable. `reason()` calls the pluggable `ReasoningEngine`. `parse_findings()` is deterministic JSON parsing to typed `Finding` objects. Each step is independently testable and auditable.

---

## The Fifteen Structural Mystery Agents

| # | Agent | Domain | Authority |
|---|-------|--------|-----------|
| 01 | InstitutionalMemoryAgent | Knowledge capture | Advise |
| 02 | SupplierStressAgent | Risk · stress signals | Recommend |
| 03 | NegotiationIntelligenceAgent | Counterparty intelligence | Advise |
| 04 | SpecificationInflationAgent | Specification analysis | Recommend |
| 05 | WorkingCapitalOptimiserAgent | Treasury integration | Recommend |
| 06 | ShouldCostAgent | Cost intelligence | Recommend |
| 07 | DemandIntelligenceAgent | Pre-signal demand | Advise |
| 08 | SupplierInnovationAgent | Innovation capture | Advise |
| 09 | ComplianceVerificationAgent | Regulatory exposure | Escalate |
| 10 | DecisionCopilotAgent | Cognitive load | Advise |
| 11 | ValueRealisationAgent | Savings delivery | Recommend |
| 12 | TradeScenarioAgent | Trade policy | Recommend |
| 13 | DecisionAuditAgent | Accountability · AI Act | Escalate |
| 14 | TotalCostIntelligenceAgent | Lifecycle cost | Recommend |
| 15 | ProcurementBiasDetectorAgent | Evaluation bias | Escalate |

**Authority levels:** *Advise* — provides analysis, human decides. *Recommend* — proposes specific action, human approves. *Escalate* — findings require senior leadership attention.

---

## Quick Start

### Install dependencies

```bash
git clone https://github.com/mohammedakbaransari/skein
cd skein
pip install pyyaml requests       # minimal — for Ollama local inference
```

### Run all 100 tests (no LLM required)

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Run an agent with Ollama

```bash
ollama pull llama3.1

python -m agents.supply_risk.supplier_stress \
    --data data/synthetic/supplier_transactions.json

# --dry-run skips the LLM call entirely
python -m agents.supply_risk.supplier_stress --dry-run
```

### Switch LLM provider

Edit `config/config.yaml` or use environment variables:

```yaml
# config/config.yaml
llm:
  provider: anthropic
  model: claude-opus-4-5
  api_key: your_key_here
```

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o
export LLM_API_KEY=sk-...
```

No agent code changes required.

---

## Multi-Agent Workflow

```python
from framework.orchestration.orchestrator import WorkflowBuilder, TaskOrchestrator
from framework.core.types import SessionId

workflow = (
    WorkflowBuilder("quarterly-supplier-review")
    .session(SessionId.generate())
    .step("SupplierStressAgent",     payload={"transaction_data": erp_export})
    .parallel(
        ("ShouldCostAgent",          {"commodity_prices": price_data}),
        ("ValueRealisationAgent",    {"savings_tracking": contract_data}),
        ("ComplianceVerificationAgent", {"compliance_records": cert_data}),
    )
    .then("DecisionAuditAgent",      payload={"decision_logs": decision_data})
    .then("ProcurementBiasDetectorAgent", payload={"sourcing_evaluations": eval_data})
    .build()
)

result = orchestrator.run_workflow(workflow)
print(f"Agents run: {len(result.task_results)}")
print(f"Total findings: {len(result.all_findings)}")
```

---

## Adding a New Agent — 3 Steps

**Step 1: Define metadata**

```python
from framework.core.types import AgentCapability, AgentMetadata, DecisionAuthority

METADATA = AgentMetadata(
    agent_type="MyDomainAgent",
    display_name="My Structural Intelligence Agent",
    description="What structural problem this agent addresses.",
    version="1.0.0",
    capabilities=(AgentCapability(
        name="my_analysis",
        description="What this capability analyses",
        authority=DecisionAuthority.RECOMMEND,
    ),),
    tags=("domain", "tag"),
)
```

**Step 2: Implement the agent**

```python
from framework.agents.base import StructuralAgent
from framework.core.types import Finding, Severity, Task

class MyDomainAgent(StructuralAgent):
    METADATA = METADATA

    def observe(self, task: Task) -> dict:
        # Load and structure data — no LLM, no side effects
        return {"processed": task.payload.get("my_data")}

    def reason(self, observations: dict, task: Task) -> str:
        from framework.reasoning.engine import ReasoningRequest, ReasoningStrategy
        resp = self.reasoning.reason(ReasoningRequest(
            system_prompt="You are an analyst...",
            user_prompt=f"Analyse: {observations}",
            observations=observations,
            strategy=ReasoningStrategy.STRUCTURED,
            session_id=str(task.session_id),
        ))
        return resp.content

    def parse_findings(self, observations: dict, reasoning: str, task: Task) -> list:
        parsed = self._parse_llm_json(reasoning) or {}
        return [self._make_finding(
            finding_type="my_finding",
            severity=Severity.MEDIUM,
            summary=parsed.get("key_finding", ""),
        )]
```

**Step 3: Register**

```python
from framework.core.registry import get_registry
get_registry().register_class(MyDomainAgent)
```

The orchestrator routes to it automatically. The governance logger records all executions.

---

## Extending to Other Domains

SKEIN is designed to be domain-agnostic. The framework core contains no procurement-specific logic. The agents in the `agents/` directory are a procurement reference implementation.

To apply SKEIN to a different domain:

1. Create a new domain directory alongside `agents/procurement/` — for example `agents/healthcare/` or `agents/legal/`
2. Define `AgentMetadata` for your domain's structural mysteries
3. Subclass `StructuralAgent` with domain-specific `observe()` and `parse_findings()` logic
4. The entire framework — memory, governance, orchestration, reasoning — works unchanged

The structural pattern is universal: every complex enterprise function has invisible threads that determine its performance, resist current-generation AI platforms, and require a different class of intelligence to address.

---

## Test Coverage

| Suite | Tests | Scope |
|-------|-------|-------|
| `tests/unit/` | 55 | Pure domain logic, signal extraction, edge cases |
| `tests/integration/` | 29 | Registry, orchestrator, memory, governance chain |
| `tests/system/` | 16 | Multi-agent workflows, concurrency, scalability |
| **Total** | **100** | Zero external dependencies required |

---

## Research Background

This framework is the implementation companion to:

**"The 15 Structural Mysteries of Procurement AI: A Research Agenda for the Next Generation of Procurement Intelligence Systems"**
Mohammed Akbar Ansari, March 2026

Available on SSRN and arXiv. The paper argues that the 45-point gap between AI pilot rates (49%) and production deployment rates (4%) in procurement is not a technology failure — it is a failure to identify which problems are worth solving. SKEIN is the architectural response to the fifteen structural mysteries the paper identifies.

**SSRN:** [papers.ssrn.com](https://papers.ssrn.com) *(link to be added post-submission)*
**arXiv:** [arxiv.org](https://arxiv.org) *(link to be added post-submission)*

---

## Name and Namespace

**SKEIN — Structural Knowledge and Enterprise Intelligence Network**

The name is unique across GitHub, PyPI, academic databases, and commercial AI platforms (verified March 2026). When citing or referencing this framework, use the full title to ensure unambiguous identification.

- **GitHub repository:** `github.com/mohammedakbaransari/skein`
- **PyPI package:** `pip install skein-framework` *(on first release)*
- **Citation key:** `ansari2026skein`
- **Full title:** SKEIN — Structural Knowledge and Enterprise Intelligence Network

---

*Independent personal research. No vendor relationship. No platform to sell.*
