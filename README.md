# SKEIN — Structural Knowledge and Enterprise Intelligence Network

> **Independent Research · Mohammed Akbar Ansari · Navi Mumbai, India**
> *This is a personal open-source research project. No employer, vendor, or customer affiliation.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-135%20passing-brightgreen.svg)]()

SKEIN is a multi-agent framework for structural procurement intelligence. It addresses the 15 structural gaps in current-generation AI procurement tools identified in the companion research paper: *"The 15 Structural Mysteries of Procurement AI"* (Ansari, 2026).

---

## What SKEIN Does

Current AI procurement tools excel at answering questions inside structured data. SKEIN addresses a different problem: the structural knowledge gaps that sit *outside* those systems — in behavioural signals, tacit expertise, fragmented context, and untracked decision patterns.

Each of the 15 SKEIN agents targets one structural mystery:

| Agent | Mystery | What it detects |
|-------|---------|-----------------|
| `SupplierStressAgent` | M02 | Early supplier distress signals — 6–9 months before failure |
| `ShouldCostAgent` | M06 | Commodity price leverage — where supplier pricing lacks cost basis |
| `TotalCostIntelligenceAgent` | M07 | TCO gaps — assets procured on purchase price alone |
| `DecisionAuditAgent` | M13 | Decision accountability gaps — AI recommendations with no rationale |
| `ProcurementBiasDetectorAgent` | M09 | Incumbent advantage bias in sourcing evaluations |
| `ComplianceVerificationAgent` | M14 | Supplier certification gaps and CSDDD exposure |
| `ValueRealisationAgent` | M08 | Savings leakage — negotiated vs actually captured |
| `InstitutionalMemoryAgent` | M01 | Expert reasoning patterns before they walk out the door |
| `NegotiationIntelligenceAgent` | M03 | Counterparty insights and negotiation positioning |
| `SpecificationInflationAgent` | M04 | Over-specified requirements that limit supplier competition |
| `WorkingCapitalOptimiserAgent` | M05 | Payment term optimisation opportunities |
| `DemandIntelligenceAgent` | M10 | Demand consolidation and aggregation opportunities |
| `SupplierInnovationAgent` | M12 | Untapped supplier innovation pipeline |
| `DecisionCopilotAgent` | M11 | Real-time decision support during sourcing events |
| `TradeScenarioAgent` | M15 | Tariff and trade disruption scenario modelling |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   SKEIN Framework                        │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Resilience  │  │Observability │  │   Platform   │  │
│  │ RetryExecutor│  │StructuredLog │  │  Databricks  │  │
│  │CircuitBreaker│  │  Prometheus  │  │  MS Fabric   │  │
│  │  AgentPool   │  │HealthServer  │  │  Kubernetes  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │          │
│  ┌──────▼─────────────────▼──────────────────▼───────┐  │
│  │              TaskOrchestrator (DAG)                │  │
│  │   Kahn topological sort · ThreadPoolExecutor       │  │
│  │   Per-task retry · Correlation context prop.       │  │
│  └──────────────────────┬─────────────────────────────┘  │
│                         │                                │
│  ┌──────────────────────▼─────────────────────────────┐  │
│  │               Agent Layer (15 agents)              │  │
│  │   BaseAgent → StructuralAgent → DecisionAgent      │  │
│  │   observe() · reason() · parse_findings()          │  │
│  └──────┬────────────────────────┬────────────────────┘  │
│         │                        │                        │
│  ┌──────▼──────┐        ┌────────▼───────┐              │
│  │  Reasoning  │        │    Memory      │              │
│  │  Engine     │        │ Working (LRU)  │              │
│  │  +retry+cb  │        │ Institutional  │              │
│  └─────────────┘        └────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/mohammedakbaransari/skein
cd skein
pip install pyyaml requests   # core only

# Optional: your LLM provider
pip install anthropic         # for Claude
# pip install openai          # for GPT-4o
# (Ollama needs no package)
```

### 2. Configure

```bash
cp config/config.yaml config/local.yaml
# Edit config/local.yaml — set provider, model, api_key
```

Or use environment variables:
```bash
export LLM_PROVIDER=anthropic
export LLM_API_KEY=sk-ant-...
export LLM_MODEL=claude-sonnet-4-6
```

### 3. Run a single agent (DryRun — no LLM needed)

```python
from agents.supply_risk.supplier_stress import SupplierStressAgent
from framework.core.types import Task
from framework.reasoning.stubs import DryRunReasoningEngine

agent  = SupplierStressAgent(reasoning_engine=DryRunReasoningEngine())
task   = Task.create("SupplierStressAgent", {"transaction_data": your_data})
result = agent.run(task)

for finding in result.findings:
    print(f"[{finding.severity.value.upper()}] {finding.summary}")
```

### 4. Run a multi-agent workflow

```python
from framework.core.registry import get_registry
from framework.orchestration.orchestrator import TaskOrchestrator, WorkflowBuilder
from framework.core.types import SessionId

registry = get_registry()
registry.register_class(SupplierStressAgent)
registry.register_class(DecisionAuditAgent)

orch = TaskOrchestrator(registry, config=None)
sid  = SessionId.generate()

workflow = (
    WorkflowBuilder("quarterly-review")
    .session(sid)
    .step("SupplierStressAgent", {"transaction_data": transactions})
    .then("DecisionAuditAgent",  {"decision_logs":    decisions})
    .build()
)

result = orch.run_workflow(workflow)
print(f"Succeeded: {result.succeeded}")
print(f"Findings:  {len(result.all_findings)}")
```

### 5. Run the full production server

```bash
python3 -m scripts.server --config config/config.yaml

# Health endpoints
curl http://localhost:8080/health
curl http://localhost:8080/ready
curl http://localhost:8080/metrics
```

---

## Running Tests

```bash
# All 135 tests (no LLM required — all use DryRunReasoningEngine)
python3 -m unittest discover -s tests -p "test_*.py" -v

# Individual suites
python3 -m unittest tests.unit.test_retry_circuit -v      # Retry + circuit breaker
python3 -m unittest tests.unit.test_memory -v             # Memory + session isolation
python3 -m unittest tests.unit.test_supplier_stress -v    # Supplier stress agent
python3 -m unittest tests.unit.test_agents_unit -v        # All 8 major agents
python3 -m unittest tests.integration.test_framework_integration -v
python3 -m unittest tests.system.test_multi_agent_system -v
python3 -m unittest tests.scenarios.test_procurement_scenarios -v

# Load/stress tests (slower — run separately)
python3 -m unittest tests.load.test_stress_load -v
```

---

## Platform Deployment

### Kubernetes
```bash
kubectl apply -f deploy/kubernetes/deployment.yaml
kubectl rollout status deployment/skein-agents -n skein
```

### Docker
```bash
docker build -f deploy/docker/Dockerfile -t skein-framework:latest .
docker-compose -f deploy/docker/docker-compose.yml up
```

### Databricks
```python
from platform.databricks.adapter import SkeinDatabricksApp
app    = SkeinDatabricksApp.from_notebook_context()
result = app.run_supplier_risk_review(transaction_data)
```

### Microsoft Fabric
```python
from platform.fabric.adapter import SkeinFabricApp
app    = SkeinFabricApp.from_fabric_context()
result = app.run_supplier_risk_review(transaction_data)
```

---

## LLM Provider Configuration

SKEIN works with any LLM provider. Switch by changing `config/config.yaml` or environment variables — no code changes needed.

| Provider | Config | Package |
|----------|--------|---------|
| Ollama (local, free) | `provider: ollama` | none |
| Anthropic Claude | `provider: anthropic` | `pip install anthropic` |
| OpenAI GPT-4o | `provider: openai` | `pip install openai` |
| Azure OpenAI | `provider: azure` | `pip install openai` |

Optional reasoning frameworks (drop-in, no agent changes):
- **LangChain LCEL** — `reasoning.enable_langchain: true` + `pip install langchain`
- **LangGraph** — `reasoning.enable_langgraph: true` + `pip install langgraph`
- **CrewAI** — `reasoning.enable_crewai: true` + `pip install crewai`

---

## Project Structure

```
skein/
├── framework/                  Core framework
│   ├── core/                   Types, registry
│   ├── agents/                 BaseAgent, StructuralAgent, DecisionAgent
│   ├── orchestration/          TaskOrchestrator, WorkflowBuilder (DAG)
│   ├── reasoning/              ReasoningEngine + strategy plugins
│   ├── memory/                 WorkingMemory (LRU), InstitutionalMemory
│   ├── governance/             Hash-chained audit logger
│   ├── resilience/             RetryExecutor, CircuitBreaker, AgentPool
│   └── observability/          Structured logging, Prometheus, health endpoints
│
├── agents/                     15 domain agents (one per structural mystery)
│   ├── supply_risk/            M02 — Supplier Stress Signal
│   ├── cost_intelligence/      M06 Should-Cost, M07 Total Cost
│   ├── decision_audit/         M13 — Decision Accountability
│   ├── bias_detection/         M09 — Sourcing Bias
│   ├── contract_analysis/      M08 — Value Realisation
│   ├── compliance/             M14 — Compliance Verification
│   └── market_intelligence/    M01, M03, M04, M05, M10, M11, M12, M15
│
├── tests/
│   ├── unit/                   Per-component unit tests
│   ├── integration/            Cross-layer integration tests
│   ├── system/                 Multi-agent, concurrency, pool tests
│   ├── scenarios/              End-to-end procurement use-case tests
│   └── load/                   Stress, throughput, deadlock detection
│
├── platform/
│   ├── databricks/             Delta Lake memory, MLflow governance
│   └── fabric/                 OneLake memory, Fabric Lakehouse governance
│
├── deploy/
│   ├── kubernetes/             Deployment, HPA, PDB, Service, ConfigMap
│   ├── docker/                 Multi-stage Dockerfile, docker-compose
│   └── helm/                   Helm chart values
│
├── config/config.yaml          Configuration (env-var override supported)
├── scripts/server.py           Production server entry point
├── requirements.txt            Dependencies
└── data/synthetic/             Synthetic test datasets
```

---

## Research Context

This framework accompanies:

**"The 15 Structural Mysteries of Procurement AI: A Research Agenda for the Next Generation of Procurement Intelligence Systems"**
Mohammed Akbar Ansari, Independent Researcher, Navi Mumbai, India (2026)

- SSRN: [link once live]
- arXiv: [link once endorsed]
- GitHub: https://github.com/mohammedakbaransari/skein

**Disclaimer:** This is independent research. No employer, vendor, or customer affiliation. All views are my own.

---

## Licence

MIT — see [LICENSE](LICENSE)
