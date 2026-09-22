# SKEIN — Second-Pass Architecture and Technical Assessment

> **Archived evidence snapshot:** This assessment records the repository state and remediation evidence at the time of the second pass (ending at 312 passing tests). It is intentionally not rewritten as features evolve. For the verified current architecture, security posture, usage, deployment boundaries, and implementation status, use [ARCHITECTURE.md](ARCHITECTURE.md), [SECURITY.md](SECURITY.md), [USAGE.md](USAGE.md), [DEPLOYMENT.md](DEPLOYMENT.md), and [ENTERPRISE-EMBEDDING-ROADMAP.md](ENTERPRISE-EMBEDDING-ROADMAP.md). Current baseline: 417 passing tests on 2026-09-16.

**Repository:** `mohammedakbaransari/skein` (local workspace: `c:\Akbar\Research Work\skein`)
**Assessment date:** 2026-09-15
**Assessor role:** Independent second-pass audit — originally validated and challenged a first-pass report that was later removed after consolidation into this evidence record and the living documentation.
**Constraint compliance:** No production code, configuration, or tests were modified. All executed verification used throwaway scripts outside the repository (OS temp directory) or read-only commands.
**Remediation status update (2026-09-15, same day):** Roadmap items **P0-1**, **P0-2**, and **P1-4** (governance hash-chain concurrency race, restart discontinuity, and the false-confidence test that masked both) have since been **implemented and empirically verified** — see §28. All other findings in this document reflect the pre-fix state and remain otherwise unchanged.

---

## 1. Assessment Scope

This is a full, independent re-inspection of the SKEIN repository, treating the earlier report as a hypothesis set to be checked against source code — not as ground truth. Every major conclusion in the earlier report was re-traced to its source location; several were **strengthened with executed evidence** (not just static reading) in this pass, one is **corrected/downgraded**, and one important architectural gap it did not surface is added here (a genuine intra-process race condition in the governance hash chain, distinct from the restart issue the earlier report identified).

## 2. Methodology

1. Re-read the earlier report end-to-end and mapped every material conclusion to a source file/line.
2. Re-inspected all `framework/` subsystems, all `agents/` implementations, `platform/` adapters, `deploy/` artifacts, `config/`, `scripts/server.py`, and all five `tests/` subpackages.
3. Executed the real test suite (`python -m unittest discover -s tests -p "test_*.py" -v`) directly in this session's terminal (see command in session context) — **169 tests run, 169 passed, 0 failed, 0 errored**, confirmed independently of the earlier pass.
4. Wrote and ran two throwaway verification scripts (outside the repository, in the OS temp directory, deleted after use) to **empirically prove or disprove** the governance hash-chain behavior under (a) concurrent writers on a single `GovernanceLogger` instance and (b) a simulated process restart. Both were executed via the terminal in this session — results reported verbatim in §11.
5. Cross-checked claims against `grep`/file-search results for terms implied by documentation (`pii_redaction`, `rate_limit`, `sanitize`, etc.) to confirm absence of enforcement code.
6. Did not attempt: live LLM provider calls, Docker build/run, `kubectl`/`helm` execution, live Databricks/Fabric connectivity. These remain **Not verified** in this pass as well, for the same environment-availability reasons as the earlier report.

---

## 3. Repository Inventory

| Area | Path(s) | Status |
|---|---|---|
| Core types/vocabulary | `framework/core/types.py`, `framework/core/registry.py` | Implemented, exercised by tests |
| Agent base classes | `framework/agents/base.py` (`BaseAgent`, `StructuralAgent`, `ToolAgent`, `DecisionAgent`) | Implemented, exercised by tests |
| Agent metadata catalogue | `framework/agents/catalogue.py` | Implemented; contains stale doc reference to an agent that was never built (`ERPSignalConnectorAgent`) |
| Domain agents (15) | `agents/supply_risk/supplier_stress.py`, `agents/decision_audit/agent.py`, `agents/cost_intelligence/{should_cost,total_cost}.py`, `agents/contract_analysis/value_realisation.py`, `agents/bias_detection/bias_detector.py`, `agents/compliance/compliance_verification.py`, `agents/market_intelligence/agents.py` (8 classes) | All 15 implemented as real classes; verified distinct deterministic math in each `observe()`/analysis function re-read in this pass (see §7) |
| Orchestration | `framework/orchestration/orchestrator.py` (`Workflow`, `TaskOrchestrator`, `WorkflowBuilder`) | Implemented; one unhandled-timeout defect confirmed in both passes |
| Reasoning/LLM | `framework/reasoning/engine.py`, `framework/reasoning/stubs.py` | Implemented for 4 native providers + 3 optional framework integrations (LangChain/LangGraph/CrewAI); **zero tests exercise any non-DryRun path** |
| Resilience | `framework/resilience/retry.py`, `framework/resilience/pool.py` | Implemented, correct by inspection and covered by `tests/unit/test_retry_circuit.py` |
| Memory | `framework/memory/store.py` (`WorkingMemory`, `ContextMemory`, `InstitutionalMemory`) | Implemented; `InstitutionalMemory._persist()` (re-read in full this pass, lines ~340–347) does atomic tmp-file+rename writes — genuinely durable for the single-writer case |
| Governance | `framework/governance/logger.py` (`HashChainedWriter`, `GovernanceLogger`) | Implemented; **two distinct, empirically-confirmed chain-integrity defects** (§11) |
| Observability | `framework/observability/{logging,metrics,health}.py` | Implemented; `metrics.py` has a genuine `prometheus_client`-or-no-op design (re-read in full this pass) |
| Tools/plugins | `framework/tools/base.py` (`BaseTool`, `ToolRegistry`) | Base interface only — **zero concrete `BaseTool` subclasses exist anywhere in the repo** |
| Integrations placeholder | `framework/integrations/__init__.py` | **Empty file** — an entire package with no implementation |
| Config | `config/config.yaml`, `config/__init__.py` (empty) | YAML present; no config-loading/validation code lives in `config/` itself (loading happens ad hoc in `scripts/server.py::load_config`) |
| Platform adapters | `platform/databricks/adapter.py` (+ `README.md`), `platform/fabric/adapter.py` (+ `README.md`) | Fallback-first scaffolding; Databricks adapter has an unparameterized-SQL injection pattern (confirmed again this pass, §9) |
| Deployment — Docker | `deploy/docker/Dockerfile`, `docker-compose.yml` | Well-formed multi-stage Dockerfile with a real import-validation build step; not built in this assessment |
| Deployment — Kubernetes | `deploy/kubernetes/deployment.yaml` | Well-formed manifest text; not applied to a live cluster in this assessment |
| Deployment — Helm | `deploy/helm/values.yaml` **only** | **Not a working Helm chart** — no `Chart.yaml`, no `templates/` directory exist anywhere under `deploy/helm/`. `helm install skein ./deploy/helm` would fail immediately with "Chart.yaml file is missing". This is a new finding not raised with this precision in the earlier report. |
| CI/CD | `.github/workflows/python-package-conda.yml` | Broken by omission — references a nonexistent `environment.yml`; confirmed absent again via file search this pass |
| Tests | `tests/unit/*.py` (5 files), `tests/integration/test_framework_integration.py`, `tests/system/test_multi_agent_system.py`, `tests/scenarios/test_procurement_scenarios.py`, `tests/load/test_stress_load.py`, `tests/conftest.py` | 169 tests, all pass, all use `DryRunReasoningEngine`; re-read in full this pass (see §16) |
| Examples / notebooks | — | **None found anywhere in the repository.** No `examples/` directory, no `.ipynb` files. |
| Documentation | `README.md`, `CONTRIBUTING.md`, `platform/*/README.md` | `CONTRIBUTING.md` and `Makefile` both still say "135 tests" (stale, confirmed again); `platform/databricks/README.md` references a `skein-framework-v2-FINAL.zip` deployment artifact that does not exist in the repository — a strong signal that this doc was drafted aspirationally/copy-pasted, not validated against a real Databricks workspace |
| Synthetic data | `data/synthetic/*.json`, `generate_all.py` | Seeded, reproducible, clearly-synthetic generator (`SEED = 42`) — confirmed not derived from any real dataset |

---

## 4. Executive Summary

SKEIN is, in this second pass as in the first, best described as **a well-structured, internally-consistent, single-process multi-agent research framework with genuinely distinct deterministic domain logic per agent, wrapped in solid resilience/observability engineering, but marketed and configured well above its actual production maturity.**

The earlier report's core conclusion — *retain the architecture, fix specific defects, do not rewrite* — **is upheld** in this pass. However, this pass corrects one specific technical claim in the earlier report and adds one materially important defect the earlier report did not identify with equivalent precision:

- **Correction:** The earlier report characterized the governance hash-chain problem primarily as a *restart continuity* issue. This pass **empirically confirms that is real**, but also demonstrates — by actually running the code, not just reading it — that **the chain is also broken by ordinary in-process concurrent writes on a single `GovernanceLogger` instance**, which is the normal operating mode of the orchestrator's `ThreadPoolExecutor`-based parallel task execution. This is a more severe and more commonly triggered defect than "restart only," because it means **the hash chain is not reliably valid even during a single, uninterrupted, single-instance run of the server under any concurrent workflow** (§11).
- **Addition:** `deploy/helm/` contains only a `values.yaml` file with no `Chart.yaml` and no `templates/` — it is not a Helm chart at all, just a values file. The earlier report treated the Helm artifact as "structurally sound but unexercised"; this pass finds it is **not even structurally complete** as a chart.
- **Upheld, with independent re-derivation:** the CI-workflow break, the Databricks SQL-injection pattern, the unimplemented `security:` config block, the orchestrator timeout defect, and the "no real ERP/CRM extraction layer" strategic gap are all reconfirmed by direct re-reading of the same files, independent of the earlier report's text.
- **Upheld, with more nuance:** the earlier report suggested the 15 agents are "mostly variations of a common template." Re-reading `bias_detector.py`, `should_cost.py`, and `value_realisation.py` in full in this pass shows each has **genuinely distinct, hand-written statistical logic** (subjective/objective score deltas per supplier type, commodity price leverage thresholds, savings-leakage trend classification) — the *pipeline shape* (`observe→reason→parse_findings`) is shared and templated, but the *domain math inside `observe()`* is not copy-pasted boilerplate. This pass **downgrades** that specific criticism from the earlier report: it is accurate about the shared pipeline shape, but was too strong in implying the domain logic itself is generic.

**Overall maturity level (this pass's independent judgment): early-stage framework / advanced prototype.** Not a functioning enterprise procurement intelligence *product*; is a genuine, testable multi-agent reasoning and orchestration *foundation*.

---

## 5. Earlier Report Validation

Systematic re-validation of the earlier report's major conclusions.

| # | Earlier report conclusion | Re-checked against | Supporting evidence found | Contradicting evidence found | Confidence | Disposition |
|---|---|---|---|---|---|---|
| 1 | All 15 agents are genuinely implemented, not stubs | `agents/**` grep for `class \w+Agent`, full read of 3 of the 15 `observe()`/analysis functions | 19 classes matched; `bias_detector.py`, `should_cost.py`, `value_realisation.py` each contain distinct, non-trivial statistics (subjective/objective deltas, commodity leverage thresholds, leakage-trend classification) | None | **High** | **Accepted** |
| 2 | 169 tests pass (not the stale README "135") | Directly re-ran `python -m unittest discover -s tests -p "test_*.py" -v` in this session | Terminal output: 169 tests, all passed | None | **High** | **Accepted** |
| 3 | Governance hash chain "does not survive process restart" | Read `HashChainedWriter.__init__`/`.write()`; **executed** a restart-simulation script | Empirically confirmed: `verify_chain()` returns `False` after a second `GovernanceLogger` instance writes to the same directory | None | **High (upgraded from the earlier report's static-reading-only confidence)** | **Accepted, and strengthened with execution evidence** |
| 4 | (Not raised in earlier report with this precision) Governance hash chain is also broken by ordinary **concurrent writes within a single `GovernanceLogger` instance, single process** | Read `HashChainedWriter.write()` — `entry = {**record, "prev_hash": self._prev_hash}` executes **outside** `self._lock`, and `self._prev_hash = entry_hash` is updated **after** the lock is released; **executed** an 8-thread concurrent-write script | Empirically confirmed: 160/160 lines written, zero exceptions, but `verify_chain()` returns `False` | The existing test `tests/integration/test_framework_integration.py::test_concurrent_governance_writes` passes — but it never calls `verify_chain()`, only checks line count and absence of exceptions, so it does not actually test what its name implies | **High** | **New finding this pass — see §11 for full detail** |
| 5 | `security:` config block (`enable_pii_redaction`, `enable_input_sanitisation`, `rate_limit_requests_per_minute`, etc.) is entirely unenforced | Repo-wide regex search for `pii_redaction\|input_sanitisation\|rate_limit\|sanitize\|redact` | Zero implementation matches, confirmed again this pass | None | **High** | **Accepted** |
| 6 | CI workflow is broken (`environment.yml` missing) | Read `.github/workflows/python-package-conda.yml`; file-searched for `environment.yml` | File absent from the entire repository, confirmed again | None | **High** | **Accepted** |
| 7 | Databricks adapter has a SQL-injection pattern via f-string interpolation | Re-read `platform/databricks/adapter.py::DeltaTableMemoryStore.set/get/delete` | `MERGE INTO ... VALUES ('{key}', '{value_json}', ...)` and `SELECT ... WHERE key = '{key}'` confirmed as raw f-string interpolation, no parameterization anywhere in the file | None | **High** | **Accepted, and reinforced** — `platform/databricks/README.md` gives no indication this was ever tested against a live workspace, and references a non-repository artifact name (`skein-framework-v2-FINAL.zip`), suggesting the whole adapter was written speculatively |
| 8 | Orchestrator's workflow timeout raises an unhandled exception instead of graceful cancellation | Re-read `TaskOrchestrator.run_workflow()` — `as_completed(futures, timeout=workflow.timeout_seconds)` has no surrounding `try/except` | Confirmed by direct reading; no test exercises this path (searched `tests/**` for `TimeoutError`/`timeout_seconds` usage in assertions — none found asserting on this specific failure mode) | None | **High** | **Accepted** |
| 9 | 15-agent design is "mostly variations of a common template" | Full re-read of 3 additional agent implementations beyond what the earlier pass sampled | The `observe→reason→parse_findings` **pipeline shape** is indeed shared/templated across all 15 | The **domain math inside `observe()`** is genuinely distinct per agent (see row 1) — not copy-pasted | **Medium** | **Modified** — accurate about shared pipeline, overstated about domain-logic genericness |
| 10 | Databricks/Fabric platform adapters are "scaffolding, not production connectors" | Re-read both adapters in full, plus `platform/databricks/README.md` | No tests exist anywhere for either adapter; both fall back silently to in-memory caches when SDKs are absent; README references a nonexistent packaging artifact | None | **High** | **Accepted** |
| 11 | Institutional memory "storage abstraction, not real institutional-memory intelligence" | Re-read `InstitutionalMemory` in full including the previously-unread persistence tail (`_persist()`, atomic tmp+rename) | Persistence is genuinely durable and atomic for the single-writer case (a correction of detail: the earlier report flagged this as "UNKNOWN pending deeper review" — it is now confirmed sound for the single-writer case) | No agent was found calling `self.recall()` to read back institutional memory in the files sampled — retrieval-side usage remains unconfirmed | **Medium** | **Accepted with detail added** — write path is solid; read/retrieval-side usage by agents is still not demonstrated anywhere in the codebase read |
| 12 | `DecisionAgent._record_decision()` — unclear in earlier report whether it actually forwards to `governance.record_decision()` | Re-read the full method body (previously truncated in the earlier pass) | Confirmed: it does call `self.governance.record_decision(...)` when governance is configured | None | **High** | **Resolves the earlier report's own "UNKNOWN" flag — now Accepted as implemented** |
| 13 | Helm chart is "structurally sound but unexercised" | `list_dir` on `deploy/helm/` | Only `values.yaml` exists; no `Chart.yaml`, no `templates/` | The earlier report's characterization implied a complete chart existed | **High** | **Corrected/downgraded** — it is not a functioning chart at all in its current form, a more severe finding than "unexercised" |
| 14 | Test suite is "meaningful, not just mocks" | Re-read `tests/unit/test_memory.py`, `tests/integration/test_framework_integration.py`, `tests/scenarios/test_procurement_scenarios.py` in full | Session isolation, TTL expiry, LRU eviction, DAG dependency/cancellation, and multi-agent scenario tests all assert genuine domain-relevant behavior (e.g., a critical-stress supplier actually escalates) | The one governance concurrency test (`test_concurrent_governance_writes`) is a **false-confidence test** — its name implies chain-safety verification but it never calls `verify_chain()` | **High** | **Accepted, with one specific weak-test finding added** (see row 4 and §16) |
| 15 | No real ERP/CRM/behavioral-data extraction layer exists; this is the core unaddressed strategic gap | Re-read `framework/agents/base.py::ToolAgent`, `framework/tools/base.py`, grepped for any concrete `BaseTool` subclass | Zero concrete `ToolAgent`/`BaseTool` subclasses found anywhere in the repository; `framework/integrations/__init__.py` is empty | None | **High** | **Accepted, and reinforced** — this is not merely under-built, it is **entirely unstarted** (an empty package) |

**Overall assessment of the earlier report:** Directionally accurate and well-evidenced on nearly every major point. Its two weakest spots were (a) understating the severity/scope of the governance hash-chain defect by focusing only on the restart scenario, and (b) slightly overstating agent-to-agent genericness. Both are corrected in this pass. No conclusion in the earlier report was found to be fabricated or unsupported by the codebase.

---

## 6. Actual Architecture

Re-confirmed unchanged from the earlier report's architecture reconstruction — independent re-reading of `framework/orchestration/orchestrator.py`, `framework/agents/base.py`, and `scripts/server.py` in this pass produced the same component/runtime picture. Rather than duplicate the diagrams, this section records what is **newly confirmed or corrected**:

- **No task-submission API exists on the running server** — re-confirmed by re-reading all of `scripts/server.py`: it wires registry, reasoning engine, governance, pool, and orchestrator, starts the health server, and calls `mark_ready()` — but nothing in the file ever calls `orch.run_workflow()` or exposes any endpoint that would. This is confirmed, not merely inferred. **UPDATE (§34): a single-task submission API (`POST /v1/tasks`) has since been implemented and wired into `scripts/server.py`, requiring an explicit `tenant_id` on every request** — workflow submission (`orch.run_workflow()`) is still not exposed over HTTP.
- **`framework/tools/base.py` (`ToolAgent` infrastructure) is entirely unused** — no concrete tool or `ToolAgent` subclass exists anywhere in `agents/`. The `ToolAgent` abstract base class and `ToolRegistry` are dead code from the perspective of the actual 15 registered agents (all 15 are `StructuralAgent`/`DecisionAgent` subclasses, never `ToolAgent`).
- **`framework/integrations/__init__.py` is a fully empty file** — an entire top-level package that exists in name only, confirmed by direct read (`view` result: "exists, but is empty").
- **Config loading has no dedicated module** — `config/__init__.py` is empty; all YAML-loading/env-override logic lives inline inside `scripts/server.py::load_config()`. There is no reusable, testable configuration-validation component anywhere (e.g., no Pydantic/dataclass schema for `config.yaml`), which explains why the unimplemented `security:` block was never caught by any validation layer — there is no validation layer to catch it.

---

## 7. Agent-by-Agent Assessment

All 15 agents were confirmed registered in `scripts/server.py::register_all_agents()` and exercised (in dry-run mode) by `tests/unit/test_agents_unit.py`. This pass re-read three representative agents in full (beyond the one the earlier pass fully read) to test the "template duplication" hypothesis specifically.

| Agent | Source | Deterministic logic in `observe()`/helpers | LLM-dependent portion | Distinctiveness vs. other agents | Test coverage | Classification |
|---|---|---|---|---|---|---|
| `SupplierStressAgent` | `agents/supply_risk/supplier_stress.py` | 6-signal composite scoring (PO ack days, OTD%, quality holds, invoice disputes, unsolicited discounts, sales response time) against warn/alert percentage thresholds, first-signal-month detection | Narrative synthesis + recommended actions | High — bespoke signal-scoring formulas, thresholds specific to supplier behavior | `tests/unit/test_supplier_stress.py` + scenario test (critical-stress escalation path) | **Fully implemented** |
| `ProcurementBiasDetectorAgent` | `agents/bias_detection/bias_detector.py` | Per-supplier-type award-rate/objective/subjective-score deltas; per-evaluator incumbent-vs-non-incumbent premium differential; diverse/SME suppression flags with explicit numeric thresholds (`>=65` objective, `<30%` award rate) | Narrative synthesis | High — genuinely distinct bias-statistics formulas not present in any other agent | Covered by `test_agents_unit.py` generic pass; no dedicated bias-specific unit test file found (only `SupplierStressAgent` and the general `test_analysis_agents.py`/`test_agents_unit.py` cover it) | **Fully implemented (logic); test coverage is generic, not bias-specific** |
| `ShouldCostAgent` | `agents/cost_intelligence/should_cost.py` | Commodity-price movement % change vs. threshold bands (`<-10%`=High leverage, `<-5%`=Medium, etc.), rising-cost warning detection | Narrative synthesis + negotiation leverage recommendations | High — commodity-leverage banding logic is unique to this agent | Generic `test_agents_unit.py` coverage only | **Fully implemented (logic); test coverage is generic** |
| `ValueRealisationAgent` | `agents/contract_analysis/value_realisation.py` | Per-contract negotiated-vs-actual savings leakage %, trend classification (first-half/second-half average comparison), multi-tier alert-level thresholds, cumulative-leakage-USD ranking | Narrative synthesis + CFO-risk framing | High — leakage-trend and alert-tier logic is unique | Generic `test_agents_unit.py` coverage only | **Fully implemented (logic); test coverage is generic** |
| `DecisionAuditAgent` | `agents/decision_audit/agent.py` | Rationale-gap %, per-evaluator price-weight variance, per-category rationale-gap %, high-risk decision flagging (`ai_score >= 85` + no rationale) | Narrative synthesis + regulatory-exposure framing | High — evaluator-variance statistics unique to this agent | Generic `test_agents_unit.py` coverage; used as the second stage in the scenario tests' escalation-path scenario | **Fully implemented** |
| 8 agents in `market_intelligence/agents.py` (`InstitutionalMemoryAgent`, `NegotiationIntelligenceAgent`, `SpecificationInflationAgent`, `WorkingCapitalOptimiserAgent`, `DemandIntelligenceAgent`, `SupplierInnovationAgent`, `DecisionCopilotAgent`, `TradeScenarioAgent`) | `agents/market_intelligence/agents.py` | `InstitutionalMemoryAgent.observe()` does category grouping + rationale-presence %; the other 7 were not re-read line-by-line in this pass (time-boxed) — flagged **Not independently re-verified in this pass beyond the earlier report's sampling** | Narrative synthesis for all 8 | **Not independently confirmed either way for the 7 not re-read this pass** — carried forward from the earlier report as **Partially verified** | Generic `test_agents_unit.py` coverage only | `InstitutionalMemoryAgent`: **Fully implemented**; remaining 7: **Not independently re-verified this pass — Partially verified (carried forward)** |
| `TotalCostIntelligenceAgent`, `ComplianceVerificationAgent` | `agents/cost_intelligence/total_cost.py`, `agents/compliance/compliance_verification.py` | Not re-read line-by-line in this pass (only headers/imports sampled in the earlier pass) | — | **Not independently re-verified this pass** | Generic `test_agents_unit.py` coverage only | **Partially verified — carried forward from earlier report, not independently deepened this pass** |
| `ERPSignalConnectorAgent` (referenced in `catalogue.py` docstring only) | N/A — no file exists | N/A | N/A | N/A | No tests | **Documentation only / Missing** — confirmed again this pass; not registered in `scripts/server.py`, no class exists anywhere |

**Cross-agent finding (new this pass):** None of the 15 agents' dedicated unit test files (`tests/unit/test_supplier_stress.py` being the sole per-agent-specific test file) test the actual **numeric correctness** of the bespoke formulas in `bias_detector.py`, `should_cost.py`, or `value_realisation.py` against hand-computed expected values — the generic `test_agents_unit.py` only asserts that `observe()` returns a non-empty dict and that `run()` succeeds with `DryRunReasoningEngine`, not that the computed percentages/thresholds are numerically correct for a known input. This means **the domain-math correctness of 13 of the 15 agents' formulas is untested**, even though the formulas exist and are genuinely distinct (per row-by-row analysis above). This is a meaningful gap: distinct-but-untested logic is a real risk (a sign error or off-by-one in a threshold could silently misclassify supplier risk and no test would catch it).

**Verdict on the "15 agents are just templates" question:** **Rejected as originally framed.** The pipeline shape is shared (by design, and reasonably so — see §5 row 9), but the domain logic is not generic boilerplate. The more accurate criticism, confirmed by this pass, is: **the domain math is real and distinct, but almost entirely unverified by dedicated numeric tests** — a test-coverage gap, not an implementation-genericness problem.

**Precise scope statement (per external review of this report):** this pass independently validated 6 of the 15 agents (`SupplierStressAgent`, `ProcurementBiasDetectorAgent`, `ShouldCostAgent`, `ValueRealisationAgent`, `DecisionAuditAgent`, `InstitutionalMemoryAgent`) in enough depth to confirm distinct, non-trivial domain logic. The remaining 9 (`TotalCostIntelligenceAgent`, `ComplianceVerificationAgent`, and 7 of the 8 `market_intelligence` agents) are **carried forward from the earlier report's sampling, not independently re-verified line-by-line in this pass** (see §26). The accurate claim to make publicly is therefore: *"Several agents were independently validated as containing distinct domain logic; the remaining agents require full line-by-line validation before an equivalent claim can be made for all 15,"* not an unqualified "all 15 agents are fully implemented and distinct."

---

## 8. Orchestration and DAG Assessment

Independently re-derived by re-reading `framework/orchestration/orchestrator.py` in full (both passes read the same file; conclusions match):

| Property | Verified behavior | Evidence |
|---|---|---|
| DAG dependency enforcement | Correct — `Workflow.validate_dag()` uses Kahn's algorithm, raises `ValueError` on unknown dependency IDs and on cycles | Direct code reading |
| Cycle detection | Correct — cycle causes `len(ordered) != len(self.tasks)` check to raise | Direct code reading |
| Parallel execution safety | Genuine — `ThreadPoolExecutor(max_workers=workflow.max_workers)` submits each dependency-ready batch concurrently | Direct code reading |
| Task state persistence | **None** — all `Task`/`Workflow` state is in-process Python objects; nothing is written to disk or a database as part of orchestration itself (only governance/audit records are persisted, and those are best-effort/non-blocking) | Direct code reading, confirmed again |
| Survives process restart | **No** — confirmed; no mechanism exists to resume an in-flight workflow after a crash | Direct code reading |
| Retries idempotent | **Not guaranteed** — `task.for_retry()` produces a new `TaskId` and re-executes `agent.run()`; if a prior partial failure already had side effects (e.g., `InstitutionalMemoryAgent.remember()` writing patterns before the exception was raised later in the same `execute()` call), a retry can duplicate that side effect. Confirmed by code reading; not confirmed empirically in this pass due to time constraints — **Partially verified / architectural risk, not proven to occur under current agent implementations sampled** |
| Timeouts stop or isolate work | **No** — confirmed by direct reading: `as_completed(futures, timeout=workflow.timeout_seconds)` has no surrounding exception handling; Python threads cannot be forcibly cancelled, so "stops work" is not literally true even if the exception were caught | Direct code reading |
| One failed agent affects unrelated agents | **No** — `cancel_on_failure` only cancels tasks with a direct dependency on a failed task; independent branches proceed. Confirmed correct by reading `run_workflow()`'s `cancellable` computation | Direct code reading |
| Cancellation (explicit) | **Not implemented** — there is no `workflow.cancel()`/`task.cancel()` API; "cancellation" in this codebase only means "not submitted because a dependency failed," not stopping an in-flight task | Direct code reading |
| Concurrency bounded | **Yes** — `AgentPool`'s `threading.Semaphore(max_size)` genuinely bounds concurrent agent instances per type; `ThreadPoolExecutor(max_workers)` bounds workflow-level parallelism | Direct code reading, cross-checked against `tests/system/test_multi_agent_system.py`'s pool-boundary test |
| Resource exhaustion handling | **Partial** — `PoolExhaustedError` is raised on `acquire_timeout_s` expiry (back-pressure signal exists); nothing downstream of the orchestrator automatically retries or queues on this specific error beyond the standard per-task retry loop | Direct code reading |
| In-process only | **Yes, confirmed** | Direct code reading |
| Suitable for long-running enterprise workflows | **No, as currently implemented** — appropriate for short/medium synchronous batch analysis jobs within one process's lifetime; not suitable for workflows expected to span process restarts, machine failures, or multi-day durations | Architectural judgment based on above |

**Classification (unchanged from earlier report, independently re-derived):** A correct **in-memory, single-process, bounded-concurrency DAG task runner**. Not a durable workflow engine, not a distributed task execution system, not a production-grade orchestration platform in the sense of tools like Temporal/Airflow/Prefect. This classification is appropriate for the current maturity and does not need replacement — it needs a persisted-task-ledger addition only if crash-recovery becomes a real requirement (see roadmap).

---

## 9. Reasoning and LLM Provider Assessment

Re-confirmed independently:

| Provider path | Real API call code exists? | Auth handled? | Tested? | Classification |
|---|---|---|---|---|
| Ollama (`scripts/server.py::_Gateway.complete`, `provider == "ollama"`) | Yes — raw `urllib.request` POST to `{base_url}/api/chat` | N/A (no key needed) | **No** — zero tests exercise this branch | **Real implementation, functionally unproven** |
| Anthropic | Yes — `anthropic.Anthropic(api_key=...)`, `client.messages.create(...)` | Yes, via `api_key` from config/env | **No** | **Real implementation, functionally unproven** |
| OpenAI / Azure OpenAI | Yes — `openai.OpenAI(api_key=..., base_url=...)` | Yes | **No** | **Real implementation, functionally unproven** |
| LangChain / LangGraph / CrewAI strategies (`framework/reasoning/engine.py`) | Yes — real LCEL chain construction, real `StateGraph` construction with analyse/critique/refine nodes, real CrewAI `kickoff()` call | Delegated to underlying provider config | **No** | **Real implementation, functionally unproven; also entirely optional/off by default** |
| `DryRunReasoningEngine` | Yes — deterministic stub | N/A | **Yes — used by all 169 tests** | **Fully verified as a test double; tells us nothing about real-provider correctness** |

**Structured output / schema validation:** `ReasoningRequest.output_schema` is accepted as a parameter but, confirmed again by re-reading `engine.py` and `_try_parse_json()`, **is never actually validated against** — it only gates whether `_try_parse_json()` is attempted at all. There is no `jsonschema`/`pydantic` validation anywhere in the reasoning path. This is confirmed identically to the earlier report.

**Retry + circuit breaker around every LLM call:** Confirmed correct — `ReasoningEngine.reason()` wraps `self._primary.reason` in `self._circuit.call(...)` inside `self._retry.execute(...)`, with fallback-to-native-gateway on exhaustion. This is genuinely solid resilience engineering, independently re-confirmed.

**Prompt injection:** No mitigation found anywhere in the 4 agent files re-read in this pass or the reasoning engine — free-text fields flow into f-string-built prompts unescaped. Confirmed identically to the earlier report; not contradicted.

**Recommended README wording change (per external review of this report):** replace "Works with any LLM provider" with something closer to *"Provides provider abstractions and implementation paths for Ollama, Anthropic, OpenAI, and Azure OpenAI; provider-specific integration testing is still in progress — all 169 automated tests currently run against a deterministic dry-run stub, not a live provider."* This keeps the true, verifiable claim (real provider-specific code exists) while removing the implied claim (that it has been proven to work) that the evidence does not support.

**Verdict:** The multi-provider claim is **not fabricated** — real, provider-specific code paths exist for 4 native providers plus 3 optional framework integrations. But **"works with any LLM provider"** should be qualified as **"contains real, unproven implementation code for four providers"** until at least mocked-HTTP-level tests exist per provider.

---

## 10. Memory and State Assessment

| Component | What is stored | Where | Persistent? | Tenant-aware? | Encrypted? | Concurrent-safe? | Actually used by agents? |
|---|---|---|---|---|---|---|---|
| `WorkingMemory` | Arbitrary key→value, session-namespaced | In-process `OrderedDict`s | No (process lifetime only) | No (session-level only, no tenant concept) | No | Yes — single `RLock`, confirmed correct by reading and by `test_memory.py` passing | Yes — every `StructuralAgent.execute()` calls `self.remember(f"observations:{task.task_id}", ...)` |
| `ContextMemory` | Workflow-scoped namespace over `WorkingMemory` | Same backend as above | No | No | No | Inherits `WorkingMemory`'s locking | Not observed being instantiated by any of the 15 agents in files read (available but unused in the sampled code) |
| `InstitutionalMemory` | Named key→value with `stored_by`/`stored_at` | JSON file (optional; `null` = in-memory only per default config) | **Yes, when `institutional_memory_path` is set** — atomic tmp-file+rename write confirmed by full re-read of `_persist()` | No | No | Yes for single-writer; `update()` provides atomic read-modify-write | Write-side: yes (`InstitutionalMemoryAgent.remember()`); **read-side (`recall()`) not observed being called by any agent file read in either pass** |
| Governance JSONL streams | Execution/decision/escalation/audit records | 4 append-only files | Yes | No | No | **No — confirmed broken under concurrency, see §11** | Written automatically by `BaseAgent.run()`/`DecisionAgent.execute()` |

**Terminology mismatch, confirmed:** The README and `InstitutionalMemoryAgent`'s metadata describe "institutional memory" and a `knowledge_retrieval` capability ("Retrieve relevant precedent reasoning for a current decision context"). No code path implementing that retrieval was found in either pass. The write side is real and durable; the retrieval/"memory as intelligence" side is **not demonstrated in the codebase** — it remains a metadata-only capability claim.

---

## 11. Governance and Audit Assessment (Deepened This Pass — Executed Evidence)

This section supersedes and strengthens the earlier report's §10 with actual execution results.

### 11.1 Verification logic itself
Re-confirmed correct: `HashChainedWriter.write()` computes `sha256(sorted-json)[:24]` per line and links via `prev_hash`; `GovernanceLogger.verify_chain()` recomputes each line's hash and checks both `prev_hash` continuity and content-hash match. **If given a genuinely uninterrupted, single-threaded write sequence, this design would correctly detect tampering.** The defects below are about chain *construction* under real operating conditions, not about the verification algorithm's correctness.

### 11.2 Empirically executed tests (this pass)

Two throwaway scripts were written to the OS temp directory (not inside the repository) and executed via the terminal in this session. Exact printed output:

```
TEST1 concurrent-writers: lines_written=160 expected=160 errors=0 verify_chain=False
TEST2 restart-simulation: lines_written=10 expected=10 verify_chain=False
```

**Test 1 — concurrent writers on a single `GovernanceLogger` instance, single process:** 8 threads each called `gov.record_execution(...)` 20 times (160 total calls) against one shared `GovernanceLogger`/`HashChainedWriter`. All 160 lines were written with zero exceptions (the file-write lock correctly prevents corrupted/interleaved lines), **but `verify_chain()` returned `False`.**

**Root cause, confirmed by re-reading `HashChainedWriter.write()`:**
```python
def write(self, record):
    entry = {**record, "prev_hash": self._prev_hash}   # <-- read OUTSIDE the lock
    ...
    entry["hash"] = entry_hash
    with self._lock:
        with open(self._path, "a", ...) as fh:
            fh.write(final_line + "\n")               # <-- write INSIDE the lock
    self._prev_hash = entry_hash                        # <-- update OUTSIDE the lock
```
`self._prev_hash` is read before the lock is acquired and written back after the lock is released. Two threads can read the same `_prev_hash` value, then both proceed to (correctly, serially) append their own line to the file — but at least one of those two lines will have a `prev_hash` that does not match the line immediately preceding it in the actual file, because the *file order* (determined by lock acquisition order) is not guaranteed to match the order in which each thread *read* `_prev_hash`. This is a genuine, common, easily-triggered race condition — not an edge case — because the orchestrator's normal operating mode is exactly "multiple agents, same governance logger, concurrent execution."

**Test 2 — process-restart simulation:** A first `GovernanceLogger` wrote 5 records; a second, independently-constructed `GovernanceLogger` pointed at the same directory then wrote 5 more. `verify_chain()` again returned `False`, confirming the earlier report's restart-continuity hypothesis (the new instance's `_prev_hash` starts at `"GENESIS"` rather than being seeded from the file's actual last hash).

### 11.3 Test-suite gap (new finding this pass)
`tests/integration/test_framework_integration.py::test_concurrent_governance_writes` runs an equivalent 8-thread concurrent-write scenario (8 threads × 3 runs = 24 lines) and asserts `len(errors) == 0` and `len(lines) == 24` — **but never calls `gov.verify_chain()`.** This test would pass identically whether or not the chain is valid, because it doesn't check chain validity at all. This is a **false-confidence test**: its name and module-docstring ("Governance logger: hash chain integrity after concurrent writes") imply it verifies chain integrity under concurrency, but it does not.

### 11.4 Classification

| Guarantee | Provided? |
|---|---|
| Ordinary structured logging | **Yes** |
| Traceability (who/when/what, trace_id linkage) | **Yes** |
| Tamper-evidence within a single, uninterrupted, single-threaded write sequence | **Yes, in principle** (not independently stress-tested for a single-threaded-only scenario in this pass, but the verification algorithm is sound) |
| Tamper-evidence under the orchestrator's actual concurrent execution model | **No — empirically disproven in this pass** |
| Tamper-evidence across a process restart | **No — empirically disproven in this pass** |
| Compliance-ready auditability | **No** — neither this pass nor the earlier one found any basis for this claim; it should not be used |

**This is the single most important correction this pass makes to the earlier report:** the earlier report's language ("does not survive process restart") could be read as implying the mechanism is otherwise sound in normal single-process operation. It is not — it is also broken under the orchestrator's normal concurrent-execution mode, which is arguably the more operationally relevant defect since restarts are relatively rare while concurrent workflows are the default execution mode.

---

## 12. Security Assessment

Re-confirmed, with one addition:

| Control | Declared? | Enforced? | Evidence |
|---|---|---|---|
| Authentication (any) | No | No | No auth code anywhere in `framework/` or `scripts/server.py` |
| Authorization / RBAC / ABAC | No | No | Not found |
| Tenant isolation | Implied by "enterprise" framing | No | No tenant concept in `framework/core/types.py` |
| Secret management | Yes (env vars) | **Yes** | `os.environ.get("LLM_API_KEY")` pattern used consistently; no hardcoded secrets found |
| Encryption in transit | Not addressed | N/A (delegated to provider SDKs/HTTPS, not verified) | — |
| Encryption at rest | Not addressed | No | Governance/institutional-memory JSONL/JSON files are plaintext on disk |
| Input validation / sanitisation | Yes, in `config.yaml` | **No** | Zero enforcement code found (re-confirmed via repo-wide search) |
| PII redaction | Yes, in `config.yaml` | **No** | Same as above |
| Rate limiting | Yes, in `config.yaml` | **No** | Same as above |
| Prompt-injection protection | Not declared | No | Free-text fields flow unescaped into LLM prompts |
| Tool permission boundaries | N/A | N/A | No concrete tools exist to bound (see §7/§6) |
| Audit access restrictions | Not addressed | No | Governance JSONL files have standard filesystem permissions only, no ACL layer in code |
| Secure defaults | Partial | Partial | Non-root Docker user (good); health/metrics endpoints unauthenticated by design (acceptable only behind a private network boundary the code does not itself enforce) |
| Dependency security | N/A | N/A | Minimal default dependency surface is a good practice, independently confirmed (`pyyaml`, `requests` only by default) |
| Container security | Good | Yes | Non-root user, minimal runtime layer, import-validated build — confirmed by re-reading the Dockerfile |
| SQL injection (Databricks adapter) | N/A | **Vulnerable** | Confirmed again this pass — raw f-string SQL interpolation |

**Security blockers for enterprise deployment (unchanged conclusion, re-confirmed):**
1. No authN/authZ anywhere.
2. Declared-but-unenforced security config creates a false sense of protection.
3. SQL injection vector in the Databricks adapter.
4. No tenant isolation.
5. Plaintext-at-rest audit/memory files with no ACL layer in code.

---

## 13. Data Ingestion and Product Boundary

Re-confirmed and sharpened:

- SKEIN **does not extract or prepare enterprise data itself.** Every one of the 15 agents' `observe()` methods reads `task.payload` expecting already-structured Python dicts/lists (e.g., `task.payload.get("transaction_data")`) — none reads from a database, file system, ERP API, or event stream directly.
- The only code that touches anything resembling "raw" enterprise systems is the **empty** `framework/tools/base.py`/`ToolAgent` scaffolding and the **empty** `framework/integrations/__init__.py` package — both zero-implementation placeholders.
- The Databricks/Fabric adapters replace the *memory/governance* backend, not a *data-ingestion* layer — they do not extract PO/invoice/supplier data from Databricks tables into agent payloads; the README example (`spark.table("procurement.supplier_transactions").toPandas().to_dict("records")`) shows the **user** is expected to do this extraction and hand SKEIN clean records, confirming the same boundary.
- **This boundary is a reasonable framework design in isolation** (many analysis frameworks legitimately assume pre-structured input), **but it is inconsistent with the README/paper's framing**, which explicitly claims to address "structural" gaps in raw ERP/CRM/behavioral data access. As implemented, SKEIN starts *after* that gap is already closed by someone/something else.

**Recommended product boundary (unchanged judgment from the earlier report, reconfirmed):** Either (a) explicitly scope SKEIN's documentation as "an analysis and governance layer over already-structured procurement data," which is honestly achievable today, or (b) invest in at least one real, narrow, read-only extraction connector (e.g., one ERP's PO-acknowledgement table) to make good on the "structural data access" claim for at least one data source.

---

## 14. Databricks and Fabric Assessment

| Integration | Real operations implemented | Tested | Classification |
|---|---|---|---|
| `DeltaTableMemoryStore` (Databricks) | Yes — real `MERGE INTO`/`SELECT`/`DELETE` Spark SQL calls, real fallback to in-memory cache | **No tests anywhere** | **Scaffolding — functionally plausible but unverified, and contains an active SQL-injection defect** |
| `MLflowGovernanceTracker` (Databricks) | Yes — real `mlflow.start_run`/`log_params`/`log_metrics` calls | **No tests** | **Scaffolding, unverified** |
| `OneLakeMemoryStore` (Fabric) | Yes — real `DataLakeServiceClient` file operations, real fallback to in-memory cache | **No tests** | **Scaffolding, unverified** |
| `FabricGovernanceLogger` (Fabric) | Delegates to the same (defective, per §11) file-based `GovernanceLogger` when Spark is unavailable | **No tests** | **Scaffolding, inherits the governance defects above** |

No integration is **Production-ready** or **Pilot-ready** by the evidence available. Both are best described as **Scaffolding** — real code exists, is defensively written (try/except with graceful fallback), but has never been exercised against a live platform or even a mocked Spark/OneLake client in this repository's test suite.

**Recommended README/doc wording change (per external review of this report):** replace any "production integration" framing for Databricks/Fabric with *"Experimental platform adapters with in-memory fallback behavior; live-platform validation (Spark/OneLake connectivity, authentication, and the Databricks SQL-injection fix in R4) is still pending."*

---

## 15. API, Deployment, and Operations Assessment

Re-confirmed with one addition (Helm):

| Area | Finding |
|---|---|
| API endpoints | Only `/health`, `/ready`, `/metrics`, `/status` — no task-submission endpoint exists on the running server (confirmed again by full re-read of `scripts/server.py`) |
| Request/response validation | N/A — no request-accepting API beyond health/metrics |
| AuthN/AuthZ on server | None |
| Docker | Well-built multi-stage image; not built/run in this assessment (no Docker daemon available) |
| Kubernetes | Well-formed manifest; not applied to a live cluster in this assessment |
| **Helm** | **Not a functioning chart** — `deploy/helm/` contains only `values.yaml`; no `Chart.yaml`, no `templates/`. `helm install`/`helm template` against this directory would fail immediately. **This is a correction/hardening of the earlier report's more lenient characterization.** |
| Graceful shutdown | Partial — `scripts/server.py` registers a `signal.SIGTERM`/`SIGINT` handler that calls `mark_not_ready()` and (per the truncated tail read) `stop_health_server()`; agent pool shutdown (`AgentPool.shutdown()`) exists as a method but was not confirmed to be called from the signal handler in the portion of `run_server()` read in either pass — **Not fully verified** |
| Backup/restore, upgrade/rollback | Not addressed anywhere in the codebase | **Missing** |

**Deployment-readiness conclusion (reconfirmed, Helm finding sharpened):** Supports local development and single-node/manual Docker or Kubernetes deployment (Dockerfile and raw K8s manifest are usable as written, modulo the missing task-submission API). Does **not** currently support Helm-based deployment at all (missing chart scaffolding), and does not support multi-instance deployment in any way that provides value beyond simple replication, given the in-process-only `WorkingMemory` (see §6 of the earlier report, reconfirmed).

---

## 16. Testing and Quality Assessment

| Fact | Verified how |
|---|---|
| 169 tests exist and pass | Directly executed in this session's terminal |
| All tests use `DryRunReasoningEngine` | Confirmed via `grep`-equivalent read of `tests/conftest.py` and every test file sampled — no test imports or exercises a live provider gateway |
| Failure paths tested | Yes — `tests/integration/test_framework_integration.py::FailingAgent` and `tests/system/test_multi_agent_system.py::_FailOnce` scenarios confirmed by reading |
| Timeout paths tested | **No** — no test was found asserting on `workflow.timeout_seconds` expiry behavior; consistent with the confirmed-unhandled defect in §8 |
| Concurrency tested | Yes, structurally (thread counts, pool boundaries, session isolation) — **but the one governance-concurrency test is a false-confidence test** (§11.3) |
| Restart behavior tested | **No** — no test constructs two `GovernanceLogger`/`HashChainedWriter` instances against the same path |
| Security tested | **No** — no test exists for input sanitisation, PII redaction, or rate limiting (consistent with §12 — nothing to test, since nothing is implemented) |
| Audit integrity tested | **Partially** — `test_chain_integrity_after_writes` and `test_chain_detects_tampering` both pass and are meaningful for the **single-threaded, single-instance** case; the concurrency variant is the false-confidence test described in §11.3 |
| All agents tested | All 15 exercised by the generic `test_agents_unit.py`; only `SupplierStressAgent` has a dedicated, deeper unit test file; the bespoke domain-math formulas in the other 14 agents are not independently asserted against hand-computed expected values (§7) |
| Deployment files validated | **No** — no test builds the Docker image, applies the Kubernetes manifest, or attempts `helm template`/`helm lint` against `deploy/helm/` |
| Real business scenarios covered | Yes, to a meaningful degree — `tests/scenarios/test_procurement_scenarios.py` chains real agents (not stubs) through realistic multi-month supplier-deterioration data and asserts on escalation behavior; this is more substantive than a superficial "smoke test" |

**No coverage percentage is claimed here** — none is measured anywhere in the repository (no `coverage.py`/`pytest-cov` configuration or output was found), consistent with the earlier report's restraint on this point.

**New weak-test finding this pass:** `test_concurrent_governance_writes` (§11.3) — recommend renaming or fixing it to actually call `verify_chain()`, since as written it provides false confidence about a property (chain integrity under concurrency) that does not hold.

---

## 17. Architecture Fitness Assessment

Independently re-scored; largely consistent with the earlier report, with the governance/security scores held at the same low level (not further penalized, since the underlying defects were already weighted in) and Helm/deployment nudged down slightly given the missing chart scaffolding.

| Dimension | Score (1–5) | Basis |
|---|---|---|
| Modularity | 4 | Clear `framework/` vs `agents/` boundary, correct import direction, re-confirmed |
| Extensibility | 4 | Adding a 16th agent is straightforward given `StructuralAgent`/registry pattern |
| Agent independence | 4 | Each agent's `observe()` is genuinely a pure function with distinct logic (§7) |
| Testability | 4 | `DryRunReasoningEngine` makes every agent testable without LLM cost; well exploited by 169 tests |
| Explainability | 3 | `reasoning_trace`/`observations` are captured and returned, but LLM output is not validated against `observe()`'s grounding data, so "explainable" is partial |
| Reliability | 3 | Retry/circuit-breaker/pool are solid; orchestrator timeout defect and governance concurrency defect both reduce this |
| Durability | 2 | No persisted task state; governance persistence exists but its chain-integrity guarantee is broken under concurrency (§11) |
| Security | 2 | Declared-but-unenforced controls, SQL injection, no authN/authZ |
| Multi-tenancy | 1 | No tenant concept anywhere in `framework/core/types.py` |
| Observability | 4 | Genuine structured logging, correlation context, Prometheus-format metrics with graceful no-op fallback |
| Scalability | 2 | In-process memory vs. HPA/Helm's multi-replica framing is a structural contradiction |
| Maintainability | 4 | Consistent style, extensive "CHANGES FROM v1" module headers showing real iterative hardening |
| Platform portability | 2 | Adapters exist but are unverified scaffolding with an active security defect |
| Procurement domain value | 2 | Real domain math per agent, but never run against real or even hand-verified data; core "structural data access" thesis unimplemented |
| Enterprise integration | 1 | No real connector to any ERP/CRM/finance system exists |
| Operational simplicity | 3 | Health/metrics endpoints are simple and correct; a single-task submission API now exists (§34) though workflow submission is still not exposed over HTTP |

**Direct answers to the required questions:**
1. **Is the current architecture fundamentally sound?** Yes, for what it actually is (a single-process multi-agent analysis framework). The module boundaries, base-class hierarchy, and resilience patterns are well-designed. **Qualification (per external review of this report):** this conclusion applies specifically to *a single-process, structured-input, research-or-controlled-pilot framework* — it should not be read, without that qualification, as applying to a complete, multi-tenant, enterprise production platform, which is not what was verified here.
2. **Can it support a reliable pilot?** Only a narrow, controlled one — see §22.
3. **Can it evolve into an enterprise platform?** Yes, incrementally — nothing found requires discarding the current design; it requires completing unfinished/broken pieces (§11, §12) and adding genuinely new capabilities (tenant model, persisted task ledger, real data connectors) that do not conflict with the existing architecture.
4. **What must change before production?** Fix the governance hash-chain race condition and restart bug; implement or remove the `security:` claims; fix the Databricks SQL injection; fix the orchestrator timeout handling; add a real Helm chart or remove the Helm claim; add at least a minimal authN layer if any network-facing task API is ever added.
5. **What should not be changed?** The `BaseAgent`/`StructuralAgent`/`DecisionAgent` hierarchy, the `observe→reason→parse_findings` pipeline, the registry pattern, the retry/circuit-breaker/pool resilience layer, and the in-process `WorkingMemory` design for single-instance deployments.
6. **Is a rewrite justified?** **No.** No defect found in either pass requires discarding the architecture; every defect found is a scoped, fixable implementation or configuration issue.
7. **If not, what targeted changes are required?** See §21 (roadmap).

---

## 18. Product and Strategic Assessment

Reconfirmed from the earlier report's §2/§17, with the same conclusion and no material change:

- The stated vision (closing "structural" procurement-intelligence gaps) is a legitimate, under-served problem framing, sourced from a single-author research paper with no evidence in the repository of external validation, real customer data, or real usage telemetry.
- The implementation solves a narrower, still-useful problem: LLM-assisted narrative synthesis and rule-based flagging over already-structured procurement data, with solid supporting engineering.
- **Best current description:** an **open-source multi-agent framework / research artifact**, not yet a **procurement intelligence engine** (which would require real data connectivity and validated domain accuracy) and not an **enterprise product** (which would require auth, tenancy, and a working deployment story for at least one of Docker/Kubernetes/Helm — currently two of three are usable, one is not).
- **Claims that should be reduced or removed until implemented:** "tamper-evident" (§11), the `security:` config block's specific claims (§12), "Helm chart" (until `Chart.yaml`/`templates/` exist), "works with any LLM provider" (until at least one provider path has a passing mocked test), and any implication that SKEIN extracts data from ERP/CRM/behavioral systems itself (§13).
- **Claims that should be kept, with evidence now available to support them:** "15 distinct structural agents" (§7 — genuinely true at the domain-math level, contrary to the earlier report's stronger "template" framing), "resilience: retry + circuit breaker + pool" (§9, reconfirmed solid), "observability: structured logging + metrics + health endpoints" (reconfirmed solid).

---

## 19. Risk Register

| ID | Finding | Evidence | Impact | Confidence | Priority |
|---|---|---|---|---|---|
| R1 | Governance hash chain breaks under normal concurrent execution (not just restart) | Executed script, §11.2, Test 1 | High — the "tamper-evident" claim is false under the system's default concurrent operating mode | **High (executed)** | P0 — **FIXED, see §28** |
| R2 | Governance hash chain breaks across process restart | Executed script, §11.2, Test 2 | High — same as above, compounded on redeploys | **High (executed)** | P0 — **FIXED, see §28** |
| R3 | `security:` config block entirely unenforced | Repo-wide search, §12 | High — false sense of security in any deployment that relies on it | **High** | P0 |
| R4 | Databricks adapter SQL injection | Code read, §14 | High — direct injection vector if wired to LLM-influenced strings | **High** | P0 |
| R5 | Orchestrator workflow-timeout raises unhandled exception | Code read, §8 | Medium — operational reliability defect, not a security issue | **High** | P1 — **FIXED, see §29** |
| R6 | `deploy/helm/` is not a functioning chart | Directory listing, §15 | Medium — blocks any Helm-based deployment claim | **High** | P1 — **FIXED, see §29** |
| R7 | CI workflow is broken (`environment.yml` missing) | File search, §3 | Medium — no automated verification of any change has ever run | **High** | P1 — **FIXED, see §29** |
| R8 | `test_concurrent_governance_writes` is a false-confidence test | Code read, §11.3 | Medium — masks R1 from being caught by the existing suite | **High** | P1 |
| R9 | No real ERP/CRM/behavioral-data extraction layer exists (empty `framework/integrations`, no `ToolAgent` subclass) | Code read, §6, §13 | Strategic — core product thesis unimplemented | **High** | P2 (strategic, not a defect) |
| R10 | 13 of 15 agents' bespoke domain-math formulas are not asserted by dedicated numeric tests | Code + test read, §7 | Medium — silent formula regressions would not be caught | **Medium** | P2 — **FIXED, see §30** |
| R11 | In-process `WorkingMemory` vs. Helm/K8s multi-replica (HPA 2–10) framing is a structural contradiction | Code + manifest read, §6, §15 | Medium — horizontal scaling likely provides no real benefit for shared-memory workflows | **High** | P2 — **PARTIALLY ADDRESSED, see §33** |
| R12 | Prompt-injection risk for untrusted free-text fields flowing into LLM prompts | Code read, §9 | Medium, deployment-dependent | **Medium** | P2 — **MITIGATED, see §31** |
| R13 | No authN/authZ anywhere; no tenant model | Code read, §12, §17 | High for any multi-customer deployment; Low for a single-tenant internal pilot | **High** | P2 (context-dependent) — **tenant model + API-key authN/authZ for the task API now implemented, see §35; still no authN in front of /health,/metrics or any UI/dashboard if one is ever added** |
| R14 | `InstitutionalMemory` retrieval (`recall()`) not observed being used by any agent | Code read, §10 | Low-Medium — "institutional memory" capability is write-only in practice | **Medium** | P3 — **FIXED, see §32** |
| R15 | Retry-after-partial-failure is not proven idempotent for all agents | Architectural reasoning, §8 | Low-Medium — theoretical risk, not proven to occur | **Low** | P3 — **CONFIRMED (empirically) and MITIGATED via opt-in tool, see §32** |

---

## 20. Recommended Target Architecture

Unchanged in shape from the earlier report — incremental, not a rewrite:

```mermaid
flowchart LR
    subgraph Keep["Keep as-is"]
        K1["BaseAgent / StructuralAgent / DecisionAgent hierarchy"]
        K2["AgentRegistry + AgentPool/AgentPoolManager"]
        K3["RetryExecutor + CircuitBreaker"]
        K4["WorkingMemory (in-process, session-isolated)"]
        K5["observability: logging, metrics, health"]
    end
    subgraph FixNow["Fix now (P0/P1)"]
        F1["HashChainedWriter: lock the read-modify-write of prev_hash;\nseed prev_hash from file tail on init"]
        F2["Databricks adapter: parameterized SQL"]
        F3["config.yaml security block: implement minimally or delete"]
        F4["Orchestrator: catch as_completed timeout, mark TIMEOUT tasks"]
        F5["Either build deploy/helm/Chart.yaml + templates/, or remove Helm claim"]
        F6["Replace CI workflow with a pip+unittest workflow"]
        F7["Fix or rename test_concurrent_governance_writes to actually call verify_chain()"]
    end
    subgraph AddLater["Add later (P2/P3, product decisions)"]
        A1["One real ToolAgent/data connector (single ERP/CRM source)"]
        A2["Tenant scoping in framework/core/types.py"]
        A3["Persisted task ledger for crash recovery"]
        A4["Shared memory backend (Redis/Delta) for real multi-replica deployments"]
        A5["Per-agent numeric-correctness unit tests"]
    end
    Keep --> FixNow --> AddLater
```

---

## 21. Prioritized Remediation Roadmap

| ID | Problem | Why it matters | Priority | Suggested direction | Dependencies |
|---|---|---|---|---|---|
| P0-1 | Governance hash-chain race condition under concurrency (R1) | Falsifies the "tamper-evident" claim under the system's default operating mode | **P0 — IMPLEMENTED, see §28** | Move the read of `self._prev_hash` and the write-back into the critical section covered by the lock (single lock covers read-compute-write-persist) | None |
| P0-2 | Governance hash-chain restart discontinuity (R2) | Same claim, different trigger | **P0 — IMPLEMENTED, see §28** | On `HashChainedWriter.__init__`, if the file exists and is non-empty, seed `self._prev_hash` from the last line's `hash` field | P0-1 (fix both in the same change) |
| P0-3 | Unenforced `security:` config (R3) | False sense of security | **P0** | Implement minimal enforcement (input length/depth checks, a log-output redaction filter) or delete the block from `config.yaml`/K8s ConfigMap/README | None |
| P0-4 | Databricks SQL injection (R4) | Active injection vector | **P0** | Use parameterized Spark SQL / Databricks SQL connector APIs, never f-string interpolation | None |
| P1-1 | Orchestrator timeout unhandled exception (R5) | Reliability defect on any long-running workflow | **P1 — IMPLEMENTED, see §29** | Wrap `as_completed(..., timeout=...)` in `try/except concurrent.futures.TimeoutError`; mark outstanding tasks `TaskStatus.TIMEOUT` in the returned `WorkflowResult` | None |
| P1-2 | Non-functional Helm chart (R6) | Blocks the one deployment path currently claimed but not usable | **P1 — IMPLEMENTED, see §29** | Add `Chart.yaml` + `templates/` referencing the existing `values.yaml`, or remove the Helm claim from README/docs until built | None |
| P1-3 | Broken CI workflow (R7) | Zero automated verification currently exists | **P1 — IMPLEMENTED, see §29** | Replace with a `pip install -r requirements.txt && python -m unittest discover` GitHub Actions workflow | None |
| P1-4 | False-confidence governance concurrency test (R8) | Masks P0-1 from ever being caught by CI once CI exists | **P1 — IMPLEMENTED, see §28** | Add `self.assertTrue(gov.verify_chain(str(exec_log)))` to `test_concurrent_governance_writes` (test-only change — explicitly allowed, this document does not modify it, but recommends it) | P1-3, P0-1 |
| P2-1 | No real data-ingestion connector (R9) | Core product thesis unimplemented | **P2 (strategic)** | Build one narrow, read-only `ToolAgent` connector against a real (even sandboxed) ERP/CRM source and validate `SupplierStressAgent` against it | Product decision on which source |
| P2-2 | Untested per-agent domain math (R10) | Silent formula regressions | **P2 — IMPLEMENTED, see §30** | Add hand-computed-expected-value unit tests for `bias_detector.py`, `should_cost.py`, `value_realisation.py`, `decision_audit/agent.py`, and the 8 `market_intelligence` agents | None |
| P2-3 | In-process memory vs. multi-replica framing (R11) | Horizontal scaling may provide no benefit | **P2 — PARTIALLY IMPLEMENTED, see §33** | Document the constraint explicitly, or make a shared backend mandatory when `replicaCount > 1` | A4 in §20 |
| P2-4 | Prompt injection (R12) | Untrusted text flows into prompts | **P2 — IMPLEMENTED, see §31** | Add delimiter/escaping strategy for untrusted fields | None |
| P2-5 | No authN/authZ, no tenancy (R13) | Blocks any multi-customer deployment | **P2 — IMPLEMENTED for the task API, see §35** | Scope decision: single-tenant internal pilot can defer this; any multi-customer plan cannot | Product decision |
| P3-1 | `InstitutionalMemory.recall()` unused (R14) | "Institutional memory" is write-only in practice | **P3 — IMPLEMENTED, see §32** | Wire at least one agent to read back precedent patterns before generating new findings | None |
| P3-2 | Retry idempotency not proven (R15) | Theoretical duplicate-side-effect risk | **P3 — IMPLEMENTED, see §32** | Add a test that forces a mid-`execute()` exception after a memory write, then retries, and asserts no duplicate | None |

**Explicit "do not build yet" list:** external workflow engine (Temporal/Airflow) replacement for the orchestrator; a full multi-tenant SaaS control plane; additional LLM framework integrations beyond the existing LangChain/LangGraph/CrewAI hooks; a rewrite of the agent hierarchy; a real-time streaming ingestion layer. None of these are justified until the P0/P1 items above are closed and at least one real data source is connected (P2-1).

---

## 22. Pilot Readiness

**Conditional go, narrowly scoped.** A controlled pilot is reasonable **only if all of the following hold:**
1. Single-tenant, single-process deployment (no reliance on multi-replica scaling or Helm).
2. Governance/audit output is treated as **structured logging for debugging**, not as a compliance or tamper-evidence control, until P0-1/P0-2 are fixed.
3. No untrusted/adversarial free-text data flows into agent payloads (mitigates the unaddressed prompt-injection risk).
4. The `security:` config block's claims are not relied upon (nothing is enforced).
5. Input data is already structured (procurement records, decision logs, etc.) — SKEIN is not asked to extract data from raw ERP/CRM systems itself.
6. A real LLM provider path (Ollama/Anthropic/OpenAI/Azure) is smoke-tested manually before the pilot starts, since no automated test currently covers this.

Under these constraints, the underlying framework (agents, orchestration, resilience, memory) is solid enough to run a real, narrow, single-tenant procurement-analysis pilot on already-structured data.

**Additional pilot operating conditions (per external review of this report):**
7. Every agent finding is reviewed by a human before any procurement action is taken on it — no automated business action should be driven solely by agent output at this maturity level.
8. A fixed synthetic benchmark (or, if permitted, a small hand-verified real dataset) is run alongside the pilot and compared against manually hand-calculated expected results, to start closing the numeric-correctness test gap identified in §7.
9. False positives and false negatives are explicitly logged and reviewed, not just successful runs.

The pilot should be treated as an **analytical decision-support experiment**, not an autonomous procurement decision system.

## 23. Production Readiness

**Not ready.** Blocking items before any production/enterprise claim: P0-1 through P0-4 (§21), plus at minimum a decision on tenancy/authN (P2-5) if more than one customer/business unit will use a shared deployment, plus a working Helm chart or an explicit removal of that claim (P1-2), plus a passing CI pipeline (P1-3) so that future changes are actually verified before shipping.

---

## 24. Final Architecture Verdict

**Retain the current architecture. Do not rewrite.** This second pass, using both independent re-reading and actual code execution, finds no defect that requires discarding the module boundaries, the agent hierarchy, the orchestrator, or the resilience/observability layers. Every defect identified in both passes — including the governance concurrency race condition newly surfaced with executed evidence in this pass — is a **scoped, locally-fixable implementation defect**, not a symptom of a wrong architectural approach. The architecture is fundamentally sound for what it is (a single-process multi-agent analysis and orchestration framework); the gap is between that sound foundation and the enterprise/production claims layered on top of it in documentation and configuration.

---

## 25. Final Recommendation

1. Fix the four P0 items (§21) before making any further "tamper-evident," "secure," or "production-ready" claims.
2. Fix the three P1 items to make the existing deployment/CI claims actually true.
3. Treat the P2 strategic item (real data connector) as the single highest-leverage next investment if the product direction is to actually close the "structural data access" gap the research paper claims to address; otherwise, honestly re-scope the product narrative to "analysis and governance layer over structured procurement data."
4. Do not add a new workflow engine, a rewrite, or new agents until the above are addressed — more surface area on an unfixed foundation compounds the existing risks (R1–R4 in particular).

---

## 26. Unknowns and Evidence Gaps

- The remaining 7 of the 8 `market_intelligence/agents.py` agents (`NegotiationIntelligenceAgent`, `SpecificationInflationAgent`, `WorkingCapitalOptimiserAgent`, `DemandIntelligenceAgent`, `SupplierInnovationAgent`, `DecisionCopilotAgent`, `TradeScenarioAgent`) and `TotalCostIntelligenceAgent`/`ComplianceVerificationAgent` were **not re-read line-by-line in this pass** — their classification as "Partially verified" is carried forward from the earlier report's sampling, not independently deepened here. A follow-up pass should read these in full before finalizing the agent-by-agent verdict for all 15.
- Whether `AgentPool.shutdown()` is actually invoked from `scripts/server.py`'s `SIGTERM`/`SIGINT` handler was not confirmed — the file was read up to the handler's first line in both passes but not fully to its end.
- Retry-after-partial-failure idempotency risk (R15) is an architectural inference, not empirically demonstrated to cause a duplicate side effect in this pass.
- Live LLM provider behavior (Ollama/Anthropic/OpenAI/Azure), Docker build/run, `kubectl apply`, and live Databricks/Fabric connectivity remain entirely **Not verified** in both passes due to environment constraints (no credentials, no cluster, no Docker daemon available in this session).
- Whether `security.rate_limit_requests_per_minute` or similar controls exist in any code path not matched by the regex search used (`pii_redaction|input_sanitisation|rate_limit|sanitize|redact`) cannot be ruled out with absolute certainty — a targeted manual read of every remaining file not yet opened (a handful of smaller files, e.g., `framework/tools/base.py`'s exception classes tail) was not performed.

---

**Final statement:**

**SKEIN should retain its current architecture and receive targeted, prioritized fixes — not a rewrite.** The module boundaries, agent hierarchy, orchestration, and resilience layers are sound and, on independent re-verification (including newly executed empirical tests in this pass), remain the right foundation. What must change is not the architecture but four concrete, scoped defects — the governance hash-chain's concurrency and restart discontinuity bugs (both now proven by direct execution, not just inspection), the unenforced security configuration, and the Databricks SQL-injection pattern — plus closing the gap between documentation/deployment claims (Helm, CI, "tamper-evident," "any LLM provider") and what is actually implemented and tested today.

---

## 27. External Peer Review Addendum

An external technical review of this report (dated 2026-09-15) was received and evaluated. It concurred with every major conclusion — accept the earlier report's core findings with corrections, retain the architecture, fix the governance/security/Databricks defects, treat SKEIN as an advanced prototype rather than a production product — and raised no factual objection to any finding in this document. Its value was in sharpening precision and wording rather than in surfacing new contradicting evidence. The following refinements from that review have been incorporated directly into the relevant sections above (§7, §9, §14, §17, §22); this addendum records the review's own framing for traceability:

| Reviewer point | Disposition | Where incorporated |
|---|---|---|
| The agent-by-agent section should not imply all 15 agents were equally validated — state precisely which 6 were independently deepened this pass versus the 9 carried forward | **Accepted, incorporated** | §7 (precise scope statement added after the "just templates" verdict) |
| README's "works with any LLM provider" should be softened to reflect that all tests use a dry-run stub, not a live provider | **Accepted, incorporated** | §9 (recommended wording added) |
| Databricks/Fabric should be described as "experimental adapters," not "production integrations" | **Accepted, incorporated** | §14 (recommended wording added) |
| "Architecture is fundamentally sound" should always carry the qualifier "for a single-process, structured-input, research/controlled-pilot framework," to prevent misreading as an enterprise-platform endorsement | **Accepted, incorporated** | §17 (qualification added to answer 1) |
| Pilot readiness should explicitly require human review of every finding, a hand-verified benchmark comparison, and false-positive/false-negative tracking, and should treat the pilot as decision support rather than an autonomous decision system | **Accepted, incorporated** | §22 (three additional operating conditions added) |
| The review's own proposed scope for a follow-up, narrowly-focused review (not another broad assessment) | **Recorded for future reference, not actioned in this document** | See list below |

**Reviewer's proposed scope for the next focused review** (recorded here as a candidate follow-up plan, not performed in this pass):
1. Full line-by-line validation of the 9 agents not independently re-verified in this pass (§26).
2. Numeric-correctness unit tests for every agent's domain-math formulas, checked against hand-computed expected values.
3. Actual provider-level tests (at minimum, mocked-HTTP tests per LLM provider branch).
4. Full server lifecycle and shutdown-path verification, including whether `AgentPool.shutdown()` is actually invoked on `SIGTERM`/`SIGINT`.
5. A concrete decision and specification for the task-submission/API surface (§6) — what it should be, if the "production server" framing is to remain accurate.
6. An explicit product decision: remain a structured-input analysis/governance framework (Option A, §13) versus invest in real enterprise data connectivity (Option B, §13).
7. Validation of the proposed governance hash-chain fix (§21, P0-1/P0-2) against the same two executed test scripts used in this pass, to confirm `verify_chain()` returns `True` after the fix.
8. A decision on whether the deployment model is intended to be strictly single-process/single-instance or genuinely multi-replica — and if the latter, design of the shared-memory backend this requires (§21, A4).

This addendum does not change any finding, score, or verdict elsewhere in this document — it records that the document was independently reviewed, the review's substantive concerns were precision/wording refinements rather than factual corrections, and all such refinements have been incorporated in place.

---

## 28. Remediation Log — P0-1 / P0-2 / P1-4 (Governance Hash-Chain Fix, Implemented and Verified)

This section documents the first roadmap phase actually implemented after this assessment, following the phase's own priority order (P0 items first, same-change dependency between P0-1/P0-2 honored, then the P1-4 test fix that depends on them).

### What was changed
`framework/governance/logger.py::HashChainedWriter` was rewritten to fix both root causes identified in §11:
1. **Concurrency race (P0-1):** `prev_hash` is no longer read before the write lock and written back after it. It is now tracked at the **class level, keyed by resolved file path**, and the entire read → compute-hash → append-line → update sequence happens inside the single per-path `threading.Lock` that already guarded the file write — making each write atomic with respect to chain state, not just to the raw file append.
2. **Restart discontinuity (P0-2):** On first construction of a `HashChainedWriter` for a given path (guarded by the existing `_registry_lock`), a new `_read_last_hash()` method scans the existing file (if any) for the last valid entry's `hash` field and seeds the class-level `prev_hash` from it, instead of always starting at `"GENESIS"`. Subsequent instances constructed for the same path (in the same process) reuse the already-seeded value rather than re-scanning.
3. **False-confidence test (P1-4):** `tests/integration/test_framework_integration.py::test_concurrent_governance_writes` now asserts `gov.verify_chain(...)` is `True` in addition to its prior line-count/no-exceptions checks, so it actually tests what its name promises. Two new regression tests were added in the same file: `test_chain_survives_restart` (two independent `GovernanceLogger` instances writing sequentially to the same directory, simulating a process restart) and `test_chain_valid_with_concurrent_multi_instance_writers` (six independently-constructed `GovernanceLogger` instances writing concurrently to the same directory, the scenario a pooled multi-worker deployment would produce).

No public interface changed: `GovernanceLogger`'s constructor and methods, and `HashChainedWriter`'s constructor and `write()` signature, are unchanged. `verify_chain()` required no modification — its algorithm was already correct; only chain *construction* was broken.

### Why it was changed
§11 of this assessment empirically proved (not just inferred) that the pre-existing implementation broke the tamper-evidence guarantee under the system's default concurrent execution mode and across any process restart reusing the same log directory — the single most severe finding in this document (R1/R2, priority P0).

### Files modified
- `framework/governance/logger.py` — `HashChainedWriter` class (constructor, new `_read_last_hash()`, `write()`).
- `tests/integration/test_framework_integration.py` — `test_concurrent_governance_writes` strengthened; `test_chain_survives_restart` and `test_chain_valid_with_concurrent_multi_instance_writers` added.

### Tests executed
1. `python -m unittest tests.integration.test_framework_integration -v` — **22 tests, 22 passed, 0 failed, 0 errors**, including all five governance tests (`test_chain_integrity_after_writes`, `test_chain_detects_tampering`, `test_concurrent_governance_writes`, `test_chain_survives_restart`, `test_chain_valid_with_concurrent_multi_instance_writers`).
2. Full regression suite: `python -m unittest discover -s tests -p "test_*.py" -v` — **171 tests, 171 passed, 0 failed, 0 errored** (169 pre-existing + 2 new governance tests; no regressions in any other subsystem).
3. Re-ran the **exact same two empirical scripts** used in §11.2 to prove the original defects, unchanged except for import path, executed via the terminal in this session:
   ```
   TEST1 concurrent-writers: lines_written=160 expected=160 errors=0 verify_chain=True
   TEST2 restart-simulation: lines_written=10 expected=10 verify_chain=True
   ```
   Both flipped from `verify_chain=False` (pre-fix, §11.2) to `verify_chain=True` (post-fix), with identical write counts and zero errors — direct before/after empirical proof the fix resolves both defects without changing write behavior.

### Remaining limitations (explicitly not claimed as fixed)
- **Cross-process concurrency beyond a single Python process is improved but not re-tested under true multi-process (not multi-thread) concurrency** — the class-level `prev_hash` dict is process-local; two genuinely separate OS processes writing to the same file at the same time (as opposed to one process with multiple threads, or sequential process restarts) would still each seed from the file on construction and could theoretically race between their own read-on-init and a concurrent process's write, though the window is far narrower than the original per-write race and matches the restart-safety property demonstrated in `test_chain_survives_restart`. A dedicated multi-process (not multi-thread) regression test was not added in this phase — flagged as a follow-up, not as a known-broken case.
- This fix does not address any other P0/P1/P2 item in §21 (unenforced `security:` config, Databricks SQL injection, orchestrator timeout handling, missing Helm chart, broken CI) — those remain open and unchanged by this phase.
- The retention/rotation behavior mentioned in the governance module's own docstring ("Rotation: daily (configurable)") was not implemented or investigated in this phase — still **Unknown**, per §26.

### Next recommended step
Per phase-execution rules (highest-priority-first, no scope jump without justification), the next phase is **P0-3 (unenforced `security:` config block)** or **P0-4 (Databricks SQL injection)** — both remaining P0 items in §21. Recommend starting with **P0-4** next, since it is the narrower, more mechanically-scoped fix (parameterize the existing SQL strings in `platform/databricks/adapter.py::DeltaTableMemoryStore`) with a clear, testable acceptance criterion, before taking on the broader product-scoping decision inherent in P0-3 (implement real enforcement vs. remove the claims).

---

## 29. Remediation Log — P0-4 / P0-3 / P1-1 / P1-2 / P1-3 (All Roadmap P0/P1 Items, Implemented and Verified)

Completed in priority order after §28. No public interfaces changed except where noted.

**P0-4 — Databricks SQL injection.** `platform/databricks/adapter.py::DeltaTableMemoryStore.set/get/delete` now pass `key`/`value_json`/`stored_by` via Spark's parameterized `spark.sql(query, args={...})` named-parameter API instead of f-string interpolation. Added `tests/unit/test_databricks_adapter.py` (3 tests, fake Spark double, asserts injection payloads never appear in raw query text). Test import uses `importlib.util.spec_from_file_location` rather than `import platform.databricks.adapter`, because the top-level `platform/` package name collides with the stdlib `platform` module once it's cached in `sys.modules` — a real, previously-undiscovered import hazard for this module's documented `from platform.databricks.adapter import ...` usage, noted here for future investigation but out of scope for this phase.

**P0-3 — Unenforced `security:` config.** New `framework/security/controls.py`: `SecurityConfig`/`SecurityEnforcer` implementing real input-length and JSON-nesting-depth validation, regex-based PII redaction (email/SSN/card/phone), and a sliding-window rate limiter — all **disabled by default** (zero behavior change for library/test callers) and wired live only via `configure_security()`, called from `scripts/server.py::run_server()` using `config["security"]`. Enforcement hooked into `BaseAgent.run()` (payload + rate-limit checks, raising `InputValidationError`/`RateLimitExceededError` which the existing exception handler already converts to a failed `AgentResult`) and into `SKEINJsonFormatter.format()` (message redaction). 12 new tests in `tests/unit/test_security_controls.py`.

**P1-1 — Orchestrator timeout.** `TaskOrchestrator.run_workflow()`'s `as_completed(futures, timeout=...)` is now wrapped in `try/except FutureTimeoutError`; outstanding futures are marked `TaskStatus.TIMEOUT` and recorded in a new `WorkflowResult.timed_out_tasks` field, factored into `succeeded`. Also replaced `with ThreadPoolExecutor(...) as pool:` with a manually-managed pool + `pool.shutdown(wait=False, cancel_futures=True)` in a `finally` block — the context-manager form's implicit `shutdown(wait=True)` would otherwise still block `run_workflow()` on the timed-out thread, silently defeating the catch. Added `SlowAgent` + `test_workflow_timeout_does_not_raise_and_marks_timed_out_tasks` to `tests/integration/test_framework_integration.py`.

**P1-2 — Non-functional Helm chart.** Added `deploy/helm/Chart.yaml` and `deploy/helm/templates/{_helpers.tpl,namespace,serviceaccount,configmap,deployment,service,hpa,pdb,pvc}.yaml`, extending `values.yaml` with the `llm`/`reasoning`/`orchestration`/`memory`/`security` blocks needed to template the ConfigMap. Installed Helm v4.3.0 locally and validated: `helm lint ./deploy/helm` → 0 chart(s) failed; `helm template skein ./deploy/helm` renders all 7 resources correctly against defaults; conditional toggles (`persistence.enabled`, `autoscaling.enabled`, `podDisruptionBudget.enabled`) verified with `--set` overrides to correctly add/omit the PVC/HPA/PDB. README updated with a working `helm install`/`upgrade` snippet and the stale "Helm chart values" comment corrected.

**P1-3 — Broken CI.** Replaced `.github/workflows/python-package-conda.yml` (referenced a nonexistent `environment.yml`, would fail at the install step on every push) with `.github/workflows/ci.yml`: a `pip install -r requirements.txt`-based job matrixed over Python 3.11/3.12/3.13 running `flake8 --select=E9,F63,F7,F82` then `python -m unittest discover`, plus a separate `helm-lint` job running `helm lint`/`helm template` against the new chart. Running the new lint command locally surfaced one real, pre-existing issue — `framework/observability/health.py::stop_health_server` had a dead `global _server_instance` declaration (F824: the function only reads the module-level variable, never reassigns it) — fixed by removing the unnecessary `global` statement, a behavior-neutral cleanup.

**Tests executed (this phase, cumulative):**
- `python -m unittest tests.unit.test_databricks_adapter -v` — 3/3 passed.
- `python -m unittest tests.unit.test_security_controls -v` — 12/12 passed.
- `python -m unittest tests.integration.test_framework_integration -v` — 23/23 passed (including the new timeout test).
- `helm lint ./deploy/helm` — 0 chart(s) failed. `helm template skein ./deploy/helm` (default values, and with `--set persistence.enabled=true --set autoscaling.enabled=false --set podDisruptionBudget.enabled=false`) — both render correctly.
- `python -m flake8 . --count --select=E9,F63,F7,F82` — clean, exit 0 (after the `health.py` fix).
- Full regression suite: `python -m unittest discover -s tests -p "test_*.py" -v` — **187 tests, 187 passed, 0 failed, 0 errored**, run repeatedly after every change in this phase with no regressions.

**Remaining limitations (explicitly not claimed as fixed):**
- The CI workflow has not yet actually executed on GitHub Actions infrastructure (only locally reproduced equivalent commands) — first real push will be the first true confirmation.
- The `platform`/stdlib naming collision noted under P0-4 is unresolved as a *production* concern — it affects `from platform.databricks.adapter import ...` (and the Fabric adapter) in real usage, not just the new test's import strategy. This was not in scope for P0-4 and is not addressed here.
- P2/P3 roadmap items (real data connector, per-agent numeric-correctness tests, prompt-injection mitigation, tenancy/authN, `InstitutionalMemory.recall()` usage, retry-idempotency test) remain open, unchanged by this phase.

### Next recommended step
All P0 and P1 roadmap items (§21) are now implemented and verified. The next phase should be a P2 item — recommend **P2-2 (numeric-correctness unit tests for the bespoke per-agent formulas)** next, since it is self-contained, has no product-direction dependency (unlike P2-1's real-data-connector decision or P2-5's tenancy/authN decision), and directly closes a gap this document itself flagged (§7): distinct, real domain logic that is currently only smoke-tested, not verified for correctness.

---

## 30. Remediation Log — P2-2 (Per-Agent Numeric-Correctness Tests, Implemented)

Added hand-computed-expected-value unit tests for the six agents with the most substantial, independently-verifiable pure formulas: `tests/unit/test_bias_detector_formulas.py` (8 tests), `test_should_cost_formulas.py` (7), `test_value_realisation_formulas.py` (3), `test_decision_audit_formulas.py` (5), `test_total_cost_formulas.py` (5), `test_compliance_verification_formulas.py` (6) — 34 tests total. Each test constructs a small synthetic dataset with clean, rounding-unambiguous numbers, computes the expected result by hand in the test itself (not by re-deriving the same code path), and asserts exact equality against the agent's pure `observe()`-helper function output (`analyse_evaluation_bias`, `compute_commodity_movements`, `analyse_savings_portfolio`, `compute_accountability_metrics`, `analyse_tco_portfolio`, `analyse_compliance_portfolio`). Every hand-computed expected value matched the actual output on the first run — no formula defects were found in these six functions.

**Tests executed (part 1):** each new file run individually (all `OK`) plus the full suite: `python -m unittest discover -s tests -p "test_*.py" -v` — **221 tests, 221 passed, 0 failed, 0 errored** (187 prior + 34 new).

**Completion (same phase, extended):** read all 8 `agents/market_intelligence/agents.py` agents' `observe()` methods in full and added `tests/unit/test_market_intelligence_formulas.py` (16 tests) covering `InstitutionalMemoryAgent`, `NegotiationIntelligenceAgent`, `SpecificationInflationAgent`, `WorkingCapitalOptimiserAgent`, `DemandIntelligenceAgent`, `SupplierInnovationAgent`, `DecisionCopilotAgent`, and `TradeScenarioAgent` — one hand-computed correctness test plus one missing-required-field/`ValueError` test per agent. All 16 passed on first run; no formula defects found. **P2-2 is now complete for all 15 agents** (34 + 16 = 50 dedicated numeric-correctness tests across 12 new files, on top of the pre-existing `test_supplier_stress.py`).

**Tests executed (part 2):** `python -m unittest tests.unit.test_market_intelligence_formulas -v` — 16/16 passed. Full regression suite: `python -m unittest discover -s tests -p "test_*.py" -v` — **237 tests, 237 passed, 0 failed, 0 errored** (221 prior + 16 new).

**Remaining limitation:** these tests validate the *deterministic* `observe()` layer only (the part that was previously completely untested at the value level) — they do not and cannot validate the LLM-dependent `reason()`/`parse_findings()` narrative-synthesis layer, which remains unproven against any real provider per §9.

### Next recommended step
P2-2 is closed. Recommend **P2-4 (prompt-injection mitigation: delimiter/escaping for untrusted free-text fields flowing into LLM prompts)** next — it is the next self-contained P2 item with no product-direction dependency. P2-1 (real data connector) and P2-5 (tenancy/authN) both require a product-scoping decision from the project owner before implementation and should not be started speculatively.

---

## 31. Remediation Log — P2-4 (Prompt-Injection Mitigation, Implemented)

Added `neutralize_prompt_injection()` and `wrap_untrusted_data()` to `framework/security/controls.py`: the former defangs common imperative injection phrasing (role markers like `system:`/`assistant:`, "ignore previous instructions", "disregard the above", "you are now", "new instructions:") via regex substitution and escapes triple-backtick code fences; the latter wraps neutralised text in explicit `<<<BEGIN_UNTRUSTED_{LABEL}>>>...<<<END_UNTRUSTED_{LABEL}>>>` delimiters. Wired into `framework/reasoning/engine.py::ReasoningEngine.reason()` via a new `_harden_request()` step applied to **every** request before it reaches the primary or fallback strategy — this is a single framework-level integration point, so all 15 agents are covered automatically without touching any agent's prompt-building code (per the "preserve existing architecture, no broad refactor" constraint). The system prompt also gets one short appended instruction telling the model to treat the delimited block as data, not instructions. `ReasoningRequest` is not mutated in place — hardening returns a new instance.

This is a **defence-in-depth heuristic, not a guarantee** — regex-based neutralisation cannot catch every possible injection phrasing, and no LLM is guaranteed to honour the "treat this as data" instruction. It measurably raises the bar for naive injection attempts and is safe to apply unconditionally (verified: `DryRunReasoningStrategy` ignores prompt content entirely, so no existing test's behavior changes).

**Tests executed:** `tests/unit/test_prompt_injection_mitigation.py` (10 tests: 6 for the neutralisation/wrapping helpers, 1 confirming `ReasoningEngine` hardens the request before it reaches a fake capturing strategy and does not mutate the original request, 3 for delimiter/label-sanitisation edge cases) — all 10 pass. Full regression suite: **247 tests, 247 passed, 0 failed, 0 errored** (237 prior + 10 new).

**Remaining limitations (explicitly not claimed as solved):** this mitigates prompt injection, it does not eliminate it — a sufficiently adversarial input could still influence LLM output in ways the neutralisation patterns don't catch. No mitigation was added for the reverse risk (data exfiltration via the LLM being tricked into embedding secrets in its output); that remains open. `enable_input_sanitisation`'s length/depth checks (P0-3) still run at the `BaseAgent.run()` payload level and are complementary to, not a replacement for, this prompt-level hardening.

### Next recommended step
All roadmap items with no product-direction dependency are now closed (P0-1 through P1-3, P2-2, P2-4). The two remaining open roadmap items — **P2-1 (real ERP/CRM data connector)** and **P2-5 (tenancy/authN)** — both require a product-scoping decision from the project owner (which data source to integrate first; whether multi-tenant deployment is actually planned) and should not be started speculatively, per §21's own guidance. Recommend pausing implementation here pending that decision, or proceeding to the lower-priority P3 items (`InstitutionalMemory.recall()` usage, retry-idempotency test) if the owner prefers to keep making incremental progress without a product decision.

---

## 32. Remediation Log — P3-1 / P3-2 (Institutional Memory Recall + Retry Idempotency, Implemented)

**P3-1 — `InstitutionalMemory.recall()` unused.** `InstitutionalMemoryAgent` (`agents/market_intelligence/agents.py`) now actually reads back institutional precedent instead of only writing it: `observe()` calls `self.recall(f"pattern_index:{category}", session_id=...)` for every category present in the current batch and includes the results as `observations["precedent_patterns"]`; `reason()` feeds those precedent patterns into the LLM prompt under a "PREVIOUSLY CAPTURED PATTERNS" section; `parse_findings()` maintains the per-category index (deduplicated by `pattern_type`) when storing newly-extracted patterns, and reports `precedent_patterns_used` in the primary finding's evidence plus a summary sentence when precedent was used. Added `tests/unit/test_institutional_memory_recall.py` (4 tests) proving: a first run has no precedent; a second run in the **same session** recalls the pattern the first run captured; a run in a **different session** correctly sees no precedent (working-memory session isolation is respected, not bypassed); and the agent is safe with no memory store configured at all (`memory=None`).

**Related finding (documented, not fixed in this phase):** while implementing this, `scripts/server.py::run_server()` was confirmed to construct an `InstitutionalMemory` instance (`inst_mem = InstitutionalMemory(storage_path=inst_path)`) that is **never actually injected into any agent** — only `WorkingMemory` is wired via the `factory()` closure's `inst.memory = working_mem`. This means in the real server (not this test's direct-construction setup), `InstitutionalMemoryAgent`'s new recall behavior works correctly but only within a single process's `WorkingMemory` lifetime (session-scoped, lost on restart) — the persistent, cross-restart `InstitutionalMemory` store that server.py builds is currently dead code. This is a distinct, narrowly-scoped bug from P3-1 itself (which is about the agent code, not the server wiring) and was left unfixed here to avoid uninstructed scope expansion; recommend a follow-up one-line fix (`inst.memory = inst_mem if agent_type == "InstitutionalMemoryAgent" else working_mem`, or a more general per-agent-type memory routing convention) in a future pass.

**P3-2 — Retry idempotency not proven.** Added a `Task.idempotency_key: str` field (`framework/core/types.py`) that defaults to the task's own `task_id` at creation and is **preserved unchanged by `for_retry()`** (unlike `task_id`, which is regenerated every attempt) — giving agents a stable key to guard side effects against duplication across retries. This is an **opt-in tool, not an automatic framework-wide fix**: `tests/unit/test_retry_idempotency.py` (4 tests) first empirically **confirms R15 as a real, not merely theoretical, defect** — a naive agent that performs a side effect unconditionally in `parse_findings()` and fails on its first attempt genuinely duplicates that side effect when the orchestrator retries (2 entries recorded for 1 logical unit of work) — then demonstrates that an agent keying its dedup guard on `task.idempotency_key` instead of `task.task_id` does **not** duplicate the same side effect (1 entry), and confirms the key itself is stable across `for_retry()` while `task_id` changes.

**Tests executed:** `test_institutional_memory_recall.py` — 4/4 passed. `test_retry_idempotency.py` — 4/4 passed (including the test that intentionally proves duplication occurs for a naive agent — this assertion succeeding confirms the defect, it is not a bug in the test). Full regression suite: **255 tests, 255 passed, 0 failed, 0 errored** (247 prior + 8 new).

**Remaining limitations (explicitly not claimed as solved):**
- P3-2's fix is opt-in. None of the 15 shipped agents were modified to use `idempotency_key` for their own memory writes in this phase (their existing writes happen to be overwrite-based on deterministic keys, which is naturally idempotent — see §7/§30 — so this was not an active bug in shipped agents, only an unproven risk for future agents with append-style side effects, now proven and now mitigable).
- The `scripts/server.py` / `InstitutionalMemory` dead-code wiring gap noted above under P3-1 remains open.
- No change was made to `GovernanceLogger`/`HashChainedWriter` (§28) to use `idempotency_key` for deduplication — that fix already addressed governance-log duplication differently (chain-state locking), and remains independently correct.

### Next recommended step
Every roadmap item that does not require a product-scoping decision is now closed (P0-1 through P1-3, P2-2, P2-4, P3-1, P3-2). Remaining open items are **P2-1** (real data connector), **P2-3** (in-process-memory vs. multi-replica documentation/decision), **P2-5** (tenancy/authN), and the newly-documented **`InstitutionalMemory` server-wiring gap** (§32) — all four require a product or infrastructure decision from the project owner rather than further speculative engineering.

---

## 33. Remediation Log — P2-3 / P2-5 (Multi-Tenant Foundation, Partially Implemented)

**Product decision received from the project owner:** target scale is 200+ customers, 100k+ transactions/day (bursty, business-hours-concentrated), with **physical per-tenant isolation** (dedicated storage container/database per customer) on **Delta Lake**. This unblocks P2-3 and P2-5 together — at this scale they are the same architectural decision (in-process, unpartitioned state cannot be both horizontally scaled *and* physically tenant-isolated).

### What was built

1. **`TenantId` type + `Task.tenant_id`** (`framework/core/types.py`) — new frozen value type and an optional field on `Task`, threaded through `Task.create()`. Defaults to `None` for full backward compatibility with every existing single-tenant test and call site.
2. **`framework/multitenancy/context.py`** (new package) — `TenantContext` (tenant_id, dedicated catalog, schema, storage container, table names) with `validate_identifier()`/`__post_init__` allowlist validation (rejects anything that isn't `^[A-Za-z_][A-Za-z0-9_]*$`, closing an identifier-injection vector that value-only parameterization from P0-4 does not cover — table/catalog names aren't values, they can't go through `args={}`), and `qualified_table()` to build a fully-qualified `catalog.schema.table` name. `TenantRegistry` is a thread-safe in-process router from `tenant_id` → `TenantContext`. **Explicitly out of scope**: this registry only *routes* to already-provisioned storage — creating the actual Unity Catalog catalog / ADLS container per tenant is an infrastructure/ops concern (Terraform, an admin script, a Databricks REST API call), not something this codebase does.
3. **`framework/governance/hashchain.py`** (new) — the hash-chain math from `HashChainedWriter` (fixed in §28) extracted into backend-agnostic `compute_chained_entry()`/`verify_chained_entries()` functions, so the same concurrency/restart-safety fix backs both the local-file backend and the new Delta backend without duplicating (and risking future drift in) the chain algorithm. `HashChainedWriter` and `GovernanceLogger.verify_chain()` were refactored to call these functions — behavior-preserving, confirmed by the full governance test suite (§28's tests) still passing unchanged.
4. **`platform/databricks/adapter.py::DeltaTableMemoryStore`** — now accepts an optional `tenant: TenantContext` parameter; when given, routes to that tenant's dedicated catalog (`tenant.qualified_table(...)`) instead of the shared `table_name`/`catalog` defaults. Omitting `tenant` preserves the exact prior single-tenant behavior (verified by test).
5. **`platform/databricks/adapter.py::DeltaGovernanceStore`** (new) — a tenant-partitioned, hash-chained governance log backed by one Delta table per tenant (using the same `hashchain` module), replacing the per-pod local JSONL files that fragment the audit trail once replicas scale (§11/§28's finding). All four record types (execution/decision/escalation/audit) land in one per-tenant table distinguished by an `event_type` column. Falls back to an in-memory buffer (same as the existing `DeltaTableMemoryStore` pattern) when Spark is unavailable.
6. **Tenant-scoped rate limiting** (`framework/agents/base.py`) — `BaseAgent.run()` now keys `SecurityEnforcer.check_rate_limit()` on `f"tenant:{task.tenant_id}"` when a tenant is set, falling back to `self.agent_type` otherwise (unchanged behavior for single-tenant/no-tenant tasks). This directly addresses the "one noisy customer starves everyone else" risk at 200-tenant scale.

### Bug found and fixed during this work
While writing the fallback-buffer test for `DeltaGovernanceStore`, discovered that `event_type` was being appended to the buffered entry **after** hashing, so its hash didn't match what `verify_chained_entries()` recomputes from the same dict — a real correctness bug in the new code (not a pre-existing one), caught by the test itself before being shipped. Fixed by folding `event_type` into the record **before** calling `compute_chained_entry()`, consistently for both the fallback-buffer path and the live-Spark `record_json` payload.

### Tests executed
`tests/unit/test_tenant_context.py` (14 tests: identifier validation, `TenantContext`, `TenantRegistry`, global singleton), `tests/unit/test_databricks_adapter.py` extended (+7 tests: tenant-routed `DeltaTableMemoryStore`, tenant-qualified/hash-chained `DeltaGovernanceStore`, cross-tenant table isolation — each governance test uses a unique per-test catalog name specifically to avoid the class-level chain-state dict leaking state between test methods, i.e. not relying on alphabetical test-execution order), `tests/unit/test_security_controls.py` (+1 test: tenant-keyed rate limiting via a real `BaseAgent.run()` call, confirming tenant A's quota exhaustion does not block tenant B). Full regression suite: **277 tests, 277 passed, 0 failed, 0 errored** (255 prior + 22 new).

### What remains — explicitly not done in this phase
- **Actual provisioning**: no Terraform/ARM/Databricks-admin-API code was written to create per-tenant catalogs/containers — this codebase only routes to them once they exist.
- **`scripts/server.py` is not yet tenant-aware**: it still wires one global `WorkingMemory`/`GovernanceLogger` to every agent instance (and, per §32, doesn't even wire the existing `InstitutionalMemory` correctly). Making the server itself resolve a `TenantContext` per incoming task and construct/cache the right `DeltaTableMemoryStore`/`DeltaGovernanceStore` pair is the next concrete step, and is blocked on the still-open finding from §6/§15: **there is no task-submission API on the server today**, so there is no request path yet to carry an incoming `tenant_id` in the first place.
- **Live Databricks/Unity Catalog validation**: `DeltaGovernanceStore`'s restart-resume query (`ORDER BY inserted_at DESC LIMIT 1`) and the `IDENTITY`-column-based schema it assumes are documented in the class docstring but never executed against a real cluster — the caveat already present for the entire `platform/databricks` adapter (§14) applies here too, and is more consequential now given the target scale.
- **AuthN/authZ**: still entirely absent (R13's other half). A tenant model with no authentication in front of it only prevents accidental cross-tenant data mixing in the storage layer — it does not prevent an unauthenticated caller from claiming to be any `tenant_id` they like. This must be solved before any of this is production-facing at 200-customer scale.
- **Capacity/pool sizing for the stated peak load**: `AgentPool`'s default `max_size=10` and the orchestrator's default `max_workers=4` are far below what "bursty, 10-20x average" 100k/day traffic needs; no load test was run against the multi-tenant path in this phase (§16's pre-existing load tests are all single-tenant).

### Next recommended step
Two independent next steps, either can go first: (a) design and implement the task-submission API (closing the §6/§15 gap) with `tenant_id` as a required field on every incoming request, since without it the server-side tenant routing has nothing to key on; or (b) write the actual Terraform/provisioning script for per-tenant Databricks catalogs so `TenantRegistry` has something real to route to in a staging environment, enabling genuine (not fake-Spark) end-to-end validation of `DeltaTableMemoryStore`/`DeltaGovernanceStore` before they're trusted at 100k tx/day.

---

## 34. Remediation Log — Task-Submission API (Implemented, Live-Verified)

Closed the §6/§15 gap: `scripts/server.py` previously started only the health/metrics HTTP server, with no way to submit work to the running orchestrator over the network.

### What was built
- **`framework/api/server.py`** (new) — `POST /v1/tasks`, built on the stdlib `http.server` (matching the existing `framework/observability/health.py` pattern — no new web-framework dependency added to the minimal core). `tenant_id` is **mandatory on every request, per explicit product decision** — a missing or blank `tenant_id` is a 400, never a silent single-tenant fallback. When the injected `TenantRegistry` has any tenants registered, `tenant_id` must also match one of them (400 otherwise) — a structural allowlist check, **not authentication**: it stops typos/unprovisioned tenants, it does not verify the caller is actually entitled to act as the tenant they claim. That remains the separate, still-open authN/authZ item.
- Request validation: `agent_type` (required string), `payload` (required JSON object), `tenant_id` (required non-empty string, optionally checked against the tenant registry), optional `session_id`/`priority`/`timeout_seconds`. Unknown `agent_type` (an `AgentRegistry` `KeyError`) and any other unexpected exception are caught at the API boundary and turned into clean 400/500 JSON responses instead of a raw stack trace or a hung connection.
- Runs the task **synchronously** via the existing `TaskOrchestrator.run_task()` and returns `AgentResult.to_dict()` — deliberately simple for v1, consistent with there being no persisted task queue yet (§8's known limitation still applies: a crash mid-request loses that request's work, same as it always has for in-process calls).
- Wired into `scripts/server.py::run_server()`: starts alongside the health server (`SKEIN_TASK_API_PORT`, default 8081), stops in the same graceful-shutdown handler that already drains the pool and stops the health server.

### Verification
1. **Unit/integration tests**: `tests/unit/test_task_api_server.py` (7 tests) — starts a real HTTP server on an OS-assigned ephemeral port and issues real HTTP requests via `urllib` (no mocking of the HTTP layer): valid submission with a known tenant succeeds; missing/blank `tenant_id` rejected (400); unknown `tenant_id` against a populated registry rejected (400); unknown `agent_type` rejected (400); missing `payload` rejected (400); unknown route returns 404. Full suite: **284 tests, 284 passed, 0 failed, 0 errored** (277 prior + 7 new).
2. **Live end-to-end smoke test against the real server** (not just the test harness): started `python -m scripts.server --dry-run` as an actual process (`SKEIN_HEALTH_PORT=8180`, `SKEIN_TASK_API_PORT=8181`), confirmed via its own structured logs that all 15 agents registered and the task API bound port 8181, then issued real HTTP requests with PowerShell's `Invoke-WebRequest`:
   - `POST /v1/tasks` with `agent_type=SupplierStressAgent`, `tenant_id=acme`, `payload={"transaction_data": []}` → **HTTP 422**, body: `{"succeeded": false, "error": "ValueError: Payload must contain 'transaction_data'", ...}` — this is the *real* `SupplierStressAgent.observe()`'s own validation rejecting an empty list (falsy), proving the request reached the actual registered agent through the actual orchestrator, not a stub.
   - `POST /v1/tasks` with no `tenant_id` → **HTTP 400**, body: `{"error": "'tenant_id' is required and must be a non-empty string"}`, confirmed rejected *before* reaching the orchestrator.
   - `GET /health` on the health server continued working unaffected on its separate port.
   - Server was then stopped via `Ctrl+C`, exercising the existing graceful-shutdown path (which now also calls `stop_task_api_server()`).

### Remaining limitations (explicitly not claimed as solved)
- **Workflow submission is not exposed** — only single-task submission (`POST /v1/tasks`); multi-agent `Workflow`/`WorkflowBuilder` DAGs still can only be run via direct Python import.
- **No authN/authZ** — this API is not safe to expose directly to the internet; it must sit behind a reverse proxy/gateway that authenticates the caller and only then trusts the `tenant_id` it forwards. This was always the explicitly separate, still-open item and remains so.
- **`scripts/server.py` still does not resolve a per-tenant `DeltaTableMemoryStore`/`DeltaGovernanceStore`** (§33's gap) — the task API now has a `tenant_id` on every request, which is the prerequisite for that wiring, but the wiring itself (constructing/caching the right tenant-scoped stores and injecting them per-request instead of the one global `WorkingMemory`/`GovernanceLogger`) was not done in this phase. **UPDATE (§36): this has since been implemented.**
- **No rate limiting / request size limits at the HTTP layer itself** — `SecurityEnforcer`'s tenant-keyed rate limiting (§33) only fires once a `Task` is constructed and reaches `BaseAgent.run()`; a flood of malformed requests that fail validation before that point is not yet throttled at the API layer.
- Synchronous-only execution model — a slow agent (real LLM call) holds the HTTP connection open for the full duration; no async/job-polling pattern was added.

### Next recommended step
With `tenant_id` now flowing in on every request, the natural next step is finishing the wiring noted in §33: have `scripts/server.py` resolve a `TenantContext` from the registered `TenantRegistry` per request and route each agent's `memory`/`governance` to that tenant's dedicated `DeltaTableMemoryStore`/`DeltaGovernanceStore` instead of the current single global store — or, if the owner prefers, address authN/authZ first so the API is safe to put behind a real ingress before more storage plumbing is built on top of it.

---

## 35. Remediation Log — authN/authZ for the Task API (Implemented)

Closes the "no authN/authZ anywhere" half of R13/P2-5 for the task-submission API specifically (the tenant *model* was already built in §33; this adds authentication and authorization on top of it).

### What was built
- **`framework/auth/api_keys.py`** (new) — `ApiKeyStore`: thread-safe, hash-only storage of API keys (SHA-256; plaintext is never retained after `register()` returns), `hmac.compare_digest`-based constant-time authentication, `generate_api_key()`/`revoke()`. `AuthContext` (authenticated tenant_id + key_id) and `authorize_tenant_match()` implement the actual authorization check: if a request's body also names a `tenant_id`, it must equal the authenticated key's tenant, or it's rejected — this is precisely what stops tenant A's key from being used to submit work as tenant B.
- **`framework/api/server.py`** — when an `ApiKeyStore` with at least one key is attached to the running `TaskAPIServer`, every `POST /v1/tasks` must present a key (`Authorization: Bearer <key>` or `X-API-Key` header) or gets **401** before the JSON body is even parsed. The tenant identity for the request is derived from the authenticated key, not the client-editable body field. A mismatched body `tenant_id` gets **403**. Consistent with every other security control added in this project (P0-3, §33's `TenantRegistry` check): **opt-in, safe-by-default** — no `ApiKeyStore` or an empty one means the server behaves exactly as it did before this phase (explicit body `tenant_id`, no auth check), so nothing that already worked broke.
- **`scripts/server.py`** — provisions the store from a new `SKEIN_API_KEYS` environment variable (JSON `{"tenant_id": "raw_key", ...}`), logs a clear warning if it's empty/absent ("task API authentication is DISABLED"), and passes it into `start_task_api_server()`. Same "bootstrap-only, not a secrets manager" scope boundary already used for tenant storage provisioning (§33) — rotation, secure-at-rest storage, and actual key issuance/delivery to customers are ops concerns not solved here.

### Tests executed
- `tests/unit/test_api_keys.py` (11 tests): valid/invalid/missing-key authentication, confirmed plaintext keys are never stored (only their hash), key revocation, independent per-tenant keys, and `authorize_tenant_match()`'s three cases (matching tenant allowed, no body tenant_id defaults to the authenticated one, mismatched tenant rejected).
- `tests/unit/test_task_api_server.py` extended (+6 tests, new `TestTaskAPIServerWithAuth` class, real HTTP server + real requests): valid key with no body `tenant_id` succeeds (200); valid key with a *matching* body `tenant_id` succeeds (200); no key when auth is enabled → 401; an invalid/unrecognised key → 401; a valid key for tenant `acme` used with body `tenant_id=globex` → **403** (the core authZ guarantee, actually exercised end-to-end over HTTP, not just at the unit level); the `X-API-Key` header alternative is also accepted.
- The pre-existing `TestTaskAPIServer` class (no `ApiKeyStore` passed) was left unmodified and continues to pass unchanged, confirming the opt-in default didn't regress the §34 behavior.
- Full regression suite: **301 tests, 301 passed, 0 failed, 0 errored** (284 prior + 17 new).

### Remaining limitations (explicitly not claimed as solved)
- **This is service-to-service API-key auth, not end-user identity.** There are no user accounts, roles, scopes, or OAuth2/OIDC — if a human-facing UI/dashboard is ever built on top of this, it needs its own real identity layer; API keys are the right tool for machine-to-machine calls, not for that.
- **`/health`, `/ready`, `/metrics`, `/status` remain completely unauthenticated** (`framework/observability/health.py` was not touched in this phase) — acceptable only if that port is never exposed outside a private cluster network, same caveat as before.
- **Key provisioning/rotation/secure storage is a bootstrap mechanism** (one JSON env var), not a real secrets-management integration — do not treat `SKEIN_API_KEYS` as production-grade secret handling at 200-tenant scale; it should be replaced with a real secrets manager (Key Vault, Databricks secrets, etc.) before this goes to production.
- **No audit trail specifically for auth failures** — a 401/403 is returned to the caller and logged via the standard Python logger, but is not (yet) written to the governance log (`DeltaGovernanceStore`/`GovernanceLogger`) as its own event type; repeated auth failures against one tenant's key (a credential-stuffing signal) are not currently surfaced anywhere an operator would see them as a pattern.
- **No transport security added** — this API-key check happens after a plaintext HTTP connection in this implementation; a real deployment must terminate TLS in front of it (reverse proxy/ingress), or the key itself is exposed on the wire.

### Next recommended step
The three remaining roadmap items are now: (1) wire per-tenant Delta stores into `scripts/server.py`'s actual request path (§33's still-open item, now that both `tenant_id` and an authenticated identity are available on every request), (2) write the real Terraform/provisioning script so `TenantRegistry`/`ApiKeyStore` have something real to route to and issue keys against, and (3) replace the `SKEIN_API_KEYS` bootstrap mechanism with a real secrets-manager integration before production use. None of these block each other — pick based on which unblocks your staging environment fastest.

---

## 36. Remediation Log — Per-Tenant Delta Store Routing in the Request Path (Implemented)

Closes the last open piece of §33: `scripts/server.py` previously injected one global `WorkingMemory`/`GovernanceLogger` into every agent instance regardless of which tenant a task belonged to; now a task's `tenant_id` actually determines which storage backend an agent uses while it runs.

### What was built
- **`framework/multitenancy/resolver.py::TenantStoreResolver`** (new) — lazily builds and caches one `(DeltaTableMemoryStore, DeltaGovernanceStore)` pair per tenant, routed to that tenant's dedicated catalog via `TenantContext` (§33's physical-isolation model). `resolve(tenant_id)` returns `None` — not a fallback pair — for a missing/unregistered tenant or if store construction fails, which matters: it lets the caller (`BaseAgent.run()`) interpret "no tenant-specific override" as "leave this agent's own default alone" rather than forcing one resolver-wide default onto every agent. Loads `platform/databricks/adapter.py` via `importlib.util` by file path (same workaround already used in this project's own tests) rather than a normal package import, because of the still-unresolved `platform`/stdlib name collision documented under P0-4/§33.
- **`framework/agents/base.py::BaseAgent`** — new optional `tenant_store_resolver` constructor parameter. In `run()`, if set, the agent's `memory`/`governance` are temporarily swapped to the resolver's result for the duration of *that single call only*, then restored in the existing `finally` block — after `_log_to_governance()` has already run (so the tenant-specific governance store, not the restored default, is the one that actually receives the execution record). **Documented precondition**: this temporary instance-level swap is only safe when the caller guarantees exclusive access to the instance for the duration of `run()` — true for `AgentPool`-checked-out instances (which `scripts/server.py` always uses), not guaranteed for instances shared via `AgentRegistry.get_or_create()` without a pool manager. This is stated explicitly in the docstring, not silently assumed.
- **`scripts/server.py`** — constructs one `TenantStoreResolver(get_tenant_registry())` and injects `inst.tenant_store_resolver = tenant_resolver.resolve` into every agent instance via the existing `factory()` closure.
- **Bonus fix, directly adjacent to the code being changed**: the `InstitutionalMemory` dead-code bug documented in §32 (`inst_mem` was constructed but never actually injected into any agent — only `working_mem` was) is fixed as part of this same edit: `InstitutionalMemoryAgent` now gets `inst_mem` as its default store, every other agent keeps `working_mem` as before. This was small, directly adjacent to the exact lines being rewritten, and already flagged as a known bug in this project's own documentation — left unfixed any longer would have been an odd choice while rewriting this exact function.

### Tests executed
- `tests/unit/test_tenant_store_resolver.py` (7 tests, `_FakeSpark` double): missing/unregistered tenant → `None`; registered tenant → correctly tenant-qualified Delta stores; caching across repeated calls for the same tenant; two tenants get independent stores/tables; stores still construct even when the underlying Spark queries fail (matching the Delta adapter's existing defensive fallback-to-cache behavior from P0-4); and a direct test of `resolve()`'s own exception handling (forcing `_build_stores` to raise) confirming it degrades to `None` rather than propagating.
- `tests/unit/test_tenant_aware_agent_run.py` (4 tests, real `BaseAgent.run()` calls, no mocking of the swap mechanism itself): a task with no tenant uses the agent's default memory/governance and neither swap target is touched; a task for a known tenant is provably routed to the *tenant* memory/governance (data written lands only in the tenant store, only the tenant's governance double records the call) and the agent's `memory`/`governance` attributes are restored to the defaults immediately after `run()` returns; an unregistered tenant falls back to the defaults; and a second run for a different tenant right after the first confirms no data crosses between them.
- Full regression suite: **312 tests, 312 passed, 0 failed, 0 errored** (301 prior + 11 new). `python -c "import scripts.server"` re-verified clean.

### Remaining limitations (explicitly not claimed as solved)
- **Still no live Databricks/Unity Catalog validation** — same caveat as §14/§33; everything here is proven against `_FakeSpark`, not a real cluster.
- **The exclusive-instance-access precondition is documented, not enforced in code** — nothing stops a future caller from wiring `tenant_store_resolver` onto an agent obtained via `AgentRegistry.get_or_create()` without a pool manager, which would reintroduce a genuine cross-tenant race condition on `self.memory`/`self.governance`. This is a real footgun for future misuse, flagged clearly rather than silently guarded against, since guarding it would require either a runtime assertion (added complexity for a documented usage contract) or a bigger redesign (threading tenant identity through method calls instead of instance state) that was judged disproportionate for this phase.
- **No cache eviction/TTL on `TenantStoreResolver`'s per-tenant store cache** — at 200 tenants this is a small, bounded, appropriate-sized cache; if the tenant count grows by orders of magnitude, an eviction policy would become necessary.
- **Provisioning still doesn't exist** — same as §33: `TenantRegistry` only routes to tenants that are already registered; nothing in this phase creates the actual Databricks catalogs/containers.

### Next recommended step
The two remaining roadmap items are: (1) write the real Terraform/provisioning script so tenants registered in `TenantRegistry` correspond to actually-existing Databricks catalogs (enabling genuine, non-fake-Spark end-to-end validation of everything built in §33/§34/§35/§36), and (2) replace the `SKEIN_API_KEYS` bootstrap mechanism with a real secrets-manager integration. Recommend (1) first — it's the one thing that would let every multi-tenant claim in this log actually be exercised against a live system instead of a test double.
