"""
framework/agents/catalogue.py
================================
Complete specification of all 15 Procurement Intelligence Agents.

This module is the authoritative agent catalogue — it defines the
AgentMetadata, capabilities, and design contract for all 15 agents
derived from "The 15 Structural Mysteries of Procurement AI".

AGENT HIERARCHY:
  ProcurementAgent (8 agents)
    - SupplierStressAgent        Mystery 02
    - ShouldCostAgent            Mystery 06
    - ValueRealisationAgent      Mystery 11
    - SpecificationInflationAgent Mystery 04
    - DemandIntelligenceAgent    Mystery 07
    - SupplierInnovationAgent    Mystery 08
    - ComplianceVerificationAgent Mystery 09
    - TradeScenarioAgent         Mystery 12

  DecisionAgent (4 agents)
    - InstitutionalMemoryAgent   Mystery 01
    - NegotiationIntelligenceAgent Mystery 03
    - WorkingCapitalOptimiser    Mystery 05
    - StrategicProcurementAgent  (orchestrating agent)

  ToolAgent (1 agent)
    - ERPSignalConnectorAgent    (data infrastructure)

  Specialised (2 agents)
    - DecisionAuditAgent         Mystery 13
    - TotalCostIntelligenceAgent Mystery 14
    - ProcurementBiasDetectorAgent Mystery 15
"""

from framework.core.types import (
    AgentCapability, AgentMetadata, DecisionAuthority,
)

AUTHOR = "Mohammed Akbar Ansari"

# ---------------------------------------------------------------------------
# Agent 01 — Institutional Memory
# ---------------------------------------------------------------------------

INSTITUTIONAL_MEMORY_METADATA = AgentMetadata(
    agent_type="InstitutionalMemoryAgent",
    display_name="Institutional Memory Capture Agent",
    description=(
        "Captures and preserves tacit procurement expertise from experienced "
        "practitioners — the reasoning behind decisions, not just the decisions. "
        "Addresses the $31.5B annual institutional knowledge loss problem."
    ),
    version="1.0.0",
    capabilities=(
        AgentCapability(
            name="expertise_capture",
            description="Extract decision reasoning patterns from practitioner interactions",
            input_types=("decision_logs", "negotiation_transcripts", "category_reviews"),
            output_types=("reasoning_patterns", "expertise_profiles"),
            authority=DecisionAuthority.ADVISE,
        ),
        AgentCapability(
            name="knowledge_retrieval",
            description="Retrieve relevant precedent reasoning for a current decision context",
            input_types=("current_context", "supplier_id", "category"),
            output_types=("precedent_reasoning", "similar_cases"),
            authority=DecisionAuthority.ADVISE,
        ),
    ),
    tags=("knowledge", "expertise", "institutional"),
    author=AUTHOR,
    mystery_refs=("mystery_01",),
)

# ---------------------------------------------------------------------------
# Agent 02 — Supplier Stress Signal
# ---------------------------------------------------------------------------

SUPPLIER_STRESS_METADATA = AgentMetadata(
    agent_type="SupplierStressAgent",
    display_name="Supplier Financial Stress Early Warning Agent",
    description=(
        "Detects early warning signatures of supplier financial or operational "
        "distress from internal ERP transaction data — 6–9 months before any "
        "external source carries the signal."
    ),
    version="2.0.0",
    capabilities=(
        AgentCapability(
            name="supplier_stress_detection",
            description="Score 6 behavioural signals per supplier on a 0-12 composite scale",
            input_types=("supplier_transactions", "po_data", "quality_records"),
            output_types=("stress_profiles", "risk_alerts", "advance_warning"),
            authority=DecisionAuthority.RECOMMEND,
        ),
    ),
    tags=("supplier", "risk", "early_warning", "erp"),
    author=AUTHOR,
    mystery_refs=("mystery_02",),
)

# ---------------------------------------------------------------------------
# Agent 03 — Negotiation Psychology
# ---------------------------------------------------------------------------

NEGOTIATION_INTELLIGENCE_METADATA = AgentMetadata(
    agent_type="NegotiationIntelligenceAgent",
    display_name="Counterparty Negotiation Intelligence Agent",
    description=(
        "Builds per-supplier counterparty intelligence models — concession patterns, "
        "authority structure, financial position, historical movement — to generate "
        "negotiation strategy specific to this supplier in this market context."
    ),
    version="1.0.0",
    capabilities=(
        AgentCapability(
            name="counterparty_profiling",
            description="Build negotiation profile per supplier from historical data",
            input_types=("negotiation_history", "supplier_financials", "commodity_prices"),
            output_types=("counterparty_profiles", "concession_patterns"),
            authority=DecisionAuthority.ADVISE,
        ),
        AgentCapability(
            name="negotiation_strategy",
            description="Generate specific negotiation tactics for current context",
            input_types=("supplier_id", "negotiation_context", "target_outcomes"),
            output_types=("negotiation_recommendations", "tactical_guidance"),
            authority=DecisionAuthority.RECOMMEND,
        ),
    ),
    tags=("negotiation", "counterparty", "strategy"),
    author=AUTHOR,
    mystery_refs=("mystery_03",),
)

# ---------------------------------------------------------------------------
# Agent 04 — Specification Inflation
# ---------------------------------------------------------------------------

SPECIFICATION_INFLATION_METADATA = AgentMetadata(
    agent_type="SpecificationInflationAgent",
    display_name="Specification Inflation & Competitive Tension Agent",
    description=(
        "Operates upstream in specification workflows to detect requirements that "
        "artificially limit the competitive pool — certifications held by only one "
        "supplier, proprietary interfaces, unnecessary technical constraints."
    ),
    version="1.0.0",
    capabilities=(
        AgentCapability(
            name="specification_bias_detection",
            description="Flag specification requirements that limit competitive pool",
            input_types=("specifications", "supplier_database", "certification_registry"),
            output_types=("bias_findings", "alternative_specifications"),
            authority=DecisionAuthority.RECOMMEND,
        ),
    ),
    tags=("specification", "competition", "upstream"),
    author=AUTHOR,
    mystery_refs=("mystery_04",),
)

# ---------------------------------------------------------------------------
# Agent 05 — Working Capital Triangle
# ---------------------------------------------------------------------------

WORKING_CAPITAL_METADATA = AgentMetadata(
    agent_type="WorkingCapitalOptimiserAgent",
    display_name="Working Capital Triangle Optimisation Agent",
    description=(
        "Treats payment terms, supplier financial health, and treasury position "
        "as one continuous optimisation problem — making the trade-off calculation "
        "in real time before the supplier needs to ask for help."
    ),
    version="1.0.0",
    capabilities=(
        AgentCapability(
            name="working_capital_optimisation",
            description="Compute optimal payment terms given supplier health and treasury position",
            input_types=("payment_terms", "supplier_health_signals", "treasury_position"),
            output_types=("optimisation_recommendations", "trade_off_analysis"),
            authority=DecisionAuthority.RECOMMEND,
        ),
    ),
    tags=("finance", "working_capital", "treasury", "payments"),
    author=AUTHOR,
    mystery_refs=("mystery_05",),
)

# ---------------------------------------------------------------------------
# Agent 06 — Should-Cost Intelligence
# ---------------------------------------------------------------------------

SHOULD_COST_METADATA = AgentMetadata(
    agent_type="ShouldCostAgent",
    display_name="Should-Cost Intelligence Agent",
    description=(
        "Maintains real-time cost models across every spend category by continuously "
        "integrating commodity price indices, regional labour costs, and energy data. "
        "Exposes structural information asymmetry in supplier pricing."
    ),
    version="2.0.0",
    capabilities=(
        AgentCapability(
            name="should_cost_modelling",
            description="Build bottom-up cost model from commodity and labour inputs",
            input_types=("commodity_prices", "labour_indices", "energy_data", "supplier_margins"),
            output_types=("cost_models", "price_benchmarks", "negotiation_leverage"),
            authority=DecisionAuthority.RECOMMEND,
        ),
    ),
    tags=("cost", "negotiation", "commodity", "should_cost"),
    author=AUTHOR,
    mystery_refs=("mystery_06",),
)

# ---------------------------------------------------------------------------
# Agent 07 — Pre-Signal Demand Intelligence
# ---------------------------------------------------------------------------

DEMAND_INTELLIGENCE_METADATA = AgentMetadata(
    agent_type="DemandIntelligenceAgent",
    display_name="Pre-Signal Demand Intelligence Agent",
    description=(
        "Monitors exogenous leading indicators — building permits, hiring patterns, "
        "shipping data, weather, policy filings — to generate procurement positioning "
        "recommendations 3–6 months before demand materialises in the order book."
    ),
    version="1.0.0",
    capabilities=(
        AgentCapability(
            name="leading_indicator_monitoring",
            description="Track exogenous signals correlated with category demand",
            input_types=("macro_data", "sector_indicators", "category_mappings"),
            output_types=("demand_forecasts", "positioning_recommendations"),
            authority=DecisionAuthority.ADVISE,
        ),
    ),
    tags=("demand", "forecasting", "market_intelligence", "leading_indicators"),
    author=AUTHOR,
    mystery_refs=("mystery_07",),
)

# ---------------------------------------------------------------------------
# Agent 08 — Supplier Innovation
# ---------------------------------------------------------------------------

SUPPLIER_INNOVATION_METADATA = AgentMetadata(
    agent_type="SupplierInnovationAgent",
    display_name="Supplier Innovation Intelligence Agent",
    description=(
        "Monitors patent filings, product announcements, R&D hiring patterns, "
        "and conference participation across the supplier base — cross-referencing "
        "against the buying organisation's product roadmap and sustainability agenda."
    ),
    version="1.0.0",
    capabilities=(
        AgentCapability(
            name="innovation_signal_detection",
            description="Surface relevant supplier R&D before suppliers present it",
            input_types=("patent_data", "supplier_announcements", "buyer_roadmap"),
            output_types=("innovation_opportunities", "supplier_capability_updates"),
            authority=DecisionAuthority.ADVISE,
        ),
    ),
    tags=("innovation", "r_and_d", "supplier_intelligence", "patents"),
    author=AUTHOR,
    mystery_refs=("mystery_08",),
)

# ---------------------------------------------------------------------------
# Agent 09 — Compliance Verification
# ---------------------------------------------------------------------------

COMPLIANCE_VERIFICATION_METADATA = AgentMetadata(
    agent_type="ComplianceVerificationAgent",
    display_name="Supply Chain Compliance Verification Agent",
    description=(
        "Triangulates supplier certification claims against independent signals — "
        "satellite monitoring, worker sentiment platforms, trade flow analysis — "
        "for CSDDD, CSRD, and forced labour regulation compliance."
    ),
    version="1.0.0",
    capabilities=(
        AgentCapability(
            name="compliance_triangulation",
            description="Verify certifications against independent observable signals",
            input_types=("certification_database", "trade_flows", "external_signals"),
            output_types=("compliance_findings", "verification_gaps", "regulatory_risks"),
            authority=DecisionAuthority.ESCALATE,
        ),
    ),
    tags=("compliance", "esg", "csddd", "csrd", "forced_labour"),
    author=AUTHOR,
    mystery_refs=("mystery_09",),
)

# ---------------------------------------------------------------------------
# Agent 10 — Cognitive Load / Decision Co-Pilot
# ---------------------------------------------------------------------------

DECISION_COPILOT_METADATA = AgentMetadata(
    agent_type="DecisionCopilotAgent",
    display_name="Category Manager Decision Co-Pilot",
    description=(
        "Prioritises AI-generated outputs for human review — surfaces the decisions "
        "that genuinely require human judgment this week, provides context, and "
        "defers or handles everything else. Addresses the cognitive load crisis."
    ),
    version="1.0.0",
    capabilities=(
        AgentCapability(
            name="decision_prioritisation",
            description="Rank pending decisions by urgency and human judgment requirement",
            input_types=("pending_alerts", "category_context", "user_preferences"),
            output_types=("prioritised_decisions", "deferred_items", "auto_handled"),
            authority=DecisionAuthority.ADVISE,
        ),
    ),
    tags=("cognitive_load", "prioritisation", "decision_support", "copilot"),
    author=AUTHOR,
    mystery_refs=("mystery_10",),
)

# ---------------------------------------------------------------------------
# Agent 11 — Value Realisation
# ---------------------------------------------------------------------------

VALUE_REALISATION_METADATA = AgentMetadata(
    agent_type="ValueRealisationAgent",
    display_name="Procurement Value Realisation Monitoring Agent",
    description=(
        "Continuously tracks the gap between negotiated savings commitments and "
        "actual P&L capture — identifying specific leakage causes and recommending "
        "targeted interventions while there is still time to recover value."
    ),
    version="2.0.0",
    capabilities=(
        AgentCapability(
            name="savings_leakage_detection",
            description="Detect and quantify the gap between negotiated and realised savings",
            input_types=("contract_tracking", "spend_data", "erp_actuals"),
            output_types=("leakage_profiles", "intervention_recommendations"),
            authority=DecisionAuthority.RECOMMEND,
        ),
    ),
    tags=("savings", "value_realisation", "contract_compliance", "pnl"),
    author=AUTHOR,
    mystery_refs=("mystery_11",),
)

# ---------------------------------------------------------------------------
# Agent 12 — Trade Policy / Scenario Planning
# ---------------------------------------------------------------------------

TRADE_SCENARIO_METADATA = AgentMetadata(
    agent_type="TradeScenarioAgent",
    display_name="Trade Policy Scenario Intelligence Agent",
    description=(
        "Maintains a continuous model of sourcing network exposure across multiple "
        "geopolitical scenarios — calculating the NPV of strategic sourcing options "
        "and the window to act before options close."
    ),
    version="1.0.0",
    capabilities=(
        AgentCapability(
            name="trade_scenario_modelling",
            description="Model sourcing exposure and option NPV across geopolitical scenarios",
            input_types=("sourcing_network", "tariff_data", "geopolitical_signals"),
            output_types=("scenario_models", "option_valuations", "strategic_recommendations"),
            authority=DecisionAuthority.RECOMMEND,
        ),
    ),
    tags=("trade_policy", "geopolitics", "scenario_planning", "tariffs"),
    author=AUTHOR,
    mystery_refs=("mystery_12",),
)

# ---------------------------------------------------------------------------
# Agent 13 — Decision Accountability
# ---------------------------------------------------------------------------

DECISION_AUDIT_METADATA = AgentMetadata(
    agent_type="DecisionAuditAgent",
    display_name="Procurement Decision Accountability Agent",
    description=(
        "Surfaces the accountability vacuum in AI-assisted procurement decisions. "
        "Detects missing rationale, evaluator bias, override patterns, and "
        "regulatory exposure (EU AI Act, OECD transparency principles)."
    ),
    version="2.0.0",
    capabilities=(
        AgentCapability(
            name="accountability_gap_detection",
            description="Identify AI-assisted decisions without traceable reasoning records",
            input_types=("decision_logs", "evaluator_records", "override_history"),
            output_types=("accountability_gaps", "regulatory_risks", "remediation_plan"),
            authority=DecisionAuthority.ESCALATE,
        ),
    ),
    tags=("governance", "accountability", "eu_ai_act", "audit", "transparency"),
    author=AUTHOR,
    mystery_refs=("mystery_13",),
)

# ---------------------------------------------------------------------------
# Agent 14 — Total Cost Intelligence
# ---------------------------------------------------------------------------

TOTAL_COST_METADATA = AgentMetadata(
    agent_type="TotalCostIntelligenceAgent",
    display_name="Total Cost of Ownership Intelligence Agent",
    description=(
        "Computes full lifecycle cost profiles at sourcing decision time — energy, "
        "maintenance, downtime risk, switching costs — revealing the gap between "
        "what procurement optimises (invoice price) and what the business cares about."
    ),
    version="2.0.0",
    capabilities=(
        AgentCapability(
            name="tco_analysis",
            description="Compute TCO vs purchase price for capital and service categories",
            input_types=("asset_data", "energy_costs", "maintenance_history", "downtime_data"),
            output_types=("tco_profiles", "price_vs_tco_gap", "portfolio_risk"),
            authority=DecisionAuthority.RECOMMEND,
        ),
    ),
    tags=("tco", "lifecycle_cost", "capital_procurement", "finance"),
    author=AUTHOR,
    mystery_refs=("mystery_14",),
)

# ---------------------------------------------------------------------------
# Agent 15 — Procurement Bias Detection
# ---------------------------------------------------------------------------

BIAS_DETECTOR_METADATA = AgentMetadata(
    agent_type="ProcurementBiasDetectorAgent",
    display_name="Procurement Evaluation Bias Detector",
    description=(
        "Surfaces incumbent advantage bias and diverse/SME supplier suppression "
        "patterns that are invisible in individual decisions but statistically "
        "significant in aggregate — and amplified when AI is trained on biased data."
    ),
    version="2.0.0",
    capabilities=(
        AgentCapability(
            name="incumbent_bias_detection",
            description="Measure incumbent award rate premium beyond objective performance",
            input_types=("sourcing_evaluations", "award_history", "objective_scores"),
            output_types=("bias_profiles", "evaluator_flags", "corrective_recommendations"),
            authority=DecisionAuthority.ESCALATE,
        ),
        AgentCapability(
            name="diverse_supplier_suppression_detection",
            description="Identify systematic underperformance of diverse/SME suppliers",
            input_types=("sourcing_evaluations", "supplier_diversity_data"),
            output_types=("suppression_findings", "process_recommendations"),
            authority=DecisionAuthority.ESCALATE,
        ),
    ),
    tags=("bias", "incumbent", "diversity", "fairness", "evaluation"),
    author=AUTHOR,
    mystery_refs=("mystery_15",),
)

# ---------------------------------------------------------------------------
# Complete catalogue — all 15 agents
# ---------------------------------------------------------------------------

ALL_AGENT_METADATA = [
    INSTITUTIONAL_MEMORY_METADATA,
    SUPPLIER_STRESS_METADATA,
    NEGOTIATION_INTELLIGENCE_METADATA,
    SPECIFICATION_INFLATION_METADATA,
    WORKING_CAPITAL_METADATA,
    SHOULD_COST_METADATA,
    DEMAND_INTELLIGENCE_METADATA,
    SUPPLIER_INNOVATION_METADATA,
    COMPLIANCE_VERIFICATION_METADATA,
    DECISION_COPILOT_METADATA,
    VALUE_REALISATION_METADATA,
    TRADE_SCENARIO_METADATA,
    DECISION_AUDIT_METADATA,
    TOTAL_COST_METADATA,
    BIAS_DETECTOR_METADATA,
]


def print_catalogue() -> None:
    """Print a formatted summary of all 15 agents."""
    print(f"\n{'='*70}")
    print("  SKEIN — Procurement Intelligence Agent Catalogue")
    print(f"{'='*70}")
    for i, m in enumerate(ALL_AGENT_METADATA, 1):
        caps = ", ".join(c.name for c in m.capabilities)
        print(f"\n  {i:02d}. {m.display_name}")
        print(f"      Type:         {m.agent_type}")
        print(f"      Mystery refs: {', '.join(m.mystery_refs)}")
        print(f"      Capabilities: {caps}")
        print(f"      Tags:         {', '.join(m.tags)}")
    print(f"\n{'='*70}")
    print(f"  Total: {len(ALL_AGENT_METADATA)} agents")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print_catalogue()
