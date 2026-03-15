"""
Synthetic data generator for the Procurement AI Mysteries framework.

Generates realistic, reproducible data for all agents:
  - Supplier transaction records (Mystery 02)
  - Commodity price history (Mystery 06)
  - Contract savings tracking (Mystery 11)
  - Procurement decision logs (Mystery 13)
  - Purchase order lifecycle costs (Mystery 14)
  - Sourcing evaluation records (Mystery 15)

All generators are seeded for reproducibility.
Run:  python generate_all.py
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
random.seed(SEED)
BASE_DATE = datetime(2024, 1, 1)
MONTHS = 12
OUT = Path(__file__).parent


def month_label(n: int) -> str:
    return (BASE_DATE + timedelta(days=30 * n)).strftime("%Y-%m")


# ── Mystery 02: Supplier transaction data ─────────────────────────────────────

def gen_healthy(sid, name):
    return [{"supplier_id": sid, "supplier_name": name, "month": month_label(m),
             "po_ack_days": round(random.uniform(1.5, 2.5), 1),
             "otd_pct": round(random.uniform(94, 99), 1),
             "quality_hold_pct": round(random.uniform(0.4, 0.9), 2),
             "invoice_disputes": random.randint(0, 1),
             "unsolicited_discounts": 0,
             "sales_response_hours": round(random.uniform(2, 8), 1),
             "partial_shipments": random.randint(0, 1),
             "scenario": "healthy"} for m in range(MONTHS)]


def gen_crisis(sid, name, crisis_at=9):
    rows = []
    for m in range(MONTHS):
        stress = max(0, m - 2)
        if m >= crisis_at:
            rows.append({"supplier_id": sid, "supplier_name": name, "month": month_label(m),
                         "po_ack_days": round(random.uniform(9, 16), 1),
                         "otd_pct": round(random.uniform(52, 68), 1),
                         "quality_hold_pct": round(random.uniform(7, 13), 2),
                         "invoice_disputes": random.randint(9, 16),
                         "unsolicited_discounts": random.randint(2, 4),
                         "sales_response_hours": round(random.uniform(52, 100), 1),
                         "partial_shipments": random.randint(4, 9),
                         "scenario": "CRISIS"})
        else:
            rows.append({"supplier_id": sid, "supplier_name": name, "month": month_label(m),
                         "po_ack_days": round(2.0 + stress * 0.7, 1),
                         "otd_pct": round(97.0 - stress * 3.1, 1),
                         "quality_hold_pct": round(0.8 + stress * 0.8, 2),
                         "invoice_disputes": random.randint(0, 1) + stress,
                         "unsolicited_discounts": 1 if stress >= 2 else 0,
                         "sales_response_hours": round(4.0 + stress * 4.2, 1),
                         "partial_shipments": random.randint(0, 1) + (stress // 2),
                         "scenario": f"pre_crisis_stress_{stress}"})
    return rows


def gen_noisy(sid, name):
    rows = []
    for m in range(MONTHS):
        spike = random.random() < 0.25
        rows.append({"supplier_id": sid, "supplier_name": name, "month": month_label(m),
                     "po_ack_days": round(random.uniform(6, 10) if spike else random.uniform(1.5, 3), 1),
                     "otd_pct": round(random.uniform(78, 85) if spike else random.uniform(93, 98), 1),
                     "quality_hold_pct": round(random.uniform(3, 6) if spike else random.uniform(0.5, 1.5), 2),
                     "invoice_disputes": random.randint(4, 8) if spike else random.randint(0, 2),
                     "unsolicited_discounts": 0,
                     "sales_response_hours": round(random.uniform(2, 8), 1),
                     "partial_shipments": random.randint(0, 1),
                     "scenario": "spike" if spike else "normal"})
    return rows


def generate_supplier_transactions():
    data = (gen_healthy("SUP-001", "Apex Components") +
            gen_healthy("SUP-002", "Meridian Industrial") +
            gen_crisis("SUP-003", "Vertex Precision Parts", crisis_at=9) +
            gen_noisy("SUP-004", "Orion Packaging"))
    return data


# ── Mystery 06: Commodity prices ──────────────────────────────────────────────

def generate_commodity_prices():
    categories = {
        "steel_hrc_usd_ton": (620, 40, -0.5),        # (base, volatility, monthly_drift%)
        "copper_lme_usd_ton": (8800, 300, 0.3),
        "hdpe_resin_usd_ton": (1050, 80, -1.2),
        "labour_index_mfg": (100, 2, 0.15),
        "energy_index": (100, 5, 0.08),
    }
    records = []
    for m in range(MONTHS):
        row = {"month": month_label(m)}
        for cat, (base, vol, drift) in categories.items():
            prev = records[-1][cat] if records else base
            change = random.gauss(drift / 100 * prev, vol)
            row[cat] = round(prev + change, 2)
        records.append(row)
    return records


# ── Mystery 11: Contract savings tracking ─────────────────────────────────────

def generate_savings_tracking():
    contracts = [
        {"contract_id": "CTR-001", "category": "Packaging",
         "negotiated_savings_pct": 14.0, "negotiated_date": "2024-02-01",
         "annual_spend": 2_400_000},
        {"contract_id": "CTR-002", "category": "Industrial Gases",
         "negotiated_savings_pct": 9.5, "negotiated_date": "2024-03-15",
         "annual_spend": 1_800_000},
        {"contract_id": "CTR-003", "category": "Logistics 3PL",
         "negotiated_savings_pct": 11.0, "negotiated_date": "2024-01-10",
         "annual_spend": 5_600_000},
    ]
    leakage_scenarios = {
        "CTR-001": {"actual_savings_pct": [14, 13.5, 12.8, 10.2, 9.8, 8.1],
                    "causes": ["spec_change", "maverick_spend"]},
        "CTR-002": {"actual_savings_pct": [9.5, 9.4, 9.5, 9.3, 9.4, 9.2],
                    "causes": []},
        "CTR-003": {"actual_savings_pct": [11, 10.5, 9.8, 8.2, 7.5, 6.1],
                    "causes": ["volume_shortfall", "erp_not_updated", "maverick_spend"]},
    }
    tracking = []
    for c in contracts:
        cid = c["contract_id"]
        scenario = leakage_scenarios[cid]
        for i, actual in enumerate(scenario["actual_savings_pct"]):
            month_n = i + 1
            base_spend = c["annual_spend"] / 12
            expected_saving = base_spend * c["negotiated_savings_pct"] / 100
            actual_saving = base_spend * actual / 100
            tracking.append({
                "contract_id": cid,
                "category": c["category"],
                "month": month_label(i + 1),
                "negotiated_savings_pct": c["negotiated_savings_pct"],
                "actual_savings_pct": actual,
                "leakage_pct": round(c["negotiated_savings_pct"] - actual, 2),
                "leakage_amount_usd": round(expected_saving - actual_saving, 0),
                "leakage_causes": scenario["causes"],
                "monthly_spend": round(base_spend, 0),
            })
    return tracking


# ── Mystery 13: Decision audit trail ─────────────────────────────────────────

def generate_decision_logs():
    decisions = []
    suppliers = [("SUP-001", "Apex Components"), ("SUP-002", "Meridian Industrial"),
                 ("SUP-003", "GlobalTech Parts"), ("SUP-004", "FastForward Logistics")]
    categories = ["Packaging", "Raw Materials", "MRO", "Logistics", "IT Hardware"]
    outcomes = ["awarded", "rejected", "re-evaluated", "escalated"]

    for i in range(40):
        sup = random.choice(suppliers)
        cat = random.choice(categories)
        outcome = random.choice(outcomes)
        score = round(random.uniform(55, 98), 1)
        decisions.append({
            "decision_id": f"DEC-{i+1:04d}",
            "date": (BASE_DATE + timedelta(days=random.randint(0, 360))).strftime("%Y-%m-%d"),
            "category": cat,
            "supplier_id": sup[0],
            "supplier_name": sup[1],
            "ai_recommendation": outcome,
            "ai_score": score,
            "human_override": random.random() < 0.18,   # 18% override rate
            "final_outcome": outcome if random.random() > 0.18 else random.choice(outcomes),
            "rationale_logged": random.random() < 0.45,  # only 45% have rationale
            "factors_weighted": {
                "price": round(random.uniform(0.3, 0.5), 2),
                "quality": round(random.uniform(0.2, 0.4), 2),
                "delivery": round(random.uniform(0.1, 0.3), 2),
                "risk": round(random.uniform(0.05, 0.2), 2),
            },
        })
    return decisions


# ── Mystery 14: Total cost vs purchase price ──────────────────────────────────

def generate_tco_data():
    assets = []
    categories = [
        ("industrial_pump", 45000, {"energy": 0.25, "maintenance": 0.08, "downtime_risk": 0.05}),
        ("conveyor_system", 120000, {"energy": 0.18, "maintenance": 0.12, "downtime_risk": 0.06}),
        ("cnc_machine", 280000, {"energy": 0.10, "maintenance": 0.09, "downtime_risk": 0.04}),
        ("forklift", 52000, {"energy": 0.06, "maintenance": 0.11, "downtime_risk": 0.03}),
    ]
    for asset_type, base_price, cost_drivers in categories:
        for variant in range(3):
            purchase_price = base_price * random.uniform(0.88, 1.15)
            annual_revenue_at_risk = purchase_price * random.uniform(3, 8)
            lifecycle_years = random.randint(7, 15)
            energy_annual = purchase_price * cost_drivers["energy"] * random.uniform(0.9, 1.1)
            maintenance_annual = purchase_price * cost_drivers["maintenance"] * random.uniform(0.85, 1.2)
            downtime_annual = annual_revenue_at_risk * cost_drivers["downtime_risk"] * random.uniform(0.7, 1.3)
            total_tco = (purchase_price +
                         (energy_annual + maintenance_annual + downtime_annual) * lifecycle_years)
            assets.append({
                "asset_id": f"ASSET-{asset_type[:4].upper()}-{variant+1:02d}",
                "asset_type": asset_type,
                "supplier_id": f"SUP-{random.randint(1,4):03d}",
                "purchase_price_usd": round(purchase_price, 0),
                "lifecycle_years": lifecycle_years,
                "annual_energy_cost_usd": round(energy_annual, 0),
                "annual_maintenance_cost_usd": round(maintenance_annual, 0),
                "annual_downtime_risk_usd": round(downtime_annual, 0),
                "total_tco_usd": round(total_tco, 0),
                "tco_vs_purchase_ratio": round(total_tco / purchase_price, 2),
                "procurement_decided_on_price_alone": random.random() < 0.62,  # 62% price-only
            })
    return assets


# ── Mystery 15: Sourcing bias detection ──────────────────────────────────────

def generate_sourcing_evaluations():
    evals = []
    supplier_types = [
        ("incumbent", 0.72),    # (type, historical_award_rate)
        ("new_entrant", 0.21),
        ("diverse_owned", 0.18),
        ("sme", 0.24),
    ]
    for i in range(60):
        stype, base_award_rate = random.choice(supplier_types)
        obj_score = round(random.uniform(60, 95), 1)
        # Incumbents get inflated scores in subjective criteria
        subj_score = obj_score + (random.uniform(4, 12) if stype == "incumbent" else
                                  random.uniform(-5, 5))
        subj_score = round(min(100, max(0, subj_score)), 1)
        awarded = random.random() < (base_award_rate + (0.12 if stype == "incumbent" else 0))
        evals.append({
            "eval_id": f"EVAL-{i+1:04d}",
            "date": (BASE_DATE + timedelta(days=random.randint(0, 360))).strftime("%Y-%m-%d"),
            "supplier_type": stype,
            "category": random.choice(["Packaging", "Raw Materials", "MRO", "IT"]),
            "objective_score": obj_score,      # price, delivery, quality — measurable
            "subjective_score": subj_score,    # relationship, fit — assessor judgement
            "combined_score": round((obj_score * 0.6 + subj_score * 0.4), 1),
            "awarded": awarded,
            "evaluator_id": f"EVL-{random.randint(1, 8):03d}",
            "incumbent_advantage_flag": stype == "incumbent",
        })
    return evals


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    datasets = {
        "supplier_transactions.json": generate_supplier_transactions(),
        "commodity_prices.json": generate_commodity_prices(),
        "savings_tracking.json": generate_savings_tracking(),
        "decision_logs.json": generate_decision_logs(),
        "tco_data.json": generate_tco_data(),
        "sourcing_evaluations.json": generate_sourcing_evaluations(),
    }
    for filename, data in datasets.items():
        path = OUT / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {filename}: {len(data)} records")

    print(f"\nAll datasets written to {OUT}")


if __name__ == "__main__":
    main()
