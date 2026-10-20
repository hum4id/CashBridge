"""Sandbox financing product catalog."""
from __future__ import annotations

import json

PRODUCTS = [
    {
        "id": "quick_stock_advance",
        "name": "Quick Stock Advance",
        "provider": "Sandbox Bank",
        "flat_fee_pct": 1.0,
        "tenor_days": 30,
        "max_amount": 15_000_000,
        "min_coverage_ratio": 1.15,
        "description": "30-day working-capital advance for stock purchases, repaid weekly.",
    },
    {
        "id": "receivable_bridge",
        "name": "Receivable Bridge",
        "provider": "Sandbox Bank",
        "flat_fee_pct": 1.4,
        "tenor_days": 45,
        "max_amount": 25_000_000,
        "min_coverage_ratio": 1.20,
        "description": "45-day facility sized to the cash gap and repaid as customer payouts land.",
    },
    {
        "id": "weekly_micro",
        "name": "Weekly Micro Loan",
        "provider": "Sandbox Bank",
        "flat_fee_pct": 1.5,
        "tenor_days": 56,
        "max_amount": 10_000_000,
        "min_coverage_ratio": 1.25,
        "description": "Small 8-week facility with weekly installments for quick needs.",
    },
]


# Eligibility rules kept in code (versioned with the policy, not editable from the DB).
RULES = {
    "quick_stock_advance": {"min_history_days": 56, "max_income_cv": 0.50,
                            "requires_repayment_source": False},
    "receivable_bridge":   {"min_history_days": 56, "max_income_cv": 0.80,
                            "requires_repayment_source": True},
    "weekly_micro":        {"min_history_days": 84, "max_income_cv": 0.40,
                            "requires_repayment_source": False},
}


def ensure_products(conn) -> None:
    for p in PRODUCTS:
        conn.execute(
            """INSERT OR REPLACE INTO products
               (id, name, provider, flat_fee_pct, tenor_days, max_amount, min_coverage_ratio, description)
               VALUES (?,?,?,?,?,?,?,?)""",
            (p["id"], p["name"], p["provider"], p["flat_fee_pct"], p["tenor_days"],
             p["max_amount"], p["min_coverage_ratio"], p["description"]),
        )
    conn.commit()


def list_products(conn) -> list:
    return [{**dict(r), **RULES.get(r["id"], {})}
            for r in conn.execute("SELECT * FROM products ORDER BY tenor_days")]


def get_product(conn, product_id: str) -> dict:
    row = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
    if not row:
        raise KeyError(f"Unknown product: {product_id}")
    return {**dict(row), **RULES.get(product_id, {})}
