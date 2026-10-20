"""Receivable reconciliation: match bank credits to open invoices.

Deterministic, explainable rules (no model):
  * candidate = an incoming transaction not yet linked to any receivable, dated on
    or after the invoice date, within ``WINDOW_DAYS`` after the expected date;
  * amount must match within ``AMOUNT_TOL`` (0.5%), because partial payments are
    a separate workflow;
  * confidence "high" when a debtor name token also appears in the description or
    counterparty, otherwise "medium";
  * one-to-one greedy assignment: high before medium, then closest to the expected
    date, so one credit can never settle two invoices.

Proposals change nothing. The owner confirms each match (``service.confirm_match``),
which marks the invoice settled and is written to the audit chain. Unreconciled
invoices matter for cash: the forecast still expects their money to arrive.
"""
from __future__ import annotations

import re
from datetime import timedelta

from . import util

AMOUNT_TOL = 0.005
WINDOW_DAYS = 60
_STOP = {"pt", "cv", "tbk", "the", "and", "of", "payout", "payment", "-", "&"}


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z]{3,}", (text or "").lower()) if t not in _STOP}


def propose_matches(conn, business_id: int, as_of=None) -> dict:
    as_of = util.parse_date(as_of) if as_of else util.today()
    receivables = [dict(r) for r in conn.execute(
        "SELECT * FROM receivables WHERE business_id = ? AND status = 'open' AND issued_date <= ?",
        (business_id, as_of.isoformat()))]
    credits = [dict(r) for r in conn.execute(
        """SELECT * FROM transactions WHERE business_id = ? AND direction = 'in' AND date <= ?
             AND category NOT IN ('financing', 'other_income') AND is_cash_sale = 0
             AND id NOT IN (SELECT settled_transaction_id FROM receivables
                            WHERE settled_transaction_id IS NOT NULL)""",
        (business_id, as_of.isoformat()))]

    candidates = []
    for r in receivables:
        issued = util.parse_date(r["issued_date"])
        latest = util.parse_date(r["expected_settlement_date"]) + timedelta(days=WINDOW_DAYS)
        names = _tokens(r["debtor"])
        for t in credits:
            d = util.parse_date(t["date"])
            if not (issued <= d <= latest):
                continue
            if abs(float(t["amount"]) - float(r["amount"])) > AMOUNT_TOL * float(r["amount"]):
                continue
            name_hit = bool(names & (_tokens(t["description"]) | _tokens(t.get("counterparty") or "")))
            candidates.append({
                "receivable_id": r["id"], "transaction_id": t["id"],
                "debtor": r["debtor"], "amount": float(r["amount"]),
                "invoice_expected": r["expected_settlement_date"], "paid_on": t["date"],
                "days_late": util.days_between(r["expected_settlement_date"], t["date"]),
                "description": t["description"],
                "confidence": "high" if name_hit else "medium",
                "_rank": (0 if name_hit else 1,
                          abs(util.days_between(r["expected_settlement_date"], t["date"]))),
            })
    candidates.sort(key=lambda c: c["_rank"])
    used_r, used_t, matches = set(), set(), []
    for c in candidates:
        if c["receivable_id"] in used_r or c["transaction_id"] in used_t:
            continue
        used_r.add(c["receivable_id"])
        used_t.add(c["transaction_id"])
        matches.append({k: v for k, v in c.items() if k != "_rank"})

    overdue = [{"receivable_id": r["id"], "debtor": r["debtor"], "amount": float(r["amount"]),
                "expected": r["expected_settlement_date"],
                "days_overdue": util.days_between(r["expected_settlement_date"], as_of)}
               for r in receivables
               if r["id"] not in used_r and util.parse_date(r["expected_settlement_date"]) < as_of]
    overdue.sort(key=lambda o: -o["days_overdue"])
    return {
        "as_of": as_of.isoformat(),
        "matches": matches,
        "matched_total": util.money(sum(m["amount"] for m in matches)),
        "overdue_unmatched": overdue,
        "overdue_total": util.money(sum(o["amount"] for o in overdue)),
        "unmatched_credits": [{"transaction_id": t["id"], "date": t["date"], "amount": float(t["amount"]),
                               "description": t["description"]}
                              for t in credits if t["id"] not in used_t][-10:],
    }
