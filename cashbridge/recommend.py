"""Alternatives-first recommendation engine.

Order is enforced here in code, not in the LLM prompt:
  1. non-loan options (accelerate collections, postpone restock, cut spend),
     each simulated with the same deterministic forecast engine,
  2. loan options, sized proportionally to the gap and rejected when
     repayment capacity does not hold.

Every monetary figure returned here is computed, never generated.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from . import db, products, sizing, util
from .forecast import run_forecast
from .gap import detect_gaps


def _has_outflow_category(conn, business_id: int, category: str, as_of, window_days: int = 28) -> bool:
    row = conn.execute(
        """SELECT COUNT(*) AS n FROM transactions
           WHERE business_id = ? AND category = ? AND direction = 'out' AND date > ?""",
        (business_id, category,
         util.format_date(util.add_days(as_of, -window_days))),
    ).fetchone()
    return row["n"] > 0


def recommend(conn, business_id: int, as_of=None, safety_balance: Optional[float] = None,
              forecast: Optional[dict] = None, persist: bool = True) -> dict:
    as_of = util.parse_date(as_of) if as_of else util.today()
    business = db.get_business(conn, business_id)
    if business is None:
        raise KeyError(f"Unknown business: {business_id}")
    safety = float(safety_balance if safety_balance is not None else business["safety_balance"])

    forecast = forecast or run_forecast(conn, business_id, as_of=as_of, persist=persist)
    gaps = detect_gaps(conn, business_id, forecast, safety, persist=persist)
    base_min_low = min(row["balance_low"] for row in forecast["days"])

    result = {
        "business_id": business_id,
        "business_name": business["name"],
        "as_of": forecast["as_of"],
        "safety_balance": safety,
        "forecast": forecast,
        "gap": None,
        "non_loan": [],
        "loan": None,
        "decision": "no_loan_needed",
        "message": None,
    }

    if not gaps:
        result["message"] = (
            f"No cash gap projected in the next {forecast['horizon_days']} days: the lowest "
            f"projected balance ({util.fmt_money(base_min_low)}) stays above the safety "
            f"threshold ({util.fmt_money(safety)})."
        )
        return result

    gap = gaps[0]
    result["gap"] = gap
    horizon = forecast["horizon_days"]
    receivables = db.open_receivables(conn, business_id)
    due_within = [r for r in receivables
                  if 1 <= (util.parse_date(r["expected_settlement_date"]) - as_of).days <= horizon]

    # ------------------------------------------------- non-loan options
    p_before = float(gap.get("p_breach", 0.0))
    depth_before = max(0.0, safety - base_min_low)   # same definition as depth_after

    def _simulate(adjustments: dict) -> dict:
        sim = run_forecast(conn, business_id, as_of=as_of, horizon_days=horizon,
                           adjustments=adjustments, persist=False)
        new_min = min(row["balance_low"] for row in sim["days"])
        p_after = float(sim["risk"]["p_breach"])
        depth_after = max(0.0, safety - new_min)
        return {
            "new_min_low": util.money(new_min),
            "improvement": util.money(max(0.0, new_min - base_min_low)),
            "clears_gap": new_min >= safety,
            "p_breach_after": round(p_after, 4),
            "p_breach_drop_pts": round(100 * (p_before - p_after), 1),
            # material = cuts breach probability by >= 5 points or the gap depth by >= 10%
            "material": (p_before - p_after) >= 0.05
                        or (depth_before > 0 and depth_after <= 0.9 * depth_before),
        }

    options = []
    # v2 targeted collection: receivables expected to arrive late AND after the gap starts
    gap_off = max(1, util.days_between(as_of, gap["start_date"]))
    paths_fc = run_forecast(conn, business_id, as_of=as_of, horizon_days=horizon,
                            persist=False, return_paths=True)
    late = []
    for j, r in enumerate(paths_fc["_receivables"]):
        due_off = util.days_between(as_of, r["expected_settlement_date"])
        median_arrival = float(sorted(paths_fc["_arrivals"][:, j])[len(paths_fc["_arrivals"]) // 2])
        if median_arrival > max(due_off, 1) and median_arrival >= gap_off and due_off <= gap_off + 7:
            late.append(r)
    if late:
        late_total = sum(float(r["amount"]) for r in late)
        options.append({
            "id": "collect_late_receivables",
            "title": "Chase the invoices that are likely to pay late",
            "detail": (f"{len(late)} invoices worth {util.fmt_money(late_total)} are predicted to arrive "
                       f"after their due date and after the gap begins. Simulation: they are collected "
                       f"on the due date."),
            "receivable_ids": [r["id"] for r in late],
            "impact": _simulate({"collect_on_due_ids": [r["id"] for r in late]}),
        })
    options.append({
        "id": "defer_bills",
        "title": "Ask to pay bills due before the gap 5 days later",
        "detail": ("Recurring bills due before the gap move 5 days later. Payroll and tax are never "
                   "deferred."),
        "defer_days": 5,
        "impact": _simulate({"defer_bills_days": 5, "defer_before_offset": gap_off}),
    })
    if due_within:
        total_due = sum(float(r["amount"]) for r in due_within)
        options.append({
            "id": "accelerate_collections",
            "title": "Accelerate receivables collection",
            "detail": (f"{len(due_within)} payouts worth {util.fmt_money(total_due)} are expected "
                       f"within {horizon} days. Simulation: the largest debtors settle 3 days earlier."),
            "impact": _simulate({"payout_shift_days": -3}),
        })
    if _has_outflow_category(conn, business_id, "inventory", as_of):
        options.append({
            "id": "delay_restock",
            "title": "Postpone restocking by one week",
            "detail": "Shift discretionary stock purchases 7 days later and run on current stock.",
            "impact": _simulate({"inventory_delay_days": 7}),
        })
    if _has_outflow_category(conn, business_id, "other_opex", as_of):
        options.append({
            "id": "cut_discretionary",
            "title": "Cut discretionary spending by 40%",
            "detail": "Temporarily reduce ads and non-essential spending by 40% for two weeks.",
            "reduction_pct": 40,   # a computed parameter, so prose citing "40%" stays grounded
            "impact": _simulate({"category_factors": {"other_opex": 0.6}}),
        })

    options = [o for o in options if o["impact"]["material"]]
    options.sort(key=lambda o: (o["impact"]["p_breach_after"], -o["impact"]["improvement"]))
    for rank, option in enumerate(options, start=1):
        option["rank"] = rank
    result["non_loan"] = options

    if not forecast.get("data_sufficient", True):
        result["decision"] = "insufficient_data"
        result["loan"] = None
        result["message"] = (
            f"Only {forecast['history_days']} days of history: at least 8 weeks are needed before "
            f"financing can be assessed. The forecast is shown with low confidence."
        )
        return result

    # ------------------------------------------------- loan options
    products.ensure_products(conn)
    principal = sizing.size_principal(gap)
    evaluated = [sizing.evaluate_product(conn, business_id, gap, p, as_of)
                 for p in products.list_products(conn)]
    eligible = sorted((e for e in evaluated if e["eligible"]),
                      key=lambda e: e["total_repayment"])
    result["loan"] = {
        "principal_needed": principal,
        "products": evaluated,
        "eligible": eligible,
        "blocked": not eligible,
        "block_reason": None if eligible else _block_reason(evaluated),
    }

    if eligible:
        result["decision"] = "loan_available"
        best = eligible[0]
        result["message"] = (
            f"Cash gap of {util.fmt_money(gap['depth'])} projected starting {gap['start_date']}. "
            f"A {util.fmt_money(principal)} loan over {best['tenor_days']} days is affordable "
            f"(coverage {best['coverage_ratio']}x), but non-loan options should be considered first."
        )
    else:
        result["decision"] = "loan_blocked"
        result["message"] = (
            f"Loan not recommended: {result['loan']['block_reason']}. "
            f"Non-loan options above are the priority."
        )
    return result


def _block_reason(evaluated: list) -> str:
    sim_fails = [e for e in evaluated if e.get("reason_code") in ("repay_prob", "does_not_help")]
    if sim_fails:
        best = max(sim_fails, key=lambda e: e["repay_prob"] or 0)
        return best["reason"].rstrip(".")
    capacity_fails = [e for e in evaluated if e.get("reason_code") == "capacity"]
    if capacity_fails:
        best = max(capacity_fails, key=lambda e: e["coverage_ratio"])
        return (f"projected repayment capacity covers only {best['coverage_ratio']}x of "
                f"installments over {best['tenor_days']} days "
                f"(at least {best['min_coverage_ratio']}x required)")
    return "no financing product fits the required amount"
