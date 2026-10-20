"""Loan sizing, repayment schedules, and repayment checks.

A product is eligible only when ALL of these hold (every number is computed here):
  1. principal <= product cap,
  2. affordability: recent weekly net cash x tenor covers the installments
     ``min_coverage_ratio`` times (historical sanity check),
  3. repayment probability: simulating the loan inside the Monte Carlo forecast
     (disbursement in, every installment out), the balance stays >= 0 on every
     installment date in at least ``REPAY_MIN_PROB`` of paths,
  4. the loan actually helps: P(breach) over the 14-day horizon after the loan is
     below ``forecast.P_ACT``.
Thresholds are design parameters shown to the owner, not industry standards.
"""
from __future__ import annotations

import math
from datetime import timedelta
from typing import Optional

from . import config, db, util

REPAY_MIN_PROB = 0.80


def size_principal(gap: dict, max_principal: Optional[float] = None) -> float:
    cap = max_principal if max_principal is not None else config.settings.sandbox_max_principal
    principal = max(float(gap["shortfall"]), float(config.MIN_LOAN_PRINCIPAL))
    principal = util.round_100k(principal)
    return min(principal, util.round_100k(cap))


def repayment_total(principal: float, flat_fee_pct: float) -> float:
    return util.money(principal * (1.0 + flat_fee_pct / 100.0))


def recent_weekly_net(conn, business_id: int, as_of,
                      window_days: Optional[int] = None) -> float:
    """Average weekly net operating cash flow over the recent baseline window."""
    window = int(window_days or config.HISTORY_BASELINE_DAYS)
    start = util.parse_date(as_of) - timedelta(days=window)
    row = conn.execute(
        """SELECT COALESCE(SUM(CASE WHEN direction = 'in' THEN amount ELSE -amount END), 0) AS net
           FROM transactions
           WHERE business_id = ? AND date > ? AND date <= ?
             AND category NOT IN ('financing', 'other_income')""",
        (business_id, util.format_date(start), util.format_date(as_of)),
    ).fetchone()
    return float(row["net"]) / (window / 7.0)


def coverage_ratio(weekly_net: float, tenor_days: int, repayment: float) -> float:
    if repayment <= 0:
        return 0.0
    return round(weekly_net * (tenor_days / 7.0) / repayment, 2)


def _nearest_receivable(receivables: list, due, used: set) -> Optional[dict]:
    best, best_delta = None, None
    for r in receivables:
        if r["id"] in used:
            continue
        expected = util.parse_date(r["expected_settlement_date"])
        delta = abs((expected - due).days)
        if delta <= 21 and (best_delta is None or delta < best_delta):
            best, best_delta = r, delta
    return best


def build_schedule(principal: float, product: dict, start, receivables: list) -> list:
    """Weekly installments, each linked to the nearest expected receivable."""
    total = repayment_total(principal, product["flat_fee_pct"])
    installments = max(1, math.ceil(product["tenor_days"] / 7))
    base = util.round_1k(total / installments)
    start = util.parse_date(start)
    used: set = set()
    schedule = []
    remaining = total
    for i in range(1, installments + 1):
        due = start + timedelta(days=7 * i)
        amount = base if i < installments else util.money(remaining)  # schedule sums to the disclosed total
        link = _nearest_receivable(receivables, due, used)
        if link:
            used.add(link["id"])
        schedule.append({
            "due_date": due.isoformat(),
            "amount": amount,
            "linked_receivable_id": link["id"] if link else None,
            "linked_debtor": link["debtor"] if link else None,
        })
        remaining -= amount
    return schedule


def repayment_simulation(conn, business_id: int, principal: float, schedule: list, as_of,
                         horizon: int) -> dict:
    """Run the forecast with the loan in it; return repayment probability and new risk."""
    from .forecast import run_forecast  # local import: forecast imports nothing from sizing
    as_of = util.parse_date(as_of)
    flows = [(1, float(principal))]
    due_offsets = []
    for inst in schedule:
        off = util.days_between(as_of, inst["due_date"])
        flows.append((off, -float(inst["amount"])))
        due_offsets.append(off)
    sim_h = max(horizon, max(due_offsets, default=horizon))
    fc = run_forecast(conn, business_id, as_of=as_of, horizon_days=sim_h, extra_flows=flows,
                      persist=False, return_paths=True)
    paths = fc["_paths"]
    cols = [o - 1 for o in due_offsets if 1 <= o <= sim_h]
    repay_ok = (paths[:, cols] >= 0).all(axis=1) if cols else (paths >= 0).all(axis=1)
    safety = fc["risk"]["safety_balance"]
    p_after = float((paths[:, :horizon].min(axis=1) < safety).mean())
    return {"repay_prob": round(float(repay_ok.mean()), 4), "p_breach_after": round(p_after, 4)}


def cost_disclosure(principal: float, schedule: list, as_of, fee_pct: float) -> dict:
    """Comprehensive financing-cost disclosure for one offer.

    The annualised cost is the internal rate of return of the actual cash flows
    (principal out on day 0, each installment on its due date), so the fee and its
    timing are both reflected. ``apr_nominal`` = daily IRR x 365; ``apr_effective``
    = (1 + daily IRR)^365 - 1. The cap check uses the higher (effective) figure.
    """
    as_of = util.parse_date(as_of)
    flows = [(util.days_between(as_of, i["due_date"]), float(i["amount"])) for i in schedule]
    total = sum(a for _, a in flows)
    if not flows or principal <= 0 or total <= principal:
        r = 0.0
    else:
        lo, hi = 0.0, 0.05
        for _ in range(100):           # bisection on the daily rate
            mid = (lo + hi) / 2
            pv = sum(a / (1 + mid) ** d for d, a in flows)
            lo, hi = (mid, hi) if pv > principal else (lo, mid)
        r = (lo + hi) / 2
    cap, juris = config.active_rate_cap()
    apr_eff = (1 + r) ** 365 - 1
    return {
        "principal": util.money(principal),
        "interest": 0.0,
        "fees": [{"name": "Flat facility fee", "amount": util.money(total - principal),
                  "rate_pct_of_principal": fee_pct, "charged_by": "Sandbox Bank",
                  "collected": "spread over installments"}],
        "total_repayment": util.money(total),
        "total_cost": util.money(total - principal),
        "apr_nominal": round(r * 365, 4),
        "apr_effective": round(apr_eff, 4),
        "contingent_costs": "None in the sandbox: no late fee, no prepayment penalty.",
        "rate_cap": cap,
        "rate_cap_jurisdiction": juris,
        "within_cap": apr_eff <= cap,
        "no_other_fees": True,
    }


def _weekly_inflow_cv(conn, business_id: int, as_of, weeks: int = 8) -> Optional[float]:
    as_of = util.parse_date(as_of)
    sums = []
    for w in range(weeks):
        end = as_of - timedelta(days=7 * w)
        start = end - timedelta(days=7)
        row = conn.execute(
            """SELECT COALESCE(SUM(amount), 0) s FROM transactions WHERE business_id = ?
               AND direction = 'in' AND category = 'revenue' AND date > ? AND date <= ?""",
            (business_id, start.isoformat(), end.isoformat())).fetchone()
        sums.append(float(row["s"]))
    mean = sum(sums) / len(sums)
    if mean <= 0:
        return None
    var = sum((x - mean) ** 2 for x in sums) / (len(sums) - 1)
    return round(var ** 0.5 / mean, 3)


def _inflow_30d(conn, business_id: int, as_of) -> float:
    as_of = util.parse_date(as_of)
    row = conn.execute(
        """SELECT COALESCE(SUM(amount), 0) s FROM transactions WHERE business_id = ?
           AND direction = 'in' AND category = 'revenue' AND date > ? AND date <= ?""",
        (business_id, (as_of - timedelta(days=30)).isoformat(), as_of.isoformat())).fetchone()
    return float(row["s"])


def _has_active_loan(conn, business_id: int) -> bool:
    row = conn.execute("""SELECT COUNT(*) n FROM loans l JOIN offers o ON o.id = l.offer_id
                          WHERE o.business_id = ? AND l.status = 'active'""", (business_id,)).fetchone()
    return row["n"] > 0


def _history_days(conn, business_id: int, as_of) -> int:
    row = conn.execute("SELECT MIN(date) m FROM transactions WHERE business_id = ? AND date <= ?",
                       (business_id, util.format_date(as_of))).fetchone()
    return util.days_between(row["m"], as_of) if row["m"] else 0


def _repayment_source(conn, business_id: int, gap: dict, tenor_days: int, as_of) -> float:
    """Open receivables expected after the gap starts and within the tenor."""
    start = util.parse_date(gap["start_date"])
    end = util.parse_date(as_of) + timedelta(days=tenor_days)
    return sum(float(r["amount"]) for r in db.open_receivables(conn, business_id)
               if start <= util.parse_date(r["expected_settlement_date"]) <= end)


def evaluate_product(conn, business_id: int, gap: dict, product: dict, as_of) -> dict:
    """Size the loan for this gap and check it against one product's rules."""
    principal = size_principal(gap)
    total = repayment_total(principal, product["flat_fee_pct"])
    weekly = recent_weekly_net(conn, business_id, as_of)
    coverage = coverage_ratio(weekly, product["tenor_days"], total)
    p_before = float(gap.get("p_breach", 0.0))

    eligible, reason, code = True, None, "eligible"
    repay = {"repay_prob": None, "p_breach_after": None}
    receivables = db.open_receivables(conn, business_id)
    draft_schedule = build_schedule(principal, product, as_of, receivables)
    disclosure = cost_disclosure(principal, draft_schedule, as_of, product["flat_fee_pct"])
    history = _history_days(conn, business_id, as_of)
    cv = _weekly_inflow_cv(conn, business_id, as_of)
    inflow30 = _inflow_30d(conn, business_id, as_of)
    per_30d = total / max(product["tenor_days"], 1) * 30
    source = _repayment_source(conn, business_id, gap, product["tenor_days"], as_of)

    if _has_active_loan(conn, business_id):
        eligible, code = False, "active_loan"
        reason = "An earlier sandbox loan is still active; one loan at a time."
    elif history < product.get("min_history_days", 56):
        eligible, code = False, "insufficient_history"
        reason = (f"Only {history} days of history; this product needs "
                  f"{product.get('min_history_days', 56)}.")
    elif cv is None or cv > product.get("max_income_cv", 1.0):
        eligible, code = False, "income_unstable"
        reason = (f"Weekly income varies too much (coefficient of variation {cv}, "
                  f"limit {product.get('max_income_cv')}).")
    elif not disclosure["within_cap"]:
        eligible, code = False, "rate_cap"
        reason = (f"Annualised cost {disclosure['apr_effective']:.1%} exceeds the "
                  f"{disclosure['rate_cap_jurisdiction']} cap of {disclosure['rate_cap']:.0%}.")
    elif product.get("requires_repayment_source") and source < total:
        eligible, code = False, "no_repayment_source"
        reason = (f"Receivables expected after the gap and within {product['tenor_days']} days total "
                  f"{util.fmt_money(source)}, less than the repayment of {util.fmt_money(total)}.")
    elif inflow30 <= 0 or per_30d > config.DSR_MAX * inflow30:
        eligible, code = False, "dsr"
        reason = (f"Repayments of {util.fmt_money(per_30d)} per 30 days exceed "
                  f"{config.DSR_MAX:.0%} of the last 30 days' income ({util.fmt_money(inflow30)}).")
    elif principal > product["max_amount"]:
        eligible, code = False, "over_cap"
        reason = (f"Required principal {util.fmt_money(principal)} exceeds the product cap "
                  f"of {util.fmt_money(product['max_amount'])}.")
    elif coverage < product["min_coverage_ratio"]:
        eligible, code = False, "capacity"
        reason = (f"Repayment capacity too thin: projected net cash over {product['tenor_days']} days "
                  f"covers only {coverage}x of the installment total "
                  f"(needs {product['min_coverage_ratio']}x).")

    schedule = draft_schedule if eligible else []
    if eligible:
        repay = repayment_simulation(conn, business_id, principal, schedule, as_of,
                                     config.HORIZON_DAYS)
        if repay["repay_prob"] < REPAY_MIN_PROB:
            eligible, code = False, "repay_prob"
            reason = (f"In simulation the installments are paid on time in only "
                      f"{repay['repay_prob']:.0%} of cases (needs {REPAY_MIN_PROB:.0%}).")
        elif repay["p_breach_after"] >= min(0.30, max(p_before, 0.10)):
            eligible, code = False, "does_not_help"
            reason = (f"The loan does not reduce the shortfall risk enough "
                      f"({p_before:.0%} -> {repay['p_breach_after']:.0%}).")
        if not eligible:
            schedule = []

    return {
        "product_id": product["id"],
        "name": product["name"],
        "provider": product["provider"],
        "description": product["description"],
        "flat_fee_pct": product["flat_fee_pct"],
        "tenor_days": product["tenor_days"],
        "min_coverage_ratio": product["min_coverage_ratio"],
        "principal": principal,
        "total_repayment": total,
        "coverage_ratio": coverage,
        "weekly_net_baseline": util.money(weekly),
        "repay_prob": repay["repay_prob"],
        "p_breach_before": p_before,
        "p_breach_after": repay["p_breach_after"],
        "eligible": eligible,
        "reason": reason,
        "reason_code": code,
        "schedule": schedule,
        "cost_disclosure": disclosure,
        "checks": {"history_days": history, "income_cv": cv, "inflow_30d": util.money(inflow30),
                   "repayment_per_30d": util.money(per_30d), "repayment_source": util.money(source)},
    }
