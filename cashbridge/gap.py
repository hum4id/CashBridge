"""Cash-gap detection against the business safety threshold, based on simulated paths.

Severity is set by the **path** breach probability p = P(min_t B_t < safety):
  * ``expected`` when p >= 0.30 (and the shortfall is material),
  * ``risk``     when 0.10 <= p < 0.30,
  * no gap below 0.10.
Depth is the 90th percentile of per-path shortfalls; funding adds a 10% buffer.
The window (start/end) is where the daily P10 sits below safety, so the chart and
the banner always agree.
"""
from __future__ import annotations

from . import config, util
from .forecast import P_ACT, P_WARN

MIN_MATERIAL_GAP = 500_000


def severity_of(p_breach: float, depth: float) -> str | None:
    if p_breach >= P_ACT and depth >= MIN_MATERIAL_GAP:
        return "expected"
    if p_breach >= P_WARN:
        return "risk"
    return None


def detect_gaps(conn, business_id: int, forecast: dict, safety_balance: float,
                persist: bool = True) -> list:
    days = forecast["days"]
    risk = forecast.get("risk") or {}
    p = float(risk.get("p_breach", 0.0))
    depth = float(risk.get("shortfall_q90", 0.0))
    if safety_balance != risk.get("safety_balance"):
        # Caller overrode the threshold: recompute from the stored daily bands.
        lows = [row["balance_low"] for row in days]
        depth = max(0.0, safety_balance - min(lows))
        below = [row["balance_low"] < safety_balance for row in days]
        p = max(p, 0.10 if any(below) else 0.0)
    severity = severity_of(p, depth)
    if severity is None:
        return []

    below_idx = [i for i, row in enumerate(days) if row["balance_low"] < safety_balance] or [0]
    first_breach = risk.get("first_breach_date")
    start = first_breach if first_breach and first_breach >= days[below_idx[0]]["date"] \
        else days[below_idx[0]]["date"]
    shortfall = util.round_100k(depth * 1.10) if depth > 0 else 0.0
    if shortfall and shortfall < config.MIN_LOAN_PRINCIPAL:
        shortfall = float(config.MIN_LOAN_PRINCIPAL)
    gap = {
        "business_id": business_id,
        "forecast_id": forecast.get("forecast_id"),
        "start_date": min(start, days[below_idx[0]]["date"]),
        "end_date": days[below_idx[-1]]["date"],
        "min_balance": util.money(min(row["balance_low"] for row in days)),
        "depth": util.money(depth),
        "duration_days": below_idx[-1] - below_idx[0] + 1,
        "shortfall": shortfall,
        "severity": severity,
        "p_breach": p,
        "first_breach_date": first_breach,
    }
    if persist:
        cur = conn.execute(
            """INSERT INTO gaps
               (business_id, forecast_id, start_date, end_date, min_balance, depth,
                duration_days, shortfall, severity)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (gap["business_id"], gap["forecast_id"], gap["start_date"], gap["end_date"],
             gap["min_balance"], gap["depth"], gap["duration_days"], gap["shortfall"],
             gap["severity"]),
        )
        gap["gap_id"] = cur.lastrowid
        conn.commit()
    return [gap]
