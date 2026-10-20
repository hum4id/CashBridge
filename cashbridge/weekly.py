"""Week-over-week movement: what actually changed since last week.

The weekly delta view answers the returning owner's first question — "what's
different from last week?" — using booked transactions only. No LLM, no
projection; the numbers are sums of what happened.
"""
from __future__ import annotations

from datetime import date

from . import util

_MOVERS_LIMIT = 4


def _window_totals(conn, business_id: int, start, end) -> dict:
    rows = conn.execute(
        """SELECT direction, category, SUM(amount) AS total
           FROM transactions
           WHERE business_id = ? AND date > ? AND date <= ?
           GROUP BY direction, category""",
        (business_id, util.format_date(start), util.format_date(end)),
    ).fetchall()
    inflow = outflow = revenue = 0.0
    by_category: dict = {}
    for row in rows:
        total = float(row["total"] or 0.0)
        if row["direction"] == "in":
            inflow += total
            if row["category"] == "revenue":
                revenue += total
        else:
            outflow += total
            by_category[row["category"]] = by_category.get(row["category"], 0.0) + total
    return {
        "inflow": util.money(inflow),
        "outflow": util.money(outflow),
        "net": util.money(inflow - outflow),
        "revenue": util.money(revenue),
        "by_category": {k: util.money(v) for k, v in by_category.items()},
    }


def weekly_delta(conn, business_id: int, as_of=None) -> dict:
    """Compare the last 7 days with the 7 days before them."""
    as_of = util.parse_date(as_of) if as_of else util.today()
    current = _window_totals(conn, business_id, util.add_days(as_of, -7), as_of)
    previous = _window_totals(conn, business_id, util.add_days(as_of, -14),
                              util.add_days(as_of, -7))

    categories = set(current["by_category"]) | set(previous["by_category"])
    movers = []
    for category in categories:
        now = current["by_category"].get(category, 0.0)
        before = previous["by_category"].get(category, 0.0)
        if now or before:
            movers.append({"category": category, "current": now, "previous": before,
                           "change": util.money(now - before)})
    movers.sort(key=lambda m: abs(m["change"]), reverse=True)

    return {
        "as_of": as_of.isoformat(),
        "current": current,
        "previous": previous,
        "delta": {
            "net": util.money(current["net"] - previous["net"]),
            "inflow": util.money(current["inflow"] - previous["inflow"]),
            "outflow": util.money(current["outflow"] - previous["outflow"]),
            "revenue": util.money(current["revenue"] - previous["revenue"]),
        },
        "top_movers": movers[:_MOVERS_LIMIT],
    }
