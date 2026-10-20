"""Honest evaluation of the forecast, gap detector and loan sizer.

Three experiments, all out-of-sample (the model only ever sees history up to as_of):

  A. In-distribution: synthetic businesses, future drawn from the same generator.
  B. Adversarial: the same history, but the future breaks the model's assumptions:
     Student-t(3) shocks on sales, a -25% sales regime shift on a random day,
     one surprise expense the model cannot know about, and 10% of customer
     payments arriving 30 days late.
  C. Real data: daily sales of a real online retailer (UCI Online Retail II,
     CC BY 4.0, 2009-12..2011-12) used as the cash-sales stream of a business with
     fixed weekly costs; 14-day backtests from many origins.

Metrics: P10 / P90 one-sided coverage (target 0.90 each), P10-P90 band coverage
(target 0.80), pinball loss of the median vs a seasonal-naive baseline (repeat
last week's daily net flow), gap detection precision/recall with Wilson 95% CIs,
Brier score of p_breach vs a constant base-rate forecast, and loan sufficiency.

Usage: python scripts/evaluate_forecast.py [--dates 8] [--seeds 42,7,11,23] [--quick]
Writes reports/eval_report.md and reports/eval_results.json.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cashbridge import db, evaluate, seed, util  # noqa: E402
from cashbridge.forecast import P_ACT, run_forecast  # noqa: E402
from cashbridge.recommend import recommend  # noqa: E402

H = 14
INTENDED = {"sembako": "no_loan_needed", "katering": "loan_blocked", "online": "loan_available"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(centre - half, 3), round(centre + half, 3))


def pinball(y, q, tau=0.5) -> float:
    d = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def seasonal_naive(conn, bid, as_of, opening) -> list:
    """Repeat the last 7 days of net cash flow (a standard naive baseline)."""
    nets = []
    for k in range(7, 0, -1):
        d = (as_of - timedelta(days=k - 1)).isoformat()
        row = conn.execute("""SELECT COALESCE(SUM(CASE WHEN direction='in' THEN amount ELSE -amount END),0) n
                              FROM transactions WHERE business_id=? AND date=?
                              AND category NOT IN ('financing','other_income')""", (bid, d)).fetchone()
        nets.append(float(row["n"]))
    path, bal = [], opening
    for t in range(H):
        bal += nets[t % 7]
        path.append(bal)
    return path


# ------------------------------------------------------------- adversarial

def perturb_future(conn, rng: np.random.Generator, as_of: date) -> None:
    for (bid,) in conn.execute("SELECT id FROM businesses").fetchall():
        rows = conn.execute("""SELECT id, date, amount, direction, category, description
                               FROM ground_truth_transactions WHERE business_id = ?""", (bid,)).fetchall()
        shift_day = int(rng.integers(1, H + 1))
        rev14 = 0.0
        for r in rows:
            off = util.days_between(as_of, r["date"])
            amt = float(r["amount"])
            if r["direction"] == "in" and r["category"] == "revenue":
                if r["description"].startswith("Credit payment") and rng.random() < 0.10:
                    newd = util.add_days(r["date"], 30).isoformat()   # 30 days late
                    conn.execute("UPDATE ground_truth_transactions SET date=? WHERE id=?", (newd, r["id"]))
                    continue
                shock = max(0.05, 1 + 0.25 * rng.standard_t(3) / math.sqrt(3))
                amt *= shock * (0.75 if off >= shift_day else 1.0)
                if off <= H:
                    rev14 += amt
                conn.execute("UPDATE ground_truth_transactions SET amount=? WHERE id=?", (round(amt), r["id"]))
        surprise_day = util.add_days(as_of, int(rng.integers(1, H + 1))).isoformat()
        conn.execute("""INSERT INTO ground_truth_transactions (business_id, date, description, counterparty,
                        amount, direction, category, is_cash_sale)
                        VALUES (?,?,?,?,?,'out','other_opex',0)""",
                     (bid, surprise_day, "Surprise repair", "Vendor", round(0.15 * rev14)))
    conn.commit()


def run_synthetic(dates: list, seeds: list, adversarial: bool) -> dict:
    rows = []
    for s in seeds:
        rng = np.random.default_rng(s * 7919)
        for as_of in dates:
            conn = db.connect(":memory:")
            db.init_db(conn)
            ids = seed.run_seed(conn, as_of=as_of, seed=s)
            if adversarial:
                perturb_future(conn, rng, as_of)
            for key, bid in ids.items():
                business = db.get_business(conn, bid)
                safety = float(business["safety_balance"])
                fc = run_forecast(conn, bid, as_of=as_of, horizon_days=H, persist=False)
                actual = [r["balance"] for r in evaluate.ground_truth_balance_path(conn, bid, as_of, H)]
                lo = [d["balance_low"] for d in fc["days"]]
                mid = [d["balance_mid"] for d in fc["days"]]
                hi = [d["balance_high"] for d in fc["days"]]
                naive = seasonal_naive(conn, bid, as_of, fc["opening_balance"])
                rec = recommend(conn, bid, as_of=as_of, forecast=fc, persist=False)
                min_actual = min(actual)
                principal = None
                if rec["decision"] == "loan_available":
                    principal = rec["loan"]["principal_needed"]
                rows.append({
                    "business": key, "seed": s, "as_of": as_of.isoformat(),
                    "p10_cov": float(np.mean(np.array(actual) >= np.array(lo))),
                    "p90_cov": float(np.mean(np.array(actual) <= np.array(hi))),
                    "band_cov": float(np.mean((np.array(actual) >= np.array(lo)) & (np.array(actual) <= np.array(hi)))),
                    "pinball_model": pinball(actual, mid), "pinball_naive": pinball(actual, naive),
                    "p_breach": fc["risk"]["p_breach"],
                    "pred_act": bool(rec["gap"] and rec["gap"]["severity"] == "expected"),
                    "pred_any": bool(rec["gap"]),
                    "actual_breach": bool(min_actual < safety),
                    "actual_shortfall": max(0.0, safety - min_actual),
                    "decision": rec["decision"], "intended": INTENDED[key],
                    "principal": principal,
                })
            conn.close()
    return summarise(rows)


def summarise(rows: list) -> dict:
    n = len(rows)
    out = {"n": n}
    for k in ("p10_cov", "p90_cov", "band_cov"):
        out[k] = round(statistics.mean(r[k] for r in rows), 3)
    out["pinball_model"] = round(statistics.mean(r["pinball_model"] for r in rows))
    out["pinball_naive"] = round(statistics.mean(r["pinball_naive"] for r in rows))
    if "actual_breach" in rows[0]:
        pos = [r for r in rows if r["actual_breach"]]
        base = len(pos) / n
        for name, key in (("act", "pred_act"), ("any", "pred_any")):
            tp = sum(1 for r in rows if r[key] and r["actual_breach"])
            pp = sum(1 for r in rows if r[key])
            out[f"precision_{name}"] = (round(tp / pp, 3) if pp else None, wilson(tp, pp))
            out[f"recall_{name}"] = (round(tp / len(pos), 3) if pos else None, wilson(tp, len(pos)))
        out["brier_model"] = round(statistics.mean((r["p_breach"] - r["actual_breach"]) ** 2 for r in rows), 3)
        out["brier_base_rate"] = round(statistics.mean((base - r["actual_breach"]) ** 2 for r in rows), 3)
        out["breach_rate"] = round(base, 3)
        loans = [r for r in rows if r["principal"] and r["actual_breach"]]
        if loans:
            ratios = sorted(r["principal"] / r["actual_shortfall"] for r in loans if r["actual_shortfall"] > 0)
            out["loan_cover_ratio_median"] = round(statistics.median(ratios), 2) if ratios else None
            out["loan_sufficient_share"] = round(
                sum(1 for r in loans if r["principal"] >= r["actual_shortfall"]) / len(loans), 3)
        out["loan_offers_without_actual_breach"] = sum(1 for r in rows if r["principal"] and not r["actual_breach"])
        out["story_hit"] = round(sum(r["decision"] == r["intended"] for r in rows) / n, 3)
        out["by_business"] = {b: {k: round(statistics.mean(r[k] for r in rows if r["business"] == b), 3)
                                  for k in ("band_cov", "p10_cov", "p90_cov")}
                              for b in sorted({r["business"] for r in rows})}
    return out


# --------------------------------------------------------------- real data

def run_real(step_days: int = 14) -> dict:
    import csv
    path = ROOT / "data" / "external" / "online_retail_ii_daily.csv"
    if not path.exists():
        return {"error": f"{path} missing; run scripts/fetch_datasets.py"}
    series = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            series[date.fromisoformat(r["date"][:10])] = float(r["sales"]) - float(r["refunds"])
    days = sorted(series)
    SCALE = 20_000.0                           # GBP -> IDR-like magnitude; scale-free metrics
    mean_daily = statistics.mean(series.values()) * SCALE
    weekly_cost = 0.9 * 7 * mean_daily          # costs ~90% of average sales: thin margin
    rows = []
    origin = days[0] + timedelta(days=150)
    while origin + timedelta(days=H) <= days[-1]:
        conn = db.connect(":memory:")
        db.init_db(conn)
        bid = conn.execute("INSERT INTO businesses (name, sector, safety_balance) VALUES (?,?,?)",
                           ("Real retailer", "retail", 0)).lastrowid
        start = origin - timedelta(days=150)
        conn.execute("""INSERT INTO transactions (business_id, date, description, counterparty, amount,
                        direction, category, is_cash_sale, source)
                        VALUES (?,?,?,?,?,'in','other_income',0,'eval')""",
                     (bid, start.isoformat(), "Opening balance", "Owner", 30 * mean_daily))
        d = start
        while d <= origin + timedelta(days=H):
            table = "transactions" if d <= origin else "ground_truth_transactions"
            extra = ", source" if table == "transactions" else ""
            qs = ",?" if table == "transactions" else ""
            sales = series.get(d, 0.0) * SCALE
            if sales > 0:
                conn.execute(f"""INSERT INTO {table} (business_id, date, description, counterparty, amount,
                                 direction, category, is_cash_sale{extra}) VALUES (?,?,?,?,?,'in','revenue',1{qs})""",
                             (bid, d.isoformat(), "Daily sales", "Customers", round(sales), *(["eval"] if qs else [])))
            elif sales < 0:
                conn.execute(f"""INSERT INTO {table} (business_id, date, description, counterparty, amount,
                                 direction, category, is_cash_sale{extra}) VALUES (?,?,?,?,?,'out','other_opex',0{qs})""",
                             (bid, d.isoformat(), "Refunds", "Customers", round(-sales), *(["eval"] if qs else [])))
            if d.weekday() == 0:
                conn.execute(f"""INSERT INTO {table} (business_id, date, description, counterparty, amount,
                                 direction, category, is_cash_sale{extra}) VALUES (?,?,?,?,?,'out','inventory',0{qs})""",
                             (bid, d.isoformat(), "Weekly stock + wages", "Supplier", round(weekly_cost),
                              *(["eval"] if qs else [])))
            d += timedelta(days=1)
        conn.commit()
        fc = run_forecast(conn, bid, as_of=origin, horizon_days=H, persist=False)
        actual = [r["balance"] for r in evaluate.ground_truth_balance_path(conn, bid, origin, H)]
        lo = np.array([x["balance_low"] for x in fc["days"]])
        hi = np.array([x["balance_high"] for x in fc["days"]])
        mid = [x["balance_mid"] for x in fc["days"]]
        a = np.array(actual)
        rows.append({"p10_cov": float(np.mean(a >= lo)), "p90_cov": float(np.mean(a <= hi)),
                     "band_cov": float(np.mean((a >= lo) & (a <= hi))),
                     "pinball_model": pinball(actual, mid),
                     "pinball_naive": pinball(actual, seasonal_naive(conn, bid, origin, fc["opening_balance"])),
                     "origin": origin.isoformat()})
        conn.close()
        origin += timedelta(days=step_days)
    out = summarise(rows)
    out["origins"] = len(rows)
    out["worst_band_cov"] = round(min(r["band_cov"] for r in rows), 3)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", type=int, default=8)
    ap.add_argument("--seeds", default="42,7,11,23")
    ap.add_argument("--quick", action="store_true", help="2 dates x 1 seed, real data every 28 days")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    n_dates = args.dates
    if args.quick:
        seeds, n_dates = seeds[:1], 2
    dates = [date(2026, 10, 2) + timedelta(days=7 * i) for i in range(n_dates)]   # Fridays
    results = {
        "A_in_distribution": run_synthetic(dates, seeds, adversarial=False),
        "B_adversarial": run_synthetic(dates, seeds, adversarial=True),
        "C_real_online_retail_ii": run_real(28 if args.quick else 14),
        "config": {"dates": [d.isoformat() for d in dates], "seeds": seeds, "horizon": H, "P_ACT": P_ACT},
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "eval_results.json").write_text(json.dumps(results, indent=2, default=str))
    (reports / "eval_report.md").write_text(render(results))
    print(render(results))
    return 0


def _fmt(v):
    if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[1], (list, tuple)):
        return f"{v[0]} [{v[1][0]}–{v[1][1]}]"
    return str(v)


def render(r: dict) -> str:
    keys = ["n", "band_cov", "p10_cov", "p90_cov", "pinball_model", "pinball_naive", "precision_act",
            "recall_act", "precision_any", "recall_any", "brier_model", "brier_base_rate", "breach_rate",
            "loan_cover_ratio_median", "loan_sufficient_share", "loan_offers_without_actual_breach",
            "story_hit", "origins", "worst_band_cov"]
    cols = ["A_in_distribution", "B_adversarial", "C_real_online_retail_ii"]
    lines = ["# Forecast evaluation (generated by scripts/evaluate_forecast.py)", "",
             f"Config: {json.dumps(r['config'])}", "",
             "| metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(_fmt(r[c].get(k, "")) for c in cols) + " |")
    lines += ["", "Targets: band_cov 0.80, p10_cov 0.90, p90_cov 0.90. Pinball in IDR (lower is better).",
              "Precision/recall shown with Wilson 95% intervals. 'act' = gap flagged as expected "
              "(p_breach >= 0.30 and depth >= IDR 500k); 'any' = any flagged gap (p_breach >= 0.10).",
              "C uses real daily sales (UCI Online Retail II, CC BY 4.0, doi:10.24432/C5CG6D) with "
              "synthetic fixed costs; it tests the sales-noise model only, not receivables.", ""]
    for c in cols[:2]:
        if "by_business" in r[c]:
            lines.append(f"{c} by business: {json.dumps(r[c]['by_business'])}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
