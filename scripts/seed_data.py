"""Generate the demo database and print a diagnostic summary.

Usage:
    python scripts/seed_data.py [--db path] [--as-of YYYY-MM-DD] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cashbridge import config, db, evaluate, seed, util
from cashbridge.ingest import export_sample_csv
from cashbridge.recommend import recommend
from app import service


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="SQLite path (default: CASHBRIDGE_DB)")
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--if-empty", action="store_true",
                        help="do nothing when the database already holds businesses (container start)")
    parser.add_argument("--samples-dir", default=str(config.BASE_DIR / "data" / "samples"))
    args = parser.parse_args(argv)

    as_of = util.parse_date(args.as_of) if args.as_of else util.today()
    conn = db.connect(args.db)
    db.init_db(conn)
    if args.if_empty and db.list_businesses(conn):
        print("database already seeded; skipping")
        return 0
    db.reset_all(conn)  # idempotent regeneration: wipe previous demo state first
    ids = seed.run_seed(conn, as_of=as_of, seed=args.seed)
    from app import auth
    auth.ensure_demo_owner(conn)  # the demo login owns the regenerated businesses
    for business_id in ids.values():
        # onboarding: the owner grants data-use consent before any analysis
        service.grant_consent(conn, business_id, actor="owner")

    print(f"CashBridge seed - as_of {as_of.isoformat()}, seed {args.seed}")
    print("=" * 76)
    for key, business_id in ids.items():
        business = db.get_business(conn, business_id)
        balance = db.opening_balance(conn, business_id, as_of)
        receivables = db.open_receivables(conn, business_id)
        rec_total = sum(r["amount"] for r in receivables)

        result = recommend(conn, business_id, as_of=as_of)
        forecast = result["forecast"]
        min_low = min(r["balance_low"] for r in forecast["days"])
        min_mid = min(r["balance_mid"] for r in forecast["days"])
        min_high = min(r["balance_high"] for r in forecast["days"])

        actual = evaluate.ground_truth_balance_path(conn, business_id, as_of)
        coverage = evaluate.band_coverage(forecast["days"], actual)
        gt_gap = evaluate.ground_truth_gap(conn, business_id, as_of, business["safety_balance"])
        gt_min = min(p["balance"] for p in actual)

        print(f"\n[{key}] {business['name']}  (sector: {business['sector']})")
        print(f"  balance today            : {util.fmt_money(balance)}")
        print(f"  safety threshold         : {util.fmt_money(business['safety_balance'])}")
        print(f"  data-use consent         : "
              f"{'active' if service.consent_active(conn, business_id) else 'revoked'}")
        print(f"  open receivables         : {len(receivables)} worth {util.fmt_money(rec_total)}")
        print(f"  forecast min low/mid/high: {util.fmt_money(min_low)} | {util.fmt_money(min_mid)} | {util.fmt_money(min_high)}")
        if result["gap"]:
            gap = result["gap"]
            print(f"  gap detected             : {gap['start_date']}..{gap['end_date']} "
                  f"({gap['severity']}, depth {util.fmt_money(gap['depth'])}, shortfall {util.fmt_money(gap['shortfall'])})")
        else:
            print("  gap detected             : none")
        print(f"  actual (GT) min balance  : {util.fmt_money(gt_min)}  (GT gap: {gt_gap['has_gap']})")
        print(f"  band coverage vs GT      : {coverage['coverage']:.0%} "
              f"({coverage['days_inside']}/{coverage['days_checked']} days)")
        print(f"  decision                 : {result['decision']}")
        if result["non_loan"]:
            top = result["non_loan"][0]
            print(f"  top non-loan option      : {top['title']} "
                  f"(improvement {util.fmt_money(top['impact']['improvement'])}, clears gap: {top['impact']['clears_gap']})")
        if result["loan"] and result["loan"]["eligible"]:
            best = result["loan"]["eligible"][0]
            print(f"  best loan                : {best['name']} {util.fmt_money(best['principal'])} / "
                  f"{best['tenor_days']}d, coverage {best['coverage_ratio']}x")
        elif result["loan"]:
            print(f"  loan blocked             : {result['loan']['block_reason']}")
        if result["message"]:
            print(f"  message                  : {result['message']}")

        sample_path = Path(args.samples_dir) / f"{key}_transactions.csv"
        export_sample_csv(conn, business_id, sample_path, as_of=as_of)

    print(f"\nSample upload CSVs written to {args.samples_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
