"""End-to-end sandbox run that produces the submission's execution evidence.

Drives the real web app over HTTP (login, CSRF, owner approval) on a fresh demo
database and records every call. Writes to evidence/:
  operation_log.jsonl      one line per HTTP call: time, method, path, status, summary
  audit_log.json           the full hash-chained audit trail
  chain_verification.json  result of recomputing every hash
  sandbox_ledger.csv       simulated money movements (disbursement, repayments)
  summary.md               what happened, in order

Usage: python scripts/demo_run.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MOCK_LLM", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app import audit, service  # noqa: E402
from app.main import create_app  # noqa: E402
from cashbridge import db, seed, util  # noqa: E402

OUT = ROOT / "evidence"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp()) / "evidence.db"
    conn = db.connect(tmp)
    db.init_db(conn)
    ids = seed.run_seed(conn, as_of=None, seed=42)
    for bid in ids.values():
        service.grant_consent(conn, bid, actor="demo")
    online, katering = ids["online"], ids["katering"]
    client = TestClient(create_app(tmp))
    log, steps = [], []

    def call(method, path, note, **kw):
        t0 = time.perf_counter()
        r = client.request(method, path, **kw)
        try:
            body = r.json()
        except ValueError:
            body = None
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "method": method,
                 "path": path, "status": r.status_code, "ms": round(1000 * (time.perf_counter() - t0)),
                 "step": note}
        if isinstance(body, dict):
            entry["response"] = {k: body[k] for k in list(body)[:8]
                                 if not isinstance(body[k], (list, dict)) or k in ("detail",)}
        log.append(entry)
        steps.append(f"{len(steps) + 1}. {note} -> HTTP {r.status_code}")
        return r, body

    r = client.post("/login", data={"username": "demo", "password": "cashbridge-demo"}, follow_redirects=False)
    log.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "method": "POST",
                "path": "/login", "status": r.status_code, "step": "Owner logs in"})
    steps.append(f"1. Owner logs in -> HTTP {r.status_code}")
    csrf = conn.execute("SELECT csrf FROM sessions ORDER BY expires_at DESC LIMIT 1").fetchone()[0]
    client.headers.update({"X-CSRF-Token": csrf})

    call("GET", f"/api/business/{online}/forecast", "Forecast 14 days (Monte Carlo)")
    _, rec = call("GET", f"/api/business/{online}/recommendation", "Recommendation: non-loan options first, then loan check")
    product = rec["loan"]["eligible"][0]["product_id"]
    call("POST", f"/api/business/{katering}/ask", "Ask the copilot (katering, grounded template answer)",
         json={"question": "Can I get a loan?", "engine": "mock"})
    call("POST", "/api/bank/offers", "Loan draft for katering is refused by the rules (repayment capacity)",
         json={"business_id": katering, "product_id": "receivable_bridge"})
    _, draft = call("POST", "/api/bank/offers", f"Draft offer ({product}) for online store",
                    json={"business_id": online, "product_id": product})
    oid = draft["offer_id"]
    call("POST", f"/api/bank/offers/{oid}/disburse", "Disburse WITHOUT owner approval (must fail)",
         headers={"Idempotency-Key": "evidence-no-approval"})
    _, appr = call("POST", f"/api/bank/offers/{oid}/approve", "Owner approves -> single-use token bound to terms", json={})
    token = appr["approval_token"]
    call("POST", f"/api/bank/offers/{oid}/disburse", "Sandbox disbursement with token + idempotency key",
         headers={"Idempotency-Key": "evidence-disburse-1", "X-Approval-Token": token})
    call("POST", f"/api/bank/offers/{oid}/disburse", "Replay of the same disbursement (must fail)",
         headers={"Idempotency-Key": "evidence-disburse-2", "X-Approval-Token": token})
    _, loans = call("GET", f"/api/bank/loans?business_id={online}", "Read loan + repayment schedule")
    call("POST", f"/api/bank/repayments/{loans['repayments'][0]['id']}/pay", "Pay first installment (sandbox)")

    # reconciliation: the bank shows a payout that matches an open invoice
    inv = conn.execute("""SELECT * FROM receivables WHERE business_id=? AND status='open'
                          ORDER BY expected_settlement_date LIMIT 1""", (katering,)).fetchone()
    conn.execute("""INSERT INTO transactions (business_id, date, description, counterparty, amount, direction,
                    category, is_cash_sale, source) VALUES (?,?,?,?,?,'in','revenue',0,'csv')""",
                 (katering, util.today().isoformat(), f"Transfer {inv['debtor']}", inv["debtor"], inv["amount"]))
    conn.commit()
    _, recon = call("GET", f"/api/business/{katering}/reconcile", "Reconciliation proposes invoice matches")
    m = recon["matches"][0]
    call("POST", f"/api/business/{katering}/reconcile", "Owner confirms the match",
         json={"receivable_id": m["receivable_id"], "transaction_id": m["transaction_id"]})
    call("POST", f"/api/business/{katering}/review", "Owner contests the automated loan decision (human review)")
    call("GET", "/api/bank/audit?limit=5", "Read audit trail + chain check")

    with open(OUT / "operation_log.jsonl", "w") as f:
        for e in log:
            f.write(json.dumps(e, default=str) + "\n")
    rows = [dict(r) for r in conn.execute("SELECT * FROM audit_log ORDER BY id")]
    (OUT / "audit_log.json").write_text(json.dumps(rows, indent=2))
    chain = audit.verify_chain(conn)
    (OUT / "chain_verification.json").write_text(json.dumps(chain, indent=2))
    ledger = [dict(r) for r in conn.execute("SELECT * FROM sandbox_ledger ORDER BY id")]
    with open(OUT / "sandbox_ledger.csv", "w", newline="") as f:
        if ledger:
            w = csv.DictWriter(f, fieldnames=list(ledger[0]))
            w.writeheader()
            w.writerows(ledger)
    summary = ["# Sandbox execution evidence", "",
               f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by scripts/demo_run.py "
               f"(demo clock {util.today().isoformat()}, all money simulated).", "", *steps, "",
               f"Audit entries: {len(rows)}; hash chain intact: {chain['ok']}.",
               f"Ledger rows: {len(ledger)} (see sandbox_ledger.csv)."]
    (OUT / "summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
