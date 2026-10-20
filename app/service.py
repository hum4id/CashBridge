"""Domain service layer: consent, dashboards, offers, approvals, sandbox actions.

This is the only place that mutates money-adjacent state (offers, loans,
repayments, ledger). Every mutation goes through the policy engine and writes
an audit entry. The LLM never calls these directly — it goes through the tool
layer, which uses this service.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Optional

from . import approval, audit, policy
from cashbridge import config, db, products as products_mod, recommend as recommend_mod, util

# Consent passport: separate, individually revocable purposes (PIPL Art. 14/29/39,
# UU PDP Pasal 20-22). ``external_ai`` = sending (pseudonymised) data to an LLM
# provider that may sit outside the user's country, so it is never on by default.
SCOPES = {
    "cashflow_analysis": "Analyse my transactions and forecast my cash",
    "financing": "Assess financing and prepare loan offers for my approval",
    "external_ai": "Send pseudonymised figures to an external AI provider for explanations",
}
DEFAULT_SCOPE = "cashflow_analysis,financing"


def _scopes(raw: str) -> set:
    legacy = {"transactions:read": "cashflow_analysis", "loan:simulate": "financing"}
    return {legacy.get(s.strip(), s.strip()) for s in (raw or "").split(",") if s.strip()}


class ServiceError(Exception):
    """Raised for not-found or invalid-state operations."""


# ------------------------------------------------------------------ consent

def grant_consent(conn, business_id: int, scope: str = DEFAULT_SCOPE, actor: str = "owner") -> dict:
    if db.get_business(conn, business_id) is None:
        raise ServiceError(f"Unknown business: {business_id}")
    unknown = _scopes(scope) - set(SCOPES)
    if unknown:
        raise ServiceError(f"Unknown consent scope(s): {', '.join(sorted(unknown))}")
    cur = conn.execute("INSERT INTO consents (business_id, scope) VALUES (?,?)", (business_id, scope))
    conn.commit()
    entry = audit.append(conn, actor, "consent_granted", "business", str(business_id), {"scope": scope})
    return {"consent_id": cur.lastrowid, "business_id": business_id, "scope": scope, "audit": entry}


def active_scopes(conn, business_id: int) -> set:
    rows = conn.execute("SELECT scope FROM consents WHERE business_id = ? AND revoked_at IS NULL",
                        (business_id,)).fetchall()
    out: set = set()
    for r in rows:
        out |= _scopes(r["scope"])
    return out


def revoke_consent(conn, business_id: int, actor: str = "owner", scope: Optional[str] = None) -> dict:
    """Revoke everything (default) or only the given scope(s); other scopes stay granted."""
    before = active_scopes(conn, business_id)
    cur = conn.execute(
        "UPDATE consents SET revoked_at = datetime('now') WHERE business_id = ? AND revoked_at IS NULL",
        (business_id,),
    )
    keep = before - _scopes(scope) if scope else set()
    if keep:
        conn.execute("INSERT INTO consents (business_id, scope) VALUES (?,?)",
                     (business_id, ",".join(sorted(keep))))
    conn.commit()
    entry = audit.append(conn, actor, "consent_revoked", "business", str(business_id),
                         {"revoked": sorted(before - keep), "still_granted": sorted(keep)})
    return {"business_id": business_id, "revoked_count": cur.rowcount, "audit": entry}


def consent_active(conn, business_id: int, scope: str = "cashflow_analysis") -> bool:
    return scope in active_scopes(conn, business_id)


# ---------------------------------------------------------------- dashboards

def list_overview(conn, as_of=None) -> list:
    as_of = util.parse_date(as_of) if as_of else util.today()
    out = []
    for business in db.list_businesses(conn):
        out.append({
            "id": business["id"],
            "name": business["name"],
            "sector": business["sector"],
            "safety_balance": business["safety_balance"],
            "balance": db.opening_balance(conn, business["id"], as_of),
            "consent_active": consent_active(conn, business["id"]),
        })
    return out


def dashboard(conn, business_id: int, as_of=None) -> dict:
    result = recommend_mod.recommend(conn, business_id, as_of=as_of)
    result["consent_active"] = consent_active(conn, business_id)
    result["offers"] = list_offers(conn, business_id)
    result["sandbox"] = sandbox_summary(conn, business_id)
    return result


# ------------------------------------------------------------------- offers

def _offer_row(conn, offer_id: int) -> dict:
    row = conn.execute("SELECT * FROM offers WHERE id = ?", (offer_id,)).fetchone()
    if row is None:
        raise ServiceError(f"Unknown offer: {offer_id}")
    return dict(row)


def list_offers(conn, business_id: int) -> list:
    rows = conn.execute(
        "SELECT * FROM offers WHERE business_id = ? ORDER BY id DESC", (business_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def draft_offer(conn, business_id: int, product_id: str, actor: str = "agent",
                granted_tier: str = "T2", as_of=None) -> dict:
    """Draft a proportional short-term loan offer for one eligible product."""
    decision = policy.check_action("draft_offer", granted_tier=granted_tier,
                                   consent_active=consent_active(conn, business_id, "financing"))
    if not decision["allow"]:
        audit.append(conn, actor, "offer_draft_denied", "business", str(business_id),
                     {"product_id": product_id, "reason": decision["reason"], "code": decision["code"]})
        return {"error": decision}

    products_mod.ensure_products(conn)
    recommendation = recommend_mod.recommend(conn, business_id, as_of=as_of)
    if not recommendation["gap"]:
        return {"error": {"code": "no_gap", "reason": "No cash gap projected; a loan is not warranted."}}

    evaluation = next(
        (e for e in recommendation["loan"]["products"] if e["product_id"] == product_id), None
    )
    if evaluation is None:
        raise ServiceError(f"Unknown product for this business: {product_id}")
    if not evaluation["eligible"]:
        audit.append(conn, actor, "offer_draft_blocked", "business", str(business_id),
                     {"product_id": product_id, "reason": evaluation["reason"]})
        return {"error": {"code": "ineligible", "reason": evaluation["reason"]}}

    gap = recommendation["gap"]
    rationale = (
        f"Gap of {util.fmt_money(gap['depth'])} projected starting {gap['start_date']}. "
        f"Principal targets the shortfall with a 10% buffer. "
        f"Repayment capacity covers {evaluation['coverage_ratio']}x of installments "
        f"(minimum {evaluation['min_coverage_ratio']}x)."
    )
    cur = conn.execute(
        """INSERT INTO offers
           (business_id, gap_id, product_id, principal, tenor_days, total_repayment,
            coverage_ratio, schedule, rationale, status)
           VALUES (?,?,?,?,?,?,?,?,?,'draft')""",
        (business_id, gap.get("gap_id"), product_id, evaluation["principal"],
         evaluation["tenor_days"], evaluation["total_repayment"], evaluation["coverage_ratio"],
         json.dumps(evaluation["schedule"]), rationale),
    )
    conn.execute("UPDATE offers SET disclosure = ? WHERE id = ?",
                 (json.dumps(evaluation["cost_disclosure"]), cur.lastrowid))
    conn.commit()
    offer_id = cur.lastrowid
    audit.append(conn, actor, "offer_drafted", "offer", str(offer_id),
                 {"business_id": business_id, "product_id": product_id,
                  "principal": evaluation["principal"], "coverage_ratio": evaluation["coverage_ratio"]})
    return {**_offer_row(conn, offer_id), "evaluation": evaluation, "recommendation_message": recommendation["message"]}


def decide_offer(conn, offer_id: int, decision: str, actor: str = "owner",
                 note: Optional[str] = None) -> dict:
    """Owner approves or rejects a draft offer. This is the human-in-the-loop gate.

    Approving returns an ``approval_token`` bound to the exact terms; disbursement
    requires that token. Only an owner can approve: agents and tools never can.
    """
    if decision not in ("approved", "rejected"):
        raise ServiceError(f"Decision must be 'approved' or 'rejected', got {decision!r}")
    if decision == "approved" and (actor in ("agent", "llm", "bank_sandbox", "system")
                                   or not actor):
        audit.append(conn, actor or "unknown", "offer_approval_denied", "offer", str(offer_id),
                     {"code": "owner_only"})
        return {"error": {"code": "tier_denied", "reason": "Only the business owner can approve an offer."}}
    offer = _offer_row(conn, offer_id)
    if offer["status"] != "draft":
        return {"error": {"code": "invalid_state",
                          "reason": f"Offer {offer_id} is '{offer['status']}', only drafts can be decided."}}
    conn.execute("INSERT INTO approvals (offer_id, actor, decision, note) VALUES (?,?,?,?)",
                 (offer_id, actor, decision, note))
    conn.execute("UPDATE offers SET status = ? WHERE id = ?", (decision, offer_id))
    conn.commit()
    audit.append(conn, actor, f"offer_{decision}", "offer", str(offer_id),
                 {"note": note, "principal": offer["principal"], "product_id": offer["product_id"],
                  "terms_hash": approval.terms_hash(offer)})
    row = _offer_row(conn, offer_id)
    if decision == "approved":
        token, _nonce = approval.issue(actor, offer)
        row["approval_token"] = token
    return row


def _still_valid(conn, offer: dict) -> Optional[str]:
    """Re-run the analysis at execution time: data may have changed since the draft."""
    rec = recommend_mod.recommend(conn, offer["business_id"], persist=False)
    if not rec["gap"]:
        return "No cash gap is projected anymore; the loan is not needed."
    ev = next((e for e in rec["loan"]["products"] if e["product_id"] == offer["product_id"]), None)
    if ev is None or not ev["eligible"]:
        return "The offer no longer passes the repayment checks with the latest data."
    if ev["principal"] != offer["principal"]:
        return "The required amount changed with the latest data; draft a new offer."
    return None


def disburse_offer(conn, offer_id: int, idempotency_key: Optional[str] = None,
                   actor: str = "agent", approval_token: Optional[str] = None) -> dict:
    """Execute the sandbox disbursement for an approved offer. Simulated funds only.

    Requires the owner's approval token for these exact terms (single use), and
    re-checks the offer against the latest data before any money moves.
    """
    offer = _offer_row(conn, offer_id)
    key = idempotency_key or f"offer-{offer_id}-disburse"
    if key in {r["idempotency_key"] for r in conn.execute(
            "SELECT idempotency_key FROM sandbox_ledger WHERE idempotency_key IS NOT NULL")}:
        audit.append(conn, actor, "disbursement_denied", "offer", str(offer_id),
                     {"code": "duplicate_action", "idempotency_key": key})
        return {"error": {"code": "duplicate_action",
                          "reason": f"Idempotency key '{key}' was already used; refusing to execute twice."}}
    try:
        token_payload = approval.verify(conn, approval_token, offer)
    except approval.ApprovalError as exc:
        audit.append(conn, actor, "disbursement_denied", "offer", str(offer_id),
                     {"code": exc.code, "reason": exc.reason})
        return {"error": {"code": exc.code, "reason": exc.reason}}
    stale = _still_valid(conn, offer)
    if stale:
        conn.execute("UPDATE offers SET status = 'expired' WHERE id = ?", (offer_id,))
        conn.commit()
        audit.append(conn, actor, "disbursement_denied", "offer", str(offer_id),
                     {"code": "offer_changed", "reason": stale})
        return {"error": {"code": "offer_changed", "reason": stale}}
    used_keys = {r["idempotency_key"] for r in
                 conn.execute("SELECT idempotency_key FROM sandbox_ledger WHERE idempotency_key IS NOT NULL")}
    approved = conn.execute(
        "SELECT 1 FROM approvals WHERE offer_id = ? AND decision = 'approved' LIMIT 1", (offer_id,)
    ).fetchone() is not None

    decision = policy.check_action(
        "disburse_sandbox", granted_tier="T3",
        consent_active=consent_active(conn, offer["business_id"], "financing"),
        amount=offer["principal"], max_amount=config.settings.sandbox_max_principal,
        idempotency_key=key, used_keys=used_keys,
        approved=approved and offer["status"] == "approved",
    )
    if not decision["allow"]:
        audit.append(conn, actor, "disbursement_denied", "offer", str(offer_id),
                     {"code": decision["code"], "reason": decision["reason"]})
        return {"error": decision}
    today_total = conn.execute(
        """SELECT COALESCE(SUM(amount), 0) s FROM sandbox_ledger WHERE business_id = ?
           AND entry_type = 'disbursement' AND date(created_at) = date('now')""",
        (offer["business_id"],)).fetchone()["s"]
    if today_total + offer["principal"] > config.settings.sandbox_daily_limit:
        reason = (f"Daily sandbox limit {util.fmt_money(config.settings.sandbox_daily_limit)} would be "
                  f"exceeded ({util.fmt_money(today_total)} already disbursed today).")
        audit.append(conn, actor, "disbursement_denied", "offer", str(offer_id),
                     {"code": "daily_limit", "reason": reason})
        return {"error": {"code": "daily_limit", "reason": reason}}

    cur = conn.execute(
        "INSERT INTO loans (offer_id, principal, status, disbursed_at) VALUES (?,?,'active',datetime('now'))",
        (offer_id, offer["principal"]),
    )
    loan_id = cur.lastrowid
    schedule = json.loads(offer["schedule"] or "[]")
    for installment in schedule:
        conn.execute(
            """INSERT INTO repayments (loan_id, due_date, amount, status, linked_receivable_id)
               VALUES (?,?,?,'scheduled',?)""",
            (loan_id, installment["due_date"], installment["amount"],
             installment.get("linked_receivable_id")),
        )
    conn.execute(
        "INSERT INTO sandbox_ledger (business_id, loan_id, entry_type, amount, idempotency_key) VALUES (?,?,'disbursement',?,?)",
        (offer["business_id"], loan_id, offer["principal"], key),
    )
    # The simulated money lands in the business account, so the balance and the
    # forecast reflect the loan (installments are picked up as scheduled outflows).
    conn.execute(
        """INSERT INTO transactions (business_id, date, description, counterparty, amount, direction,
                                     category, is_cash_sale, source)
           VALUES (?,?,?,?,?,'in','financing',0,'sandbox')""",
        (offer["business_id"], util.today().isoformat(), f"Loan disbursement (sandbox) #{loan_id}",
         "Sandbox Bank", offer["principal"]))
    approval.consume(conn, token_payload)
    conn.execute("UPDATE offers SET status = 'disbursed' WHERE id = ?", (offer_id,))
    conn.commit()
    audit.append(conn, actor, "loan_disbursed_sandbox", "loan", str(loan_id),
                 {"offer_id": offer_id, "principal": offer["principal"],
                  "installments": len(schedule), "idempotency_key": key,
                  "approved_by": token_payload["o"], "approval_nonce": token_payload["n"]})
    return {"loan_id": loan_id, "offer_id": offer_id, "principal": offer["principal"],
            "installments": len(schedule), "idempotency_key": key}


# ------------------------------------------------------------------ sandbox

def sandbox_summary(conn, business_id: int) -> dict:
    loans = [dict(r) for r in conn.execute(
        """SELECT l.*, o.product_id, o.status AS offer_status
           FROM loans l JOIN offers o ON o.id = l.offer_id
           WHERE o.business_id = ? ORDER BY l.id""", (business_id,))]
    repayments = [dict(r) for r in conn.execute(
        """SELECT rp.* FROM repayments rp JOIN loans l ON l.id = rp.loan_id
           JOIN offers o ON o.id = l.offer_id
           WHERE o.business_id = ? ORDER BY rp.due_date""", (business_id,))]
    ledger = [dict(r) for r in conn.execute(
        "SELECT * FROM sandbox_ledger WHERE business_id = ? ORDER BY id", (business_id,))]
    return {"loans": loans, "repayments": repayments, "ledger": ledger,
            "total_disbursed": sum(e["amount"] for e in ledger if e["entry_type"] == "disbursement"),
            "total_repaid": sum(e["amount"] for e in ledger if e["entry_type"] == "repayment")}


def pay_repayment(conn, repayment_id: int, actor: str = "owner") -> dict:
    """Simulate an installment being paid from incoming receivables."""
    row = conn.execute(
        """SELECT rp.*, l.id AS loan_id, o.business_id
           FROM repayments rp JOIN loans l ON l.id = rp.loan_id JOIN offers o ON o.id = l.offer_id
           WHERE rp.id = ?""", (repayment_id,)).fetchone()
    if row is None:
        raise ServiceError(f"Unknown repayment: {repayment_id}")
    repayment = dict(row)
    if repayment["status"] == "paid":
        return {"error": {"code": "duplicate_action", "reason": "This installment is already marked paid."}}

    key = f"repay-{repayment_id}"
    used_keys = {r["idempotency_key"] for r in
                 conn.execute("SELECT idempotency_key FROM sandbox_ledger WHERE idempotency_key IS NOT NULL")}
    decision = policy.check_action("schedule_repayment", granted_tier="T3",
                                   consent_active=consent_active(conn, repayment["business_id"]),
                                   amount=repayment["amount"],
                                   max_amount=config.settings.sandbox_max_principal,
                                   idempotency_key=key, used_keys=used_keys, approved=True)
    if not decision["allow"]:
        audit.append(conn, actor, "repayment_denied", "repayment", str(repayment_id),
                     {"code": decision["code"], "reason": decision["reason"]})
        return {"error": decision}

    conn.execute("UPDATE repayments SET status = 'paid', paid_at = datetime('now') WHERE id = ?",
                 (repayment_id,))
    conn.execute(
        "INSERT INTO sandbox_ledger (business_id, loan_id, entry_type, amount, idempotency_key) VALUES (?,?,'repayment',?,?)",
        (repayment["business_id"], repayment["loan_id"], repayment["amount"], key),
    )
    conn.execute(
        """INSERT INTO transactions (business_id, date, description, counterparty, amount, direction,
                                     category, is_cash_sale, source)
           VALUES (?,?,?,?,?,'out','financing',0,'sandbox')""",
        (repayment["business_id"], util.today().isoformat(),
         f"Loan installment (sandbox) #{repayment['loan_id']}", "Sandbox Bank", repayment["amount"]))
    remaining = conn.execute(
        "SELECT COUNT(*) AS n FROM repayments WHERE loan_id = ? AND status != 'paid'",
        (repayment["loan_id"],)).fetchone()["n"]
    if remaining == 0:
        conn.execute("UPDATE loans SET status = 'settled' WHERE id = ?", (repayment["loan_id"],))
        conn.execute("UPDATE offers SET status = 'settled' WHERE id = (SELECT offer_id FROM loans WHERE id = ?)",
                     (repayment["loan_id"],))
    conn.commit()
    audit.append(conn, actor, "repayment_paid_sandbox", "repayment", str(repayment_id),
                 {"amount": repayment["amount"], "loan_id": repayment["loan_id"], "remaining": remaining})
    return {"repayment_id": repayment_id, "status": "paid", "remaining": remaining}


# ------------------------------------------------------------ reconciliation

def confirm_match(conn, receivable_id: int, transaction_id: int, actor: str = "owner") -> dict:
    """Owner confirms a proposed match: the invoice is marked settled by that credit."""
    from cashbridge import reconcile
    rec = conn.execute("SELECT * FROM receivables WHERE id = ?", (receivable_id,)).fetchone()
    if rec is None:
        raise ServiceError(f"Unknown receivable: {receivable_id}")
    proposals = reconcile.propose_matches(conn, rec["business_id"])["matches"]
    if not any(m["receivable_id"] == receivable_id and m["transaction_id"] == transaction_id
               for m in proposals):
        return {"error": {"code": "invalid_state",
                          "reason": "That pair is not a current match proposal (already settled or changed)."}}
    conn.execute("UPDATE receivables SET status = 'settled', settled_transaction_id = ? WHERE id = ?",
                 (transaction_id, receivable_id))
    conn.commit()
    audit.append(conn, actor, "receivable_reconciled", "receivable", str(receivable_id),
                 {"transaction_id": transaction_id, "amount": float(rec["amount"])})
    return {"receivable_id": receivable_id, "transaction_id": transaction_id, "status": "settled"}


def request_human_review(conn, business_id: int, subject: str, detail: dict, actor: str) -> dict:
    """Right to an explanation and to contest an automated decision
    (PIPL Art. 24(3); UU PDP Pasal 10). Creates a review ticket and audits it."""
    cur = conn.execute(
        "INSERT INTO review_requests (business_id, subject, detail, requested_by) VALUES (?,?,?,?)",
        (business_id, subject, json.dumps(detail, default=str), actor))
    conn.commit()
    audit.append(conn, actor, "human_review_requested", "business", str(business_id),
                 {"subject": subject, "review_id": cur.lastrowid})
    return {"review_id": cur.lastrowid, "status": "open"}


def delete_business_data(conn, business_id: int, actor: str) -> dict:
    """Erase the owner's operational data (PIPL Art. 47; UU PDP Pasal 8 & 16).

    Kept: loan contracts and ledger entries (financial records with a legal retention
    period) and the audit chain (proof of what was done). Their personal fields are
    minimal: amounts, dates and ids.
    """
    if db.get_business(conn, business_id) is None:
        raise ServiceError(f"Unknown business: {business_id}")
    counts = {}
    conn.execute("UPDATE offers SET gap_id = NULL WHERE business_id = ?", (business_id,))
    for table in ("transactions", "receivables", "ground_truth_transactions", "gaps",
                  "review_requests"):
        counts[table] = conn.execute(f"DELETE FROM {table} WHERE business_id = ?",
                                     (business_id,)).rowcount
    fids = [r[0] for r in conn.execute("SELECT id FROM forecasts WHERE business_id = ?", (business_id,))]
    for fid in fids:
        conn.execute("DELETE FROM forecast_days WHERE forecast_id = ?", (fid,))
    counts["forecasts"] = conn.execute("DELETE FROM forecasts WHERE business_id = ?", (business_id,)).rowcount
    conn.execute("UPDATE consents SET revoked_at = datetime('now') WHERE business_id = ? AND revoked_at IS NULL",
                 (business_id,))
    conn.commit()
    audit.append(conn, actor, "business_data_deleted", "business", str(business_id), counts)
    return {"business_id": business_id, "deleted": counts}
