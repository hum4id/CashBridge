"""Service-layer tests: consent, offer lifecycle, sandbox execution, repayments."""
from __future__ import annotations

import pytest

from app import audit as audit_mod
from app import service
from cashbridge import db, seed
from cashbridge.recommend import recommend


@pytest.fixture()
def env(tmp_path):
    conn = db.connect(tmp_path / "svc.db")
    db.init_db(conn)
    ids = seed.run_seed(conn, as_of=None, seed=42)
    yield {"conn": conn, "ids": ids}
    conn.close()


def _eligible_product(conn, business_id) -> str:
    recommendation = recommend(conn, business_id)
    assert recommendation["decision"] == "loan_available", recommendation["decision"]
    return recommendation["loan"]["eligible"][0]["product_id"]


def _disburse_flow(env):
    """consent -> draft -> approve -> disburse, returning (offer, disbursement)."""
    conn, online = env["conn"], env["ids"]["online"]
    service.grant_consent(conn, online)
    offer = service.draft_offer(conn, online, _eligible_product(conn, online))
    assert "error" not in offer, offer
    assert offer["status"] == "draft"
    decided = service.decide_offer(conn, offer["id"], "approved", actor="owner")
    assert decided["status"] == "approved"
    env["token"] = decided["approval_token"]
    return offer, service.disburse_offer(conn, offer["id"], approval_token=decided["approval_token"])


def test_consent_lifecycle(env):
    conn, online = env["conn"], env["ids"]["online"]
    assert not service.consent_active(conn, online)
    service.grant_consent(conn, online)
    assert service.consent_active(conn, online)
    service.revoke_consent(conn, online)
    assert not service.consent_active(conn, online)
    assert audit_mod.verify_chain(conn)["ok"]


def test_draft_requires_consent(env):
    conn, online = env["conn"], env["ids"]["online"]
    result = service.draft_offer(conn, online, "receivable_bridge")
    assert result["error"]["code"] == "consent_revoked"
    assert audit_mod.verify_chain(conn)["ok"]


def test_offer_lifecycle_and_idempotency(env):
    conn = env["conn"]
    offer, disbursement = _disburse_flow(env)
    assert disbursement["loan_id"]
    assert disbursement["installments"] >= 4

    # replaying the same action is refused (same idempotency key)
    again = service.disburse_offer(conn, offer["id"], approval_token=env["token"])
    assert again["error"]["code"] == "duplicate_action"

    # a fresh idempotency key cannot replay either: the approval token is single-use
    replay = service.disburse_offer(conn, offer["id"], idempotency_key="manual-replay",
                                    approval_token=env["token"])
    assert replay["error"]["code"] == "duplicate_action"
    # and without a token there is no approval at all
    assert service.disburse_offer(conn, offer["id"], idempotency_key="k2")["error"]["code"] == "approval_missing"

    summary = service.sandbox_summary(conn, offer["business_id"])
    assert summary["total_disbursed"] == offer["principal"]
    assert len(summary["repayments"]) == disbursement["installments"]
    assert all(r["status"] == "scheduled" for r in summary["repayments"])
    assert audit_mod.verify_chain(conn)["ok"]


def test_disburse_requires_approval(env):
    conn, online = env["conn"], env["ids"]["online"]
    service.grant_consent(conn, online)
    offer = service.draft_offer(conn, online, _eligible_product(conn, online))
    result = service.disburse_offer(conn, offer["id"])
    assert result["error"]["code"] == "approval_missing"


def test_revoke_blocks_disbursement(env):
    conn, online = env["conn"], env["ids"]["online"]
    service.grant_consent(conn, online)
    offer = service.draft_offer(conn, online, _eligible_product(conn, online))
    token = service.decide_offer(conn, offer["id"], "approved", actor="owner")["approval_token"]
    service.revoke_consent(conn, online)
    result = service.disburse_offer(conn, offer["id"], approval_token=token)
    assert result["error"]["code"] == "consent_revoked"


def test_ineligible_offer_blocked(env):
    conn, katering = env["conn"], env["ids"]["katering"]
    service.grant_consent(conn, katering)
    result = service.draft_offer(conn, katering, "receivable_bridge")
    assert result["error"]["code"] == "ineligible"
    assert "capacity" in result["error"]["reason"].lower()
    # the refusal is audited, not silent
    actions = [e["action"] for e in audit_mod.recent(conn, 10)]
    assert "offer_draft_blocked" in actions


def test_repayments_settle_loan(env):
    conn = env["conn"]
    offer, _ = _disburse_flow(env)
    summary = service.sandbox_summary(conn, offer["business_id"])
    expected_total = sum(r["amount"] for r in summary["repayments"])

    first = service.pay_repayment(conn, summary["repayments"][0]["id"])
    assert first["status"] == "paid"
    duplicate = service.pay_repayment(conn, summary["repayments"][0]["id"])
    assert duplicate["error"]["code"] == "duplicate_action"

    for repayment in summary["repayments"][1:]:
        assert service.pay_repayment(conn, repayment["id"])["status"] == "paid"

    summary = service.sandbox_summary(conn, offer["business_id"])
    assert all(r["status"] == "paid" for r in summary["repayments"])
    assert summary["loans"][0]["status"] == "settled"
    assert abs(summary["total_repaid"] - expected_total) < 0.01
    offer_status = conn.execute("SELECT status FROM offers WHERE id = ?",
                                (offer["id"],)).fetchone()["status"]
    assert offer_status == "settled"
    assert audit_mod.verify_chain(conn)["ok"]


def test_agent_cannot_approve(env):
    conn, online = env["conn"], env["ids"]["online"]
    service.grant_consent(conn, online)
    offer = service.draft_offer(conn, online, _eligible_product(conn, online))
    result = service.decide_offer(conn, offer["id"], "approved", actor="agent")
    assert result["error"]["code"] == "tier_denied"
    assert conn.execute("SELECT status FROM offers WHERE id=?", (offer["id"],)).fetchone()[0] == "draft"


def test_token_bound_to_terms(env):
    conn, online = env["conn"], env["ids"]["online"]
    service.grant_consent(conn, online)
    offer = service.draft_offer(conn, online, _eligible_product(conn, online))
    token = service.decide_offer(conn, offer["id"], "approved", actor="owner")["approval_token"]
    conn.execute("UPDATE offers SET principal = principal * 3 WHERE id = ?", (offer["id"],))
    result = service.disburse_offer(conn, offer["id"], approval_token=token)
    assert result["error"]["code"] == "approval_terms_changed"


def test_forged_token_rejected(env):
    conn, online = env["conn"], env["ids"]["online"]
    service.grant_consent(conn, online)
    offer = service.draft_offer(conn, online, _eligible_product(conn, online))
    token = service.decide_offer(conn, offer["id"], "approved", actor="owner")["approval_token"]
    forged = token.split(".")[0] + "." + "0" * 64
    assert service.disburse_offer(conn, offer["id"], approval_token=forged)["error"]["code"] == "approval_invalid"


def test_disbursement_moves_balance(env):
    conn = env["conn"]
    offer, _ = _disburse_flow(env)
    bal_tx = conn.execute("SELECT amount FROM transactions WHERE source='sandbox' AND direction='in'").fetchone()
    assert bal_tx["amount"] == offer["principal"]


def test_audit_log_is_append_only(env):
    import sqlite3
    conn = env["conn"]
    service.grant_consent(conn, env["ids"]["online"])
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE audit_log SET actor = 'x'")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM audit_log")
