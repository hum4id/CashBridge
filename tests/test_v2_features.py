"""Features ported from the v2 concept: targeted options, loan rules + cost disclosure,
scoped consent, reconciliation, human review, erasure, limits, data minimum."""
from __future__ import annotations

import pytest

from app import audit as audit_mod
from app import service
from cashbridge import config, db, reconcile, seed, sizing
from cashbridge.recommend import recommend


@pytest.fixture()
def env(tmp_path):
    conn = db.connect(tmp_path / "v2.db")
    db.init_db(conn)
    ids = seed.run_seed(conn, as_of=None, seed=42)
    for bid in ids.values():
        service.grant_consent(conn, bid)
    yield {"conn": conn, "ids": ids}
    conn.close()


def test_non_loan_options_are_material_and_ranked(env):
    rec = recommend(env["conn"], env["ids"]["online"])
    assert rec["non_loan"], "at least one material non-loan option"
    for o in rec["non_loan"]:
        assert o["impact"]["material"]
    ids = [o["id"] for o in rec["non_loan"]]
    assert "collect_late_receivables" in ids
    p_after = [o["impact"]["p_breach_after"] for o in rec["non_loan"]]
    assert p_after == sorted(p_after)


def test_defer_bills_never_moves_payroll(env):
    from cashbridge.forecast import ScenarioParams, build_profiles, simulate_paths
    from cashbridge import util
    conn, bid = env["conn"], env["ids"]["sembako"]
    as_of = util.today()
    base = simulate_paths(conn, bid, as_of, 14, ScenarioParams(), n_paths=50)
    prof = build_profiles(conn, bid, as_of, 14)
    payroll = [i for i in prof["scheduled_items"] if i["category"] == "payroll"]
    moved = simulate_paths(conn, bid, as_of, 14,
                           ScenarioParams(defer_bills_days=5, defer_before_offset=15), n_paths=50)
    for item in payroll:
        t = item["day_offset"] - 1
        assert moved["base_out"][t] >= item["amount"] - 1   # payroll still on its day
    assert moved["base_out"].sum() <= base["base_out"].sum()


def test_cost_disclosure_and_rate_cap(env):
    rec = recommend(env["conn"], env["ids"]["online"])
    for p in rec["loan"]["products"]:
        d = p["cost_disclosure"]
        assert d["total_cost"] == pytest.approx(p["total_repayment"] - p["principal"], abs=1)
        assert d["apr_effective"] >= d["apr_nominal"] > 0
        assert d["within_cap"] and d["rate_cap"] == 0.24


def test_irr_matches_closed_form():
    # one payment of 101 after 30 days on 100: daily r = 1.01^(1/30) - 1
    d = sizing.cost_disclosure(100.0, [{"due_date": "2026-12-06", "amount": 101.0}], "2026-11-06", 1.0)
    r = 1.01 ** (1 / 30) - 1
    assert d["apr_nominal"] == pytest.approx(r * 365, abs=1e-4)


def test_expensive_product_blocked_by_cap(env, monkeypatch):
    from cashbridge import products
    rules = [dict(p) for p in products.PRODUCTS]
    for p in rules:
        p["flat_fee_pct"] = 4.0
    monkeypatch.setattr(products, "PRODUCTS", rules)
    rec = recommend(env["conn"], env["ids"]["online"])
    assert rec["decision"] == "loan_blocked"
    assert {p["reason_code"] for p in rec["loan"]["products"]} == {"rate_cap"}


def test_jurisdiction_cap_selection(monkeypatch):
    monkeypatch.setattr(config, "settings", config.Settings(jurisdictions="ID"))
    assert config.active_rate_cap() == (0.365, "ID")
    monkeypatch.setattr(config, "settings", config.Settings(jurisdictions="CN,ID"))
    assert config.active_rate_cap() == (0.24, "CN")


def test_one_active_loan_at_a_time(env):
    conn, bid = env["conn"], env["ids"]["online"]
    product = recommend(conn, bid)["loan"]["eligible"][0]["product_id"]
    offer = service.draft_offer(conn, bid, product)
    token = service.decide_offer(conn, offer["id"], "approved", actor="owner")["approval_token"]
    assert "loan_id" in service.disburse_offer(conn, offer["id"], approval_token=token)
    after = recommend(conn, bid)
    if after["loan"]:
        assert all(p["reason_code"] == "active_loan" for p in after["loan"]["products"])


def test_daily_limit(env, monkeypatch):
    conn, bid = env["conn"], env["ids"]["online"]
    monkeypatch.setattr(config, "settings", config.Settings(sandbox_daily_limit=1_000_000))
    product = recommend(conn, bid)["loan"]["eligible"][0]["product_id"]
    offer = service.draft_offer(conn, bid, product)
    token = service.decide_offer(conn, offer["id"], "approved", actor="owner")["approval_token"]
    assert service.disburse_offer(conn, offer["id"], approval_token=token)["error"]["code"] == "daily_limit"


def test_scoped_consent(env):
    conn, bid = env["conn"], env["ids"]["online"]
    service.revoke_consent(conn, bid, scope="financing")
    assert service.consent_active(conn, bid, "cashflow_analysis")
    assert not service.consent_active(conn, bid, "financing")
    product = "quick_stock_advance"
    assert service.draft_offer(conn, bid, product)["error"]["code"] == "consent_revoked"
    from app import tools
    out = tools.execute_tool(conn, "get_recommendations", {"business_id": bid}, granted_tier="T1")
    assert "loan" not in out and out["decision"] == "financing_consent_required"
    with pytest.raises(service.ServiceError):
        service.grant_consent(conn, bid, "sell_my_data")


def test_reconciliation_proposes_and_confirms(env):
    conn, bid = env["conn"], env["ids"]["katering"]
    r = conn.execute("SELECT * FROM receivables WHERE business_id=? AND status='open' "
                     "ORDER BY expected_settlement_date LIMIT 1", (bid,)).fetchone()
    from cashbridge import util
    conn.execute("""INSERT INTO transactions (business_id, date, description, counterparty, amount,
                    direction, category, is_cash_sale, source) VALUES (?,?,?,?,?,'in','revenue',0,'csv')""",
                 (bid, util.today().isoformat(), f"Transfer from {r['debtor']}", r["debtor"], r["amount"]))
    conn.commit()
    out = reconcile.propose_matches(conn, bid)
    match = next(m for m in out["matches"] if m["receivable_id"] == r["id"])
    assert match["confidence"] == "high"
    before = recommend(conn, bid, persist=False)["forecast"]["open_receivables_total"]
    done = service.confirm_match(conn, r["id"], match["transaction_id"])
    assert done["status"] == "settled"
    after = recommend(conn, bid, persist=False)["forecast"]["open_receivables_total"]
    assert after == pytest.approx(before - r["amount"])
    # the same credit cannot settle a second invoice
    again = service.confirm_match(conn, r["id"], match["transaction_id"])
    assert again["error"]["code"] == "invalid_state"
    assert audit_mod.verify_chain(conn)["ok"]


def test_amount_mismatch_not_matched(env):
    conn, bid = env["conn"], env["ids"]["katering"]
    r = conn.execute("SELECT * FROM receivables WHERE business_id=? AND status='open' LIMIT 1", (bid,)).fetchone()
    from cashbridge import util
    conn.execute("""INSERT INTO transactions (business_id, date, description, counterparty, amount,
                    direction, category, is_cash_sale, source) VALUES (?,?,?,?,?,'in','revenue',0,'csv')""",
                 (bid, util.today().isoformat(), f"Transfer from {r['debtor']}", r["debtor"], r["amount"] * 0.9))
    out = reconcile.propose_matches(conn, bid)
    assert not any(m["receivable_id"] == r["id"] for m in out["matches"])


def test_human_review_and_erasure(env):
    conn, bid = env["conn"], env["ids"]["katering"]
    review = service.request_human_review(conn, bid, "loan_decision", {"decision": "loan_blocked"}, "demo")
    assert review["status"] == "open"
    result = service.delete_business_data(conn, bid, "demo")
    assert result["deleted"]["transactions"] > 0
    assert conn.execute("SELECT COUNT(*) FROM transactions WHERE business_id=?", (bid,)).fetchone()[0] == 0
    assert not service.consent_active(conn, bid)
    actions = [e["action"] for e in audit_mod.recent(conn, 5)]
    assert "business_data_deleted" in actions
    assert audit_mod.verify_chain(conn)["ok"]


def test_insufficient_history_blocks_financing(tmp_path):
    conn = db.connect(tmp_path / "new.db")
    db.init_db(conn)
    from cashbridge import util
    bid = conn.execute("INSERT INTO businesses (name, sector, safety_balance) VALUES ('New','retail',5000000)").lastrowid
    for i in range(20):
        d = util.add_days(util.today(), -i).isoformat()
        conn.execute("""INSERT INTO transactions (business_id, date, description, counterparty, amount,
                        direction, category, is_cash_sale, source) VALUES (?,?,?,?,?,'in','revenue',1,'csv')""",
                     (bid, d, "sales", "walk-in", 400_000))
    conn.execute("""INSERT INTO transactions (business_id, date, description, counterparty, amount,
                    direction, category, is_cash_sale, source) VALUES (?,?,?,?,?,'out','rent',0,'csv')""",
                 (bid, util.today().isoformat(), "rent", "landlord", 30_000_000))
    conn.commit()
    service.grant_consent(conn, bid)
    rec = recommend(conn, bid)
    assert rec["forecast"]["data_sufficient"] is False
    if rec["gap"]:
        assert rec["decision"] == "insufficient_data" and rec["loan"] is None
