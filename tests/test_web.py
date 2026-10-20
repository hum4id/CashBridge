"""Web app tests: pages, CSV upload + dedup, ask API, sandbox bank API, consent."""
from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from app import service
from app.main import create_app
from cashbridge import db, seed, util
from cashbridge.recommend import recommend


@pytest.fixture()
def env(tmp_path):
    db_path = tmp_path / "web.db"
    conn = db.connect(db_path)
    db.init_db(conn)
    ids = seed.run_seed(conn, as_of=None, seed=42)
    for business_id in ids.values():
        service.grant_consent(conn, business_id)
    conn.close()
    app = create_app(db_path)
    with TestClient(app) as client:
        csrf = _login(client)
        yield {"client": client, "ids": ids, "db_path": db_path, "csrf": csrf}


def _login(client) -> str:
    """Log in as the demo owner; the CSRF token is sent on every unsafe request."""
    response = client.post("/login", data={"username": "demo", "password": "cashbridge-demo"},
                           follow_redirects=False)
    assert response.status_code == 303, response.text
    conn = db.connect(client.app.state.db_path)
    csrf = conn.execute("SELECT csrf FROM sessions ORDER BY expires_at DESC LIMIT 1").fetchone()[0]
    conn.close()
    client.headers.update({"X-CSRF-Token": csrf})
    return csrf


def _small_csv(days_ago: int = 1) -> bytes:
    from cashbridge import util
    day = util.format_date(util.add_days(util.today(), -days_ago))
    return (
        "date,description,amount,direction\n"
        f"{day},Daily sales - cash,450000,in\n"
        f"{day},Supplier payment - flour,1200000,out\n"
    ).encode("utf-8")


# ---------------------------------------------------------------------- pages

def test_index_lists_businesses(env):
    response = env["client"].get("/")
    assert response.status_code == 200
    assert "Warung Sembako" in response.text
    assert "Kirana Online" in response.text


def test_dashboard_gap_and_offer_form(env):
    response = env["client"].get(f"/business/{env['ids']['online']}")
    assert response.status_code == 200
    text = response.text
    assert "Cash gap projected" in text
    assert "Receivable Bridge" in text
    assert "Draft offer" in text
    assert "non-loan options" in text.lower()


def test_dashboard_blocked_financing(env):
    response = env["client"].get(f"/business/{env['ids']['katering']}")
    assert response.status_code == 200
    assert "Financing not recommended" in response.text


def test_dashboard_healthy_control(env):
    response = env["client"].get(f"/business/{env['ids']['sembako']}")
    assert response.status_code == 200
    assert "No cash shortfall projected" in response.text


def test_audit_page_chain_intact(env):
    response = env["client"].get("/audit")
    assert response.status_code == 200
    assert "Chain intact" in response.text


def test_bank_page(env):
    response = env["client"].get("/bank")
    assert response.status_code == 200
    assert "receivable_bridge" in response.text


# ---------------------------------------------------------------- ask + sim

def test_ask_api_is_grounded(env):
    response = env["client"].post(f"/api/business/{env['ids']['online']}/ask",
                                  json={"question": "Will I run short on cash?",
                                        "engine": "mock"})
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "mock"
    assert data["validation"]["ok"], data["validation"]["issues"]
    assert "cash gap" in data["answer"]["summary"].lower()


def test_ask_denied_after_consent_revoked(env):
    client, online = env["client"], env["ids"]["online"]
    revoke = client.post(f"/business/{online}/consent/revoke", follow_redirects=False)
    assert revoke.status_code == 303
    response = client.post(f"/api/business/{online}/ask",
                           json={"question": "What should I do?", "engine": "mock"})
    assert response.status_code == 200
    assert "consent" in response.json()["answer"]["summary"].lower()


def test_simulate_api_and_whitelist(env):
    client, online = env["client"], env["ids"]["online"]
    response = client.post(f"/api/business/{online}/simulate",
                           json={"adjustments": {"payout_shift_days": -3}})
    assert response.status_code == 200
    data = response.json()
    assert data["improvement"] >= 0
    assert "simulated_min_low" in data

    bad = client.post(f"/api/business/{online}/simulate",
                      json={"adjustments": {"wire_everything": 1}})
    assert bad.status_code == 400
    assert bad.json()["detail"]["code"] == "no_adjustments"


# ------------------------------------------------------------------- upload

def test_upload_and_dedup(env):
    client, online = env["client"], env["ids"]["online"]
    first = client.post(f"/business/{online}/upload",
                        files={"file": ("week.csv", _small_csv(), "text/csv")})
    assert first.status_code == 200
    assert "2 new transactions imported" in first.text

    second = client.post(f"/business/{online}/upload",
                         files={"file": ("week.csv", _small_csv(), "text/csv")})
    assert second.status_code == 200
    assert "duplicates skipped" in second.text

    conn = db.connect(env["db_path"])
    count = conn.execute("SELECT COUNT(*) AS n FROM transactions WHERE business_id = ? "
                         "AND source = 'csv'", (online,)).fetchone()["n"]
    conn.close()
    assert count == 2


# ------------------------------------------------------------ sandbox bank

def _first_eligible_product(env) -> str:
    conn = db.connect(env["db_path"])
    recommendation = recommend(conn, env["ids"]["online"])
    product = recommendation["loan"]["eligible"][0]["product_id"]
    conn.close()
    return product


def test_bank_offer_flow_and_idempotency(env):
    client, online = env["client"], env["ids"]["online"]
    draft = client.post("/api/bank/offers",
                        json={"business_id": online, "product_id": _first_eligible_product(env)})
    assert draft.status_code == 201, draft.text
    offer_id = draft.json()["offer_id"]

    approve = client.post(f"/api/bank/offers/{offer_id}/approve", json={})
    assert approve.status_code == 200
    assert approve.json()["status"] == "approved"
    token = approve.json()["approval_token"]

    key = "web-test-disburse-1"
    disburse = client.post(f"/api/bank/offers/{offer_id}/disburse",
                           headers={"Idempotency-Key": key, "X-Approval-Token": token})
    assert disburse.status_code == 201, disburse.text
    payload = disburse.json()
    assert payload["installments"] >= 4

    replay = client.post(f"/api/bank/offers/{offer_id}/disburse",
                         headers={"Idempotency-Key": key, "X-Approval-Token": token})
    assert replay.status_code == 409
    assert replay.json()["detail"]["code"] == "duplicate_action"

    loans = client.get(f"/api/bank/loans?business_id={online}").json()
    assert len(loans["loans"]) == 1
    assert len(loans["repayments"]) == payload["installments"]

    paid = client.post(f"/api/bank/repayments/{loans['repayments'][0]['id']}/pay")
    assert paid.status_code == 200
    assert paid.json()["status"] == "paid"


def test_disburse_without_approval_is_409(env):
    client, online = env["client"], env["ids"]["online"]
    draft = client.post("/api/bank/offers",
                        json={"business_id": online, "product_id": _first_eligible_product(env)})
    offer_id = draft.json()["offer_id"]
    disburse = client.post(f"/api/bank/offers/{offer_id}/disburse",
                           headers={"Idempotency-Key": "no-approval"})
    assert disburse.status_code == 409
    assert disburse.json()["detail"]["code"] == "approval_missing"


def test_bank_blocks_ineligible_product(env):
    client, katering = env["client"], env["ids"]["katering"]
    response = client.post("/api/bank/offers",
                           json={"business_id": katering, "product_id": "receivable_bridge"})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "ineligible"


# ------------------------------------------------------------------ onboard

def test_onboard_first_run(tmp_path):
    db_path = tmp_path / "first_run.db"
    app = create_app(db_path)
    with TestClient(app) as client:
        _login(client)
        page = client.get("/onboard")
        assert page.status_code == 200

        response = client.post("/onboard", follow_redirects=False,
                               data={"name": "Toko Baru", "sector": "retail",
                                     "safety_balance": "5.000.000",
                                     "opening_balance": "20.000.000"},
                               files={"file": ("start.csv", _small_csv(), "text/csv")})
        assert response.status_code == 303
        location = response.headers["location"]
        assert location.startswith("/business/")

        dashboard = client.get(location)
        assert dashboard.status_code == 200
        assert "Toko Baru" in dashboard.text
        assert "consent active" in dashboard.text

        conn = db.connect(db_path)
        business = conn.execute("SELECT * FROM businesses").fetchone()
        conn.close()
        assert business["name"] == "Toko Baru"
        assert float(business["safety_balance"]) == 5_000_000


# ------------------------------------------------------------------ security

def test_unauthenticated_is_rejected(env, tmp_path):
    anon = TestClient(env["client"].app)
    assert anon.get("/", follow_redirects=False).headers["location"] == "/login"
    assert anon.get("/api/businesses").status_code == 401
    online = env["ids"]["online"]
    assert anon.post("/api/bank/offers", json={"business_id": online,
                                               "product_id": "receivable_bridge"}).status_code == 401
    assert anon.get("/api/health").status_code == 200


def test_csrf_required_for_unsafe_requests(env):
    client, online = env["client"], env["ids"]["online"]
    no_csrf = client.post(f"/api/business/{online}/simulate",
                          json={"adjustments": {"payout_shift_days": -3}},
                          headers={"X-CSRF-Token": ""})
    assert no_csrf.status_code == 403
    assert no_csrf.json()["detail"]["code"] == "csrf"


def test_web_approve_requires_confirmation_and_executes_once(env):
    client, online = env["client"], env["ids"]["online"]
    offer_id = client.post("/api/bank/offers", json={"business_id": online,
                           "product_id": _first_eligible_product(env)}).json()["offer_id"]
    unconfirmed = client.post(f"/offers/{offer_id}/approve", data={"csrf_token": env["csrf"]})
    assert "Tick the box" in unconfirmed.text
    done = client.post(f"/offers/{offer_id}/approve", data={"csrf_token": env["csrf"], "confirm": "yes"})
    assert "Sandbox disbursement executed" in done.text
    again = client.post(f"/offers/{offer_id}/approve", data={"csrf_token": env["csrf"], "confirm": "yes"})
    assert "Sandbox disbursement executed" not in again.text
    loans = client.get(f"/api/bank/loans?business_id={online}").json()
    assert len(loans["loans"]) == 1


def test_disburse_needs_idempotency_key(env):
    client, online = env["client"], env["ids"]["online"]
    offer_id = client.post("/api/bank/offers", json={"business_id": online,
                           "product_id": _first_eligible_product(env)}).json()["offer_id"]
    assert client.post(f"/api/bank/offers/{offer_id}/disburse").status_code == 400


def test_other_owner_cannot_touch_business(env):
    from app import auth
    conn = db.connect(env["db_path"])
    conn.execute("INSERT INTO owners (username, display_name, password_hash) VALUES (?,?,?)",
                 ("mallory", "Mallory", auth.hash_password("mallory-password")))
    conn.commit()
    conn.close()
    other = TestClient(env["client"].app)
    other.post("/login", data={"username": "mallory", "password": "mallory-password"})
    conn = db.connect(env["db_path"])
    csrf = conn.execute("SELECT csrf FROM sessions WHERE username='mallory'").fetchone()[0]
    conn.close()
    online = env["ids"]["online"]
    assert other.get(f"/api/business/{online}/forecast").status_code == 403
    assert other.get("/api/businesses").json()["businesses"] == []
    assert other.post("/api/bank/offers", headers={"X-CSRF-Token": csrf},
                      json={"business_id": online, "product_id": "receivable_bridge"}).status_code == 403


def test_login_lockout(env):
    anon = TestClient(env["client"].app)
    for _ in range(5):
        anon.post("/login", data={"username": "demo", "password": "wrong"})
    locked = anon.post("/login", data={"username": "demo", "password": "cashbridge-demo"})
    assert "Too many failed logins" in locked.text
