"""CashBridge web app: FastAPI factory + server-rendered UI.

Pages (Jinja2):
  /                     business overview
  /business/{id}        dashboard: forecast, gap, options, offer flow, weekly
                        delta, bill reminders, ask-the-copilot, simulations
  /business/{id}/upload CSV (re-)upload
  /onboard              first-run: create a business, upload its transactions
  /bank                 sandbox bank page (products, loans, ledger, API docs)
  /audit                tamper-evident audit trail

The JSON + sandbox bank API lives in ``app.api``.

Run it:
    uvicorn app.main:create_app --factory --port 8000
or
    python scripts/run_web.py
"""
from __future__ import annotations

import json
import mimetypes
from datetime import date
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import api, audit, auth, service
from .guard import LoginRequired, guard
from cashbridge import config, db, forecast as forecast_mod
from cashbridge import ingest, recommend as recommend_mod, util, weekly

APP_DIR = Path(__file__).resolve().parent
mimetypes.add_type("font/woff2", ".woff2")  # bundled webfonts serve with the right type


def _short_money(value) -> str:
    """Compact money for ticker chips: 48.5M, 1.2B, 940K."""
    v = float(value or 0)
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.1f}B".replace(".0B", "B")
    if a >= 1e6:
        return f"{v / 1e6:.1f}M".replace(".0M", "M")
    if a >= 1e3:
        return f"{v / 1e3:.0f}K"
    return f"{v:.0f}"


templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
templates.env.filters["idr"] = lambda v: util.fmt_money(float(v or 0))
templates.env.filters["signed"] = lambda v: ("+" if float(v or 0) > 0 else "") + util.fmt_money(float(v or 0))
templates.env.filters["short"] = _short_money
templates.env.filters["signed_short"] = lambda v: (
    ("+" if float(v or 0) > 0 else "") + _short_money(float(v or 0))
)
templates.env.filters["pretty"] = lambda s: json.dumps(json.loads(s or "{}"), indent=2, sort_keys=True)


def get_conn(request: Request):
    conn = db.connect(request.app.state.db_path)
    try:
        yield conn
    finally:
        conn.close()


def _redirect(url: str, msg: str = None, level: str = "info") -> RedirectResponse:
    if msg:
        from urllib.parse import quote
        joiner = "&" if "?" in url else "?"
        url = f"{url}{joiner}msg={quote(msg)}&level={level}"
    return RedirectResponse(url=url, status_code=303)


def _parse_money(text, default: float = 0.0) -> float:
    """Accept '15000000', '15.000.000', '15,000,000', '15jt' is not supported."""
    if text is None:
        return default
    s = str(text).strip().replace("Rp", "").replace("rp", "").replace(" ", "")
    if not s:
        return default
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = (head.replace(",", "") + "." + tail) if len(tail) in (1, 2) else s.replace(",", "")
    elif "." in s:
        head, _, tail = s.rpartition(".")
        if len(tail) == 3:
            s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return default


def _engine_label() -> str:
    if config.settings.mock_llm or not config.settings.openrouter_api_key:
        return "template engine (offline)"
    return f"LLM: {config.settings.openrouter_model}"


def _business_strip(conn, active_id=None, owned=None) -> list:
    """Business ticker rows for the strip under the top nav (name, balance, 7-day delta)."""
    rows = []
    for overview in service.list_overview(conn):
        if owned is not None and overview["id"] not in owned:
            continue
        delta = weekly.weekly_delta(conn, overview["id"])
        rows.append({
            "id": overview["id"],
            "name": overview["name"],
            "sector": overview["sector"],
            "balance": overview["balance"],
            "safety_balance": overview["safety_balance"],
            "consent_active": overview["consent_active"],
            "delta7": float(delta["delta"]["net"] or 0),
            "active": overview["id"] == active_id,
        })
    return rows


def _dashboard_context(conn, business_id: int, msg: str = None, level: str = "info") -> dict:
    business = db.get_business(conn, business_id)
    recommendation = recommend_mod.recommend(conn, business_id)
    forecast = recommendation["forecast"]
    days = forecast["days"]
    chart = {
        "labels": [day["date"][5:] for day in days],
        "low": [day["balance_low"] for day in days],
        "mid": [day["balance_mid"] for day in days],
        "high": [day["balance_high"] for day in days],
        "safety": recommendation["safety_balance"],
    }
    offers = service.list_offers(conn, business_id)
    for offer in offers:
        last = conn.execute(
            "SELECT decision, actor, note, decided_at FROM approvals "
            "WHERE offer_id = ? ORDER BY id DESC LIMIT 1", (offer["id"],)).fetchone()
        offer["last_approval"] = dict(last) if last else None
    for offer in offers:
        offer["disclosure_obj"] = json.loads(offer["disclosure"]) if offer.get("disclosure") else None
    sandbox = service.sandbox_summary(conn, business_id)
    from cashbridge import reconcile
    risk = forecast["risk"]
    p = float(risk["p_breach"])
    status = ("act" if recommendation["gap"] and recommendation["gap"]["severity"] == "expected"
              else "watch" if p >= 0.10 else "safe")
    upload = dict(conn.execute(
        "SELECT COUNT(*) AS n, MAX(created_at) AS last FROM transactions "
        "WHERE business_id = ? AND source = 'csv'", (business_id,)).fetchone())
    return {
        "business": business,
        "balance": db.opening_balance(conn, business_id, recommendation["as_of"]),
        "consent_active": service.consent_active(conn, business_id),
        "scopes": service.SCOPES,
        "active_scopes": service.active_scopes(conn, business_id),
        "status": status,
        "p_breach": p,
        "recon": reconcile.propose_matches(conn, business_id),
        "reviews": [dict(r) for r in conn.execute(
            "SELECT * FROM review_requests WHERE business_id = ? ORDER BY id DESC LIMIT 3", (business_id,))],
        "rec": recommendation,
        "chart": chart,
        "offers": offers,
        "sandbox": sandbox,
        "scheduled": [r for r in sandbox["repayments"] if r["status"] == "scheduled"],
        "bills": forecast_mod.upcoming_bills(conn, business_id),
        "delta": weekly.weekly_delta(conn, business_id),
        "upload": upload,
        "audit_entries": audit.recent(conn, 6),
        "engine_label": _engine_label(),
        "strip": _business_strip(conn, active_id=business_id),
        "nav_active": "businesses",
        "msg": msg,
        "level": level,
    }


MAX_UPLOAD_BYTES = 1_000_000
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
       "base-uri 'self'; form-action 'self'; object-src 'none'")


def create_app(db_path=None) -> FastAPI:
    auth.check_production_config()
    app = FastAPI(title="CashBridge", docs_url=None if auth.production() else "/api/docs",
                  openapi_url=None if auth.production() else "/openapi.json", redoc_url=None,
                  dependencies=[Depends(guard)])
    app.state.db_path = str(db_path or config.settings.db_path)
    app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
    app.include_router(api.router)

    conn = db.connect(app.state.db_path)
    db.init_db(conn)
    auth.ensure_demo_owner(conn)
    conn.close()

    @app.exception_handler(LoginRequired)
    async def _to_login(request: Request, exc: LoginRequired):
        return RedirectResponse(url="/login", status_code=303)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if not request.url.path.startswith("/api/docs"):
            response.headers.setdefault("Content-Security-Policy", CSP)
        if auth.production():
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
        return response

    # ------------------------------------------------------------------ auth

    @app.get("/login")
    def login_form(request: Request, msg: str = None, level: str = "info"):
        return templates.TemplateResponse(request, "login.html", {
            "msg": msg, "level": level, "strip": [], "nav_active": None,
            "engine_label": _engine_label(), "demo": not auth.production()})

    @app.post("/login")
    def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
        conn = db.connect(app.state.db_path)
        try:
            result = auth.login(conn, username.strip().lower(), password)
        except HTTPException as exc:
            return _redirect("/login", exc.detail["reason"], "error")
        finally:
            conn.close()
        if result is None:
            return _redirect("/login", "Wrong username or password.", "error")
        token, _csrf = result
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(auth.COOKIE, token, httponly=True, samesite="lax",
                            secure=auth.production(), max_age=auth.SESSION_TTL, path="/")
        return response

    @app.post("/logout")
    def logout(request: Request, conn=Depends(get_conn)):
        auth.logout(conn, request.cookies.get(auth.COOKIE))
        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie(auth.COOKIE, path="/")
        return response

    # ------------------------------------------------------------------ pages

    @app.get("/")
    def index(request: Request, msg: str = None, level: str = "info", conn=Depends(get_conn)):
        businesses = _business_strip(conn, owned=request.state.owned)
        for business in businesses:
            recommendation = recommend_mod.recommend(conn, business["id"], persist=False)
            business["decision"] = recommendation["decision"]
            business["gap"] = recommendation["gap"]
        return templates.TemplateResponse(request, "index.html", {
            "businesses": businesses, "strip": businesses, "nav_active": "businesses",
            "engine_label": _engine_label(), "msg": msg, "level": level})

    @app.get("/business/{business_id}")
    def dashboard(request: Request, business_id: int, msg: str = None, level: str = "info",
                  conn=Depends(get_conn)):
        if db.get_business(conn, business_id) is None:
            return _redirect("/", f"Unknown business {business_id}", "error")
        context = _dashboard_context(conn, business_id, msg=msg, level=level)
        context["strip"] = _business_strip(conn, active_id=business_id, owned=request.state.owned)
        return templates.TemplateResponse(request, "dashboard.html", context)

    # ---------------------------------------------------------------- actions

    @app.post("/business/{business_id}/consent/grant")
    def consent_grant(business_id: int, conn=Depends(get_conn)):
        service.grant_consent(conn, business_id, actor="owner")
        return _redirect(f"/business/{business_id}", "Data-use consent granted.", "success")

    @app.post("/business/{business_id}/consent/{scope}/{action}")
    def consent_scope(request: Request, business_id: int, scope: str, action: str, conn=Depends(get_conn)):
        if scope not in service.SCOPES or action not in ("grant", "revoke"):
            return _redirect(f"/business/{business_id}", "Unknown consent option.", "error")
        if action == "grant":
            service.grant_consent(conn, business_id, scope, actor=request.state.owner)
        else:
            service.revoke_consent(conn, business_id, actor=request.state.owner, scope=scope)
        return _redirect(f"/business/{business_id}",
                         f"Consent updated: {service.SCOPES[scope]} — {action}ed.", "success")

    @app.post("/business/{business_id}/reconcile")
    def reconcile_confirm(request: Request, business_id: int, receivable_id: int = Form(...),
                          transaction_id: int = Form(...), conn=Depends(get_conn)):
        owner_rec = conn.execute("SELECT business_id FROM receivables WHERE id = ?", (receivable_id,)).fetchone()
        if not owner_rec or owner_rec["business_id"] != business_id:
            return _redirect(f"/business/{business_id}", "Unknown invoice.", "error")
        result = service.confirm_match(conn, receivable_id, transaction_id, actor=request.state.owner)
        if "error" in result:
            return _redirect(f"/business/{business_id}", result["error"]["reason"], "error")
        return _redirect(f"/business/{business_id}", "Invoice marked as paid. Forecast updated.", "success")

    @app.post("/business/{business_id}/review")
    def review_request(request: Request, business_id: int, conn=Depends(get_conn)):
        rec = recommend_mod.recommend(conn, business_id, persist=False)
        detail = {"decision": rec["decision"], "message": rec["message"],
                  "reason_codes": [p["reason_code"] for p in (rec["loan"] or {}).get("products", [])]}
        result = service.request_human_review(conn, business_id, "loan_decision", detail,
                                              actor=request.state.owner)
        return _redirect(f"/business/{business_id}",
                         f"Review #{result['review_id']} requested. A person will re-check this decision; "
                         "the automated result is not final.", "success")

    @app.post("/business/{business_id}/delete-data")
    def delete_data(request: Request, business_id: int, confirm: str = Form(None), conn=Depends(get_conn)):
        if confirm != "DELETE":
            return _redirect(f"/business/{business_id}", "Type DELETE to confirm erasing your data.", "error")
        service.delete_business_data(conn, business_id, actor=request.state.owner)
        return _redirect("/", "Your transactions, invoices and forecasts were erased. "
                              "Loan records and the audit trail are kept as required.", "success")

    @app.post("/business/{business_id}/consent/revoke")
    def consent_revoke(business_id: int, conn=Depends(get_conn)):
        service.revoke_consent(conn, business_id, actor="owner")
        return _redirect(f"/business/{business_id}",
                         "Consent revoked. Analysis actions are now blocked until you re-grant.",
                         "info")

    @app.post("/business/{business_id}/offers")
    def offer_draft(business_id: int, product_id: str = Form(...), conn=Depends(get_conn)):
        result = service.draft_offer(conn, business_id, product_id, actor="owner")
        if "error" in result:
            return _redirect(f"/business/{business_id}", result["error"]["reason"], "error")
        return _redirect(f"/business/{business_id}",
                         f"Offer #{result['id']} drafted. Review it, then approve or reject.",
                         "success")

    @app.post("/offers/{offer_id}/approve")
    def offer_approve(request: Request, offer_id: int, confirm: str = Form(None), conn=Depends(get_conn)):
        """Owner approval and sandbox execution in one deliberate action.

        The approval token is minted and consumed server-side within this request;
        it never reaches the browser, the LLM, or the bank API caller.
        """
        if confirm != "yes":
            return _offer_redirect(conn, offer_id, {"error": {"reason": (
                "Tick the box to confirm you read the amount, fee and repayment dates.")}}, "")
        decided = service.decide_offer(conn, offer_id, "approved", actor=request.state.owner)
        if "error" in decided:
            return _offer_redirect(conn, offer_id, decided, "")
        result = service.disburse_offer(conn, offer_id, actor=request.state.owner,
                                        approval_token=decided["approval_token"])
        return _offer_redirect(conn, offer_id, result,
                               "Approved. Sandbox disbursement executed; repayment schedule created.")

    @app.post("/offers/{offer_id}/reject")
    def offer_reject(request: Request, offer_id: int, conn=Depends(get_conn)):
        result = service.decide_offer(conn, offer_id, "rejected", actor=request.state.owner)
        return _offer_redirect(conn, offer_id, result, "Offer rejected.")

    def _offer_redirect(conn, offer_id: int, result: dict, success_msg: str):
        row = conn.execute("SELECT business_id FROM offers WHERE id = ?", (offer_id,)).fetchone()
        target = f"/business/{row['business_id']}" if row else "/"
        if "error" in result:
            return _redirect(target, result["error"]["reason"], "error")
        return _redirect(target, success_msg, "success")

    @app.post("/repayments/{repayment_id}/pay")
    def repayment_pay(repayment_id: int, conn=Depends(get_conn)):
        row = conn.execute(
            """SELECT o.business_id FROM repayments rp
               JOIN loans l ON l.id = rp.loan_id JOIN offers o ON o.id = l.offer_id
               WHERE rp.id = ?""", (repayment_id,)).fetchone()
        target = f"/business/{row['business_id']}" if row else "/"
        result = service.pay_repayment(conn, repayment_id, actor="owner")
        if "error" in result:
            return _redirect(target, result["error"]["reason"], "error")
        note = ("All installments paid; the sandbox loan is settled."
                if result["remaining"] == 0
                else f"Installment paid (sandbox). {result['remaining']} remaining.")
        return _redirect(target, note, "success")

    # ----------------------------------------------------------------- upload

    @app.post("/business/{business_id}/upload")
    async def upload(business_id: int, file: UploadFile = File(...), conn=Depends(get_conn)):
        if db.get_business(conn, business_id) is None:
            return _redirect("/", f"Unknown business {business_id}", "error")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            return _redirect(f"/business/{business_id}", "File too large (limit 1 MB).", "error")
        try:
            rows, warnings = ingest.parse_csv(data)
        except ValueError as exc:
            return _redirect(f"/business/{business_id}", str(exc), "error")
        if rows.empty:
            return _redirect(f"/business/{business_id}",
                             "No usable rows found in the file; check the column headers.", "error")
        result = ingest.ingest_rows(conn, business_id, rows, source="csv")
        audit.append(conn, "owner", "csv_uploaded", "business", str(business_id),
                     {"file": file.filename, "inserted": result["inserted"],
                      "duplicates": result["duplicates"], "warnings": len(warnings),
                      "needs_review": result["review_count"]})
        parts = [f"{result['inserted']} new transactions imported"]
        if result["duplicates"]:
            parts.append(f"{result['duplicates']} duplicates skipped")
        if warnings:
            parts.append(f"{len(warnings)} rows need review")
        if result["review_count"]:
            parts.append(f"{result['review_count']} categorized with low confidence")
        level = "success" if result["inserted"] else "info"
        return _redirect(f"/business/{business_id}", "; ".join(parts) + ".", level)

    # ---------------------------------------------------------------- onboard

    @app.get("/onboard")
    def onboard_form(request: Request, msg: str = None, level: str = "info", conn=Depends(get_conn)):
        return templates.TemplateResponse(request, "onboard.html",
                                          {"msg": msg, "level": level,
                                           "strip": _business_strip(conn, owned=request.state.owned),
                                           "nav_active": "onboard",
                                           "engine_label": _engine_label()})

    @app.post("/onboard")
    async def onboard_submit(request: Request,
                             name: str = Form(...),
                             sector: str = Form("retail"),
                             safety_balance: str = Form("0"),
                             opening_balance: str = Form("0"),
                             file: UploadFile = File(None),
                             conn=Depends(get_conn)):
        cur = conn.execute("INSERT INTO businesses (name, sector, safety_balance) VALUES (?,?,?)",
                           (name.strip(), sector.strip() or "retail",
                            _parse_money(safety_balance)))
        business_id = cur.lastrowid
        opening = _parse_money(opening_balance)
        if opening > 0:
            conn.execute(
                """INSERT INTO transactions
                   (business_id, date, description, counterparty, amount, direction, category,
                    is_cash_sale, source)
                   VALUES (?,?,?,?,?,'in','other_income',0,'onboard')""",
                (business_id, util.format_date(util.today()), "Opening balance", "Owner",
                 util.money(opening)))
        conn.execute("INSERT OR IGNORE INTO owner_businesses (username, business_id) VALUES (?,?)",
                     (request.state.owner, business_id))
        conn.commit()
        service.grant_consent(conn, business_id, actor=request.state.owner)

        imported = None
        if file is not None and file.filename:
            data = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                data = b""   # oversized: skipped, the owner can upload a smaller file later
            try:
                rows, warnings = ingest.parse_csv(data)
                if not rows.empty:
                    imported = ingest.ingest_rows(conn, business_id, rows, source="csv")
                    audit.append(conn, "owner", "csv_uploaded", "business", str(business_id),
                                 {"file": file.filename, "inserted": imported["inserted"],
                                  "duplicates": imported["duplicates"], "warnings": len(warnings)})
            except ValueError as exc:
                return _redirect(f"/business/{business_id}",
                                 f"Business created, but the CSV was rejected: {exc}", "error")

        msg = "Business created with consent granted."
        if imported:
            msg += f" {imported['inserted']} transactions imported."
        return _redirect(f"/business/{business_id}", msg, "success")

    # ------------------------------------------------------------- reference

    @app.get("/audit")
    def audit_page(request: Request, limit: int = 100, conn=Depends(get_conn)):
        return templates.TemplateResponse(request, "audit.html", {
            "chain": audit.verify_chain(conn),
            "entries": audit.recent(conn, min(max(limit, 10), 500)),
            "strip": _business_strip(conn, owned=request.state.owned),
            "nav_active": "audit",
            "engine_label": _engine_label()})

    @app.get("/bank")
    def bank_page(request: Request, conn=Depends(get_conn)):
        ledger = [dict(r) for r in conn.execute(
            """SELECT sl.*, b.name AS business_name FROM sandbox_ledger sl
               JOIN businesses b ON b.id = sl.business_id ORDER BY sl.id DESC LIMIT 50""")]
        loans = [dict(r) for r in conn.execute(
            """SELECT l.*, o.product_id, o.status AS offer_status, o.principal,
                      b.name AS business_name
               FROM loans l JOIN offers o ON o.id = l.offer_id
               JOIN businesses b ON b.id = o.business_id ORDER BY l.id DESC LIMIT 50""")]
        products = [dict(r) for r in conn.execute("SELECT * FROM products ORDER BY tenor_days")]
        return templates.TemplateResponse(request, "bank.html", {
            "products": products, "loans": loans, "ledger": ledger,
            "chain": audit.verify_chain(conn),
            "strip": _business_strip(conn, owned=request.state.owned),
            "nav_active": "bank",
            "engine_label": _engine_label()})

    return app
