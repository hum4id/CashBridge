"""JSON API: dashboard data for the UI + the sandbox bank endpoints.

Two groups:
  /api/business/...   — read models and safe actions for the web UI
  /api/bank/...       — the simulated bank: products, offers, disbursement,
                        repayment. These are the only endpoints that move
                        sandbox money, and every one of them funnels through
                        the same policy engine + audit chain as everything else.

Error contract: service/policy errors come back as JSON
{"detail": {"code": ..., "reason": ...}} with the HTTP status mapped below.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from . import audit, orchestrator, service, tools
from cashbridge import db, forecast as forecast_mod, products as products_mod, recommend as recommend_mod, weekly

router = APIRouter(prefix="/api")

_CODE_STATUS = {
    "not_found": 404,
    "unknown_tool": 404,
    "tier_denied": 403,
    "consent_revoked": 403,
    "duplicate_action": 409,
    "approval_missing": 409,
    "approval_invalid": 409,
    "approval_expired": 409,
    "approval_wrong_offer": 409,
    "approval_terms_changed": 409,
    "offer_changed": 409,
    "invalid_state": 409,
    "amount_cap": 422,
    "ineligible": 422,
    "no_gap": 422,
    "invalid_args": 400,
    "no_adjustments": 400,
}


def get_conn(request: Request):
    conn = db.connect(request.app.state.db_path)
    try:
        yield conn
    finally:
        conn.close()


def _fail(error: dict):
    code = error.get("code", "error")
    raise HTTPException(status_code=_CODE_STATUS.get(code, 400),
                        detail={"code": code, "reason": error.get("reason", "")})


def _business_or_404(conn, business_id: int) -> dict:
    business = db.get_business(conn, business_id)
    if business is None:
        raise HTTPException(status_code=404, detail={"code": "not_found",
                                                     "reason": f"Unknown business: {business_id}"})
    return business


# ------------------------------------------------------------------ ui data

class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    engine: str = "auto"  # auto | mock | llm — "mock" pins the template engine (tests)
    # The tier is decided by the server (T2: drafts at most), never by the client.


class SimulateBody(BaseModel):
    adjustments: dict
    horizon_days: Optional[int] = None


@router.get("/health")
def health():
    return {"ok": True}


@router.get("/businesses")
def api_businesses(request: Request, conn=Depends(get_conn)):
    owned = set(request.state.owned)
    return {"businesses": [b for b in service.list_overview(conn) if b["id"] in owned]}


@router.get("/business/{business_id}/forecast")
def api_forecast(business_id: int, horizon_days: int = 14, conn=Depends(get_conn)):
    _business_or_404(conn, business_id)
    result = tools.execute_tool(conn, "run_forecast",
                                {"business_id": business_id, "horizon_days": horizon_days},
                                granted_tier="T0", actor="ui")
    if "error" in result:
        _fail(result["error"])
    return result


@router.get("/business/{business_id}/recommendation")
def api_recommendation(business_id: int, conn=Depends(get_conn)):
    _business_or_404(conn, business_id)
    return recommend_mod.recommend(conn, business_id)


@router.get("/business/{business_id}/bills")
def api_bills(business_id: int, horizon_days: int = 14, conn=Depends(get_conn)):
    _business_or_404(conn, business_id)
    bills = forecast_mod.upcoming_bills(conn, business_id, horizon_days=horizon_days)
    sandbox = service.sandbox_summary(conn, business_id)
    repayments = [r for r in sandbox["repayments"] if r["status"] == "scheduled"]
    return {"bills": bills, "bills_total": sum(b["amount"] for b in bills),
            "scheduled_repayments": repayments}


@router.get("/business/{business_id}/delta")
def api_delta(business_id: int, conn=Depends(get_conn)):
    _business_or_404(conn, business_id)
    return weekly.weekly_delta(conn, business_id)


@router.post("/business/{business_id}/ask")
def api_ask(business_id: int, body: AskBody, conn=Depends(get_conn)):
    _business_or_404(conn, business_id)
    if body.engine not in ("auto", "mock", "llm"):
        raise HTTPException(status_code=400, detail={"code": "invalid_args",
                                                     "reason": "engine must be auto, mock, or llm"})
    force_engine = None if body.engine == "auto" else body.engine
    return orchestrator.answer_question(conn, business_id, body.question,
                                        granted_tier="T2", actor="agent",
                                        force_engine=force_engine)


@router.post("/business/{business_id}/simulate")
def api_simulate(business_id: int, body: SimulateBody, conn=Depends(get_conn)):
    _business_or_404(conn, business_id)
    args = {"business_id": business_id, "adjustments": body.adjustments}
    if body.horizon_days:
        args["horizon_days"] = body.horizon_days
    result = tools.execute_tool(conn, "simulate_scenario", args,
                                granted_tier="T2", actor="ui")
    if "error" in result:
        _fail(result["error"])
    return result


class ConsentBody(BaseModel):
    scope: str = Field(max_length=100)
    action: str = Field(pattern="^(grant|revoke)$")


class MatchBody(BaseModel):
    receivable_id: int
    transaction_id: int


@router.get("/business/{business_id}/consent")
def api_consent(business_id: int, conn=Depends(get_conn)):
    _business_or_404(conn, business_id)
    active = service.active_scopes(conn, business_id)
    return {"scopes": [{"scope": k, "purpose": v, "granted": k in active} for k, v in service.SCOPES.items()]}


@router.post("/business/{business_id}/consent")
def api_consent_change(request: Request, business_id: int, body: ConsentBody, conn=Depends(get_conn)):
    _business_or_404(conn, business_id)
    if body.scope not in service.SCOPES:
        raise HTTPException(400, detail={"code": "invalid_args", "reason": "unknown scope"})
    if body.action == "grant":
        service.grant_consent(conn, business_id, body.scope, actor=request.state.owner)
    else:
        service.revoke_consent(conn, business_id, actor=request.state.owner, scope=body.scope)
    return api_consent(business_id, conn)


@router.get("/business/{business_id}/reconcile")
def api_reconcile(business_id: int, conn=Depends(get_conn)):
    from cashbridge import reconcile
    _business_or_404(conn, business_id)
    return reconcile.propose_matches(conn, business_id)


@router.post("/business/{business_id}/reconcile")
def api_reconcile_confirm(request: Request, business_id: int, body: MatchBody, conn=Depends(get_conn)):
    row = conn.execute("SELECT business_id FROM receivables WHERE id = ?", (body.receivable_id,)).fetchone()
    if not row or row["business_id"] != business_id:
        raise HTTPException(404, detail={"code": "not_found", "reason": "unknown receivable"})
    result = service.confirm_match(conn, body.receivable_id, body.transaction_id, actor=request.state.owner)
    if "error" in result:
        _fail(result["error"])
    return result


@router.post("/business/{business_id}/review")
def api_review(request: Request, business_id: int, conn=Depends(get_conn)):
    _business_or_404(conn, business_id)
    rec = recommend_mod.recommend(conn, business_id, persist=False)
    return service.request_human_review(conn, business_id, "loan_decision",
                                        {"decision": rec["decision"], "message": rec["message"]},
                                        actor=request.state.owner)


@router.post("/business/{business_id}/delete-data")
def api_delete(request: Request, business_id: int, confirm: str = Query(...), conn=Depends(get_conn)):
    if confirm != "DELETE":
        raise HTTPException(400, detail={"code": "invalid_args", "reason": "confirm=DELETE required"})
    return service.delete_business_data(conn, business_id, actor=request.state.owner)


# ------------------------------------------------------------ sandbox bank

class OfferBody(BaseModel):
    business_id: int
    product_id: str


class DecisionBody(BaseModel):
    note: Optional[str] = Field(default=None, max_length=500)
    # The actor is the logged-in owner (from the session), never a client-supplied string.


@router.get("/bank/products")
def bank_products(conn=Depends(get_conn)):
    products_mod.ensure_products(conn)
    return {"products": products_mod.list_products(conn)}


@router.get("/bank/loans")
def bank_loans(request: Request, business_id: Optional[int] = None, conn=Depends(get_conn)):
    owned = request.state.owned
    if business_id is not None:
        _business_or_404(conn, business_id)
        if business_id not in owned:
            raise HTTPException(403, detail={"code": "not_owner", "reason": "You do not own this business."})
        return service.sandbox_summary(conn, business_id)
    out = {"loans": [], "repayments": [], "ledger": []}
    for bid in owned:
        summary = service.sandbox_summary(conn, bid)
        for key in out:
            out[key].extend(summary[key])
    return out


@router.post("/bank/offers", status_code=201)
def bank_draft_offer(request: Request, body: OfferBody, conn=Depends(get_conn)):
    _business_or_404(conn, body.business_id)
    if body.business_id not in request.state.owned:
        raise HTTPException(403, detail={"code": "not_owner", "reason": "You do not own this business."})
    result = service.draft_offer(conn, body.business_id, body.product_id,
                                 actor="bank_sandbox", granted_tier="T2")
    if "error" in result:
        _fail(result["error"])
    return {"offer_id": result["id"], "status": result["status"],
            "product_id": result["product_id"], "principal": result["principal"],
            "tenor_days": result["tenor_days"], "total_repayment": result["total_repayment"],
            "coverage_ratio": result["coverage_ratio"],
            "requires_approval": "disbursement happens only after an explicit owner approval"}


@router.post("/bank/offers/{offer_id}/approve")
def bank_approve(request: Request, offer_id: int, body: DecisionBody, conn=Depends(get_conn)):
    """Owner approval (session + CSRF). Returns a single-use token bound to the terms."""
    result = service.decide_offer(conn, offer_id, "approved", actor=request.state.owner, note=body.note)
    if "error" in result:
        _fail(result["error"])
    return {"offer_id": offer_id, "status": result["status"],
            "approval_token": result["approval_token"], "expires_in_seconds": 300}


@router.post("/bank/offers/{offer_id}/reject")
def bank_reject(request: Request, offer_id: int, body: DecisionBody, conn=Depends(get_conn)):
    result = service.decide_offer(conn, offer_id, "rejected", actor=request.state.owner, note=body.note)
    if "error" in result:
        _fail(result["error"])
    return {"offer_id": offer_id, "status": result["status"]}


@router.post("/bank/offers/{offer_id}/disburse", status_code=201)
def bank_disburse(offer_id: int, idempotency_key: Optional[str] = Header(default=None),
                  x_approval_token: Optional[str] = Header(default=None),
                  conn=Depends(get_conn)):
    """Execute only with the owner's approval token (``X-Approval-Token``) and an
    ``Idempotency-Key``. The token is single-use and bound to the offer terms."""
    if not idempotency_key or len(idempotency_key) < 8:
        raise HTTPException(400, detail={"code": "invalid_args",
                                         "reason": "Idempotency-Key header (>= 8 chars) is required."})
    result = service.disburse_offer(conn, offer_id, idempotency_key=idempotency_key,
                                    actor="bank_sandbox", approval_token=x_approval_token)
    if "error" in result:
        _fail(result["error"])
    return result


@router.post("/bank/repayments/{repayment_id}/pay")
def bank_pay(repayment_id: int, conn=Depends(get_conn)):
    # duplicate attempts are guarded by the repayment row state itself
    result = service.pay_repayment(conn, repayment_id, actor="bank_sandbox")
    if "error" in result:
        _fail(result["error"])
    return result


@router.get("/bank/audit")
def bank_audit(limit: int = Query(25, ge=1, le=500), conn=Depends(get_conn)):
    chain = audit.verify_chain(conn)
    return {"chain": chain, "entries": audit.recent(conn, limit)}
