"""App-wide request guard: login, CSRF and ownership for every route.

Registered as a FastAPI app-level dependency, so a new route cannot be added
without it. Static files are a mount (not a route) and are not affected.
"""
from __future__ import annotations

from fastapi import HTTPException, Request

from . import auth
from cashbridge import db

PUBLIC = {"/login", "/api/health"}
PUBLIC_DEV = {"/api/docs", "/openapi.json"}


class LoginRequired(Exception):
    """Raised for unauthenticated page requests; handled by a redirect to /login."""


async def guard(request: Request) -> None:
    path = request.url.path
    if path in PUBLIC or (path in PUBLIC_DEV and not auth.production()):
        return
    conn = db.connect(request.app.state.db_path)
    try:
        sess = auth.session(conn, request)
        if sess is None:
            if path.startswith("/api/"):
                raise HTTPException(401, detail={"code": "login_required", "reason": "Log in first."})
            raise LoginRequired()
        if request.method not in ("GET", "HEAD", "OPTIONS") and not await auth.csrf_ok(request, sess):
            raise HTTPException(403, detail={"code": "csrf", "reason": "Missing or invalid CSRF token."})
        owner = sess["username"]
        params = request.path_params
        business_id = None
        if "business_id" in params:
            business_id = int(params["business_id"])
        elif "offer_id" in params:
            row = conn.execute("SELECT business_id FROM offers WHERE id = ?", (int(params["offer_id"]),)).fetchone()
            business_id = row["business_id"] if row else None
        elif "repayment_id" in params:
            row = conn.execute(
                """SELECT o.business_id FROM repayments rp JOIN loans l ON l.id = rp.loan_id
                   JOIN offers o ON o.id = l.offer_id WHERE rp.id = ?""", (int(params["repayment_id"]),)).fetchone()
            business_id = row["business_id"] if row else None
        if business_id is not None and not auth.owns(conn, owner, business_id):
            raise HTTPException(403, detail={"code": "not_owner", "reason": "You do not own this business."})
        request.state.owner = owner
        request.state.csrf = sess["csrf"]
        request.state.owned = auth.owned_ids(conn, owner)
    finally:
        conn.close()
