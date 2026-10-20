"""Owner authentication for the web app: password login, server-side sessions, CSRF.

* Passwords: scrypt (stdlib). Sessions: random 32-byte token in an httpOnly,
  SameSite=Lax cookie; only its SHA-256 is stored. CSRF: per-session token that every
  state-changing request must echo (hidden form field or ``X-CSRF-Token`` header).
* Ownership: every business-scoped route checks ``owner_businesses``.
* Demo mode (default) creates one owner, ``demo``, who owns every business. The
  password comes from ``DEMO_PASSWORD`` (required when APP_ENV=production).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from typing import Optional

from fastapi import HTTPException, Request

from . import audit

COOKIE = "cb_session"
SESSION_TTL = 12 * 3600
MAX_FAILURES, FAILURE_WINDOW = 5, 15 * 60
DEMO_OWNER = "demo"


def production() -> bool:
    return os.getenv("APP_ENV", "development").lower() == "production"


def demo_password() -> str:
    pw = os.getenv("DEMO_PASSWORD", "")
    if pw:
        return pw
    if production():
        raise RuntimeError("DEMO_PASSWORD must be set when APP_ENV=production")
    return "cashbridge-demo"


def check_production_config() -> None:
    if not production():
        return
    problems = []
    if len(os.getenv("CASHBRIDGE_SECRET", "")) < 32:
        problems.append("CASHBRIDGE_SECRET (>= 32 chars) is required")
    if len(os.getenv("DEMO_PASSWORD", "")) < 12:
        problems.append("DEMO_PASSWORD (>= 12 chars) is required")
    if problems:
        raise RuntimeError("Unsafe production config: " + "; ".join(problems))


# ------------------------------------------------------------------ passwords

def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, salt, dk = stored.split("$")
        got = hashlib.scrypt(pw.encode(), salt=base64.b64decode(salt), n=2 ** 14, r=8, p=1, dklen=32)
        return hmac.compare_digest(got, base64.b64decode(dk))
    except Exception:
        return False


def ensure_demo_owner(conn) -> None:
    row = conn.execute("SELECT 1 FROM owners WHERE username = ?", (DEMO_OWNER,)).fetchone()
    if row is None:
        conn.execute("INSERT INTO owners (username, display_name, password_hash) VALUES (?,?,?)",
                     (DEMO_OWNER, "Demo owner", hash_password(demo_password())))
    conn.execute("""INSERT OR IGNORE INTO owner_businesses (username, business_id)
                    SELECT ?, id FROM businesses""", (DEMO_OWNER,))
    conn.commit()


# ------------------------------------------------------------------- sessions

def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def login(conn, username: str, password: str) -> Optional[tuple[str, str]]:
    """Return (session_token, csrf) or None. Locks out after repeated failures."""
    now = time.time()
    conn.execute("DELETE FROM login_failures WHERE at < ?", (now - FAILURE_WINDOW,))
    fails = conn.execute("SELECT COUNT(*) FROM login_failures WHERE username = ?", (username,)).fetchone()[0]
    if fails >= MAX_FAILURES:
        raise HTTPException(429, detail={"code": "locked", "reason": "Too many failed logins; try again in 15 minutes."})
    row = conn.execute("SELECT * FROM owners WHERE username = ?", (username,)).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        conn.execute("INSERT INTO login_failures (username, at) VALUES (?,?)", (username, now))
        conn.commit()
        audit.append(conn, "anonymous", "login_failed", "owner", username, {})
        return None
    conn.execute("DELETE FROM login_failures WHERE username = ?", (username,))
    conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
    conn.execute("INSERT INTO sessions (token_hash, username, csrf, expires_at) VALUES (?,?,?,?)",
                 (_h(token), username, csrf, now + SESSION_TTL))
    conn.commit()
    audit.append(conn, username, "login", "owner", username, {})
    return token, csrf


def logout(conn, token: Optional[str]) -> None:
    if token:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_h(token),))
        conn.commit()


def session(conn, request: Request) -> Optional[dict]:
    token = request.cookies.get(COOKIE)
    if not token:
        return None
    row = conn.execute("SELECT * FROM sessions WHERE token_hash = ? AND expires_at >= ?",
                       (_h(token), time.time())).fetchone()
    return dict(row) if row else None


async def csrf_ok(request: Request, sess: dict) -> bool:
    sent = request.headers.get("X-CSRF-Token")
    if not sent:
        ctype = request.headers.get("content-type", "")
        if "form" in ctype:
            form = await request.form()   # cached by Starlette; endpoints can still read it
            sent = form.get("csrf_token")
    return bool(sent) and hmac.compare_digest(str(sent), sess["csrf"])


def owns(conn, username: str, business_id: int) -> bool:
    return conn.execute("SELECT 1 FROM owner_businesses WHERE username = ? AND business_id = ?",
                        (username, business_id)).fetchone() is not None


def owned_ids(conn, username: str) -> list[int]:
    return [r[0] for r in conn.execute("SELECT business_id FROM owner_businesses WHERE username = ?",
                                       (username,))]
