"""Owner approval tokens for sandbox money movement (tier T3).

An approval is not a flag in a table that anyone can set. It is an HMAC-signed,
single-use, short-lived token bound to:
  * the owner and the business,
  * the exact offer terms (principal, total repayment, schedule hash),
  * a random nonce (recorded when consumed -> no replay).
Only an authenticated owner action can mint one (the LLM has no tool for it and
never sees the server secret). Changing any term after approval invalidates it.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

from cashbridge import config

TTL_SECONDS = 300


class ApprovalError(Exception):
    def __init__(self, code: str, reason: str):
        super().__init__(reason)
        self.code, self.reason = code, reason


def _secret() -> bytes:
    env = os.getenv("CASHBRIDGE_SECRET", "")
    if env:
        return env.encode()
    path = Path(config.settings.db_path).with_suffix(".secret")
    if not path.exists():
        path.write_text(secrets.token_hex(32))
    return path.read_text().strip().encode()


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def terms_hash(offer: dict) -> str:
    """Hash of everything the owner agreed to. Schedule is part of the terms."""
    terms = {k: offer[k] for k in ("id", "business_id", "product_id", "principal",
                                   "tenor_days", "total_repayment")}
    terms["schedule"] = json.loads(offer["schedule"]) if isinstance(offer["schedule"], str) else offer["schedule"]
    return hashlib.sha256(_canon(terms).encode()).hexdigest()


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def issue(owner: str, offer: dict, ttl: int = TTL_SECONDS) -> tuple[str, str]:
    """Return (token, nonce)."""
    payload = {"o": owner, "b": offer["business_id"], "f": offer["id"], "h": terms_hash(offer),
               "n": secrets.token_hex(16), "e": int(time.time()) + ttl}
    body = _b64(_canon(payload).encode())
    sig = hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}", payload["n"]


def verify(conn, token: str | None, offer: dict) -> dict:
    """Check signature, expiry, binding to the current offer terms, and single use."""
    if not token:
        raise ApprovalError("approval_missing", "An owner approval token is required to move money.")
    try:
        body, sig = token.split(".")
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except Exception:
        raise ApprovalError("approval_invalid", "Approval token is malformed.")
    expect = hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expect, sig):
        raise ApprovalError("approval_invalid", "Approval token signature does not match.")
    if payload.get("e", 0) < time.time():
        raise ApprovalError("approval_expired", "Approval token has expired; approve again.")
    if payload.get("f") != offer["id"] or payload.get("b") != offer["business_id"]:
        raise ApprovalError("approval_wrong_offer", "Approval token belongs to a different offer.")
    if payload.get("h") != terms_hash(offer):
        raise ApprovalError("approval_terms_changed", "Offer terms changed after approval.")
    used = conn.execute("SELECT 1 FROM approval_nonces WHERE nonce = ?", (payload["n"],)).fetchone()
    if used:
        raise ApprovalError("duplicate_action", "This approval was already used.")
    return payload


def consume(conn, payload: dict) -> None:
    conn.execute("INSERT INTO approval_nonces (nonce, offer_id, owner) VALUES (?,?,?)",
                 (payload["n"], payload["f"], payload["o"]))
