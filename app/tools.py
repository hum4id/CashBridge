"""Tool registry exposed to the LLM orchestrator.

Design rules:
  * Every tool result is computed by deterministic code — the LLM never does
    arithmetic and never touches the database directly.
  * Every call goes through the policy engine (``action`` per tool) before the
    handler runs; denials are returned and audited, never silent.
  * Handlers are read-mostly: the only mutating tools are ``draft_offer``
    (T2, creates a draft) and repayment/schedule reads; execution still needs
    an explicit owner approval recorded in the service layer.
  * Untrusted strings (counterparty names from uploads) are wrapped with
    ``guardrails.wrap_untrusted`` before they reach the model.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Callable, Optional

from . import audit, guardrails, policy, service
from cashbridge import config, db, forecast as forecast_mod
from cashbridge import recommend as recommend_mod
from cashbridge import sizing, util
from cashbridge.forecast import run_forecast
from cashbridge.gap import detect_gaps

ALLOWED_ADJUSTMENTS = {"payout_shift_days", "payout_factor", "cash_rev_factor",
                       "cost_factor", "inventory_delay_days", "category_factors"}

_MAX_HORIZON = 45
_MIN_HORIZON = 7


def _as_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_of(args: dict) -> date:
    raw = args.get("as_of")
    return util.parse_date(raw) if raw else util.today()


def _require_business(conn, args: dict) -> dict:
    business = db.get_business(conn, args.get("business_id"))
    if business is None:
        raise service.ServiceError(f"Unknown business: {args.get('business_id')!r}")
    return business


def _bounded(value, default: int, low: int, high: int) -> int:
    parsed = _as_int(value)
    if parsed is None:
        parsed = default
    return max(low, min(high, parsed))


# --------------------------------------------------------------- condensing

def _condensed_forecast(f: dict) -> dict:
    days = [{"date": row["date"], "low": row["balance_low"], "mid": row["balance_mid"],
             "high": row["balance_high"]} for row in f["days"]]
    return {
        "as_of": f["as_of"],
        "horizon_days": f["horizon_days"],
        "opening_balance": f["opening_balance"],
        "open_receivables_total": f["open_receivables_total"],
        "min_balance": {
            "low": min(row["balance_low"] for row in f["days"]),
            "mid": min(row["balance_mid"] for row in f["days"]),
            "high": min(row["balance_high"] for row in f["days"]),
        },
        "forecast_days": days,
    }


_PRODUCT_FIELDS = ("product_id", "name", "provider", "description", "flat_fee_pct",
                   "tenor_days", "min_coverage_ratio", "principal", "total_repayment",
                   "coverage_ratio", "weekly_net_baseline", "eligible", "reason")


def _condense_recommendation(rec: dict) -> dict:
    out = {
        "business_id": rec["business_id"],
        "business_name": rec["business_name"],
        "as_of": rec["as_of"],
        "safety_balance": rec["safety_balance"],
        "decision": rec["decision"],
        "message": rec["message"],
        "gap": rec["gap"],
        "non_loan_options": rec["non_loan"],
        "loan": None,
        **_condensed_forecast(rec["forecast"]),
    }
    if rec["loan"] is not None:
        out["loan"] = {
            "principal_needed": rec["loan"]["principal_needed"],
            "blocked": rec["loan"]["blocked"],
            "block_reason": rec["loan"]["block_reason"],
            "products": [
                {**{k: e[k] for k in _PRODUCT_FIELDS}, "schedule": e["schedule"],
                 "apr_effective": e["cost_disclosure"]["apr_effective"],
                 "reason_code": e["reason_code"], "repay_prob": e["repay_prob"]}
                for e in rec["loan"]["products"]
            ],
        }
    return out


# ------------------------------------------------------------------ handlers

def _h_list_businesses(ctx, args) -> dict:
    return {"businesses": service.list_overview(ctx["conn"], as_of=_as_of(args))}


def _h_get_cash_overview(ctx, args) -> dict:
    conn = ctx["conn"]
    business = _require_business(conn, args)
    as_of = _as_of(args)
    balance = util.money(db.opening_balance(conn, business["id"], as_of))
    receivables = db.open_receivables(conn, business["id"])
    upcoming = [
        {"amount": util.money(r["amount"]),
         "expected_settlement_date": r["expected_settlement_date"],
         "days_until": util.days_between(as_of, r["expected_settlement_date"]),
         "debtor": guardrails.wrap_untrusted("receivable.debtor", r["debtor"])}
        for r in receivables[:5]
    ]
    return {
        "business": {"id": business["id"], "name": business["name"],
                     "sector": business["sector"]},
        "as_of": as_of.isoformat(),
        "balance": balance,
        "safety_balance": float(business["safety_balance"]),
        "headroom_above_safety": util.money(balance - float(business["safety_balance"])),
        "recent_weekly_net_cash": util.money(sizing.recent_weekly_net(conn, business["id"], as_of)),
        "open_receivables": {
            "count": len(receivables),
            "total": util.money(sum(float(r["amount"]) for r in receivables)),
            "next_expected": upcoming,
        },
    }


def _h_run_forecast(ctx, args) -> dict:
    conn = ctx["conn"]
    business = _require_business(conn, args)
    horizon = _bounded(args.get("horizon_days"), config.HORIZON_DAYS, _MIN_HORIZON, _MAX_HORIZON)
    forecast = run_forecast(conn, business["id"], as_of=_as_of(args),
                            horizon_days=horizon, persist=False)
    return {**_condensed_forecast(forecast),
            "safety_balance": float(business["safety_balance"]),
            "note": "What-if simulation only; not stored."}


def _h_detect_gaps(ctx, args) -> dict:
    conn = ctx["conn"]
    business = _require_business(conn, args)
    horizon = _bounded(args.get("horizon_days"), config.HORIZON_DAYS, _MIN_HORIZON, _MAX_HORIZON)
    forecast = run_forecast(conn, business["id"], as_of=_as_of(args),
                            horizon_days=horizon, persist=True)
    gaps = detect_gaps(conn, business["id"], forecast, float(business["safety_balance"]))
    return {
        "as_of": forecast["as_of"],
        "safety_balance": float(business["safety_balance"]),
        "opening_balance": forecast["opening_balance"],
        "min_balance_low": min(row["balance_low"] for row in forecast["days"]),
        "gap_count": len(gaps),
        "gaps": gaps,
    }


def _h_get_recommendations(ctx, args) -> dict:
    conn = ctx["conn"]
    business = _require_business(conn, args)
    out = _condense_recommendation(
        recommend_mod.recommend(conn, business["id"], as_of=_as_of(args))
    )
    if not service.consent_active(conn, business["id"], "financing"):
        # loan options are only shown under the separate financing consent
        out = {k: v for k, v in out.items() if k not in ("loan", "loan_products", "eligible_products")}
        out["financing_note"] = "Financing consent not granted: loan options are hidden."
        if out.get("decision") in ("loan_available", "loan_blocked"):
            out["decision"], out["message"] = "financing_consent_required", out["financing_note"]
    return out


def _clean_adjustments(raw) -> dict:
    """Whitelist + clamp what-if knobs. Anything unexpected is dropped."""
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        if key not in ALLOWED_ADJUSTMENTS:
            continue
        if key == "category_factors":
            if not isinstance(value, dict):
                continue
            factors = {}
            for category, factor in value.items():
                if category in forecast_mod.OUTFLOW_CATEGORIES:
                    try:
                        factors[category] = max(0.2, min(2.0, float(factor)))
                    except (TypeError, ValueError):
                        continue
            if factors:
                out[key] = factors
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if key == "payout_shift_days":
            number = max(-14.0, min(14.0, number))
        elif key == "inventory_delay_days":
            number = max(0.0, min(14.0, number))
        else:
            number = max(0.2, min(2.0, number))
        out[key] = number
    return out


def _h_simulate_scenario(ctx, args) -> dict:
    conn = ctx["conn"]
    business = _require_business(conn, args)
    horizon = _bounded(args.get("horizon_days"), config.HORIZON_DAYS, _MIN_HORIZON, _MAX_HORIZON)
    adjustments = _clean_adjustments(args.get("adjustments"))
    if not adjustments:
        return {"error": {"code": "no_adjustments", "tool": "simulate_scenario",
                          "reason": "Provide at least one adjustment, e.g. "
                                    "{'payout_shift_days': -3} or {'category_factors': {'other_opex': 0.6}}."}}
    as_of = _as_of(args)
    baseline = run_forecast(conn, business["id"], as_of=as_of, horizon_days=horizon,
                            persist=False)
    simulated = run_forecast(conn, business["id"], as_of=as_of, horizon_days=horizon,
                             adjustments=adjustments, persist=False)
    baseline_min = min(row["balance_low"] for row in baseline["days"])
    simulated_min = min(row["balance_low"] for row in simulated["days"])
    safety = float(business["safety_balance"])
    return {
        "applied_adjustments": adjustments,
        "horizon_days": horizon,
        "baseline_min_low": util.money(baseline_min),
        "simulated_min_low": util.money(simulated_min),
        "improvement": util.money(max(0.0, simulated_min - baseline_min)),
        "change_in_lowest_balance": util.money(simulated_min - baseline_min),
        "safety_balance": safety,
        "clears_safety": simulated_min >= safety,
        "simulated_forecast": _condensed_forecast(simulated),
    }


def _h_reconcile(ctx, args) -> dict:
    from cashbridge import reconcile
    conn = ctx["conn"]
    business = _require_business(conn, args)
    out = reconcile.propose_matches(conn, business["id"], as_of=_as_of(args))
    for m in out["matches"]:
        m["debtor"] = guardrails.wrap_untrusted("receivable.debtor", m["debtor"])
        m["description"] = guardrails.wrap_untrusted("transaction.description", m["description"])
    for o in out["overdue_unmatched"]:
        o["debtor"] = guardrails.wrap_untrusted("receivable.debtor", o["debtor"])
    for c in out["unmatched_credits"]:
        c["description"] = guardrails.wrap_untrusted("transaction.description", c["description"])
    return out


def _h_get_receivables(ctx, args) -> dict:
    conn = ctx["conn"]
    business = _require_business(conn, args)
    as_of = _as_of(args)
    limit = _bounded(args.get("limit"), 20, 1, 50)
    receivables = db.open_receivables(conn, business["id"])
    rows = [
        {
            "id": r["id"],
            "amount": util.money(r["amount"]),
            "expected_settlement_date": r["expected_settlement_date"],
            "days_until": util.days_between(as_of, r["expected_settlement_date"]),
            "debtor": guardrails.wrap_untrusted("receivable.debtor", r["debtor"]),
        }
        for r in receivables[:limit]
    ]
    return {
        "count": len(receivables),
        "total": util.money(sum(float(r["amount"]) for r in receivables)),
        "overdue_count": sum(1 for r in receivables
                             if util.days_between(as_of, r["expected_settlement_date"]) < 0),
        "receivables": rows,
    }


def _h_draft_offer(ctx, args) -> dict:
    conn = ctx["conn"]
    business = _require_business(conn, args)
    product_id = str(args.get("product_id") or "").strip()
    if not product_id:
        return {"error": {"code": "invalid_args", "tool": "draft_offer",
                          "reason": "product_id is required, e.g. 'receivable_bridge'."}}
    result = service.draft_offer(conn, business["id"], product_id,
                                 actor=ctx["actor"], granted_tier=ctx["granted_tier"],
                                 as_of=_as_of(args))
    if "error" in result:
        return {"error": {**result["error"], "tool": "draft_offer"}}
    return {
        "offer_id": result["id"],
        "status": result["status"],
        "product_id": result["product_id"],
        "principal": result["principal"],
        "tenor_days": result["tenor_days"],
        "total_repayment": result["total_repayment"],
        "coverage_ratio": result["coverage_ratio"],
        "rationale": result["rationale"],
        "requires_approval": True,
        "note": "Draft only. Disbursement happens after the owner approves it explicitly.",
    }


def _h_get_repayment_schedule(ctx, args) -> dict:
    conn = ctx["conn"]
    offer_id = _as_int(args.get("offer_id"))
    if offer_id is None:
        return {"error": {"code": "invalid_args", "tool": "get_repayment_schedule",
                          "reason": "offer_id (integer) is required."}}
    row = conn.execute("SELECT * FROM offers WHERE id = ?", (offer_id,)).fetchone()
    if row is None:
        raise service.ServiceError(f"Unknown offer: {offer_id}")
    repayments = [dict(r) for r in conn.execute(
        """SELECT id, due_date, amount, status, paid_at, linked_receivable_id
           FROM repayments WHERE loan_id = (SELECT id FROM loans WHERE offer_id = ?)
           ORDER BY id""", (offer_id,))]
    return {
        "offer_id": offer_id,
        "status": row["status"],
        "product_id": row["product_id"],
        "principal": row["principal"],
        "tenor_days": row["tenor_days"],
        "total_repayment": row["total_repayment"],
        "schedule": json.loads(row["schedule"] or "[]"),
        "repayments": repayments,
    }


def _h_get_sandbox_status(ctx, args) -> dict:
    conn = ctx["conn"]
    business = _require_business(conn, args)
    return service.sandbox_summary(conn, business["id"])


def _h_get_audit_trail(ctx, args) -> dict:
    conn = ctx["conn"]
    limit = _bounded(args.get("limit"), 15, 5, 50)
    chain = audit.verify_chain(conn)
    entries = [
        {"ts": e["ts"], "actor": e["actor"], "action": e["action"],
         "entity_type": e["entity_type"], "entity_id": e["entity_id"],
         "payload": json.loads(e["payload"] or "{}")}
        for e in audit.recent(conn, limit)
    ]
    return {"chain_ok": chain["ok"], "entries_total": chain["entries"], "entries": entries}


# ------------------------------------------------------------------ registry

def _schema(properties: dict, required: list) -> dict:
    return {"type": "object", "properties": properties, "required": required}


_BID = {"type": "integer", "description": "Business id"}
_AS_OF = {"type": "string", "description": "YYYY-MM-DD; defaults to today"}

TOOLS: dict = {
    "list_businesses": {
        "description": "List all businesses with balance and consent status.",
        "action": "view_data",
        "parameters": _schema({}, []),
        "handler": _h_list_businesses,
    },
    "get_cash_overview": {
        "description": "Current balance, safety threshold, weekly net cash, and open receivables for one business.",
        "action": "view_data",
        "parameters": _schema({"business_id": _BID, "as_of": _AS_OF}, ["business_id"]),
        "handler": _h_get_cash_overview,
    },
    "run_forecast": {
        "description": "14-day (or custom) balance projection in low/mid/high scenarios. Read-only.",
        "action": "view_data",
        "parameters": _schema({"business_id": _BID,
                               "horizon_days": {"type": "integer", "minimum": 7, "maximum": 45},
                               "as_of": _AS_OF}, ["business_id"]),
        "handler": _h_run_forecast,
    },
    "detect_gaps": {
        "description": "Find stretches where the projected balance dips below the safety threshold, with depth and shortfall.",
        "action": "view_data",
        "parameters": _schema({"business_id": _BID,
                               "horizon_days": {"type": "integer", "minimum": 7, "maximum": 45},
                               "as_of": _AS_OF}, ["business_id"]),
        "handler": _h_detect_gaps,
    },
    "get_recommendations": {
        "description": "Full analysis: gap, non-loan options with simulated impacts, loan options with eligibility and coverage. Always prefer this before proposing financing.",
        "action": "recommend_options",
        "parameters": _schema({"business_id": _BID, "as_of": _AS_OF}, ["business_id"]),
        "handler": _h_get_recommendations,
    },
    "simulate_scenario": {
        "description": ("What-if the forecast with adjustments: payout_shift_days (-14..14), payout_factor, "
                        "cash_rev_factor, cost_factor (0.2..2), inventory_delay_days (0..14), "
                        "category_factors {payroll|rent|utilities|inventory|other_opex: 0.2..2}."),
        "action": "simulate_scenario",
        "parameters": _schema({"business_id": _BID,
                               "adjustments": {"type": "object"},
                               "horizon_days": {"type": "integer", "minimum": 7, "maximum": 45},
                               "as_of": _AS_OF}, ["business_id", "adjustments"]),
        "handler": _h_simulate_scenario,
    },
    "get_receivables": {
        "description": "Open receivables (unpaid customer invoices) with expected settlement dates.",
        "action": "view_data",
        "parameters": _schema({"business_id": _BID,
                               "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                               "as_of": _AS_OF}, ["business_id"]),
        "handler": _h_get_receivables,
    },
    "reconcile_receivables": {
        "description": ("Propose matches between bank credits and open invoices, plus overdue invoices. "
                        "Read-only: the owner confirms each match in the UI."),
        "action": "view_data",
        "parameters": _schema({"business_id": _BID, "as_of": _AS_OF}, ["business_id"]),
        "handler": lambda ctx, args: _h_reconcile(ctx, args),
    },
    "draft_offer": {
        "description": ("Draft a proportional loan offer for ONE eligible product (e.g. 'receivable_bridge'). "
                        "Creates a draft only; the owner must approve before anything executes. "
                        "Only use when the owner asks for a financing option."),
        "action": "draft_offer",
        "parameters": _schema({"business_id": _BID,
                               "product_id": {"type": "string"},
                               "as_of": _AS_OF}, ["business_id", "product_id"]),
        "handler": _h_draft_offer,
    },
    "get_repayment_schedule": {
        "description": "Repayment schedule (installments) and payment status for an offer.",
        "action": "view_data",
        "parameters": _schema({"offer_id": {"type": "integer"}}, ["offer_id"]),
        "handler": _h_get_repayment_schedule,
    },
    "get_sandbox_status": {
        "description": "Sandbox loans, repayments, and ledger entries for a business.",
        "action": "view_data",
        "parameters": _schema({"business_id": _BID}, ["business_id"]),
        "handler": _h_get_sandbox_status,
    },
    "get_audit_trail": {
        "description": "Recent audit-log entries plus the hash-chain integrity check.",
        "action": "view_data",
        "parameters": _schema({"limit": {"type": "integer", "minimum": 5, "maximum": 50}}, []),
        "handler": _h_get_audit_trail,
    },
}


def openai_tool_specs() -> list:
    return [
        {"type": "function",
         "function": {"name": name, "description": tool["description"],
                      "parameters": tool["parameters"]}}
        for name, tool in TOOLS.items()
    ]


def execute_tool(conn, name: str, args: dict, granted_tier: str = "T2", actor: str = "agent",
                 default_business_id: Optional[int] = None) -> dict:
    """Policy-checked dispatch. Returns a plain dict, never raises for user errors."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": {"code": "unknown_tool", "reason": f"Unknown tool '{name}'."}}

    args = dict(args or {})
    if default_business_id is not None and "business_id" in tool["parameters"].get("properties", {}):
        args.setdefault("business_id", default_business_id)

    consent = True
    business_id = _as_int(args.get("business_id"))
    if business_id is not None:
        consent = service.consent_active(conn, business_id)

    decision = policy.check_action(tool["action"], granted_tier=granted_tier,
                                   consent_active=consent)
    if not decision["allow"]:
        audit.append(conn, actor, f"tool_denied:{name}", "business",
                     str(business_id) if business_id is not None else None,
                     {"code": decision["code"], "reason": decision["reason"]})
        return {"error": {"code": decision["code"], "reason": decision["reason"],
                          "tool": name}}

    try:
        return tool["handler"]({"conn": conn, "actor": actor, "granted_tier": granted_tier}, args)
    except service.ServiceError as exc:
        return {"error": {"code": "not_found", "reason": str(exc), "tool": name}}
    except (KeyError, ValueError, TypeError) as exc:
        return {"error": {"code": "invalid_args", "tool": name,
                          "reason": f"{type(exc).__name__}: {exc}"}}
