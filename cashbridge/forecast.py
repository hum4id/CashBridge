"""Probabilistic cash-flow forecast (Monte Carlo), still fully deterministic per seed.

Per simulated path n and day t (t = 1..H):

    flow[n,t] = base_in(t) - base_out(t)                 # weekday profile + recurring bills
              + resid[block_bootstrap(n,t)]              # day-to-day noise, resampled in 7-day blocks
              + receivables arriving on day t in path n  # per-debtor delay distribution
              + extra_flows(t)                           # loan disbursement / installments
    B[n,t]    = opening + sum_{s<=t} flow[n,s]

* ``balance_low/mid/high`` are the P10/P50/P90 of B[.,t] across paths.
* ``risk.p_breach`` = P(min_t B[n,t] < safety) is computed **per path**. The minimum
  of the daily P10 is *not* the 10% quantile of the path minimum: a path can dip on
  any day, so per-day quantiles understate the chance of breaching at some point.
* Receivable delays mix the debtor's own settlement history with the population
  (weight n/(n+3)), so a new debtor gets the conservative population view.
* Each path also carries a constant level error (standard error of the 4-week weekday
  means), so the band widens over time instead of only reflecting day-to-day noise.
* Block bootstrap keeps weekly patterns and runs of slow days that an i.i.d.
  resample would destroy (and with them the width of the band).

What-if ``adjustments`` reuse the same random draws (common random numbers), so the
difference between two scenarios is the effect of the change, not sampling noise.
No LLM anywhere in this module. Numbers in, numbers out.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from . import config, db, util

OUTFLOW_CATEGORIES = ["payroll", "rent", "utilities", "inventory", "other_opex"]

N_PATHS = 2000          # Monte Carlo paths (2k keeps a 56-day run well under 100 ms)
BLOCK_LEN = 7           # moving-block bootstrap length (one week)
RESID_WINDOW_DAYS = 56  # residual history window (8 weeks)
SHRINK_K = 3            # debtor history weight n/(n+K) vs population
PRIOR_DELAYS = [0, 1, 2, 3, 5, 7]   # conservative prior when the population is tiny
H_MAX = 64              # draws are always made for H_MAX days -> common random numbers
P_WARN = 0.10           # p_breach >= this -> "risk"
P_ACT = 0.30            # p_breach >= this -> "expected" gap
GAP_QUANTILE = 0.90     # shortfall reported at the 90th percentile of path shortfalls


@dataclass
class ScenarioParams:
    payout_shift_days: int = 0
    payout_factor: float = 1.0
    cash_rev_factor: float = 1.0
    cost_factor: float = 1.0
    inventory_delay_days: int = 0
    category_factors: dict = field(default_factory=dict)
    # targeted options (v2): these receivables are collected on their due date
    collect_on_due_ids: list = field(default_factory=list)
    # defer recurring bills due before ``defer_before_offset`` by N days.
    # Payroll and tax are never deferred (NON_DEFERRABLE).
    defer_bills_days: int = 0
    defer_before_offset: int = 0


# One parameter set: uncertainty now comes from the simulation, not from three
# hand-picked scenarios. What-if adjustments still override these knobs.
BASE_SCENARIOS = {"base": ScenarioParams()}

_ADJUSTABLE = ("payout_shift_days", "payout_factor", "cash_rev_factor",
               "cost_factor", "inventory_delay_days", "collect_on_due_ids",
               "defer_bills_days", "defer_before_offset")
NON_DEFERRABLE = ("payroll", "tax")
MIN_HISTORY_DAYS = 56   # below 8 weeks the forecast is shown as low-confidence and
                        # no financing is offered (reason code insufficient_history)


def _scenarios_with_adjustments(adjustments: Optional[dict]) -> dict:
    out = {}
    for name, base in BASE_SCENARIOS.items():
        params = replace(base)
        if adjustments:
            for key in _ADJUSTABLE:
                if adjustments.get(key) is not None:
                    setattr(params, key, adjustments[key])
            if adjustments.get("category_factors"):
                params.category_factors = {**params.category_factors,
                                           **adjustments["category_factors"]}
        out[name] = params
    return out


# --------------------------------------------------------------------- profiles

def _load_history(conn, business_id: int, as_of) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT date, amount, direction, category, is_cash_sale
           FROM transactions
           WHERE business_id = ? AND date <= ?""",
        conn,
        params=(business_id, util.format_date(as_of)),
        parse_dates=["date"],
    )


def _recurring_occurrences(df: pd.DataFrame, category: str, as_of: date,
                           horizon_days: int) -> Optional[list]:
    """Detect weekly/biweekly/monthly repeats. Returns occurrences in the
    horizon, or None when the category does not repeat on a stable cadence."""
    sub = df[(df["category"] == category) & (df["direction"] == "out")]
    if sub.empty:
        return None
    daily = sub.groupby(sub["date"].dt.date)["amount"].sum().sort_index()
    dates = list(daily.index)
    if len(dates) < 2:
        return None
    diffs = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    med = util.safe_median(diffs)
    if 6 <= med <= 8:
        step = 7
    elif 13 <= med <= 16:
        step = 14
    elif 27 <= med <= 32:
        step = None  # monthly via day-of-month
    else:
        return None
    amount = util.safe_median(list(daily)[-4:])
    occurrences = []
    last = dates[-1]
    if step is not None:
        k = 1
        while True:
            nxt = last + timedelta(days=step * k)
            offset = (nxt - as_of).days
            if offset > horizon_days:
                break
            if offset >= 1:
                occurrences.append({"day_offset": offset, "amount": float(amount),
                                    "category": category})
            k += 1
    else:
        dom = last.day

        def _clamped(year, month, day):
            while True:
                try:
                    return date(year, month, day)
                except ValueError:
                    day -= 1

        candidate = _clamped(as_of.year, as_of.month, dom)
        if candidate <= as_of:
            nxt = util.month_add(date(as_of.year, as_of.month, 1), 1)
            candidate = _clamped(nxt.year, nxt.month, dom)
        offset = (candidate - as_of).days
        if 1 <= offset <= horizon_days:
            occurrences.append({"day_offset": offset, "amount": float(amount),
                                "category": category})
    return occurrences


def build_profiles(conn, business_id: int, as_of: date, horizon_days: int) -> dict:
    """Learn this business's rhythm: recurring bills + variable weekday medians."""
    df = _load_history(conn, business_id, as_of)
    recent = df[df["date"].dt.date > as_of - timedelta(days=28)]

    cash_in = recent[(recent["category"] == "revenue") & (recent["is_cash_sale"] == 1)]
    cash_rev_by_wd = util.weekday_means(zip(cash_in["date"].dt.date, cash_in["amount"]), 28)
    if cash_in.empty:
        cash_rev_by_wd = {}  # no walk-in revenue at all (e.g. catering)

    scheduled_by_off: dict = {}
    scheduled_items: list = []
    variable_by_category: dict = {}
    scheduled_categories: list = []
    for category in OUTFLOW_CATEGORIES:
        occurrences = _recurring_occurrences(df, category, as_of, horizon_days)
        if occurrences is not None:
            scheduled_categories.append(category)
            for occurrence in occurrences:
                off = occurrence["day_offset"]
                scheduled_by_off[off] = scheduled_by_off.get(off, 0.0) + occurrence["amount"]
                scheduled_items.append(occurrence)
        else:
            sub = recent[(recent["category"] == category) & (recent["direction"] == "out")]
            if not sub.empty:
                variable_by_category[category] = util.weekday_means(
                    zip(sub["date"].dt.date, sub["amount"]), 28
                )
    # Settlements of credit sales (receivables) over the last 4 weeks. Used only for days
    # beyond the last open receivable: sales made after today will also settle later.
    settled = conn.execute(
        """SELECT t.date, t.amount FROM receivables r JOIN transactions t ON t.id = r.settled_transaction_id
           WHERE r.business_id = ? AND t.date > ? AND t.date <= ?""",
        (business_id, (as_of - timedelta(days=28)).isoformat(), as_of.isoformat())).fetchall()
    credit_in_by_wd = util.weekday_means([(r["date"], r["amount"]) for r in settled], 28) if settled else {}
    return {
        "credit_in_by_wd": credit_in_by_wd,
        "cash_rev_by_wd": cash_rev_by_wd,
        "scheduled_by_off": scheduled_by_off,
        "scheduled_items": scheduled_items,
        "scheduled_categories": scheduled_categories,
        "variable_by_category": variable_by_category,
        "history_rows": int(len(df)),
    }


# ------------------------------------------------------------------ simulation

def _day_flows(profiles: dict, params: ScenarioParams, day: date,
               payout_by_off: dict, offset: int):
    """Expected (deterministic) inflow/outflow for one day, before noise and receivables."""
    inflow = payout_by_off.get(offset, 0.0)
    if profiles["cash_rev_by_wd"]:
        inflow += profiles["cash_rev_by_wd"][day.weekday()] * params.cash_rev_factor

    outflow = profiles["scheduled_by_off"].get(offset, 0.0)
    for category, wd_medians in profiles["variable_by_category"].items():
        source_day = day
        if category == "inventory" and params.inventory_delay_days:
            source_day = day - timedelta(days=int(params.inventory_delay_days))
        amount = wd_medians[source_day.weekday()] * params.cost_factor
        amount *= params.category_factors.get(category, 1.0)
        outflow += amount
    return inflow, outflow


def _seed(business_id: int, as_of: date) -> int:
    """Seed depends only on (business, day): every what-if reuses the same draws."""
    digest = hashlib.sha256(f"{business_id}|{as_of.isoformat()}".encode()).hexdigest()
    return int(digest[:8], 16)


def _residuals(conn, business_id: int, as_of: date, scheduled_categories: list) -> np.ndarray:
    """Day-to-day noise of the *unscheduled* net flow over the last 8 weeks.

    Recurring bills, receivable settlements, financing and owner top-ups are modelled
    as events, so they are excluded here; what remains (walk-in sales minus variable
    spend) is centred per weekday, leaving only the noise to resample.
    """
    start = as_of - timedelta(days=RESID_WINDOW_DAYS)
    rows = conn.execute(
        """SELECT t.date, t.amount, t.direction, t.category, t.is_cash_sale
           FROM transactions t
           WHERE t.business_id = ? AND t.date > ? AND t.date <= ?
             AND t.category NOT IN ('financing', 'other_income')
             AND t.id NOT IN (SELECT settled_transaction_id FROM receivables
                              WHERE business_id = ? AND settled_transaction_id IS NOT NULL)""",
        (business_id, start.isoformat(), as_of.isoformat(), business_id)).fetchall()
    net = np.zeros(RESID_WINDOW_DAYS)
    for r in rows:
        if r["direction"] == "out" and r["category"] in scheduled_categories:
            continue
        if r["direction"] == "in" and not (r["category"] == "revenue" and r["is_cash_sale"]):
            continue
        i = (util.parse_date(r["date"]) - start).days - 1
        if 0 <= i < RESID_WINDOW_DAYS:
            net[i] += r["amount"] if r["direction"] == "in" else -r["amount"]
    weekdays = np.array([(start + timedelta(days=i + 1)).weekday() for i in range(RESID_WINDOW_DAYS)])
    centred = net.copy()
    for wd in range(7):
        mask = weekdays == wd
        centred[mask] -= net[mask].mean()
    return centred


def _open_receivables_at(conn, business_id: int, as_of: date) -> list:
    """Receivables that were open at the end of ``as_of`` (issued, not yet paid then)."""
    rows = conn.execute(
        """SELECT r.* FROM receivables r LEFT JOIN transactions t ON t.id = r.settled_transaction_id
           WHERE r.business_id = ? AND r.issued_date <= ? AND (t.id IS NULL OR t.date > ?)
           ORDER BY r.expected_settlement_date""",
        (business_id, as_of.isoformat(), as_of.isoformat())).fetchall()
    return [dict(r) for r in rows]


def _delay_history(conn, business_id: int, as_of: date) -> tuple[dict, list]:
    """Observed settlement delays (actual - expected, days) per debtor and overall,
    using only settlements known by ``as_of``."""
    rows = conn.execute(
        """SELECT r.debtor, r.expected_settlement_date, t.date AS paid
           FROM receivables r JOIN transactions t ON t.id = r.settled_transaction_id
           WHERE r.business_id = ? AND t.date <= ?""", (business_id, as_of.isoformat())).fetchall()
    by_debtor: dict = {}
    for r in rows:
        d = util.days_between(r["expected_settlement_date"], r["paid"])
        by_debtor.setdefault(r["debtor"], []).append(max(d, -7))
    population = [d for v in by_debtor.values() for d in v]
    if len(population) < 5:
        population = population + PRIOR_DELAYS
    return by_debtor, population


def _scheduled_installments(conn, business_id: int, as_of: date) -> list:
    """Unpaid installments of active sandbox loans are known future outflows."""
    rows = conn.execute(
        """SELECT rp.due_date, rp.amount FROM repayments rp
           JOIN loans l ON l.id = rp.loan_id JOIN offers o ON o.id = l.offer_id
           WHERE o.business_id = ? AND rp.status = 'scheduled'""", (business_id,)).fetchall()
    return [(max(util.days_between(as_of, r["due_date"]), 1), -float(r["amount"])) for r in rows]


def simulate_paths(conn, business_id: int, as_of: date, horizon: int, params: ScenarioParams,
                   extra_flows: Optional[list] = None, n_paths: int = N_PATHS,
                   include_loans: bool = True) -> dict:
    """Return {"paths": (N,H) balances, "arrivals": (N,R) arrival day per receivable,
    "receivables": [...], "base_in": (H,), "base_out": (H,), "opening": float}."""
    if horizon > H_MAX:
        raise ValueError(f"horizon {horizon} > H_MAX {H_MAX}")
    opening = db.opening_balance(conn, business_id, as_of)
    receivables = _open_receivables_at(conn, business_id, as_of)
    profiles = build_profiles(conn, business_id, as_of, horizon)
    resid = _residuals(conn, business_id, as_of, profiles["scheduled_categories"])
    by_debtor, population = _delay_history(conn, business_id, as_of)
    rng = np.random.default_rng(_seed(business_id, as_of))

    # --- draws in a fixed order and shape (independent of params) ---
    n_blocks = -(-H_MAX // BLOCK_LEN)
    starts = rng.integers(0, len(resid) - BLOCK_LEN + 1, size=(n_paths, n_blocks))
    u_mix = rng.random((n_paths, len(receivables)))
    u_pick = rng.random((n_paths, len(receivables)))
    z_level = rng.standard_normal(n_paths)

    if params.defer_bills_days and params.defer_before_offset:
        moved: dict = {}
        for item in profiles["scheduled_items"]:
            off = item["day_offset"]
            if item["category"] not in NON_DEFERRABLE and off < params.defer_before_offset:
                off += int(params.defer_bills_days)
            moved[off] = moved.get(off, 0.0) + item["amount"]
        profiles = {**profiles, "scheduled_by_off": moved}

    idx = (starts[..., None] + np.arange(BLOCK_LEN)).reshape(n_paths, -1)[:, :horizon]
    base_in = np.zeros(horizon)
    base_out = np.zeros(horizon)
    last_open = max([(util.parse_date(r["expected_settlement_date"]) - as_of).days for r in receivables] or [0])
    for t in range(horizon):
        day = as_of + timedelta(days=t + 1)
        base_in[t], base_out[t] = _day_flows(profiles, params, day, {}, t + 1)
        if t + 1 > last_open and profiles["credit_in_by_wd"]:
            base_in[t] += profiles["credit_in_by_wd"][day.weekday()] * params.payout_factor
    flow = (base_in - base_out)[None, :] + resid[idx] * params.cost_factor
    # Parameter uncertainty: the weekday means come from only 4 weeks of data, so each
    # path also gets a constant daily level error with the standard error of that mean.
    level_se = float(resid.std(ddof=1)) / np.sqrt(28.0) if len(resid) > 1 else 0.0
    flow += (z_level * level_se)[:, None]

    pop = np.array(population)
    arrivals = np.zeros((n_paths, len(receivables)), dtype=int)
    rec_in = np.zeros((n_paths, horizon))
    for j, r in enumerate(receivables):
        due = (util.parse_date(r["expected_settlement_date"]) - as_of).days
        overdue = max(1 - due, 0)            # must arrive on day >= 1, i.e. delay >= 1 - due
        own = np.array(by_debtor.get(r["debtor"], []))
        own_ok = own[own >= overdue] if overdue else own
        pop_ok = pop[pop >= overdue] if overdue else pop
        if len(pop_ok) == 0:
            pop_ok = np.array([overdue + 3])
        w = len(own_ok) / (len(own_ok) + SHRINK_K)
        pick_pop = pop_ok[np.minimum((u_pick[:, j] * len(pop_ok)).astype(int), len(pop_ok) - 1)]
        if len(own_ok):
            pick_own = own_ok[np.minimum((u_pick[:, j] * len(own_ok)).astype(int), len(own_ok) - 1)]
            delay = np.where(u_mix[:, j] < w, pick_own, pick_pop)
        else:
            delay = pick_pop
        if r["id"] in (params.collect_on_due_ids or []):
            delay = np.zeros_like(delay)   # collected on the due date (or today if overdue)
        arr = np.maximum(due + delay + int(params.payout_shift_days), 1)
        arrivals[:, j] = arr
        hit = arr <= horizon
        if hit.any():
            np.add.at(rec_in, (np.nonzero(hit)[0], arr[hit] - 1), float(r["amount"]) * params.payout_factor)
    flow += rec_in
    installments = _scheduled_installments(conn, business_id, as_of) if include_loans else []
    for offset, amount in list(extra_flows or []) + installments:
        if 1 <= offset <= horizon:
            flow[:, offset - 1] += amount
    paths = opening + np.cumsum(flow, axis=1)
    return {"paths": paths, "arrivals": arrivals, "receivables": receivables, "base_in": base_in,
            "base_out": base_out, "rec_in_mean": rec_in.mean(axis=0), "opening": opening}


CALIB_LAGS = (14, 21, 28)       # backtest origins (days before today)
CALIB_SCALES = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
CALIB_TARGET = 0.80             # nominal P10-P90 coverage


def calibration_scale(conn, business_id: int, as_of: date, params: ScenarioParams) -> float:
    """Widen the simulated spread until it would have covered the past.

    For a few past origins (14/21/28 days ago) the simulator is re-run with only the
    data available then, and compared with the balances that actually followed. The
    smallest spread multiplier whose P10-P90 band covers >= 80% of those days is used
    (deviations from the daily median are scaled; the median itself is unchanged).
    This is a simple backtest calibration, not a guarantee of future coverage.
    """
    devs, spreads = [], []
    for lag in CALIB_LAGS:
        origin = as_of - timedelta(days=lag)
        h = min(config.HORIZON_DAYS, lag)
        rows = conn.execute("SELECT MIN(date) m FROM transactions WHERE business_id = ?",
                            (business_id,)).fetchone()
        if not rows["m"] or util.days_between(rows["m"], origin) < RESID_WINDOW_DAYS:
            continue
        sim = simulate_paths(conn, business_id, origin, h, params, n_paths=600, include_loans=False)
        q = np.quantile(sim["paths"], [0.10, 0.50, 0.90], axis=0)
        actual = [db.opening_balance(conn, business_id, origin + timedelta(days=t + 1)) for t in range(h)]
        devs.extend(np.array(actual) - q[1])
        spreads.append(np.stack([q[1] - q[0], q[2] - q[1]]))
    if not devs:
        return 1.0
    devs = np.array(devs)
    lo = np.concatenate([sp[0] for sp in spreads])
    hi = np.concatenate([sp[1] for sp in spreads])
    for scale in CALIB_SCALES:
        inside = (devs >= -scale * lo) & (devs <= scale * hi)
        if inside.mean() >= CALIB_TARGET:
            return scale
    return CALIB_SCALES[-1]


def path_risk(paths: np.ndarray, safety: float, as_of: date) -> dict:
    """Breach probability and shortfall computed per path (not from daily quantiles)."""
    min_bal = paths.min(axis=1)
    shortfall = np.maximum(0.0, safety - min_bal)
    breach = paths < safety
    any_b = breach.any(axis=1)
    first = int(np.median(breach.argmax(axis=1)[any_b])) + 1 if any_b.any() else None
    return {
        "p_breach": round(float(any_b.mean()), 4),
        "shortfall_q90": util.money(float(np.quantile(shortfall, GAP_QUANTILE))),
        "first_breach_date": (as_of + timedelta(days=first)).isoformat() if first else None,
        "p_below_by_day": [round(float(x), 3) for x in breach.mean(axis=0)],
        "min_balance_p10": util.money(float(np.quantile(min_bal, 0.10))),
        "safety_balance": safety,
    }


def run_forecast(conn, business_id: int, as_of=None, horizon_days: Optional[int] = None,
                 adjustments: Optional[dict] = None, persist: bool = True,
                 extra_flows: Optional[list] = None, return_paths: bool = False) -> dict:
    """Project the daily balance for the next ``horizon_days``.

    ``balance_low/mid/high`` are P10/P50/P90 across simulated paths. ``risk`` holds the
    path-based breach probability. ``adjustments`` overrides the what-if knobs (pass
    ``persist=False`` for simulations); ``extra_flows`` = [(day_offset, +/-amount)].
    """
    as_of = util.parse_date(as_of) if as_of else util.today()
    horizon = int(horizon_days or config.HORIZON_DAYS)
    params = _scenarios_with_adjustments(adjustments)["base"]
    sim = simulate_paths(conn, business_id, as_of, horizon, params, extra_flows)
    scale = calibration_scale(conn, business_id, as_of, _scenarios_with_adjustments(None)["base"])
    med = np.median(sim["paths"], axis=0)
    paths = med[None, :] + scale * (sim["paths"] - med[None, :])
    q = np.quantile(paths, [0.10, 0.50, 0.90], axis=0)
    business = db.get_business(conn, business_id)
    safety = float(business["safety_balance"]) if business else 0.0
    first = conn.execute("SELECT MIN(date) m FROM transactions WHERE business_id = ? AND date <= ?",
                         (business_id, as_of.isoformat())).fetchone()["m"]
    history_days = util.days_between(first, as_of) if first else 0

    days = []
    for t in range(horizon):
        days.append({
            "date": (as_of + timedelta(days=t + 1)).isoformat(),
            "balance_low": util.money(q[0, t]),
            "balance_mid": util.money(q[1, t]),
            "balance_high": util.money(q[2, t]),
            "inflow_mid": util.money(sim["base_in"][t] + sim["rec_in_mean"][t]),
            "outflow_mid": util.money(sim["base_out"][t]),
        })

    result = {
        "business_id": business_id,
        "as_of": as_of.isoformat(),
        "horizon_days": horizon,
        "opening_balance": util.money(sim["opening"]),
        "open_receivables_total": util.money(sum(r["amount"] for r in sim["receivables"])),
        "days": days,
        "risk": path_risk(paths, safety, as_of),
        "method": (f"Monte Carlo, {paths.shape[0]} paths; weekly block bootstrap of daily noise; "
                   f"per-debtor payment delays; spread x{scale} from a 3-origin backtest. "
                   "Bands are P10-P90."),
        "calibration_scale": scale,
        "history_days": history_days,
        "data_sufficient": history_days >= MIN_HISTORY_DAYS,
        "adjustments": adjustments or {},
        "scenarios": {"base": dict(params.__dict__)},
    }
    if return_paths:
        result["_paths"] = paths
        result["_arrivals"] = sim["arrivals"]
        result["_receivables"] = sim["receivables"]
    if persist:
        cur = conn.execute(
            """INSERT INTO forecasts (business_id, as_of, horizon_days, opening_balance, params)
               VALUES (?,?,?,?,?)""",
            (business_id, result["as_of"], horizon, result["opening_balance"],
             json.dumps({"scenarios": result["scenarios"], "adjustments": result["adjustments"],
                         "risk": {k: v for k, v in result["risk"].items() if k != "p_below_by_day"}})),
        )
        forecast_id = cur.lastrowid
        conn.executemany(
            """INSERT INTO forecast_days
               (forecast_id, date, balance_low, balance_mid, balance_high, inflow_mid, outflow_mid)
               VALUES (?,?,?,?,?,?,?)""",
            [(forecast_id, r["date"], r["balance_low"], r["balance_mid"],
              r["balance_high"], r["inflow_mid"], r["outflow_mid"]) for r in days],
        )
        conn.commit()
        result["forecast_id"] = forecast_id
    return result


def min_balance(forecast: dict, scenario: str = "low") -> float:
    return min(row[f"balance_{scenario}"] for row in forecast["days"])


def upcoming_bills(conn, business_id: int, as_of=None, horizon_days: int = 14) -> list:
    """Recurring outflows expected in the next horizon (bill reminders).

    Uses the same cadence detection as the forecast, so the reminders and the
    projection can never disagree. Returns [{date, days_until, category, amount}].
    """
    as_of = util.parse_date(as_of) if as_of else util.today()
    horizon = int(horizon_days or config.HORIZON_DAYS)
    df = _load_history(conn, business_id, as_of)
    bills = []
    for category in OUTFLOW_CATEGORIES:
        occurrences = _recurring_occurrences(df, category, as_of, horizon)
        for occurrence in occurrences or []:
            bills.append({
                "date": (as_of + timedelta(days=occurrence["day_offset"])).isoformat(),
                "days_until": occurrence["day_offset"],
                "category": category,
                "amount": util.money(occurrence["amount"]),
            })
    bills.sort(key=lambda bill: (bill["date"], bill["category"]))
    return bills
