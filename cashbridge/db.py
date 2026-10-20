"""SQLite storage layer for CashBridge."""
from __future__ import annotations

import sqlite3
from typing import Optional

from . import config, util

SCHEMA = """
CREATE TABLE IF NOT EXISTS businesses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    sector TEXT NOT NULL,
    safety_balance REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    business_id INTEGER NOT NULL REFERENCES businesses(id),
    date TEXT NOT NULL,
    description TEXT NOT NULL,
    counterparty TEXT,
    amount REAL NOT NULL,                -- always positive
    direction TEXT NOT NULL CHECK (direction IN ('in','out')),
    category TEXT NOT NULL,              -- revenue|inventory|payroll|rent|utilities|other_opex|other_income|financing
    is_cash_sale INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'seed', -- seed|csv|sandbox
    dedup_hash TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_tx_business_date ON transactions(business_id, date);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tx_dedup ON transactions(business_id, dedup_hash) WHERE dedup_hash IS NOT NULL;

-- Future transactions used ONLY by evaluation/tests; the app never reads this table.
CREATE TABLE IF NOT EXISTS ground_truth_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    business_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    description TEXT NOT NULL,
    counterparty TEXT,
    amount REAL NOT NULL,
    direction TEXT NOT NULL,
    category TEXT NOT NULL,
    is_cash_sale INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_gt_business_date ON ground_truth_transactions(business_id, date);

CREATE TABLE IF NOT EXISTS receivables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    business_id INTEGER NOT NULL REFERENCES businesses(id),
    debtor TEXT NOT NULL,
    amount REAL NOT NULL,
    issued_date TEXT NOT NULL,
    expected_settlement_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',   -- open|settled
    settled_transaction_id INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_recv_business_status ON receivables(business_id, status);

CREATE TABLE IF NOT EXISTS consents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    business_id INTEGER NOT NULL REFERENCES businesses(id),
    scope TEXT NOT NULL,
    granted_at TEXT NOT NULL DEFAULT (datetime('now')),
    revoked_at TEXT
);

CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    business_id INTEGER NOT NULL REFERENCES businesses(id),
    run_at TEXT NOT NULL DEFAULT (datetime('now')),
    as_of TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    opening_balance REAL NOT NULL,
    params TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS forecast_days (
    forecast_id INTEGER NOT NULL REFERENCES forecasts(id) ON DELETE CASCADE,
    date TEXT NOT NULL,
    balance_low REAL NOT NULL,
    balance_mid REAL NOT NULL,
    balance_high REAL NOT NULL,
    inflow_mid REAL NOT NULL,
    outflow_mid REAL NOT NULL,
    PRIMARY KEY (forecast_id, date)
);

CREATE TABLE IF NOT EXISTS gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    business_id INTEGER NOT NULL REFERENCES businesses(id),
    forecast_id INTEGER NOT NULL REFERENCES forecasts(id),
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    min_balance REAL NOT NULL,
    depth REAL NOT NULL,                  -- safety_balance - min_balance
    duration_days INTEGER NOT NULL,
    shortfall REAL NOT NULL,              -- recommended funding incl. buffer
    severity TEXT NOT NULL DEFAULT 'expected',  -- expected|risk
    detected_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    flat_fee_pct REAL NOT NULL,
    tenor_days INTEGER NOT NULL,
    max_amount REAL NOT NULL,
    min_coverage_ratio REAL NOT NULL,
    description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS offers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    business_id INTEGER NOT NULL REFERENCES businesses(id),
    gap_id INTEGER REFERENCES gaps(id),
    product_id TEXT NOT NULL REFERENCES products(id),
    principal REAL NOT NULL,
    tenor_days INTEGER NOT NULL,
    total_repayment REAL NOT NULL,
    coverage_ratio REAL NOT NULL,
    schedule TEXT NOT NULL DEFAULT '[]',  -- JSON list of installments
    rationale TEXT,
    disclosure TEXT,                      -- JSON comprehensive financing-cost disclosure
    status TEXT NOT NULL DEFAULT 'draft', -- draft|approved|rejected|disbursed|settled
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS review_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    business_id INTEGER NOT NULL REFERENCES businesses(id),
    subject TEXT NOT NULL,                -- e.g. "loan_decision"
    detail TEXT,                          -- JSON snapshot of the automated decision
    requested_by TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',  -- open|resolved
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    offer_id INTEGER NOT NULL REFERENCES offers(id),
    actor TEXT NOT NULL,
    decision TEXT NOT NULL,               -- approved|rejected
    note TEXT,
    decided_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS loans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    offer_id INTEGER NOT NULL REFERENCES offers(id),
    principal REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active', -- active|settled
    disbursed_at TEXT
);

CREATE TABLE IF NOT EXISTS repayments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    loan_id INTEGER NOT NULL REFERENCES loans(id),
    due_date TEXT NOT NULL,
    amount REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'scheduled', -- scheduled|paid
    linked_receivable_id INTEGER,
    paid_at TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    payload TEXT NOT NULL DEFAULT '{}',
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approval_nonces (
    nonce TEXT PRIMARY KEY,
    offer_id INTEGER NOT NULL,
    owner TEXT NOT NULL,
    used_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Web layer: owners, ownership, sessions.
CREATE TABLE IF NOT EXISTS owners (
    username TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS owner_businesses (
    username TEXT NOT NULL,
    business_id INTEGER NOT NULL,
    PRIMARY KEY (username, business_id)
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    csrf TEXT NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS login_failures (username TEXT NOT NULL, at REAL NOT NULL);

-- The audit log is append-only at the database level too (the hash chain detects
-- tampering; these triggers prevent casual edits through the app's connection).
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;

CREATE TABLE IF NOT EXISTS sandbox_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    business_id INTEGER NOT NULL REFERENCES businesses(id),
    loan_id INTEGER REFERENCES loans(id),
    entry_type TEXT NOT NULL,             -- disbursement|repayment
    amount REAL NOT NULL,
    idempotency_key TEXT UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_RESET_ORDER = [
    "sessions",
    "owner_businesses",
    "sandbox_ledger",
    "approval_nonces",
    "repayments",
    "loans",
    "approvals",
    "offers",
    "gaps",
    "forecast_days",
    "forecasts",
    "products",
    "consents",
    "ground_truth_transactions",
    "receivables",
    "transactions",
    "audit_log",
    "businesses",
]


def connect(db_path=None) -> sqlite3.Connection:
    path = str(db_path or config.settings.db_path)
    # One connection per request; async endpoints may touch it on a different
    # worker thread than the dependency that created it, so cross-thread use
    # is allowed (strictly sequential, never shared concurrently).
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


_MIGRATIONS = [
    "ALTER TABLE offers ADD COLUMN disclosure TEXT",
]


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    for stmt in _MIGRATIONS:          # additive migrations for databases created earlier
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass                      # column already exists
    conn.commit()


def reset_all(conn: sqlite3.Connection) -> None:
    """Demo reset: wipes all demo data, including the (append-only) audit log.

    The audit triggers are dropped for the reset and re-created by ``init_db``;
    this is the only code path allowed to remove audit rows, and it is a full reset.
    """
    conn.execute("DROP TRIGGER IF EXISTS audit_no_delete")
    for table in _RESET_ORDER:
        conn.execute(f"DELETE FROM {table}")
    conn.executescript(SCHEMA)
    try:
        conn.execute("DELETE FROM sqlite_sequence")  # ids restart at 1 after a demo reset
    except sqlite3.OperationalError:
        pass
    conn.commit()


def list_businesses(conn: sqlite3.Connection) -> list:
    return [dict(r) for r in conn.execute("SELECT * FROM businesses ORDER BY id")]


def get_business(conn: sqlite3.Connection, key) -> Optional[dict]:
    """Look up a business by id, sector, or (partial) name."""
    if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
        row = conn.execute("SELECT * FROM businesses WHERE id = ?", (int(key),)).fetchone()
        return dict(row) if row else None
    row = conn.execute(
        "SELECT * FROM businesses WHERE sector = ? OR name LIKE ? ORDER BY id LIMIT 1",
        (key, f"%{key}%"),
    ).fetchone()
    return dict(row) if row else None


def opening_balance(conn: sqlite3.Connection, business_id: int, as_of) -> float:
    row = conn.execute(
        """
        SELECT COALESCE(SUM(CASE WHEN direction = 'in' THEN amount ELSE -amount END), 0) AS bal
        FROM transactions WHERE business_id = ? AND date <= ?
        """,
        (business_id, util.format_date(as_of)),
    ).fetchone()
    return float(row["bal"])


def open_receivables(conn: sqlite3.Connection, business_id: int) -> list:
    rows = conn.execute(
        "SELECT * FROM receivables WHERE business_id = ? AND status = 'open' ORDER BY expected_settlement_date",
        (business_id,),
    )
    return [dict(r) for r in rows]


def get_forecast(conn: sqlite3.Connection, forecast_id: int) -> Optional[dict]:
    row = conn.execute("SELECT * FROM forecasts WHERE id = ?", (forecast_id,)).fetchone()
    if not row:
        return None
    forecast = dict(row)
    days = conn.execute(
        "SELECT * FROM forecast_days WHERE forecast_id = ? ORDER BY date", (forecast_id,)
    )
    forecast["days"] = [dict(d) for d in days]
    return forecast
