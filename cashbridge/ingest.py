"""CSV ingestion: parse bank/ledger exports, categorize, deduplicate."""
from __future__ import annotations

import hashlib
import io
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from . import util

COLUMN_ALIASES = {
    "date": ["date", "tanggal", "transaction date", "txn date", "posting date"],
    "description": ["description", "keterangan", "desc", "detail", "memo", "narration"],
    "amount": ["amount", "jumlah", "nominal", "value", "total"],
    "direction": ["direction", "type", "tipe", "jenis", "d/c", "debit/credit"],
    "counterparty": ["counterparty", "vendor", "merchant", "pihak", "nama", "party"],
}

# Order matters: the first matching rule wins.
CATEGORY_RULES = [
    ("payroll", ["salary", "payroll", "wage", "gaji", "upah", "crew", "staff"]),
    ("rent", ["rent", "lease", "sewa", "landlord"]),
    ("utilities", ["electric", "water", "utility", "listrik", "pdam", "pln", "internet", "isp", "phone"]),
    ("inventory", ["supplier", "stock", "restock", "inventory", "ingredient", "bahan", "purchase", "gudang", "grosir"]),
    ("revenue", ["sale", "sales", "revenue", "payout", "penjualan", "income", "cod", "payment received", "credit payment"]),
    ("other_opex", ["ads", "advertis", "marketing", "packaging", "delivery", "shipping", "ongkir", "transport", "packing"]),
    ("other_income", ["opening balance", "interest", "refund", "capital injection"]),
]

IN_CATEGORIES = {"revenue", "other_income"}
_OUT_DIRECTION_MAP = {
    "in": "in", "masuk": "in", "credit": "in", "cr": "in", "kredit": "in",
    "out": "out", "keluar": "out", "debit": "out", "dr": "out",
}

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def categorize(description: str, direction: str) -> Tuple[str, float]:
    """Map a free-text description to (category, confidence)."""
    text = (description or "").lower()
    direction = "in" if direction == "in" else "out"
    for category, keywords in CATEGORY_RULES:
        for kw in keywords:
            if kw in text:
                if direction == "in":
                    if category in IN_CATEGORIES:
                        return category, 0.9
                    return "other_income", 0.5
                # money going out
                if category in IN_CATEGORIES:
                    return "other_opex", 0.6
                return category, 0.9
    return ("other_income" if direction == "in" else "other_opex"), 0.4


def _read_raw(source) -> pd.DataFrame:
    if isinstance(source, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(source))
    if isinstance(source, (str, Path)) and Path(str(source)).exists():
        return pd.read_csv(source)
    return pd.read_csv(io.StringIO(str(source)))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {str(c).strip().lower(): c for c in df.columns}
    mapping = {}
    for std, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                mapping[std] = lower[alias]
                break
    missing = [k for k in ("date", "description", "amount") if k not in mapping]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(missing)}. "
            f"Expected headers like: date, description, amount, direction, counterparty."
        )
    df = df.rename(columns={v: k for k, v in mapping.items()})
    for optional in ("direction", "counterparty"):
        if optional not in df.columns:
            df[optional] = None
    return df


def _parse_amount(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    negative = text.startswith("-") or (text.startswith("(") and text.endswith(")"))
    text = re.sub(r"[^0-9.,]", "", text)
    if not text:
        return None
    # Thousands/decimal separator heuristics.
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", text):      # 1,234,567.89
        text = text.replace(",", "")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", text):    # 1.234.567,89
        text = text.replace(".", "").replace(",", ".")
    elif "," in text and "." not in text:
        head, _, tail = text.rpartition(",")
        if head and tail.isdigit() and len(tail) == 2:
            text = head.replace(",", "") + "." + tail       # 1234,56
        else:
            text = text.replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return -abs(number) if negative else number


def parse_csv(source) -> Tuple[pd.DataFrame, list]:
    """Parse a CSV into standardized rows: date, description, counterparty, amount, direction.

    Returns (rows, warnings). Rows with unparseable dates/amounts are dropped
    and reported in warnings so the owner can confirm them manually. Raises
    ValueError only when required columns are structurally missing.
    """
    raw = _normalize_columns(_read_raw(source))
    warnings: list = []
    records = []
    raw_dates = raw["date"].astype(str).str.strip()
    dayfirst = not raw_dates.str.match(_ISO_DATE).all()

    for idx, row in raw.iterrows():
        parsed_date = pd.to_datetime(str(row["date"]).strip(), errors="coerce", dayfirst=dayfirst)
        if pd.isna(parsed_date):
            warnings.append({"row": int(idx) + 2, "issue": f"unparseable date: {row['date']!r}"})
            continue
        amount = _parse_amount(row["amount"])
        if amount is None or amount == 0:
            warnings.append({"row": int(idx) + 2, "issue": f"unparseable/zero amount: {row['amount']!r}"})
            continue

        direction = row.get("direction")
        if direction is None or (isinstance(direction, float) and pd.isna(direction)):
            direction = "in" if amount > 0 else "out"
        else:
            key = str(direction).strip().lower()
            if key in _OUT_DIRECTION_MAP:
                direction = _OUT_DIRECTION_MAP[key]
            else:
                direction = "in" if amount > 0 else "out"

        records.append({
            "date": parsed_date.date(),
            "description": str(row["description"]).strip() or "(no description)",
            "counterparty": (str(row["counterparty"]).strip()
                             if row["counterparty"] is not None and not pd.isna(row["counterparty"]) else None),
            "amount": abs(float(amount)),
            "direction": direction,
        })

    if not records:
        empty = pd.DataFrame(columns=["date", "description", "counterparty", "amount", "direction"])
        return empty, warnings
    return pd.DataFrame.from_records(records), warnings


def ingest_rows(conn: sqlite3.Connection, business_id: int, rows: pd.DataFrame,
                source: str = "csv") -> dict:
    """Insert parsed rows, skipping duplicates. Flags low-confidence categories."""
    inserted = duplicates = 0
    review = []
    for _, row in rows.iterrows():
        date_s = util.format_date(row["date"])
        description = str(row["description"])
        amount = util.money(abs(float(row["amount"])))
        direction = str(row["direction"])
        category, confidence = categorize(description, direction)
        dedup = hashlib.sha1(
            f"{business_id}|{date_s}|{description.strip().lower()}|{amount:.0f}|{direction}".encode()
        ).hexdigest()
        try:
            conn.execute(
                """INSERT INTO transactions
                   (business_id, date, description, counterparty, amount, direction, category,
                    is_cash_sale, source, dedup_hash)
                   VALUES (?,?,?,?,?,?,?,0,?,?)""",
                (business_id, date_s, description, row.get("counterparty"), amount, direction,
                 category, source, dedup),
            )
            inserted += 1
            if confidence < 0.7:
                review.append({"date": date_s, "description": description,
                               "suggested_category": category, "confidence": confidence})
        except sqlite3.IntegrityError:
            duplicates += 1
    conn.commit()
    return {
        "inserted": inserted,
        "duplicates": duplicates,
        "needs_review": review,
        "review_count": len(review),
    }


def export_sample_csv(conn: sqlite3.Connection, business_id: int, path,
                      days: int = 30, as_of=None) -> None:
    """Export the last N days of transactions as an upload-ready CSV."""
    end = util.format_date(as_of) if as_of else util.today().isoformat()
    start = util.format_date(util.add_days(end, -int(days)))
    rows = conn.execute(
        """SELECT date, description, counterparty, amount, direction
           FROM transactions
           WHERE business_id = ? AND date > ? AND date <= ?
           ORDER BY date""",
        (business_id, start, end),
    ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
