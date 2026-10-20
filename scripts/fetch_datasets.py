"""Rebuild data/external/online_retail_ii_daily.csv from the public source.

Source: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D  Licence: CC BY 4.0.
Primary URL is UCI; the fallback is the R package mirror (allanvc/onlineretail2,
which redistributes it with the donor's permission). Only the daily aggregate is
committed; the raw 1,067,371 rows are not.

Usage: python scripts/fetch_datasets.py   (needs: pandas, pyreadr for the mirror)
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "online_retail_ii_daily.csv"
UCI = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
MIRROR = "https://raw.githubusercontent.com/allanvc/onlineretail2/master/data/onlineretail2.rda"
MIRROR_SHA256 = "be2480b1fcb1fa123f90d731f63ab3cf44f6c97cba0a38eabee93759d411c6f7"


def _from_uci() -> pd.DataFrame:
    raw = urllib.request.urlopen(UCI, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.endswith(".xlsx"))
        sheets = pd.read_excel(z.open(name), sheet_name=None)
    df = pd.concat(sheets.values())
    return df.rename(columns={"Customer ID": "CustomerID"})


def _from_mirror() -> pd.DataFrame:
    import hashlib
    import tempfile
    import pyreadr
    raw = urllib.request.urlopen(MIRROR, timeout=120).read()
    if hashlib.sha256(raw).hexdigest() != MIRROR_SHA256:
        raise RuntimeError("mirror checksum mismatch")
    with tempfile.NamedTemporaryFile(suffix=".rda") as f:
        f.write(raw)
        f.flush()
        return pyreadr.read_r(f.name)["onlineretail2"]


def main() -> int:
    try:
        df = _from_uci()
    except Exception as exc:  # network policy, outage
        print(f"UCI unavailable ({exc}); using mirror", file=sys.stderr)
        df = _from_mirror()
    df["amount"] = df["Quantity"] * df["Price"]
    df["date"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
    g = df.groupby("date").agg(sales=("amount", lambda s: s[s > 0].sum()),
                               refunds=("amount", lambda s: -s[s < 0].sum()),
                               invoices=("Invoice", "nunique"),
                               customers=("CustomerID", "nunique")).round(2)
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"), fill_value=0)
    g.index.name = "date"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT)
    print(f"wrote {OUT} ({len(g)} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
