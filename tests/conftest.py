"""Shared pytest fixtures: seeded demo databases."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cashbridge import db, seed, util


@pytest.fixture(scope="session")
def seeded(tmp_path_factory):
    """One seeded database shared by read-mostly analytics tests."""
    path = tmp_path_factory.mktemp("cashbridge_seed") / "seed.db"
    conn = db.connect(path)
    db.init_db(conn)
    ids = seed.run_seed(conn, as_of=None, seed=42)
    yield {"conn": conn, "ids": ids, "as_of": util.today()}
    conn.close()


@pytest.fixture(scope="module")
def mutable_env(tmp_path_factory):
    """A fresh database for tests that mutate data (CSV ingest)."""
    path = tmp_path_factory.mktemp("cashbridge_mutable") / "mutable.db"
    conn = db.connect(path)
    db.init_db(conn)
    ids = seed.run_seed(conn, as_of=None, seed=42)
    yield {"conn": conn, "ids": ids, "as_of": util.today()}
    conn.close()
