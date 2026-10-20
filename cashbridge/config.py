"""Central configuration for CashBridge, loaded from environment/.env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _num(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


DEMO_AS_OF = "2026-11-06"  # a Friday: the three demo stories hold on this date


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("CASHBRIDGE_DB", str(BASE_DIR / "cashbridge.db")))
    )
    mock_llm: bool = field(default_factory=lambda: _flag("MOCK_LLM", True))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    openrouter_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    )
    sandbox_max_principal: float = field(
        default_factory=lambda: _num("SANDBOX_MAX_PRINCIPAL", 25_000_000)
    )
    # Demo clock. The synthetic data is generated relative to this date, so every
    # run (tests, rehearsal, live demo) tells the same story regardless of the
    # real calendar day. Set CASHBRIDGE_AS_OF=today to follow the real clock.
    as_of: str = field(default_factory=lambda: os.getenv("CASHBRIDGE_AS_OF", DEMO_AS_OF))
    # Jurisdictions whose financing-cost caps are enforced (strictest wins). See COMPLIANCE.md.
    jurisdictions: str = field(default_factory=lambda: os.getenv("CASHBRIDGE_JURISDICTIONS", "CN,ID"))
    sandbox_daily_limit: float = field(
        default_factory=lambda: _num("SANDBOX_DAILY_LIMIT", 30_000_000)
    )


settings = Settings()

# --- Domain constants ---
HORIZON_DAYS = 14          # default forecast horizon shown to the owner
MAX_TENOR_DAYS = 45        # longest repayment tenor offered in the sandbox
MIN_LOAN_PRINCIPAL = 1_000_000
HISTORY_BASELINE_DAYS = 56  # "recent normal" window for capacity heuristics
CURRENCY = "IDR"

# Annualised comprehensive financing-cost caps (fees included), as a fraction per year.
#  CN: NFRA notice on internet lending by commercial banks (2025) caps the
#      comprehensive financing cost at 24% p.a. Drafts propose a further cut to
#      4x one-year LPR (~12%) by end-2027; not in force, not applied here.
#  ID: OJK SEOJK 19/2023 caps the daily "manfaat ekonomi" for productive
#      small/medium P2P loans at 0.1% per day (= 36.5% p.a.) from 2025.
RATE_CAPS = {"CN": 0.24, "ID": 0.365}
DSR_MAX = 0.50             # repayment per 30 days <= 50% of inflow of the last 30 days


def active_rate_cap() -> tuple[float, str]:
    """Strictest cap among configured jurisdictions: (cap, jurisdiction)."""
    codes = [c.strip().upper() for c in settings.jurisdictions.split(",") if c.strip()]
    caps = [(RATE_CAPS[c], c) for c in codes if c in RATE_CAPS]
    return min(caps) if caps else (min(RATE_CAPS.values()), "CN")
