"""Deterministic synthetic data generator for the three CashBridge demo businesses.

Design goals:
- 12 weeks of history ending at ``as_of`` (transactions + receivables).
- 30 days of hidden ground-truth transactions for evaluation.
- Each business demonstrates a different story:
  * sembako  — healthy control, no cash gap expected.
  * katering — cash gap + thin repayment capacity => loan must be rejected.
  * online   — cash gap + strong repayment capacity => loan can be approved.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from . import util


@dataclass(frozen=True)
class BusinessSpec:
    key: str
    name: str
    sector: str
    safety_balance: float
    opening_balance: float


SPECS = [
    BusinessSpec("sembako", "Warung Sembako Bu Sari", "retail", 10_000_000, 42_000_000),
    BusinessSpec("katering", "Dapur Nusantara Catering", "catering", 15_000_000, 45_000_000),
    BusinessSpec("online", "Kirana Online Store", "ecommerce", 12_000_000, 10_000_000),
]

SEMBAKO_WEEKDAY_FACTOR = [0.85, 0.85, 0.90, 0.95, 1.10, 1.30, 1.15]  # Mon..Sun
CATERING_CLIENTS = [
    "PT Mitra Acara",
    "Hotel Senayan",
    "Universitas Harapan",
    "PT Sinar Event",
    "Kantor Pemerintah Kota",
]
MARKETPLACE = "Marketplace payout - Tokoo"


class Seeder:
    def __init__(self, conn, as_of: Optional[date] = None, seed: int = 42,
                 history_days: int = 84, future_days: int = 30):
        self.conn = conn
        self.as_of = util.parse_date(as_of) if as_of else util.today()
        self.seed = seed
        self.history_days = history_days
        self.future_days = future_days
        self.rng = random.Random(seed)
        self.business_id: int = 0
        self.spec: BusinessSpec = SPECS[0]

    # ------------------------------------------------------------------ helpers

    def _put(self, day_off: int, description: str, counterparty, amount: float,
             direction: str, category: str, cash_sale: bool = False) -> int:
        """Insert a transaction. day_off <= 0 -> history; > 0 -> ground truth."""
        d = (self.as_of + timedelta(days=day_off)).isoformat()
        amount = util.money(amount)
        if day_off <= 0:
            cur = self.conn.execute(
                """INSERT INTO transactions
                   (business_id, date, description, counterparty, amount, direction, category, is_cash_sale, source)
                   VALUES (?,?,?,?,?,?,?,?, 'seed')""",
                (self.business_id, d, description, counterparty, amount, direction, category, int(cash_sale)),
            )
        else:
            cur = self.conn.execute(
                """INSERT INTO ground_truth_transactions
                   (business_id, date, description, counterparty, amount, direction, category, is_cash_sale)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (self.business_id, d, description, counterparty, amount, direction, category, int(cash_sale)),
            )
        return cur.lastrowid

    def _receivable(self, issue_off: int, debtor: str, amount: float,
                    expected_delay: int, actual_delay: int) -> None:
        """Record a receivable and its (hidden) real settlement timing."""
        issue = self.as_of + timedelta(days=issue_off)
        expected = issue + timedelta(days=expected_delay)
        settle_off = issue_off + actual_delay
        amount = util.money(amount)
        if issue_off <= 0:
            settled_tx = None
            if settle_off <= 0:
                settled_tx = self._put(settle_off, f"Credit payment - {debtor}", debtor, amount, "in", "revenue")
            self.conn.execute(
                """INSERT INTO receivables
                   (business_id, debtor, amount, issued_date, expected_settlement_date, status, settled_transaction_id)
                   VALUES (?,?,?,?,?,?,?)""",
                (self.business_id, debtor, amount, issue.isoformat(), expected.isoformat(),
                 "settled" if settled_tx else "open", settled_tx),
            )
        if settle_off > 0:
            self._put(settle_off, f"Credit payment - {debtor}", debtor, amount, "in", "revenue")

    # ------------------------------------------------------------------ profile

    def run(self) -> dict:
        results = {}
        for i, spec in enumerate(SPECS):
            self.spec = spec
            self.rng = random.Random(self.seed + i * 101)
            cur = self.conn.execute(
                "INSERT INTO businesses (name, sector, safety_balance) VALUES (?,?,?)",
                (spec.name, spec.sector, spec.safety_balance),
            )
            self.business_id = cur.lastrowid
            self._put(-self.history_days, "Opening balance", "Owner", spec.opening_balance,
                      "in", "other_income")
            getattr(self, f"_gen_{spec.key}")()
            results[spec.key] = self.business_id
        self.conn.commit()
        return results

    # ------------------------------------------------------------------ stories

    def _gen_sembako(self) -> None:
        rng = self.rng
        for off in range(-self.history_days, self.future_days + 1):
            d = self.as_of + timedelta(days=off)
            sales = 2_400_000 * SEMBAKO_WEEKDAY_FACTOR[d.weekday()] * rng.uniform(0.85, 1.15)
            self._put(off, "Daily sales - cash", "Walk-in customers",
                      util.round_1k(sales), "in", "revenue", cash_sale=True)
            if rng.random() < 0.35:
                amount = util.round_1k(rng.uniform(400_000, 1_400_000))
                expected = rng.randint(3, 7)
                actual = max(1, expected + rng.randint(-1, 2))
                self._receivable(off, "Regular customer (credit book)", amount, expected, actual)
            if d.weekday() in (0, 3):  # Mon & Thu restock
                self._put(off, "Stock purchase - supplier", "PT Sumber Rejeki",
                          util.round_1k(rng.uniform(7_000_000, 8_500_000)), "out", "inventory")
            if d.weekday() == 5:  # Saturday payroll
                self._put(off, "Weekly salary - 2 staff", "Staff", 2_000_000, "out", "payroll")
            if d.day == 1:
                self._put(off, "Shop rent", "Landlord", 5_000_000, "out", "rent")
            if d.day == 5:
                self._put(off, "Electricity & water", "PLN & PDAM",
                          util.round_1k(rng.uniform(1_100_000, 1_300_000)), "out", "utilities")

    def _gen_katering(self) -> None:
        rng = self.rng
        # Schedule catering events week by week.
        event_days = []
        week = -(self.history_days // 7)
        while week * 7 < self.future_days + 7:
            count = rng.choices([1, 2, 3], weights=[0.25, 0.55, 0.20])[0]
            for day_in_week in rng.sample(range(7), count):
                off = week * 7 + day_in_week
                if -self.history_days <= off <= self.future_days:
                    event_days.append(off)
            week += 1
        for off in sorted(set(event_days)):
            invoice = util.round_1k(rng.uniform(8_000_000, 18_000_000))
            debtor = rng.choice(CATERING_CLIENTS)
            expected = rng.randint(12, 22)
            # Clients got slower recently: more slippage in the last 4 weeks.
            extra = rng.randint(0, 2) if off < -56 else rng.randint(2, 6)
            self._receivable(off, debtor, invoice, expected, expected + extra)
            ing_off = max(off - 1, -self.history_days)
            self._put(ing_off, "Ingredient purchase - event", "Pasar bahan",
                      util.round_1k(invoice * rng.uniform(0.58, 0.64)), "out", "inventory")
            self._put(off, "Event crew - freelance", "Freelance crew",
                      util.round_1k(invoice * rng.uniform(0.12, 0.15)), "out", "payroll")
        for off in range(-self.history_days, self.future_days + 1):
            d = self.as_of + timedelta(days=off)
            if d.day == 25:
                self._put(off, "Kitchen & office rent", "Landlord", 8_000_000, "out", "rent")
            if d.day == 20:
                self._put(off, "Electricity & water", "PLN & PDAM", 2_000_000, "out", "utilities")

    def _gen_online(self) -> None:
        rng = self.rng
        for off in range(-self.history_days, self.future_days + 1):
            d = self.as_of + timedelta(days=off)
            factor = 1.0
            if off % 10 == 0:
                factor = 1.8  # campaign day
            if d.weekday() >= 5:
                factor *= 1.12
            sales = util.round_1k(3_300_000 * factor * rng.uniform(0.80, 1.20))
            marketplace = util.round_1k(sales * 0.60)
            cod = sales - marketplace
            self._put(off, "Online sales - COD", "Courier COD", cod, "in", "revenue", cash_sale=True)
            expected = 15 + rng.randint(0, 4)
            actual = max(1, expected + rng.randint(-1, 3))
            self._receivable(off, MARKETPLACE, marketplace, expected, actual)
            self._put(off, "Marketplace ads", "Tokoo Ads",
                      util.round_1k(rng.uniform(320_000, 500_000)), "out", "other_opex")
            if d.weekday() == 1:  # Tuesday restock
                self._put(off, "Stock purchase - supplier", "Gudang Grosir",
                          util.round_1k(rng.uniform(11_500_000, 14_000_000)), "out", "inventory")
            if d.weekday() == 6:
                self._put(off, "Packaging supplies", "Packaging supplier", 700_000, "out", "other_opex")
            if d.day == 20:
                self._put(off, "Warehouse rent", "Landlord", 6_000_000, "out", "rent")
            if d.day == 15:
                self._put(off, "Electricity & internet", "PLN & ISP", 1_500_000, "out", "utilities")
            if d.weekday() == 4 and off % 14 == 0:  # biweekly Friday payroll
                self._put(off, "Team salary - 4 staff", "Staff", 4_500_000, "out", "payroll")
        # Campaign preparation purchases shortly before today: forces the demo dip.
        self._put(-3, "Bulk stock - campaign prep", "Gudang Grosir", 14_000_000, "out", "inventory")
        self._put(-2, "Campaign stock - second batch", "Gudang Grosir", 6_500_000, "out", "inventory")


def run_seed(conn, as_of: Optional[date] = None, seed: int = 42) -> dict:
    """Reset and regenerate all demo data. Returns {key: business_id}."""
    from . import db

    db.reset_all(conn)
    return Seeder(conn, as_of=as_of, seed=seed).run()
