"""Small shared helpers: dates, money formatting, medians."""
from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta

ISO = "%Y-%m-%d"


def parse_date(value) -> date:
    """Parse 'YYYY-MM-DD' or a date/datetime into a date."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value), ISO).date()


def today() -> date:
    """The app's notion of "today" (demo clock, see config.Settings.as_of).

    Never call ``date.today()`` directly in the app: the demo data is relative to
    this date, so mixing the two clocks makes results depend on the weekday.
    """
    from . import config  # local import: config imports nothing from util
    raw = (config.settings.as_of or "").strip().lower()
    if raw in ("", "today", "now"):
        return date.today()
    return parse_date(raw)


def format_date(value) -> str:
    return parse_date(value).isoformat()


def add_days(value, days: int) -> date:
    return parse_date(value) + timedelta(days=days)


def days_between(a, b) -> int:
    return (parse_date(b) - parse_date(a)).days


def money(value: float) -> float:
    """Round to whole currency units (store floats, avoid FP dust)."""
    return float(round(value))


def round_1k(value: float) -> float:
    return float(round(value / 1000.0) * 1000)


def round_100k(value: float) -> float:
    return float(round(value / 100_000.0) * 100_000)


def fmt_money(value: float, currency: str = "IDR") -> str:
    return f"{currency} {value:,.0f}"


def safe_median(values) -> float:
    values = [float(v) for v in values if v is not None]
    if not values:
        return 0.0
    return float(statistics.median(values))


def weekday_medians(pairs) -> dict:
    """pairs: iterable of (date, amount) -> {weekday 0..6: median amount}.

    Missing weekdays fall back to the overall median so projections never
    silently drop a day of activity.
    """
    buckets: dict[int, list[float]] = {}
    for d, amount in pairs:
        buckets.setdefault(parse_date(d).weekday(), []).append(float(amount))
    overall = safe_median([v for vs in buckets.values() for v in vs])
    return {wd: safe_median(buckets.get(wd, [overall])) if buckets.get(wd) else overall
            for wd in range(7)}


def weekday_means(pairs, window_days: int = 28) -> dict:
    """Expected daily amount per weekday: weekday total / weeks in the window.

    Mean (not median) keeps sparse categories honest: a supplier paid only on
    Mondays and Thursdays yields 0 on the other days, instead of smearing its
    typical purchase size across the whole week.
    """
    weeks = max(1.0, window_days / 7.0)
    buckets = {wd: 0.0 for wd in range(7)}
    for d, amount in pairs:
        buckets[parse_date(d).weekday()] += float(amount)
    return {wd: buckets[wd] / weeks for wd in range(7)}


def month_add(d: date, months: int) -> date:
    """Add months, clamping the day to the month's length."""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = d.day
    while True:
        try:
            return date(y, m, day)
        except ValueError:
            day -= 1
