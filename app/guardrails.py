"""Guardrails between the LLM and the numbers.

Principle: the code computes, the LLM explains. This module:
  * wraps untrusted data (uploaded descriptions) so it is treated as data,
  * validates the LLM's structured explanation against the tool outputs,
  * enforces numeric consistency: every figure the LLM states must exist in
    the set of computed values (it may never invent or fudge amounts).
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from pydantic import BaseModel, Field


class KeyFigure(BaseModel):
    label: str
    value: float
    source: str = Field(description="Tool name or tool-call id that produced this value")


class Explanation(BaseModel):
    summary: str = Field(description="One or two sentences: what is happening")
    why: str = Field(description="Why it is happening, grounded in the computed drivers")
    next_steps: list[str] = Field(default_factory=list,
                                  description="Recommended actions, non-loan options first")
    key_figures: list[KeyFigure] = Field(default_factory=list,
                                         description="Every figure cited must come from tool output")
    confidence: str = Field(default="medium", description="low | medium | high")


_NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:[,.]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)([MmKk]?)")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_LETTER_DIGIT_RE = re.compile(r"([A-Za-z])(\d)")
_DOTTED_THOUSANDS_RE = re.compile(r"\d{1,3}(?:\.\d{3})+$")


def _text_numbers(text: str) -> list:
    """Numbers in prose, with ISO dates removed first."""
    return extract_numbers(_DATE_RE.sub(" ", text or ""))


def extract_numbers(text: str) -> list:
    """Pull candidate monetary/quantity figures out of free text."""
    found = []
    # "IDR12,373,412" -> "IDR 12,373,412" so the full figure matches instead
    # of a partial tail ("373,412") that would fail grounding.
    spaced = _LETTER_DIGIT_RE.sub(r"\1 \2", text or "")
    for match in _NUMBER_RE.finditer(spaced):
        raw, suffix = match.group(1), match.group(2)
        cleaned = raw.replace(",", "")
        if _DOTTED_THOUSANDS_RE.fullmatch(cleaned):   # 12.373.412 style
            cleaned = cleaned.replace(".", "")
        elif "." in cleaned:
            head, _, tail = cleaned.rpartition(".")
            if len(tail) == 3 and head.isdigit():   # 1.250.000 style
                cleaned = head + tail
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if suffix in ("M", "m"):
            value *= 1_000_000
        elif suffix in ("K", "k"):
            value *= 1_000
        found.append(value)
    return found


def match_allowed(value: float, allowed: Iterable[float], rel_tolerance: float = 0.02) -> bool:
    for candidate in allowed:
        if candidate == 0:
            if abs(value) < 1:
                return True
            continue
        if abs(value - candidate) / abs(candidate) <= rel_tolerance:
            return True
    return False


_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|persen)", re.IGNORECASE)


def _percent_ok(pct: float, allowed: list) -> bool:
    """A stated percentage must match a computed ratio (0.72 -> 72%) or a computed
    percentage, within one percentage point (rounding in prose)."""
    for candidate in allowed:
        if abs(candidate) <= 1.0 and abs(pct - candidate * 100) <= 1.0:
            return True
        if 1.0 < abs(candidate) <= 100 and abs(pct - candidate) <= 1.0:
            return True
    return False


def _check_percentages(text: str, allowed: list, where: str, issues: list) -> str:
    """Validate and strip percentages so they are not re-read as small counts."""
    for match in _PCT_RE.finditer(text or ""):
        pct = float(match.group(1))
        if not _percent_ok(pct, allowed):
            issues.append(f"percentage {pct:g}% in {where} is not grounded in computed outputs")
    return _PCT_RE.sub(" ", text or "")


def validate_explanation(explanation: Explanation, allowed_values: Iterable[float],
                         known_sources: Optional[Iterable[str]] = None) -> dict:
    """Return {"ok": bool, "issues": [str, ...]}.

    Checks:
      1. key_figures values must come from computed tool outputs,
      2. numbers in the free text must also be matched (or be small counts),
      3. cited sources must be tools that actually ran.
    """
    allowed = [float(v) for v in allowed_values]
    known = set(known_sources or [])
    issues = []

    for figure in explanation.key_figures:
        if not match_allowed(figure.value, allowed):
            issues.append(f"key figure '{figure.label}' = {figure.value:,.0f} "
                          f"does not match any computed value")
        if known and figure.source not in known:
            issues.append(f"key figure '{figure.label}' cites unknown source '{figure.source}'")

    for field_name in ("summary", "why"):
        text = _check_percentages(getattr(explanation, field_name), allowed, f"'{field_name}'", issues)
        for number in _text_numbers(text):
            if number < 100:            # small counts (days, months) are fine
                continue
            if not match_allowed(number, allowed):
                issues.append(f"number {number:,.0f} in '{field_name}' is not grounded "
                              f"in computed outputs")

    for step in explanation.next_steps:
        step = _check_percentages(step, allowed, "a next step", issues)
        for number in _text_numbers(step):
            if number < 100:
                continue
            if not match_allowed(number, allowed):
                issues.append(f"number {number:,.0f} in a next step is not grounded "
                              f"in computed outputs")

    return {"ok": not issues, "issues": issues}


_MASK_KEYS = {"debtor", "counterparty"}
_WRAP_RE = re.compile(r'<untrusted_data source="([^"]*)">\n(.*?)\n</untrusted_data>', re.S)


def _letters(i: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA (no digits, so aliases pass the no-digit rule)."""
    out = ""
    i += 1
    while i:
        i, rem = divmod(i - 1, 26)
        out = chr(65 + rem) + out
    return out


class Pseudonymizer:
    """Replace third-party names (debtors, counterparties) with stable aliases
    before tool output is sent to an external LLM, and restore them afterwards.
    The mapping never leaves the server."""

    def __init__(self):
        self.forward: dict = {}

    def _alias(self, name: str) -> str:
        if name not in self.forward:
            self.forward[name] = "Party " + _letters(len(self.forward))
        return self.forward[name]

    def mask(self, obj):
        if isinstance(obj, dict):
            out = {}
            for key, value in obj.items():
                if key in _MASK_KEYS and isinstance(value, str) and value:
                    wrapped = _WRAP_RE.fullmatch(value)
                    out[key] = (wrap_untrusted(wrapped.group(1), self._alias(wrapped.group(2)))
                                if wrapped else self._alias(value))
                else:
                    out[key] = self.mask(value)
            return out
        if isinstance(obj, list):
            return [self.mask(v) for v in obj]
        return obj

    def unmask(self, text: str) -> str:
        # longest alias first so "Party AB" is not clobbered by "Party A"
        for name, alias in sorted(self.forward.items(), key=lambda kv: -len(kv[1])):
            text = text.replace(alias, name)
        return text


def wrap_untrusted(label: str, text: str, max_len: int = 400) -> str:
    """Wrap data that came from uploads so the LLM treats it as data, not orders."""
    clipped = (text or "")[:max_len]
    return f"<untrusted_data source=\"{label}\">\n{clipped}\n</untrusted_data>"


# --------------------------------------------------------------- placeholders
# The LLM writes language only. Every figure in its text is a {{fact_key}} that the
# code fills from tool outputs, so a model can misplace a number but never invent one.

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z0-9_.\[\]]+)\s*\}\}")
_SKIP_KEYS = {"days", "p_below_by_day", "schedule", "scenarios", "adjustments", "misses"}
_MONEY_HINTS = ("amount", "balance", "shortfall", "depth", "principal", "repayment",
                "improvement", "total", "low", "mid", "high", "inflow", "outflow", "cost", "fees")
_PCT_HINTS = ("p_breach", "apr", "prob", "rate_cap")


def build_facts(results: dict, max_list: int = 3) -> dict:
    """Flatten numeric/date leaves of tool results into {dotted_key: value}."""
    facts: dict = {}

    def walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k not in _SKIP_KEYS:
                    walk(f"{prefix}.{k}", v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:max_list]):
                walk(f"{prefix}[{i}]", v)
        elif isinstance(obj, bool) or obj is None:
            return
        elif isinstance(obj, (int, float)):
            facts[prefix] = float(obj)
        elif isinstance(obj, str) and _DATE_RE.fullmatch(obj):
            facts[prefix] = obj

    for name, result in results.items():
        walk(name, result)
    return facts


def format_fact(key: str, value) -> str:
    if isinstance(value, str):
        return value
    leaf = key.rsplit(".", 1)[-1].lower()
    if any(h in leaf for h in _PCT_HINTS):
        return f"{value:.0%}" if value <= 1 else f"{value:.1f}%"
    if any(h in leaf for h in _MONEY_HINTS):
        return f"IDR {value:,.0f}"
    if leaf.endswith("ratio"):
        return f"{value:.2f}x"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


class PlaceholderError(ValueError):
    pass


def fill_placeholders(payload: dict, facts: dict) -> dict:
    """Reject digits outside placeholders and unknown keys; fill values; build key figures."""
    used: list = []

    def fill(text: str) -> str:
        bare = _PLACEHOLDER_RE.sub("", text or "")
        if re.search(r"\d", bare):
            raise PlaceholderError("text contains digits outside {{placeholders}}")
        def repl(m):
            key = m.group(1)
            if key not in facts:
                raise PlaceholderError(f"unknown placeholder {{{{{key}}}}}")
            used.append(key)
            return format_fact(key, facts[key])
        return _PLACEHOLDER_RE.sub(repl, text or "")

    out = dict(payload)
    out["summary"] = fill(payload.get("summary", ""))
    out["why"] = fill(payload.get("why", ""))
    out["next_steps"] = [fill(step) for step in payload.get("next_steps") or []]
    out["key_figures"] = [
        {"label": key.split(".", 1)[-1], "value": facts[key], "source": key.split(".", 1)[0]}
        for key in dict.fromkeys(used) if not isinstance(facts[key], str)
    ]
    return out


def content_label(engine: str, model: str = "") -> dict:
    """Explicit + machine-readable label for generated text (CN AIGC labelling measures, 2025)."""
    ai = engine == "llm"
    return {
        "ai_generated": ai,
        "generator": f"llm:{model}" if ai else "template",
        "text": ("AI-generated explanation. Figures are computed by CashBridge, not by the AI. "
                 "Check before acting." if ai else
                 "Template explanation from computed figures (no AI model used)."),
    }
