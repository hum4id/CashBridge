"""Orchestrator tests: mock-engine determinism, grounding, gating, fallback."""
from __future__ import annotations

import pytest

from app import audit as audit_mod
from app import orchestrator, service
from cashbridge import db, seed

QUESTION = "Will I run short on cash in the next two weeks, and what should I do?"


@pytest.fixture()
def env(tmp_path):
    conn = db.connect(tmp_path / "orch.db")
    db.init_db(conn)
    ids = seed.run_seed(conn, as_of=None, seed=42)
    yield {"conn": conn, "ids": ids}
    conn.close()


def _ask(env, key, consent=True, **kwargs):
    conn = env["conn"]
    if consent:
        service.grant_consent(conn, env["ids"][key])
    kwargs.setdefault("force_engine", "mock")
    return orchestrator.answer_question(conn, env["ids"][key], QUESTION, **kwargs)


def test_mock_no_gap_is_grounded(env):
    result = _ask(env, "sembako")
    assert result["engine"] == "mock"
    assert result["validation"]["ok"], result["validation"]["issues"]
    answer = result["answer"]
    assert "no cash shortfall" in answer["summary"].lower()
    assert answer["key_figures"]
    actions = [e["action"] for e in audit_mod.recent(env["conn"], 5)]
    assert "advice_generated" in actions
    assert audit_mod.verify_chain(env["conn"])["ok"]


def test_mock_gap_puts_alternatives_first(env):
    result = _ask(env, "online")
    assert result["validation"]["ok"], result["validation"]["issues"]
    answer = result["answer"]
    steps = answer["next_steps"]
    assert steps
    option_steps = [i for i, s in enumerate(steps) if "simulated improvement" in s.lower()]
    loan_steps = [i for i, s in enumerate(steps)
                  if any(word in s.lower() for word in ("financing", "loan", "principal"))]
    assert option_steps, steps
    if loan_steps:
        assert min(option_steps) < min(loan_steps), steps
    labels = {figure["label"] for figure in answer["key_figures"]}
    assert any("Gap" in label for label in labels)
    assert "cash gap" in answer["summary"].lower()


def test_denied_without_consent(env):
    result = _ask(env, "online", consent=False)
    assert result["validation"]["ok"]
    answer = result["answer"]
    assert "consent" in answer["summary"].lower()
    assert not answer["key_figures"]
    assert [t for t in result["tool_calls"] if not t["ok"]]


def test_tier_gate_blocks_advice_at_t0(env):
    result = _ask(env, "online", granted_tier="T0")
    assert result["validation"]["ok"]
    answer = result["answer"]
    assert "tier T1" in answer["summary"]
    assert not answer["key_figures"]
    denied = [t for t in result["tool_calls"] if not t["ok"]]
    assert denied and denied[0]["name"] == "get_recommendations"


def test_llm_failure_falls_back_to_template(env, monkeypatch):
    def boom():
        raise RuntimeError("no LLM API key configured (set OPENROUTER_API_KEY in .env)")

    monkeypatch.setattr(orchestrator, "_build_client", boom)
    service.grant_consent(env["conn"], env["ids"]["online"], "external_ai")
    result = _ask(env, "online", force_engine="llm")
    assert result["engine"] == "mock_fallback"
    assert "LLM unavailable" in (result["note"] or "")
    assert result["validation"]["ok"], result["validation"]["issues"]
    assert result["answer"]["key_figures"]
