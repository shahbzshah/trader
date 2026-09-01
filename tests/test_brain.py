"""Tests for app/brain.py."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

from app.brain import (
    Decision,
    LLMClient,
    VALID_ACTIONS,
    VALID_DIRECTIONS,
    VALID_ENTRY_TYPES,
    decide,
    extract_json_object,
    load_playbooks,
    load_risk_rules,
)
from tests.fake_http import make_chat_response, recording_transport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _LLM:
    base_url: str = "http://llm.local"
    model: str = "test-model"
    temperature: float = 0.2
    timeout_seconds: float = 30.0


@dataclass
class _Risk:
    min_confidence: float = 0.6
    max_risk_pct_per_trade: float = 1.0
    max_daily_loss_pct: float = 3.0
    max_open_trades: int = 3
    max_trades_per_day: int = 5
    default_playbook: str = "breakout_retest"


@dataclass
class _Paths:
    knowledge_dir: str = "knowledge"


@dataclass
class Cfg:
    mode: str = "demo"
    llm: _LLM = field(default_factory=_LLM)
    risk: _Risk = field(default_factory=_Risk)
    paths: _Paths = field(default_factory=_Paths)
    symbols: list = field(default_factory=lambda: ["EURUSD"])
    llm_api_key: str = "sk-test"


def _client_with(transport: httpx.MockTransport) -> LLMClient:
    return LLMClient(
        base_url="http://llm.local",
        model="m",
        api_key="sk-test",
        timeout=10.0,
        transport=transport,
    )


def _snapshot() -> dict:
    return {
        "symbol": "EURUSD",
        "timeframes": {"M15": {"last_close": 1.1, "ema20": 1.1, "ema50": 1.1,
                                "rsi14": 50.0, "atr14": 0.0002, "candles": []}},
        "session": {"high": 1.101, "low": 1.099, "start_ts": 1, "end_ts": 2},
        "spread": 5,
        "swing_high": 1.102,
        "swing_low": 1.098,
        "open_positions": [],
        "account": {"balance": 10000.0, "equity": 10000.0, "trade_mode": "demo"},
    }


# ---------------------------------------------------------------------------
# extract_json_object
# ---------------------------------------------------------------------------


def test_extract_json_pure():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert extract_json_object("```json\n{\"a\": 2}\n```") == {"a": 2}


def test_extract_json_embedded_in_prose():
    text = 'Here you go:\n{"action":"no_trade","symbol":"EURUSD"}\nDone.'
    out = extract_json_object(text)
    assert out == {"action": "no_trade", "symbol": "EURUSD"}


def test_extract_json_invalid_returns_none():
    assert extract_json_object("not json at all") is None
    assert extract_json_object("") is None


# ---------------------------------------------------------------------------
# load_risk_rules / load_playbooks
# ---------------------------------------------------------------------------


def test_load_risk_rules_parses_known_repo_file():
    rules = load_risk_rules(Path("knowledge/risk_rules.md"))
    assert len(rules) >= 1
    rule_texts = " ".join(r["rule"] for r in rules)
    assert "max_risk_pct: 1.0" in rule_texts
    assert "min_confidence: 0.6" in rule_texts


def test_load_playbooks_parses_repo_templates():
    pbs = load_playbooks(Path("knowledge"))
    names = {p["name"] for p in pbs}
    # File names are breakout_retest and trend_pullback; either the file
    # stem or the "## Name" body heading counts.
    assert any("Breakout" in n or "breakout" in p["path"] for n in names for p in pbs)
    assert any("Trend" in n or "trend" in p["path"] for n in names for p in pbs)


def test_load_risk_rules_handles_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_risk_rules(tmp_path / "nope.md")


# ---------------------------------------------------------------------------
# LLMClient chat
# ---------------------------------------------------------------------------


def test_llm_client_posts_correct_payload():
    transport, captured = recording_transport(
        [make_chat_response("ok")]
    )
    client = _client_with(transport)
    resp = client.chat(
        [{"role": "user", "content": "hi"}],
        temperature=0.1,
    )
    assert resp.content == "ok"
    assert len(captured) == 1
    req = captured[0]
    body = json.loads(req.content.decode("utf-8"))
    assert body["model"] == "m"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.1
    assert req.headers.get("authorization") == "Bearer sk-test"
    assert "/chat/completions" in str(req.url)


def test_llm_client_omits_auth_when_key_empty():
    transport, captured = recording_transport([make_chat_response("ok")])
    client = LLMClient(
        base_url="http://llm.local", model="m", api_key="",
        timeout=10.0, transport=transport,
    )
    client.chat([{"role": "user", "content": "x"}])
    assert "authorization" not in {k.lower() for k in captured[0].headers.keys()} or \
           captured[0].headers.get("authorization") is None


# ---------------------------------------------------------------------------
# decide: happy path
# ---------------------------------------------------------------------------


def test_decide_parses_valid_json():
    valid = {
        "action": "no_trade",
        "symbol": "EURUSD",
        "direction": None,
        "entry_type": "market",
        "price": None,
        "stop_loss": None,
        "take_profit": None,
        "risk_pct": 0.0,
        "confidence": 0.9,
        "playbook": "breakout_retest",
        "reasoning": "no setup",
    }
    transport, captured = recording_transport(
        [make_chat_response(json.dumps(valid))]
    )
    client = _client_with(transport)
    decision, raw = decide(
        _snapshot(), [], [], {"daily_pnl": 0.0, "remaining_risk_pct": 100.0},
        Cfg(), client=client,
    )
    assert isinstance(decision, Decision)
    assert decision.action == "no_trade"
    assert decision.confidence == 0.9
    # Only one LLM call on success.
    assert len(captured) == 1


def test_decide_parses_json_embedded_in_prose():
    valid = {"action": "no_trade", "symbol": "EURUSD"}
    body = f"Sure! Here is the call:\n```json\n{json.dumps(valid)}\n```\nDone."
    transport, _ = recording_transport([make_chat_response(body)])
    client = _client_with(transport)
    decision, _ = decide(
        _snapshot(), [], [], {"daily_pnl": 0.0, "remaining_risk_pct": 100.0},
        Cfg(), client=client,
    )
    assert decision is not None
    assert decision.action == "no_trade"


# ---------------------------------------------------------------------------
# decide: invalid JSON -> retry, then succeed
# ---------------------------------------------------------------------------


def test_decide_invalid_json_then_valid_retries_once():
    bad = "I cannot comply, here is not json"
    good = {"action": "no_trade", "symbol": "EURUSD", "confidence": 0.7,
            "reasoning": "ok"}
    transport, captured = recording_transport(
        [make_chat_response(bad), make_chat_response(json.dumps(good))]
    )
    client = _client_with(transport)
    decision, raw = decide(
        _snapshot(), [], [], {"daily_pnl": 0.0, "remaining_risk_pct": 100.0},
        Cfg(), client=client,
    )
    assert decision is not None
    assert decision.action == "no_trade"
    # Two calls: the original and the retry.
    assert len(captured) == 2
    # Retry system message should be stricter.
    retry_body = json.loads(captured[1].content.decode("utf-8"))
    assert "raw JSON" in retry_body["messages"][0]["content"]


# ---------------------------------------------------------------------------
# decide: invalid JSON twice -> returns None (refusal path)
# ---------------------------------------------------------------------------


def test_decide_invalid_json_twice_returns_none():
    bad1 = "nope"
    bad2 = "still nope"
    transport, captured = recording_transport(
        [make_chat_response(bad1), make_chat_response(bad2)]
    )
    client = _client_with(transport)
    decision, raw = decide(
        _snapshot(), [], [], {"daily_pnl": 0.0, "remaining_risk_pct": 100.0},
        Cfg(), client=client,
    )
    assert decision is None
    # Both attempts captured.
    assert len(captured) == 2
    # raw dict indicates retry happened.
    assert raw.get("retried") is True


# ---------------------------------------------------------------------------
# decide: refusal / network error
# ---------------------------------------------------------------------------


def test_decide_request_error_returns_none():
    from tests.fake_http import make_raising_transport

    transport = make_raising_transport(httpx.ConnectError("boom"))
    client = LLMClient(
        base_url="http://llm.local", model="m", api_key="", timeout=1.0,
        transport=transport,
    )
    decision, raw = decide(
        _snapshot(), [], [], {"daily_pnl": 0.0, "remaining_risk_pct": 100.0},
        Cfg(), client=client,
    )
    assert decision is None
    assert "error" in raw


# ---------------------------------------------------------------------------
# decide: schema validation rejects bad shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", list(VALID_ACTIONS))
def test_decide_accepts_each_valid_action(action):
    payload = {"action": action, "symbol": "EURUSD", "confidence": 0.7}
    if action == "propose":
        payload.update({"direction": "buy", "price": 1.1,
                        "stop_loss": 1.09, "take_profit": 1.12,
                        "risk_pct": 0.5, "playbook": "breakout_retest",
                        "reasoning": "ok"})
    transport, _ = recording_transport([make_chat_response(json.dumps(payload))])
    client = _client_with(transport)
    decision, _ = decide(
        _snapshot(), [], [], {"daily_pnl": 0.0, "remaining_risk_pct": 100.0},
        Cfg(), client=client,
    )
    assert decision is not None
    assert decision.action == action


def test_decide_rejects_invalid_action_in_payload():
    payload = {"action": "sideways", "symbol": "EURUSD"}
    transport, _ = recording_transport([make_chat_response(json.dumps(payload))])
    client = _client_with(transport)
    decision, _ = decide(
        _snapshot(), [], [], {"daily_pnl": 0.0, "remaining_risk_pct": 100.0},
        Cfg(), client=client,
    )
    assert decision is None


def test_decide_rejects_non_numeric_risk_pct_then_falls_back_to_zero():
    payload = {"action": "no_trade", "symbol": "EURUSD", "risk_pct": "lots"}
    transport, _ = recording_transport([make_chat_response(json.dumps(payload))])
    client = _client_with(transport)
    decision, _ = decide(
        _snapshot(), [], [], {"daily_pnl": 0.0, "remaining_risk_pct": 100.0},
        Cfg(), client=client,
    )
    assert decision is not None
    assert decision.risk_pct == 0.0


# ---------------------------------------------------------------------------
# decision schema
# ---------------------------------------------------------------------------


def test_decision_confidence_clamped():
    with pytest.raises(Exception):
        Decision(action="no_trade", symbol="X", confidence=1.5)


def test_decision_constants_present():
    assert "no_trade" in VALID_ACTIONS
    assert "propose" in VALID_ACTIONS
    assert "close" in VALID_ACTIONS
    assert "modify" in VALID_ACTIONS
    assert "buy" in VALID_DIRECTIONS
    assert "sell" in VALID_DIRECTIONS
    assert "market" in VALID_ENTRY_TYPES
    assert "limit" in VALID_ENTRY_TYPES