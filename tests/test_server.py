"""Tests for app/server.py (signals-only dashboard) using TestClient."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.journal import Journal
from app.server import create_app
from tests.fake_client import FakeClient


@pytest.fixture
def app_and_client(tmp_path, monkeypatch):
    # Write a config that points at tmp dirs.
    cfg_text = f"""
mode: demo
symbols: [XAUUSD]
timeframes: [M15]
schedule: {{scan_interval_minutes: 15}}
session_hours: {{start: '00:00', end: '23:59'}}
risk: {{max_risk_pct_per_trade: 1.0, max_daily_loss_pct: 3.0,
       max_open_trades: 3, max_trades_per_day: 5, min_confidence: 0.6,
       default_playbook: sweep_fvg_reclaim}}
signals: {{hypothetical_balance_usd: 10000.0, feed: auto,
          telegram_enabled: true, max_signals_per_day: 5}}
paths: {{data_dir: '{tmp_path.as_posix().replace(chr(92), '/')}',
         logs_dir: '{tmp_path.as_posix().replace(chr(92), '/')}',
         reports_dir: '{tmp_path.as_posix().replace(chr(92), '/')}',
         knowledge_dir: 'knowledge',
         kill_switch_path: '{(tmp_path / 'KILL_SWITCH').as_posix()}'}}
dashboard: {{host: '127.0.0.1', port: 8765}}
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    fake = FakeClient()
    fake._connected = True
    app = create_app(cfg_path, client_factory=lambda: fake)
    return app, TestClient(app), tmp_path


def test_health(app_and_client):
    _, client, _ = app_and_client
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "feed_ok" in body
    assert "kill_switch" in body


def test_state_shape(app_and_client):
    _, client, _ = app_and_client
    r = client.get("/api/state")
    assert r.status_code == 200
    data = r.json()
    assert "kill_switch" in data
    assert "feed" in data
    assert data["feed"]["name"] == "fake"
    assert data["feed"]["feed_ok"] is True
    assert "today" in data
    assert data["today"]["decision_count"] == 0
    assert data["today"]["signals"] == 0
    assert "last_signal" in data
    assert data["last_signal"] is None


def test_state_when_feed_factory_raises(tmp_path):
    """If the feed factory raises, state must still return a valid
    response with feed_ok=False (no crash, no hang)."""

    cfg_text = f"""
mode: demo
symbols: [XAUUSD]
timeframes: [M15]
schedule: {{scan_interval_minutes: 15}}
session_hours: {{start: '00:00', end: '23:59'}}
paths: {{data_dir: '{tmp_path.as_posix().replace(chr(92), '/')}',
         logs_dir: '{tmp_path.as_posix().replace(chr(92), '/')}',
         reports_dir: '{tmp_path.as_posix().replace(chr(92), '/')}',
         knowledge_dir: 'knowledge',
         kill_switch_path: '{(tmp_path / 'KILL_SWITCH').as_posix()}'}}
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    def bad_factory():
        raise RuntimeError("feed down")

    app = create_app(cfg_path, client_factory=bad_factory)
    c = TestClient(app)
    r = c.get("/api/state")
    assert r.status_code == 200
    data = r.json()
    assert data["feed"]["feed_ok"] is False
    assert data["feed"]["name"] is None


def test_decisions_endpoint(app_and_client, tmp_path):
    _, client, tmp_path = app_and_client
    j = Journal(tmp_path / "journal.db")
    j.record_decision(symbol="XAUUSD", action="propose", confidence=0.8,
                       reasoning="ok", approved=True)
    j.record_decision(symbol="XAUUSD", action="no_trade", confidence=0.7,
                       approved=True)
    r = client.get("/api/decisions?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert {d["symbol"] for d in body["decisions"]} == {"XAUUSD"}


def test_decisions_filter_by_symbol(app_and_client, tmp_path):
    _, client, tmp_path = app_and_client
    j = Journal(tmp_path / "journal.db")
    j.record_decision(symbol="XAUUSD", action="no_trade")
    j.record_decision(symbol="EURUSD", action="no_trade")
    r = client.get("/api/decisions?symbol=XAUUSD")
    body = r.json()
    assert body["count"] == 1
    assert body["decisions"][0]["symbol"] == "XAUUSD"


def test_signals_endpoint_returns_approved_proposals(app_and_client, tmp_path):
    _, client, tmp_path = app_and_client
    j = Journal(tmp_path / "journal.db")
    raw = json.dumps({
        "direction": "buy", "price": 3900.0, "stop_loss": 3885.0,
        "take_profit": 3930.0, "risk_pct": 1.0,
    })
    j.record_decision(symbol="XAUUSD", action="propose", confidence=0.85,
                       reasoning="sweep+fvg", approved=True, raw_llm_json=raw)
    j.record_decision(symbol="XAUUSD", action="no_trade", approved=True)
    r = client.get("/api/signals")
    body = r.json()
    assert body["count"] == 1
    sig = body["signals"][0]
    assert sig["direction"] == "buy"
    assert sig["entry"] == 3900.0
    assert sig["stop_loss"] == 3885.0
    assert sig["take_profit"] == 3930.0


def test_state_last_signal(app_and_client, tmp_path):
    _, client, tmp_path = app_and_client
    j = Journal(tmp_path / "journal.db")
    raw = json.dumps({
        "direction": "sell", "price": 3900.0, "stop_loss": 3915.0,
        "take_profit": 3870.0, "risk_pct": 1.0,
    })
    j.record_decision(symbol="XAUUSD", action="propose", confidence=0.9,
                       reasoning="sweep+fvg", approved=True, raw_llm_json=raw)
    r = client.get("/api/state")
    last = r.json()["last_signal"]
    assert last is not None
    assert last["direction"] == "sell"
    assert last["entry"] == 3900.0


def test_kill_and_resume_round_trip(app_and_client, tmp_path):
    _, client, tmp_path = app_and_client
    flag = tmp_path / "KILL_SWITCH"
    assert not flag.exists()

    r = client.post("/api/kill")
    assert r.status_code == 200
    assert r.json()["kill_switch"] is True
    assert flag.exists()

    r = client.post("/api/resume")
    assert r.status_code == 200
    assert r.json()["kill_switch"] is False
    assert not flag.exists()


def test_state_reflects_kill_switch(app_and_client, tmp_path):
    _, client, tmp_path = app_and_client
    client.post("/api/kill")
    r = client.get("/api/state")
    assert r.json()["kill_switch"] is True
    client.post("/api/resume")
    r = client.get("/api/state")
    assert r.json()["kill_switch"] is False


def test_dashboard_index_served(app_and_client):
    _, client, _ = app_and_client
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    assert "Gold Signals" in body
    assert "kill switch" in body.lower()
    assert "setInterval" in body or "setinterval" in body.lower()


def test_static_files_served(app_and_client):
    _, client, _ = app_and_client
    r = client.get("/static/index.html")
    assert r.status_code == 200
    assert "Gold Signals" in r.text


def test_decisions_limit_clamped(app_and_client):
    _, client, _ = app_and_client
    # Limit > 1000 should be rejected by FastAPI Query validation.
    r = client.get("/api/decisions?limit=99999")
    assert r.status_code == 422
