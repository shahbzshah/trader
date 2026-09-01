"""Tests for app/server.py using httpx TestClient."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.journal import Journal
from app.server import create_app


@pytest.fixture
def app_and_client(tmp_path, monkeypatch):
    # Write a config that points at tmp dirs.
    cfg_text = f"""
mode: demo
symbols: [EURUSD]
timeframes: [M15]
schedule: {{scan_interval_minutes: 15}}
session_hours: {{start: '00:00', end: '23:59'}}
risk: {{max_risk_pct_per_trade: 1.0, max_daily_loss_pct: 3.0,
       max_open_trades: 3, max_trades_per_day: 5, min_confidence: 0.6,
       default_playbook: breakout_retest}}
llm: {{base_url: 'http://x', model: 'm'}}
paths: {{data_dir: '{tmp_path.as_posix().replace(chr(92), '/')}',
         logs_dir: '{tmp_path.as_posix().replace(chr(92), '/')}',
         reports_dir: '{tmp_path.as_posix().replace(chr(92), '/')}',
         knowledge_dir: 'knowledge',
         kill_switch_path: '{(tmp_path / 'KILL_SWITCH').as_posix()}'}}
dashboard: {{host: '127.0.0.1', port: 8765}}
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    # .env not needed.
    app = create_app(cfg_path, client_factory=lambda: _NoopMT5Client())
    return app, TestClient(app)


class _NoopMT5Client:
    """A test fake that looks like MT5Client but has no live methods."""

    def __init__(self):
        self._connected = False

    def connect(self, *a, **kw):
        self._connected = True
        return True

    def account_info(self):
        from app.mt5_client import AccountInfo
        return AccountInfo(
            login=12345, name="demo", server="x", currency="USD",
            trade_mode="demo", balance=10000.0, equity=10000.0,
            margin=0.0, free_margin=10000.0, leverage=100,
        )

    def positions(self, symbol=None):
        from app.mt5_client import Position
        return [Position(ticket=1, symbol="EURUSD", direction="buy",
                          volume=0.10, price_open=1.1000, price_current=1.1050,
                          sl=1.0950, tp=1.1100, profit=50.0, comment="")]

    def shutdown(self):
        self._connected = False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health(app_and_client):
    _, client = app_and_client
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_state_includes_account_positions_kill(app_and_client):
    _, client = app_and_client
    r = client.get("/api/state")
    assert r.status_code == 200
    data = r.json()
    assert "kill_switch" in data
    assert "mt5_connected" in data
    assert data["mt5_connected"] is True
    assert data["account"]["trade_mode"] == "demo"
    assert data["account"]["balance"] == 10000.0
    assert len(data["positions"]) == 1
    assert data["today"]["decision_count"] == 0


def test_state_when_mt5_factory_raises(monkeypatch, tmp_path):
    """If the MT5 client factory raises, the state endpoint must still
    return a valid response with mt5_connected=False (no crash)."""

    cfg_text = f"""
mode: demo
symbols: [EURUSD]
timeframes: [M15]
schedule: {{scan_interval_minutes: 15}}
session_hours: {{start: '00:00', end: '23:59'}}
risk: {{max_risk_pct_per_trade: 1.0, max_daily_loss_pct: 3.0,
       max_open_trades: 3, max_trades_per_day: 5, min_confidence: 0.6,
       default_playbook: breakout_retest}}
llm: {{base_url: 'http://x', model: 'm'}}
paths: {{data_dir: '{tmp_path.as_posix().replace(chr(92), '/')}',
         logs_dir: '{tmp_path.as_posix().replace(chr(92), '/')}',
         reports_dir: '{tmp_path.as_posix().replace(chr(92), '/')}',
         knowledge_dir: 'knowledge',
         kill_switch_path: '{(tmp_path / 'KILL_SWITCH').as_posix()}'}}
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    def bad_factory():
        raise RuntimeError("MT5 not installed")

    app = create_app(cfg_path, client_factory=bad_factory)
    c = TestClient(app)
    r = c.get("/api/state")
    assert r.status_code == 200
    data = r.json()
    assert data["mt5_connected"] is False
    assert data["account"] is None
    assert data["positions"] == []


def test_decisions_endpoint(app_and_client, tmp_path):
    _, client = app_and_client
    # Seed decisions directly via the journal.
    cfg_path = tmp_path / "config.yaml"
    j = Journal(tmp_path / "journal.db")
    j.record_decision(symbol="EURUSD", action="propose", confidence=0.8,
                       reasoning="ok", approved=True)
    j.record_decision(symbol="EURUSD", action="no_trade", confidence=0.7,
                       approved=True)
    r = client.get("/api/decisions?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert {d["symbol"] for d in body["decisions"]} == {"EURUSD"}


def test_decisions_filter_by_symbol(app_and_client, tmp_path):
    _, client = app_and_client
    j = Journal(tmp_path / "journal.db")
    j.record_decision(symbol="EURUSD", action="no_trade")
    j.record_decision(symbol="GBPUSD", action="no_trade")
    r = client.get("/api/decisions?symbol=EURUSD")
    body = r.json()
    assert body["count"] == 1
    assert body["decisions"][0]["symbol"] == "EURUSD"


def test_equity_endpoint(app_and_client, tmp_path):
    _, client = app_and_client
    j = Journal(tmp_path / "journal.db")
    j.record_equity(balance=10000, equity=10050)
    j.record_equity(balance=10000, equity=10075)
    r = client.get("/api/equity?days=7")
    body = r.json()
    assert body["count"] == 2
    assert body["days"] == 7
    assert [row["equity"] for row in body["equity"]] == [10050.0, 10075.0]


def test_orders_endpoint(app_and_client, tmp_path):
    _, client = app_and_client
    j = Journal(tmp_path / "journal.db")
    did = j.record_decision(symbol="EURUSD", action="propose", approved=True)
    j.record_order(decision_id=did, symbol="EURUSD", direction="buy",
                    lots=0.10, status="filled", ticket=42)
    r = client.get("/api/orders")
    body = r.json()
    assert body["count"] == 1
    assert body["orders"][0]["ticket"] == 42


def test_kill_and_resume_round_trip(app_and_client, tmp_path):
    _, client = app_and_client
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
    _, client = app_and_client
    client.post("/api/kill")
    r = client.get("/api/state")
    assert r.json()["kill_switch"] is True
    client.post("/api/resume")
    r = client.get("/api/state")
    assert r.json()["kill_switch"] is False


def test_dashboard_index_served(app_and_client):
    _, client = app_and_client
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    assert "Trader Dashboard" in body
    assert "kill-switch" in body.lower() or "kill switch" in body.lower()
    assert "<canvas" in body
    assert "setInterval" in body or "setinterval" in body.lower()


def test_static_files_served(app_and_client):
    _, client = app_and_client
    r = client.get("/static/index.html")
    assert r.status_code == 200
    assert "Trader Dashboard" in r.text


def test_decisions_limit_clamped(app_and_client):
    _, client = app_and_client
    # Limit > 1000 should be rejected by FastAPI Query validation.
    r = client.get("/api/decisions?limit=99999")
    assert r.status_code == 422


def test_equity_invalid_days_rejected(app_and_client):
    _, client = app_and_client
    r = client.get("/api/equity?days=0")
    assert r.status_code == 422