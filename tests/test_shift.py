"""Tests for the market-shift bell: app/shift.py (pure detector) and
the public GET /api/shift endpoint.

Anchor: replaying the frozen 5m backtest file
(~/workspace/trading/shift-signal/gold-m5.json, 16,904 bars) must
reproduce the backtest's counts exactly — 492 signals, 262 buys,
230 sells. The detector mirrors shift-signal-5m.pine; if this test
fails, the bot's bell and the chart's bell have diverged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.market_data import Rates
from app.server import create_app
from app.shift import detect_shifts, wilder_rsi
from tests.fake_client import FakeClient

FROZEN_M5 = Path("/home/hatch/workspace/trading/shift-signal/gold-m5.json")


def _load_frozen() -> list[dict]:
    bars = json.loads(FROZEN_M5.read_text(encoding="utf-8"))
    return [{"time": b["epoch"], "open": b["open"], "high": b["high"],
             "low": b["low"], "close": b["close"]} for b in bars]


@pytest.mark.skipif(not FROZEN_M5.exists(),
                    reason="frozen 5m file not present on this machine")
def test_anchor_replays_backtest_counts():
    candles = _load_frozen()
    assert len(candles) == 16904
    events, state = detect_shifts(candles)
    assert len(events) == 492
    assert sum(1 for e in events if e["direction"] == "BUY") == 262
    assert sum(1 for e in events if e["direction"] == "SELL") == 230
    # Events are oldest-first, identified by firing-bar epoch.
    assert events[0]["epoch"] == 1787010600
    assert events[0]["direction"] == "SELL"
    assert events[-1]["epoch"] == 1794442200
    assert events[-1]["direction"] == "BUY"
    for e in events:
        assert set(e) == {"epoch", "direction", "price", "rsi"}
        assert e["direction"] in ("BUY", "SELL")
        assert 0.0 <= e["rsi"] <= 100.0
    assert state["last_close"] == pytest.approx(3995.15)
    assert state["last_bar_epoch"] == 1794445800


def test_empty_input():
    events, state = detect_shifts([])
    assert events == []
    assert state == {"rsi": None, "armed": "none", "last_close": None,
                     "last_bar_epoch": None}


def test_determinism():
    candles = _load_frozen()[:2000] if FROZEN_M5.exists() else [
        {"time": 1_700_000_000 + i * 300, "open": 4000.0 + i % 7,
         "close": 4000.0 + (i * 3) % 11} for i in range(500)
    ]
    first, first_state = detect_shifts(candles)
    second, second_state = detect_shifts(candles)
    assert first == second
    assert first_state == second_state


def test_wilder_rsi_bounds_and_monotonic():
    rising = [100.0 + i for i in range(50)]
    rsi = wilder_rsi(rising)
    assert len(rsi) == 50
    assert all(0.0 <= v <= 100.0 for v in rsi)
    assert rsi[-1] == 100.0  # no losing bars at all
    falling = [150.0 - i for i in range(50)]
    assert wilder_rsi(falling)[-1] == 0.0
    flat = [100.0] * 50
    assert wilder_rsi(flat)[-1] == 100.0  # no losses -> RS infinite


def test_shift_state_armed_values():
    # A hard run-up then a pause: RSI stays high, arm stays sell.
    candles = [{"time": 1_700_000_000 + i * 300, "open": 4000.0 + i,
                "close": 4001.0 + i} for i in range(150)]
    _events, state = detect_shifts(candles, warmup=15)
    assert state["armed"] == "sell"
    assert state["rsi"] is not None and state["rsi"] > 70.0


# --- /api/shift endpoint -----------------------------------------------


@pytest.fixture
def shift_app(tmp_path):
    cfg_text = f"""
mode: demo
symbols: [XAUUSD]
timeframes: [M15]
schedule: {{scan_interval_minutes: 15}}
session_hours: {{start: '00:00', end: '23:59'}}
risk: {{max_risk_pct_per_trade: 1.0, max_daily_loss_pct: 3.0,
       max_open_trades: 3, max_trades_per_day: 5, min_confidence: 0.6}}
signals: {{hypothetical_balance_usd: 10000.0, feed: auto,
          telegram_enabled: true, max_signals_per_day: 5}}
paths: {{data_dir: '{tmp_path.as_posix()}',
         logs_dir: '{tmp_path.as_posix()}',
         reports_dir: '{tmp_path.as_posix()}',
         knowledge_dir: 'knowledge',
         kill_switch_path: '{(tmp_path / 'KILL_SWITCH').as_posix()}'}}
dashboard: {{host: '127.0.0.1', port: 8765}}
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    fake = FakeClient()
    fake._connected = True
    app = create_app(cfg_path, client_factory=lambda: fake)
    return TestClient(app), fake


def test_api_shift_no_feed(shift_app):
    client, fake = shift_app
    # No feed at all (connect fails): fail-soft, still 200.
    # _get_client() calls connect(), which is what flips _connected —
    # so the dead feed is simulated via connect_raises, not the flag.
    fake.connect_raises = RuntimeError("no feed")
    r = client.get("/api/shift")
    assert r.status_code == 200
    data = r.json()
    assert set(data) == {"ts", "feed_ok", "price", "rsi", "armed", "shifts"}
    assert data["feed_ok"] is False
    assert data["price"] is None
    assert data["rsi"] is None
    assert data["armed"] == "none"
    assert data["shifts"] == []


@pytest.mark.skipif(not FROZEN_M5.exists(),
                    reason="frozen 5m file not present on this machine")
def test_api_shift_seeded_matches_detector(shift_app):
    client, fake = shift_app
    candles = _load_frozen()[-300:]
    fake.rates_by_tf[("XAUUSD", "M5")] = Rates(rates=candles)
    # The endpoint drops the still-forming last bar.
    expected_events, expected_state = detect_shifts(candles[:-1])
    r = client.get("/api/shift")
    assert r.status_code == 200
    data = r.json()
    assert data["feed_ok"] is True
    assert data["price"] == expected_state["last_close"]
    assert data["rsi"] == expected_state["rsi"]
    assert data["armed"] == expected_state["armed"]
    # Newest first, at most 20.
    assert data["shifts"] == list(reversed(expected_events[-20:]))
    epochs = [s["epoch"] for s in data["shifts"]]
    assert epochs == sorted(epochs, reverse=True)
    # The fake was asked for M5 exactly.
    assert ("copy_rates", "XAUUSD", "M5", 300) in fake.calls


__all__ = []
