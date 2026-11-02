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



# --- New dashboard fields: uptime, price/day/news blocks, outcomes ---

@pytest.fixture
def fake_app(tmp_path):
    """App + client + the FakeClient itself, so tests can set ticks
    and candles before hitting the API."""
    cfg_text = f"""
mode: demo
symbols: [XAUUSD]
timeframes: [M15]
schedule: {{scan_interval_minutes: 15}}
session_hours: {{start: '00:00', end: '23:59'}}
risk: {{max_risk_pct_per_trade: 1.0, max_daily_loss_pct: 3.0,
       max_open_trades: 3, max_trades_per_day: 5, min_confidence: 0.6,
       default_playbook: sweep_fvg_reclaim}}
news: {{enabled: true, feed_url: 'https://example.invalid/ff.json',
        cache_path: '{(tmp_path / 'news_cache.json').as_posix()}',
        cache_ttl_hours: 12, currencies: [USD], min_impact: High,
        window_minutes: 30}}
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
    return app, TestClient(app), tmp_path, fake


def _record_signal(tmp_path, direction="buy", entry=3900.0, sl=3885.0,
                   tp=3930.0, risk_pct=1.0):
    j = Journal(tmp_path / "journal.db")
    raw = json.dumps({"direction": direction, "price": entry,
                      "stop_loss": sl, "take_profit": tp,
                      "risk_pct": risk_pct})
    j.record_decision(symbol="XAUUSD", action="propose", confidence=0.85,
                      reasoning="sweep+fvg", approved=True,
                      raw_llm_json=raw)


def test_health_exposes_uptime(fake_app):
    _, client, _, _ = fake_app
    body = client.get("/health").json()
    assert isinstance(body.get("uptime_s"), int)
    assert body["uptime_s"] >= 0


def test_state_exposes_dashboard_blocks(fake_app):
    _, client, _, _ = fake_app
    data = client.get("/api/state").json()
    assert isinstance(data.get("uptime_s"), int)
    assert data.get("scan_interval_minutes") == 15
    # Fake client has no tick and no candles: blocks present, empty.
    assert "price" in data and data["price"]["mid"] is None
    assert "day" in data and data["day"] is None
    assert "news" in data
    # No calendar cache at all: unavailable, never a false "all clear".
    assert data["news"]["available"] is False
    assert data["news"]["window_minutes"] == 30


def _write_news_cache(tmp_path, fetched_at, event_minutes_from_now):
    import datetime as _dt

    when = (_dt.datetime.now(_dt.timezone.utc)
            + _dt.timedelta(minutes=event_minutes_from_now))
    payload = {
        "fetched_at": fetched_at.isoformat(),
        "events": [{
            "title": "ISM Manufacturing PMI", "country": "USD",
            "date": when.isoformat(), "impact": "High",
            "forecast": "54.7", "previous": "54.5",
        }],
    }
    (tmp_path / "news_cache.json").write_text(
        json.dumps(payload), encoding="utf-8")


def test_news_stale_cache_reports_unavailable(fake_app):
    import datetime as _dt

    _, client, tmp_path, _ = fake_app
    # Cache fetched 2 days ago, still holding a future event: the
    # fail-open fallback must NOT present the week as clear.
    _write_news_cache(
        tmp_path,
        _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=2),
        event_minutes_from_now=60)
    news = client.get("/api/state").json()["news"]
    assert news["available"] is False
    assert news["next_event"] is None


def test_news_fresh_cache_reports_next_event(fake_app):
    import datetime as _dt

    _, client, tmp_path, _ = fake_app
    _write_news_cache(
        tmp_path, _dt.datetime.now(_dt.timezone.utc),
        event_minutes_from_now=60)
    news = client.get("/api/state").json()["news"]
    assert news["available"] is True
    assert news["blackout_active"] is False
    assert news["next_event"]["title"] == "ISM Manufacturing PMI"
    assert 0 < news["next_event"]["minutes_until"] <= 60


def test_signals_enriched_with_sizing(fake_app):
    _, client, tmp_path, _ = fake_app
    _record_signal(tmp_path)
    sig = client.get("/api/signals").json()["signals"][0]
    assert sig["status"] == "OPEN"
    assert sig["lots"] == pytest.approx(0.06)
    assert sig["risk_usd"] == pytest.approx(100.0)
    assert sig["per_dollar_usd"] == pytest.approx(6.0)
    assert sig["rr"] == pytest.approx(2.0)


def test_outcome_tp_hit(fake_app):
    from tests.fake_client import rates_from_rows
    import time as _time

    _, client, tmp_path, fake = fake_app
    _record_signal(tmp_path)
    t0 = int(_time.time()) + 900
    fake.rates_by_tf[("XAUUSD", "M15")] = rates_from_rows(
        [(3902.0, 3935.0, 3900.0, 3932.0)], t0=t0, step=900)
    sig = client.get("/api/signals").json()["signals"][0]
    assert sig["status"] == "TP"
    assert sig["r"] == pytest.approx(2.0)
    assert sig["close_price"] == pytest.approx(3930.0)


def test_outcome_sl_hit_and_other_side_open(fake_app):
    from tests.fake_client import rates_from_rows
    import time as _time

    _, client, tmp_path, fake = fake_app
    _record_signal(tmp_path, direction="sell", entry=3900.0,
                   sl=3915.0, tp=3870.0)
    _record_signal(tmp_path, direction="buy", entry=3900.0,
                   sl=3885.0, tp=3930.0)
    t0 = int(_time.time()) + 900
    # Candle low 3880 touches the buy SL (3885) but neither sell level
    # (high 3902 < SL 3915, low 3880 > TP 3870).
    fake.rates_by_tf[("XAUUSD", "M15")] = rates_from_rows(
        [(3900.0, 3902.0, 3880.0, 3888.0)], t0=t0, step=900)
    sigs = client.get("/api/signals").json()["signals"]
    by_dir = {s["direction"]: s for s in sigs}
    assert by_dir["sell"]["status"] == "OPEN"
    assert by_dir["buy"]["status"] == "SL"
    assert by_dir["buy"]["r"] == pytest.approx(-1.0)


def test_outcome_both_touch_same_candle_nearer_wins(fake_app):
    from tests.fake_client import rates_from_rows
    import time as _time

    _, client, tmp_path, fake = fake_app
    _record_signal(tmp_path)
    t0 = int(_time.time()) + 900
    # One candle touches both SL (3885) and TP (3930); open 3900 is
    # nearer the SL (15) than the TP (30) -> SL.
    fake.rates_by_tf[("XAUUSD", "M15")] = rates_from_rows(
        [(3900.0, 3935.0, 3880.0, 3910.0)], t0=t0, step=900)
    sig = client.get("/api/signals").json()["signals"][0]
    assert sig["status"] == "SL"
    assert sig["r"] == pytest.approx(-1.0)


def test_outcome_open_when_no_touch(fake_app):
    from tests.fake_client import rates_from_rows
    import time as _time

    _, client, tmp_path, fake = fake_app
    _record_signal(tmp_path)
    t0 = int(_time.time()) + 900
    fake.rates_by_tf[("XAUUSD", "M15")] = rates_from_rows(
        [(3900.0, 3905.0, 3895.0, 3901.0)], t0=t0, step=900)
    sig = client.get("/api/signals").json()["signals"][0]
    assert sig["status"] == "OPEN"
    assert sig["r"] is None


def test_dashboard_index_has_no_fake_values(fake_app):
    _, client, _, _ = fake_app
    body = client.get("/").text
    assert "HFT-v4" not in body
    assert "2,348" not in body
    assert "Verified MT5" not in body
    assert "No signals yet" in body
    assert "fonts.css" in body
    assert "tailwind.js" in body


def test_material_symbols_rule_sets_icon_font(fake_app):
    """Regression: icons rendered as literal text on phones because the
    .material-symbols-outlined rule never set font-family (it came from
    Google's fonts CSS, which was removed when self-hosting)."""
    _, client, _, _ = fake_app
    body = client.get("/").text
    rule = body.split(".material-symbols-outlined {", 1)[1].split("}", 1)[0]
    assert "font-family: 'Material Symbols Outlined'" in rule
    assert "-webkit-font-feature-settings: 'liga'" in rule
    css = client.get("/static/fonts/fonts.css").text
    assert "font-family:'Material Symbols Outlined'" in css
    assert client.get(
        "/static/fonts/MaterialSymbolsOutlined.woff2").status_code == 200


# --- Dashboard parity: candles chart, server stats, read-only config ---

def test_candles_endpoint_empty_feed(fake_app):
    _, client, _, _ = fake_app
    body = client.get("/api/candles?limit=200").json()
    assert body["symbol"] == "XAUUSD"
    assert body["timeframe"] == "M15"
    assert body["candles"] == []
    assert body["count"] == 0


def test_candles_endpoint_oldest_first(fake_app):
    from tests.fake_client import rates_from_rows
    import time as _time

    _, client, _, fake = fake_app
    t0 = int(_time.time())
    fake.rates_by_tf[("XAUUSD", "M15")] = rates_from_rows(
        [(3900.0, 3905.0, 3895.0, 3901.0),
         (3901.0, 3908.0, 3899.0, 3906.0)], t0=t0, step=900)
    body = client.get("/api/candles?limit=200").json()
    assert body["count"] == 2
    assert body["candles"][0]["epoch"] == t0
    assert body["candles"][0]["open"] == 3900.0
    assert body["candles"][1]["close"] == 3906.0


def test_state_stats_block_empty(fake_app):
    _, client, _, _ = fake_app
    stats = client.get("/api/state").json()["stats"]
    assert stats == {"total_signals": 0, "open": 0, "closed": 0,
                     "wins": 0, "losses": 0, "hit_rate": None,
                     "net_r": 0, "avg_r": None}


def test_state_stats_block_after_tp(fake_app):
    from tests.fake_client import rates_from_rows
    import time as _time

    _, client, tmp_path, fake = fake_app
    _record_signal(tmp_path)
    t0 = int(_time.time()) + 900
    fake.rates_by_tf[("XAUUSD", "M15")] = rates_from_rows(
        [(3902.0, 3935.0, 3900.0, 3932.0)], t0=t0, step=900)
    stats = client.get("/api/state").json()["stats"]
    assert stats["total_signals"] == 1
    assert stats["closed"] == 1
    assert stats["wins"] == 1
    assert stats["hit_rate"] == 1.0
    assert stats["net_r"] == pytest.approx(2.0)
    assert stats["avg_r"] == pytest.approx(2.0)


def test_config_endpoint_shape_no_secrets(fake_app):
    _, client, _, _ = fake_app
    body = client.get("/api/config").json()
    assert body["symbols"] == ["XAUUSD"]
    assert body["timeframe"] == "M15"
    assert body["scan_interval_minutes"] == 15
    assert body["kill_zones"] == []
    assert body["news"]["min_impact"] == "High"
    assert body["news"]["window_minutes"] == 30
    assert body["risk"]["max_risk_pct_per_trade"] == 1.0
    assert body["risk"]["default_playbook"] == "sweep_fvg_reclaim"
    assert body["signals"]["hypothetical_balance_usd"] == 10000.0
    flat = json.dumps(body).lower()
    for banned in ("api_key", "token", "secret", "password"):
        assert banned not in flat


def test_dashboard_has_chart_config_and_server_stats(fake_app):
    _, client, _, _ = fake_app
    body = client.get("/").text
    # Candle chart wired to /api/candles, levels from latest signal.
    assert "chart-canvas" in body
    assert "/api/candles" in body
    assert "chart-zoom-in" in body and "chart-zoom-out" in body
    # Read-only config section wired to /api/config.
    assert "section-config" in body
    assert "/api/config" in body
    # Stats line reads the server stats block, never a fake percentage.
    assert "s.stats" in body
    assert "hit_rate" in body
    # No external URLs: fonts + Tailwind stay self-hosted.
    assert "https://" not in body
    assert "http://" not in body
