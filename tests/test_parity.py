"""Parity tests for the gold signals bot dashboard (bot-side features).

Covers: /api/candles, live outcome tracking via ticks + ledger stats,
history snapshot/archive survival across restarts, gzip on the JSON
endpoints + dashboard page, /api/config (no secrets), and the
POST /explain round-trip.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.journal import Journal
from app.market_data import Tick
from app.server import create_app
from tests.fake_client import FakeClient, rates_from_rows


@pytest.fixture
def parity_app(tmp_path):
    cfg_text = f"""
mode: demo
symbols: [XAUUSD]
timeframes: [M15]
schedule: {{scan_interval_minutes: 15}}
session_hours: {{start: '00:00', end: '23:59'}}
kill_zones: ['07:00-10:00', '12:00-15:00']
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
    return app, TestClient(app), tmp_path, fake, cfg_path


def _record_signal(tmp_path, direction="buy", entry=3900.0, sl=3885.0,
                   tp=3930.0, risk_pct=1.0, ts=None, reasoning="sweep+fvg"):
    j = Journal(tmp_path / "journal.db")
    raw = json.dumps({"direction": direction, "price": entry,
                      "stop_loss": sl, "take_profit": tp,
                      "risk_pct": risk_pct})
    return j.record_decision(symbol="XAUUSD", action="propose",
                             confidence=0.85, reasoning=reasoning,
                             approved=True, raw_llm_json=raw, ts=ts)


def _set_tick(fake, mid, ts=None):
    fake.tick = Tick(bid=mid - 0.1, ask=mid + 0.1, spread=20.0,
                     time=int(ts or time.time()))


# --- /api/candles ----------------------------------------------------------

def test_candles_endpoint_shape(parity_app):
    _, client, _, fake, _ = parity_app
    t0 = int(time.time())
    fake.rates_by_tf[("XAUUSD", "M15")] = rates_from_rows(
        [(3900.0, 3905.0, 3895.0, 3901.0),
         (3901.0, 3910.0, 3900.0, 3908.0),
         (3908.0, 3912.0, 3905.0, 3906.0)], t0=t0, step=900)
    body = client.get("/api/candles").json()
    assert body["symbol"] == "XAUUSD"
    assert body["timeframe"] == "M15"
    assert body["count"] == 3
    assert [c["epoch"] for c in body["candles"]] == [t0, t0 + 900, t0 + 1800]
    first = body["candles"][0]
    assert first == {"epoch": t0, "open": 3900.0, "high": 3905.0,
                     "low": 3895.0, "close": 3901.0}


def test_candles_empty_when_no_feed_data(parity_app):
    _, client, _, _, _ = parity_app
    body = client.get("/api/candles").json()
    assert body["count"] == 0
    assert body["candles"] == []


def test_candles_limit_respected(parity_app):
    _, client, _, fake, _ = parity_app
    t0 = int(time.time())
    fake.rates_by_tf[("XAUUSD", "M15")] = rates_from_rows(
        [(3900.0 + i, 3905.0 + i, 3895.0 + i, 3901.0 + i)
         for i in range(10)], t0=t0, step=900)
    body = client.get("/api/candles?limit=3").json()
    assert body["count"] == 3
    # Most recent three, oldest first.
    assert body["candles"][0]["epoch"] == t0 + 7 * 900


# --- outcome tracking + stats ----------------------------------------------

def test_state_stats_honest_zeros(parity_app):
    _, client, _, _, _ = parity_app
    stats = client.get("/api/state").json()["stats"]
    assert stats == {"total_signals": 0, "open": 0, "closed": 0,
                     "wins": 0, "losses": 0, "hit_rate": None,
                     "net_r": 0.0, "avg_r": None}


def test_tick_touching_tp_resolves_win_and_stats(parity_app):
    _, client, tmp_path, fake, _ = parity_app
    _record_signal(tmp_path)  # buy 3900, SL 3885, TP 3930, rr 2.0
    _set_tick(fake, mid=3931.0)
    data = client.get("/api/state").json()
    stats = data["stats"]
    assert stats["total_signals"] == 1
    assert stats["closed"] == 1
    assert stats["wins"] == 1
    assert stats["losses"] == 0
    assert stats["open"] == 0
    assert stats["hit_rate"] == pytest.approx(1.0)
    assert stats["net_r"] == pytest.approx(2.0)
    assert stats["avg_r"] == pytest.approx(2.0)
    sig = client.get("/api/signals").json()["signals"][0]
    assert sig["status"] == "TP"
    assert sig["r"] == pytest.approx(2.0)
    assert sig["close_price"] == pytest.approx(3930.0)
    assert sig["close_ts"] is not None


def test_tick_resolves_win_and_loss_stats(parity_app):
    _, client, tmp_path, fake, _ = parity_app
    _record_signal(tmp_path, direction="buy", entry=3900.0,
                   sl=3885.0, tp=3930.0)   # rr 2.0
    _record_signal(tmp_path, direction="sell", entry=3900.0,
                   sl=3915.0, tp=3870.0)   # rr 2.0
    # One tick at 3931: buy TP hit, sell SL hit.
    _set_tick(fake, mid=3931.0)
    stats = client.get("/api/state").json()["stats"]
    assert stats["total_signals"] == 2
    assert stats["closed"] == 2
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)
    assert stats["net_r"] == pytest.approx(1.0)   # +2.0 - 1.0
    assert stats["avg_r"] == pytest.approx(0.5)


def test_open_signal_stays_open_stats(parity_app):
    _, client, tmp_path, fake, _ = parity_app
    _record_signal(tmp_path)
    _set_tick(fake, mid=3900.0)  # touches neither level
    stats = client.get("/api/state").json()["stats"]
    assert stats["total_signals"] == 1
    assert stats["open"] == 1
    assert stats["closed"] == 0
    assert stats["hit_rate"] is None
    assert stats["net_r"] == pytest.approx(0.0)
    assert stats["avg_r"] is None


def test_resolved_outcome_never_changes(parity_app):
    _, client, tmp_path, fake, _ = parity_app
    _record_signal(tmp_path)
    _set_tick(fake, mid=3931.0)
    sig = client.get("/api/signals").json()["signals"][0]
    assert sig["status"] == "TP"
    # Price collapses through the SL afterwards: the win must stand.
    _set_tick(fake, mid=3880.0)
    sig = client.get("/api/signals").json()["signals"][0]
    assert sig["status"] == "TP"
    assert sig["r"] == pytest.approx(2.0)


# --- history snapshot / archive ---------------------------------------------

def test_history_persisted_on_resolve(parity_app):
    _, client, tmp_path, fake, _ = parity_app
    _record_signal(tmp_path)
    _set_tick(fake, mid=3931.0)
    client.get("/api/state")
    snap = json.loads((tmp_path / "history_snapshot.json").read_text())
    assert len(snap) == 1
    assert snap[0]["status"] == "TP"
    assert snap[0]["r"] == pytest.approx(2.0)
    archive = (tmp_path / "signal_history.jsonl").read_text().strip().splitlines()
    assert len(archive) >= 1
    assert json.loads(archive[-1])["status"] == "TP"


def test_history_survives_restart_and_resumes_tracking(parity_app):
    _, client, tmp_path, fake, cfg_path = parity_app
    # One signal resolves now; one stays open (its SL 3930 sits below
    # the 3931 tick, its TP 3980 above it).
    _record_signal(tmp_path, direction="buy", entry=3900.0,
                   sl=3885.0, tp=3930.0)
    open_id = _record_signal(tmp_path, direction="buy", entry=3950.0,
                             sl=3930.0, tp=3980.0)
    _set_tick(fake, mid=3931.0)  # resolves the first only
    client.get("/api/state")
    assert (tmp_path / "history_snapshot.json").exists()
    snap = json.loads((tmp_path / "history_snapshot.json").read_text())
    assert {r["id"]: r["status"] for r in snap} == {1: "TP", open_id: "OPEN"}

    # Simulate a redeploy: the SQLite journal is wiped, the repo files
    # (snapshot + archive) survive.
    (tmp_path / "journal.db").unlink()
    for suffix in ("-wal", "-shm"):
        p = tmp_path / f"journal.db{suffix}"
        if p.exists():
            p.unlink()
    fake.tick = None
    # Fresh candles are set BEFORE the first post-restart poll: the
    # new app caches its first (empty) candle fetch for 60s, so the
    # TP touch must already be visible on that first poll.
    t0 = int(time.time()) + 900
    fake.rates_by_tf[("XAUUSD", "M15")] = rates_from_rows(
        [(3955.0, 3985.0, 3950.0, 3982.0)], t0=t0, step=900)
    app2 = create_app(cfg_path, client_factory=lambda: fake)
    client2 = TestClient(app2)
    sigs = client2.get("/api/signals").json()["signals"]
    assert len(sigs) == 2
    by_id = {s["id"]: s for s in sigs}
    assert by_id[1]["status"] == "TP"      # restored resolved
    # The open signal resumed TP/SL tracking against fresh candles
    # (TP 3980 is 30 pts over entry, SL 3930 is 20 under: 1.5R).
    assert by_id[open_id]["status"] == "TP"
    assert by_id[open_id]["r"] == pytest.approx(1.5)


def test_corrupt_snapshot_never_breaks_startup(parity_app):
    _, client, tmp_path, fake, cfg_path = parity_app
    (tmp_path / "history_snapshot.json").write_text(
        "{not json at all", encoding="utf-8")
    app2 = create_app(cfg_path, client_factory=lambda: fake)
    client2 = TestClient(app2)
    r = client2.get("/api/state")
    assert r.status_code == 200
    assert r.json()["stats"]["total_signals"] == 0


def test_wrong_shaped_snapshot_loads_empty(parity_app):
    _, _, tmp_path, fake, cfg_path = parity_app
    (tmp_path / "history_snapshot.json").write_text(
        json.dumps({"not": "a list"}), encoding="utf-8")
    app2 = create_app(cfg_path, client_factory=lambda: fake)
    assert TestClient(app2).get("/api/state").json()["stats"]["total_signals"] == 0


# --- gzip ---------------------------------------------------------------------

def _big_state_client(parity_app):
    """Seed enough signals that /api/state exceeds the 1024-byte gzip
    threshold (long last_signal reasoning inflates the body)."""
    _, client, tmp_path, fake, _ = parity_app
    for i in range(15):
        _record_signal(tmp_path, entry=3900.0 + i, sl=3885.0 + i,
                       tp=3930.0 + i, reasoning="R" * 1500)
    return client


def test_gzip_candles_when_requested(parity_app):
    _, client, _, fake, _ = parity_app
    t0 = int(time.time())
    fake.rates_by_tf[("XAUUSD", "M15")] = rates_from_rows(
        [(3900.0 + i, 3905.0 + i, 3895.0 + i, 3901.0 + i)
         for i in range(200)], t0=t0, step=900)
    gz = client.get("/api/candles", headers={"Accept-Encoding": "gzip"})
    plain = client.get("/api/candles", headers={"Accept-Encoding": "identity"})
    assert gz.headers.get("content-encoding") == "gzip"
    assert plain.headers.get("content-encoding") is None
    # Wire size (content-length) is far smaller gzipped; the decoded
    # bodies are identical (httpx decodes gzip transparently).
    assert int(gz.headers["content-length"]) < int(plain.headers["content-length"])
    assert gz.json() == plain.json()


def test_gzip_state_when_requested(parity_app):
    client = _big_state_client(parity_app)
    gz = client.get("/api/state", headers={"Accept-Encoding": "gzip"})
    plain = client.get("/api/state", headers={"Accept-Encoding": "identity"})
    assert gz.headers.get("content-encoding") == "gzip"
    assert plain.headers.get("content-encoding") is None
    assert int(gz.headers["content-length"]) < int(plain.headers["content-length"])
    assert gz.json()["stats"] == plain.json()["stats"]


def test_gzip_dashboard_page_when_requested(parity_app):
    _, client, _, _, _ = parity_app
    gz = client.get("/", headers={"Accept-Encoding": "gzip"})
    plain = client.get("/", headers={"Accept-Encoding": "identity"})
    assert gz.headers.get("content-encoding") == "gzip"
    assert plain.headers.get("content-encoding") is None
    assert int(gz.headers["content-length"]) < int(plain.headers["content-length"])
    assert gz.text == plain.text
    assert "Gold Signals" in gz.text


def test_no_gzip_for_small_plain_response(parity_app):
    _, client, _, _, _ = parity_app
    # Tiny body (< 1024 bytes) stays plain even when gzip is offered.
    r = client.get("/api/candles", headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") is None


# --- /api/config --------------------------------------------------------------

def test_config_shape_and_no_secrets(parity_app):
    _, client, _, _, _ = parity_app
    body = client.get("/api/config").json()
    assert body["mode"] == "demo"
    assert body["feed"] == "auto"
    assert body["symbols"] == ["XAUUSD"]
    assert body["timeframes"] == ["M15"]
    assert body["timeframe"] == "M15"
    assert body["scan_interval_minutes"] == 15
    assert body["kill_zones"] == ["07:00-10:00", "12:00-15:00"]
    assert body["news"]["enabled"] is True
    assert body["news"]["window_minutes"] == 30
    assert body["news"]["min_impact"] == "High"
    assert body["risk"]["max_risk_pct_per_trade"] == pytest.approx(1.0)
    assert body["risk"]["default_playbook"] == "sweep_fvg_reclaim"
    assert body["signals"]["hypothetical_balance_usd"] == pytest.approx(10000.0)
    assert body["signals"]["max_signals_per_day"] == 5
    blob = json.dumps(body).lower()
    for banned in ("token", "api_key", "apikey", "secret", "password",
                   "chat_id", "account_id"):
        assert banned not in blob, f"secret-like key leaked: {banned}"


# --- POST /explain --------------------------------------------------------------

def test_explain_round_trip(parity_app):
    _, client, tmp_path, _, _ = parity_app
    sig_id = _record_signal(tmp_path)
    r = client.post("/explain", json={
        "signal_id": sig_id,
        "explanation": "Sweep of Asia low, FVG reclaim, 2R to TP.",
        "model": "llama-3.2-11b-vision-instruct",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["signal_id"] == sig_id
    assert body["model"] == "llama-3.2-11b-vision-instruct"
    sig = client.get("/api/signals").json()["signals"][0]
    assert sig["explanation"] == "Sweep of Asia low, FVG reclaim, 2R to TP."
    assert sig["explanation_model"] == "llama-3.2-11b-vision-instruct"
    assert sig["explanation_ts"] is not None
    # Entry/SL/TP untouched by the explanation.
    assert sig["entry"] == pytest.approx(3900.0)
    assert sig["stop_loss"] == pytest.approx(3885.0)
    assert sig["take_profit"] == pytest.approx(3930.0)


def test_explain_by_epoch(parity_app):
    _, client, tmp_path, _, _ = parity_app
    ts = int(time.time()) - 3600
    _record_signal(tmp_path, ts=ts)
    r = client.post("/explain", json={
        "epoch": ts,
        "explanation": "Explained by epoch lookup.",
    })
    assert r.status_code == 200
    assert r.json()["signal_id"] is not None
    sig = client.get("/api/signals").json()["signals"][0]
    assert sig["explanation"] == "Explained by epoch lookup."


def test_explain_survives_restart(parity_app):
    _, client, tmp_path, fake, cfg_path = parity_app
    sig_id = _record_signal(tmp_path)
    client.post("/explain", json={
        "signal_id": sig_id, "explanation": "Kept after restart.",
    })
    (tmp_path / "journal.db").unlink()
    for suffix in ("-wal", "-shm"):
        p = tmp_path / f"journal.db{suffix}"
        if p.exists():
            p.unlink()
    app2 = create_app(cfg_path, client_factory=lambda: fake)
    sigs = TestClient(app2).get("/api/signals").json()["signals"]
    assert len(sigs) == 1
    assert sigs[0]["explanation"] == "Kept after restart."


def test_explain_unknown_signal_404(parity_app):
    _, client, _, _, _ = parity_app
    r = client.post("/explain", json={
        "signal_id": 9999, "explanation": "No such signal.",
    })
    assert r.status_code == 404


def test_explain_validation(parity_app):
    _, client, tmp_path, _, _ = parity_app
    sig_id = _record_signal(tmp_path)
    # Empty explanation.
    assert client.post("/explain", json={
        "signal_id": sig_id, "explanation": "   ",
    }).status_code == 422
    # No locator at all.
    assert client.post("/explain", json={
        "explanation": "Nowhere to attach.",
    }).status_code == 422
    # Missing explanation.
    assert client.post("/explain", json={"signal_id": sig_id}).status_code == 422
