"""End-to-end test of `python -m app.main once` with a fake feed.

The phil-rudy-v2 strategy is live (2026-11-11): a cycle evaluates
the last closed M15 bar. Non-signal bars still journal nothing —
even textbook sweep+FVG setup candles (the old strategy is gone),
and even with the kill switch engaged. Signal bars fire exactly
once, honour the gates, and flip / stop-out correctly; the
frozen-data tests below prove that against real backtested bars.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.market_data import Tick
from tests.fake_client import FakeClient, rates_from_rows

# t0 values chosen so the LAST bar lands inside the 08:00-20:00 UTC session
# on a weekday: last bar = 2023-11-15 12:00 UTC (Wednesday).
T0_M15 = 1700049600 - 104 * 900
T0_H1 = 1700049600 - 72 * 3600


def _flat_rows(n, price=1.1000, point=0.0001):
    return [(price, price + point, price - point, price)] * n


def _bullish_m15_rows():
    """100 flat bars, then a sweep of the low + bullish FVG + rally."""
    rows = [(1.1020, 1.1025, 1.1015, 1.1020)] * 100
    rows += [
        (1.1020, 1.1022, 1.1005, 1.1018),  # sweep bar
        (1.1018, 1.1025, 1.1016, 1.1023),
        (1.1023, 1.1040, 1.1030, 1.1038),  # bullish FVG
        (1.1038, 1.1045, 1.1035, 1.1042),
        (1.1042, 1.1048, 1.1038, 1.1045),  # last bar
    ]
    return rows


def _uptrend_h1_rows():
    """Clean higher-high / higher-low structure for a bullish H1 bias."""
    pivots = [1.0900, 1.0960, 1.0930, 1.0990, 1.0960,
              1.1020, 1.0990, 1.1060, 1.1030, 1.1080]
    rows = []
    px = pivots[0]
    for leg in range(len(pivots) - 1):
        for b in range(8):
            c = pivots[leg] + (pivots[leg + 1] - pivots[leg]) * b / 8
            w = 0.0002 * ((b % 3) - 1)
            o = px + w * 0.5
            rows.append((o, max(o, c) + abs(w) + 0.0001,
                         min(o, c) - abs(w) - 0.0001, c))
            px = c
    rows.append((px, px + 0.0002, px - 0.0002, pivots[-1]))
    return rows


def _setup_fake_with_setup(fake):
    """Serve setup-triggering candles: M15 sweep+FVG, H1 uptrend."""
    fake.rates_by_tf[("XAUUSD", "M15")] = rates_from_rows(
        _bullish_m15_rows(), T0_M15, 900, spread=5)
    fake.rates_by_tf[("XAUUSD", "H1")] = rates_from_rows(
        _uptrend_h1_rows(), T0_H1, 3600, spread=5)
    fake.tick = Tick(bid=1.1044, ask=1.1045, spread=10.0, time=T0_M15 + 104 * 900)


def _base_fake():
    fake = FakeClient()
    fake.tick = Tick(bid=1.1000, ask=1.1001, spread=10.0, time=1)
    return fake


def _run_once_main(tmp_path, monkeypatch, cfg, fake):
    from app import main as main_module
    monkeypatch.setattr(main_module, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(
        main_module, "build_market_client", lambda *a, **k: (fake, "fake"))
    rc = main_module.main(["--config", "config.yaml", "once"])
    assert rc == 0


def _load_cfg(tmp_path):
    from app.config import load_config

    cfg = load_config("config.yaml", env_path=tmp_path / ".env")
    cfg.paths.data_dir = str(tmp_path)
    cfg.paths.kill_switch_path = str(tmp_path / "KILL_SWITCH")
    return cfg


def _journal(tmp_path, cfg):
    from app.journal import Journal
    return Journal(Path(cfg.paths.data_dir) / "journal.db")


def test_once_cycle_scans_but_journals_nothing(
        tmp_path, monkeypatch, capsys):
    cfg = _load_cfg(tmp_path)

    fake = _base_fake()
    fake.rates_by_tf[("XAUUSD", "M15")] = rates_from_rows(
        _flat_rows(250), T0_M15, 900, spread=5)
    fake.rates_by_tf[("XAUUSD", "H1")] = rates_from_rows(
        _flat_rows(72), T0_H1, 3600, spread=5)
    _run_once_main(tmp_path, monkeypatch, cfg, fake)

    j = _journal(tmp_path, cfg)
    assert j.list_decisions(limit=20) == []
    assert j.list_orders(limit=20) == []
    out = capsys.readouterr().out
    assert "decisions=0" in out
    assert "signals=0" in out


def test_once_cycle_setup_candles_still_produce_nothing(
        tmp_path, monkeypatch, capsys):
    cfg = _load_cfg(tmp_path)

    # Textbook sweep+FVG candles, but only 105 bars: below the 300-bar
    # warmup, so the strategy cannot evaluate a signal bar and must
    # still produce no decision and no signal.
    fake = _base_fake()
    _setup_fake_with_setup(fake)
    _run_once_main(tmp_path, monkeypatch, cfg, fake)

    j = _journal(tmp_path, cfg)
    assert j.list_decisions(limit=20) == []
    assert j.list_orders(limit=20) == []
    out = capsys.readouterr().out
    assert "signals=0" in out


def test_once_cycle_completes_with_kill_switch_engaged(
        tmp_path, monkeypatch, capsys):
    cfg = _load_cfg(tmp_path)
    Path(cfg.paths.kill_switch_path).touch()

    fake = _base_fake()
    _setup_fake_with_setup(fake)
    _run_once_main(tmp_path, monkeypatch, cfg, fake)

    j = _journal(tmp_path, cfg)
    assert j.list_orders(limit=20) == []
    assert j.list_decisions(limit=20) == []
    out = capsys.readouterr().out
    assert "signals=0" in out


# --- phil-rudy-v2 end-to-end (frozen Deriv M15 bars) ----------------------
#
# Bar references (verified against ~/workspace/trading/... backtests):
#   bar 499: first V2 buy, epoch 1772598600, close 5161.77; its $10
#            stop is hit on bar 505 (backtest stop trade, pnl -$10.50).
#   bar 823: V2 buy; bar 965: the opposite V2 sell that flips it
#            (backtest flip trade, pnl +$79.98 per lot).
#   bar 702: a Phil triangle whose RUDY trend disagrees (sell, itrend
#            +1) — filtered out, must fire nothing.

_FROZEN = Path.home() / "workspace/trading/phil-backtest/gold-m15.json"


def _load_frozen_bars():
    import json as _json

    if not _FROZEN.exists():
        pytest.skip(f"frozen data not present: {_FROZEN}")
    return _json.loads(_FROZEN.read_text(encoding="utf-8"))


def _frozen_fake(upto_bar: int, tick_time: int | None = None):
    """FakeClient serving frozen bars 0..upto_bar-1 with their real
    epochs (weekend gaps preserved — uniform-step builders would
    shift every epoch and break the dedup assertions)."""
    from app.market_data import Rates

    bars = _load_frozen_bars()
    fake = FakeClient()
    fake.rates_by_tf[("XAUUSD", "M15")] = Rates(rates=[
        {"time": int(b["epoch"]), "open": float(b["open"]),
         "high": float(b["high"]), "low": float(b["low"]),
         "close": float(b["close"]), "tick_volume": 0,
         "spread": None, "real_volume": 0}
        for b in bars[:upto_bar]
    ])
    last = bars[upto_bar - 1]
    fake.tick = Tick(bid=float(last["close"]),
                     ask=float(last["close"]) + 0.05, spread=5.0,
                     time=int(tick_time or last["epoch"]) + 900)
    return fake


def _open_position(tmp_path, cfg):
    path = Path(cfg.paths.data_dir) / "open_position.json"
    if not path.exists():
        return None
    import json as _json

    return _json.loads(path.read_text(encoding="utf-8"))


def test_once_v2_buy_fires_exactly_once(tmp_path, monkeypatch, capsys):
    cfg = _load_cfg(tmp_path)
    _run_once_main(tmp_path, monkeypatch, cfg, _frozen_fake(500))

    j = _journal(tmp_path, cfg)
    decisions = j.list_decisions(limit=20)
    assert len(decisions) == 1
    row = decisions[0]
    assert row["action"] == "propose" and row["approved"] is True
    assert row["playbook"] == "phil-rudy-v2"
    assert row["confidence"] == pytest.approx(0.48)
    assert row["ts"] == 1772598600
    import json as _json

    prev = _json.loads(row["raw_llm_json"])
    assert prev["direction"] == "buy"
    assert prev["price"] == pytest.approx(5161.77)
    assert prev["stop_loss"] == pytest.approx(5151.77)
    assert prev["take_profit"] is None
    assert "RUDY itrend +1 agrees" in row["reasoning"]
    assert "no take-profit" in row["reasoning"]

    pos = _open_position(tmp_path, cfg)
    assert pos is not None
    assert pos["direction"] == "buy"
    assert pos["entry"] == pytest.approx(5161.77)
    assert pos["stop_loss"] == pytest.approx(5151.77)
    assert pos["lots"] == pytest.approx(0.10)
    assert pos["decision_id"] == row["id"]
    out = capsys.readouterr().out
    assert "decisions=1" in out and "signals=1" in out

    # Second run of the same bar: duplicate, nothing new journaled.
    _run_once_main(tmp_path, monkeypatch, cfg, _frozen_fake(500))
    assert len(j.list_decisions(limit=20)) == 1
    out = capsys.readouterr().out
    assert "duplicates=1" in out and "signals=0" in out


def test_once_v2_buy_blocked_by_kill_switch(tmp_path, monkeypatch, capsys):
    cfg = _load_cfg(tmp_path)
    Path(cfg.paths.kill_switch_path).touch()
    _run_once_main(tmp_path, monkeypatch, cfg, _frozen_fake(500))

    j = _journal(tmp_path, cfg)
    decisions = j.list_decisions(limit=20)
    assert len(decisions) == 1
    assert decisions[0]["action"] == "no_trade"
    assert decisions[0]["approved"] is False
    assert "kill switch" in decisions[0]["reject_reasons"][0]
    assert _open_position(tmp_path, cfg) is None
    out = capsys.readouterr().out
    assert "signals=0" in out and "rejected=1" in out


def test_once_v2_buy_blocked_by_blackout(tmp_path, monkeypatch, capsys):
    import datetime as _dt

    from app import news as news_mod

    cfg = _load_cfg(tmp_path)
    event = news_mod.CalEvent(
        title="CPI m/m", currency="USD",
        time_utc=_dt.datetime.now(_dt.timezone.utc), impact="High")
    monkeypatch.setattr(news_mod, "current_blackout",
                        lambda *a, **k: event)
    _run_once_main(tmp_path, monkeypatch, cfg, _frozen_fake(500))

    j = _journal(tmp_path, cfg)
    decisions = j.list_decisions(limit=20)
    assert len(decisions) == 1
    assert decisions[0]["action"] == "no_trade"
    assert decisions[0]["approved"] is False
    assert "blackout" in decisions[0]["reject_reasons"][0]
    assert _open_position(tmp_path, cfg) is None
    out = capsys.readouterr().out
    assert "signals=0" in out and "rejected=1" in out


def test_once_disagreeing_phil_fires_nothing(tmp_path, monkeypatch, capsys):
    cfg = _load_cfg(tmp_path)
    _run_once_main(tmp_path, monkeypatch, cfg, _frozen_fake(703))

    j = _journal(tmp_path, cfg)
    assert j.list_decisions(limit=20) == []
    out = capsys.readouterr().out
    assert "signals=0" in out


def test_once_flip_closes_and_opens(tmp_path, monkeypatch, capsys):
    bars = _load_frozen_bars()
    cfg = _load_cfg(tmp_path)

    # Long opens at bar 823 (buy 5117.91 per the frozen data).
    _run_once_main(tmp_path, monkeypatch, cfg, _frozen_fake(824))
    j = _journal(tmp_path, cfg)
    assert len(j.list_decisions(limit=20)) == 1
    entry = float(bars[823]["close"])
    assert entry == pytest.approx(5117.91)

    # Bar 965 is the opposite V2 sell: closes the long at the flip
    # and opens the short. No V2 signal and no stop hit in between.
    _run_once_main(tmp_path, monkeypatch, cfg, _frozen_fake(966))
    decisions = j.list_decisions(limit=20)
    assert len([d for d in decisions if d["action"] == "propose"]) == 2
    closes = j.list_positions_history()
    assert len(closes) == 1
    close = closes[0]
    exit_px = float(bars[965]["close"])
    assert close["direction"] == "buy"
    assert close["price_open"] == pytest.approx(entry)
    assert close["price_close"] == pytest.approx(exit_px)
    assert close["profit"] == pytest.approx(
        (exit_px - entry) * 0.10 * 100.0 - 0.50)
    # The close is ticketed with the journal id of the signal that
    # opened the position (the older of the two decisions: the list
    # is newest-first, so the buy is decisions[-1]).
    assert close["ticket"] == decisions[-1]["id"]
    assert decisions[-1]["ts"] == int(bars[823]["epoch"])

    # FLIP outcome written to the durable history (dashboard ledger).
    from app.history import load_snapshot

    recs = load_snapshot(Path(cfg.paths.data_dir) / "history_snapshot.json")
    flips = [r for r in recs if r["status"] == "FLIP"]
    assert len(flips) == 1
    assert flips[0]["close_price"] == pytest.approx(exit_px)
    assert flips[0]["close_ts"] == int(bars[965]["epoch"])
    assert flips[0]["take_profit"] is None
    assert flips[0]["r"] == pytest.approx(close["profit"] / 100.0)

    # The short is now the open position.
    pos = _open_position(tmp_path, cfg)
    assert pos is not None
    assert pos["direction"] == "sell"
    assert pos["entry"] == pytest.approx(exit_px)
    assert pos["stop_loss"] == pytest.approx(exit_px + 10.0)
    out = capsys.readouterr().out
    assert "signals=1" in out


def test_once_stop_out_closes_position(tmp_path, monkeypatch, capsys):
    cfg = _load_cfg(tmp_path)

    # Long opens at bar 499 (buy 5161.77, SL 5151.77).
    _run_once_main(tmp_path, monkeypatch, cfg, _frozen_fake(500))
    assert _open_position(tmp_path, cfg) is not None

    # Bar 505 runs through the stop (backtest stop trade, -$10.50
    # per lot): the position closes, no new signal fires on bar 505.
    _run_once_main(tmp_path, monkeypatch, cfg, _frozen_fake(506))
    j = _journal(tmp_path, cfg)
    closes = j.list_positions_history()
    assert len(closes) == 1
    close = closes[0]
    assert close["direction"] == "buy"
    assert close["price_close"] == pytest.approx(5151.77)
    assert close["profit"] == pytest.approx(-100.50)
    assert _open_position(tmp_path, cfg) is None
    # Only the original propose decision exists — the stop close
    # journals a position row, not a new decision.
    assert len(j.list_decisions(limit=20)) == 1
    out = capsys.readouterr().out
    assert "signals=0" in out
