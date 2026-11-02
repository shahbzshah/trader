"""End-to-end test of `python -m app.main once` with a fake feed.

Exercises the signals-only acceptance criterion: a full --once cycle on
a mocked environment must write decision rows and journal a signal for
a setup — and must NEVER write an order row (there is no execution).
Decisions come from the deterministic rules engine (no LLM); the fake
feed serves hand-crafted candles that either trigger the sweep+FVG
setup or not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.market_data import Tick
from tests.fake_client import FakeClient, rates_from_rows

# t0 values chosen so the LAST bar lands inside the 08:00-20:00 UTC session
# on a weekday: last bar = 2023-11-15 12:00 UTC (Wednesday), inside the
# 12:00-15:00 UTC New York kill zone.
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
    # Tick at the current price (last close 1.1045) so the buy-limit
    # entry (~1.103) sits below the ask, as a real limit would.
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


def test_once_cycle_writes_decision_and_no_order_for_no_trade(
        tmp_path, monkeypatch, capsys):
    cfg = _load_cfg(tmp_path)

    # Flat market: no sweep, no bias -> the rules engine says no_trade.
    fake = _base_fake()
    fake.rates_by_tf[("XAUUSD", "M15")] = rates_from_rows(
        _flat_rows(250), T0_M15, 900, spread=5)
    fake.rates_by_tf[("XAUUSD", "H1")] = rates_from_rows(
        _flat_rows(72), T0_H1, 3600, spread=5)
    _run_once_main(tmp_path, monkeypatch, cfg, fake)

    from app.journal import Journal
    j = Journal(Path(cfg.paths.data_dir) / "journal.db")
    decs = j.list_decisions(limit=20)
    assert len(decs) >= 1
    assert all(d["action"] == "no_trade" for d in decs)
    assert all(d["approved"] for d in decs)
    # Signals-only: the orders table must stay empty.
    assert j.list_orders(limit=20) == []
    out = capsys.readouterr().out
    assert "signals=0" in out


def test_once_cycle_propose_writes_decision_and_signal_no_order(
        tmp_path, monkeypatch, capsys):
    cfg = _load_cfg(tmp_path)

    fake = _base_fake()
    _setup_fake_with_setup(fake)
    _run_once_main(tmp_path, monkeypatch, cfg, fake)

    from app.journal import Journal
    j = Journal(Path(cfg.paths.data_dir) / "journal.db")
    decs = j.list_decisions(limit=20)
    orders = j.list_orders(limit=20)

    propose_decs = [d for d in decs if d["action"] == "propose"]
    assert len(propose_decs) >= 1
    assert propose_decs[0]["playbook"] == "sweep_fvg_reclaim"
    assert all(d["approved"] for d in propose_decs)
    # No order rows are ever written in signals-only mode.
    assert orders == []
    out = capsys.readouterr().out
    assert "signals=1" in out


def test_once_cycle_kill_switch_blocks(tmp_path, monkeypatch):
    cfg = _load_cfg(tmp_path)
    Path(cfg.paths.kill_switch_path).touch()

    fake = _base_fake()
    _setup_fake_with_setup(fake)
    _run_once_main(tmp_path, monkeypatch, cfg, fake)

    from app.journal import Journal
    j = Journal(Path(cfg.paths.data_dir) / "journal.db")
    assert j.list_orders(limit=20) == []
    decs = j.list_decisions(limit=20)
    assert any("kill_switch_engaged" in d["reject_reasons"] for d in decs)
