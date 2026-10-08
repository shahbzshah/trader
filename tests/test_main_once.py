"""End-to-end test of `python -m app.main once` with a mocked MT5.

Exercises the M4 acceptance criterion: a full --once cycle on a mocked
environment must write decision + order rows. Decisions now come from the
deterministic rules engine (no LLM); the fake MT5 serves hand-crafted
candles that either trigger the sweep+FVG setup or not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fake_mt5 import (
    FakeRatesRow,
    FakeSymbolInfo,
    FakeTick,
    make_fake,
)

# Timeframe constants mirror the real MT5 module (see tests/fake_mt5.py).
TF_M15 = 15
TF_H1 = 16385

# t0 values chosen so the LAST bar lands inside the 08:00-20:00 UTC session
# on a weekday: last bar = 2023-11-15 12:00 UTC (Wednesday).
T0_M15 = 1700049600 - 104 * 900
T0_H1 = 1700049600 - 72 * 3600


def _install_fake_mt5(fake) -> None:
    import app.mt5_client as client_module
    client_module._mt5 = fake


def _to_rates(rows, t0, step):
    return [
        FakeRatesRow(
            time=t0 + i * step,
            open=o, high=h, low=l, close=c,
            tick_volume=100, spread=5, real_volume=0,
        )
        for i, (o, h, l, c) in enumerate(rows)
    ]


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
    fake.rates_by_tf[TF_M15] = _to_rates(_bullish_m15_rows(), T0_M15, 900)
    fake.rates_by_tf[TF_H1] = _to_rates(_uptrend_h1_rows(), T0_H1, 3600)


def _base_fake():
    fake = make_fake()
    fake.symbol_data["EURUSD"] = FakeSymbolInfo(name="EURUSD")
    fake.symbol_data["GBPUSD"] = FakeSymbolInfo(name="GBPUSD")
    fake.symbol_data["XAUUSD"] = FakeSymbolInfo(name="XAUUSD", point=0.01,
                                               digits=2, volume_step=0.01,
                                               trade_contract_size=100.0)
    fake.tick_data["EURUSD"] = FakeTick(bid=1.1, ask=1.1001, time=1)
    fake.tick_data["GBPUSD"] = FakeTick(bid=1.25, ask=1.2501, time=1)
    fake.tick_data["XAUUSD"] = FakeTick(bid=1900.0, ask=1900.1, time=1)
    return fake


def _run_once_main(tmp_path, monkeypatch, cfg):
    from app import main as main_module
    monkeypatch.setattr(main_module, "load_config", lambda *a, **k: cfg)
    rc = main_module.main([
        "--config", "config.yaml", "once",
        "--login", "12345", "--password", "pw", "--server", "x",
    ])
    assert rc == 0


def test_once_cycle_writes_decision_and_no_order_for_no_trade(tmp_path, monkeypatch):
    from app.config import load_config

    cfg = load_config("config.yaml", env_path=tmp_path / ".env")
    cfg.paths.data_dir = str(tmp_path)
    cfg.paths.kill_switch_path = str(tmp_path / "KILL_SWITCH")

    # Flat market: no sweep, no bias -> the rules engine says no_trade.
    fake = _base_fake()
    fake.rates_data = _to_rates(_flat_rows(250), T0_M15, 900)
    _install_fake_mt5(fake)
    _run_once_main(tmp_path, monkeypatch, cfg)

    from app.journal import Journal
    j = Journal(Path(cfg.paths.data_dir) / "journal.db")
    decs = j.list_decisions(limit=20)
    assert len(decs) >= 1
    assert all(d["action"] == "no_trade" for d in decs)
    assert all(d["approved"] for d in decs)
    assert j.list_orders(limit=20) == []
    assert len(j.list_equity(limit=5)) == 1


def test_once_cycle_propose_writes_decision_and_order(tmp_path, monkeypatch):
    from app.config import load_config

    cfg = load_config("config.yaml", env_path=tmp_path / ".env")
    cfg.paths.data_dir = str(tmp_path)
    cfg.paths.kill_switch_path = str(tmp_path / "KILL_SWITCH")

    fake = _base_fake()
    _setup_fake_with_setup(fake)
    _install_fake_mt5(fake)
    _run_once_main(tmp_path, monkeypatch, cfg)

    from app.journal import Journal
    j = Journal(Path(cfg.paths.data_dir) / "journal.db")
    decs = j.list_decisions(limit=20)
    orders = j.list_orders(limit=20)

    propose_decs = [d for d in decs if d["action"] == "propose"]
    # The sweep+FVG strategy emits entry_type="limit", so the live executor
    # places a pending order (status "pending"), never a market fill.
    pending = [o for o in orders if o["status"] == "pending"]
    assert len(propose_decs) >= 1
    assert len(pending) >= 1
    assert pending[0]["decision_id"] in {d["id"] for d in propose_decs}
    assert propose_decs[0]["playbook"] == "sweep_fvg_reclaim"


def test_once_cycle_kill_switch_blocks(tmp_path, monkeypatch):
    from app.config import load_config

    cfg = load_config("config.yaml", env_path=tmp_path / ".env")
    cfg.paths.data_dir = str(tmp_path)
    cfg.paths.kill_switch_path = str(tmp_path / "KILL_SWITCH")
    Path(cfg.paths.kill_switch_path).touch()

    fake = _base_fake()
    _setup_fake_with_setup(fake)
    _install_fake_mt5(fake)
    _run_once_main(tmp_path, monkeypatch, cfg)

    from app.journal import Journal
    j = Journal(Path(cfg.paths.data_dir) / "journal.db")
    orders = j.list_orders(limit=20)
    assert all(o["status"] != "filled" for o in orders)
    decs = j.list_decisions(limit=20)
    assert any("kill_switch_engaged" in d["reject_reasons"] for d in decs)
