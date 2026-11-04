"""End-to-end test of `python -m app.main once` with a fake feed.

The strategy was removed on 2026-11-04: a cycle now only keeps the
feed warm and refreshes candles for the dashboard. It must journal
nothing, signal nothing, and never write an order row — even when the
feed serves textbook sweep+FVG setup candles, and even with the kill
switch engaged.
"""

from __future__ import annotations

from pathlib import Path

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

    # Textbook sweep+FVG candles: with the strategy removed, this must
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
