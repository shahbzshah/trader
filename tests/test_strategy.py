"""Deterministic tests for app/strategy.py.

Snapshots are built through the real scanner from hand-crafted candles so
the tests exercise the full scanner -> strategy path (no mocks).
"""

from __future__ import annotations

import pandas as pd
import pytest
from types import SimpleNamespace

from app import scanner as sc
from app.strategy import (
    PLAYBOOK_NAME,
    SETUP_CONFIDENCE,
    SETUP_RISK_PCT,
    decide,
)


T0 = 1704067200  # 2024-01-01 00:00:00 UTC (Monday)


def _df(rows, step=900, t0=T0):
    return pd.DataFrame([
        {"time": t0 + i * step, "open": o, "high": h, "low": l, "close": c,
         "tick_volume": 10, "spread": 10, "real_volume": 0}
        for i, (o, h, l, c) in enumerate(rows)
    ])


def _trend_df(pivots, bars_per_leg=8, step=3600, t0=T0, wiggle=0.0002):
    """Interpolate clean swing structure between pivot prices."""
    rows = []
    px = pivots[0]
    for leg in range(len(pivots) - 1):
        start, end = pivots[leg], pivots[leg + 1]
        for b in range(bars_per_leg):
            frac = b / bars_per_leg
            c = start + (end - start) * frac
            w = wiggle * ((b % 3) - 1)
            o = px + w * 0.5
            h = max(o, c) + abs(w) + 0.0001
            l = min(o, c) - abs(w) - 0.0001
            rows.append((o, h, l, c))
            px = c
    rows.append((px, px + 0.0002, px - 0.0002, pivots[-1]))
    return _df(rows, step=step, t0=t0)


def _bullish_h1():
    return _trend_df([1.0900, 1.0960, 1.0930, 1.0990, 1.0960,
                      1.1020, 1.0990, 1.1060, 1.1030, 1.1080])


def _bearish_h1():
    return _trend_df([1.1080, 1.1020, 1.1050, 1.0990, 1.1020,
                      1.0960, 1.0990, 1.0930, 1.0960, 1.0900])


def _bullish_m15():
    """100 flat bars, then: sweep of the low, bullish FVG, rally."""
    rows = [(1.1020, 1.1025, 1.1015, 1.1020)] * 100
    rows += [
        (1.1020, 1.1022, 1.1005, 1.1018),  # sweep bar (idx 100)
        (1.1018, 1.1025, 1.1016, 1.1023),
        (1.1023, 1.1040, 1.1030, 1.1038),  # bullish FVG (idx 102)
        (1.1038, 1.1045, 1.1035, 1.1042),
        (1.1042, 1.1048, 1.1038, 1.1045),  # last bar
    ]
    return _df(rows)


def _bearish_m15():
    """Mirror image: flat, sweep of the high, bearish FVG, drop."""
    rows = [(1.1020, 1.1025, 1.1015, 1.1020)] * 100
    rows += [
        (1.1020, 1.1035, 1.1018, 1.1022),  # sweep bar (idx 100)
        (1.1022, 1.1024, 1.1015, 1.1017),
        (1.1017, 1.1010, 1.1000, 1.1002),  # bearish FVG (idx 102)
        (1.1002, 1.1005, 1.0995, 1.0998),
        (1.0998, 1.1002, 1.0992, 1.0995),  # last bar
    ]
    return _df(rows)


def _snap(m15, h1):
    return sc.build_snapshot("EURUSD", {"M15": m15, "H1": h1}, None, [])


# ---------------------------------------------------------------------------
# The full setup fires
# ---------------------------------------------------------------------------


def test_bullish_sweep_fvg_proposes_buy():
    d = decide(_snap(_bullish_m15(), _bullish_h1()), None)
    assert d.action == "propose"
    assert d.direction == "buy"
    assert d.entry_type == "limit"
    assert d.playbook == PLAYBOOK_NAME
    assert d.confidence == SETUP_CONFIDENCE
    assert d.risk_pct == SETUP_RISK_PCT
    # Entry is the FVG midpoint, below the market.
    assert d.price == pytest.approx(1.1030, abs=1e-4)
    # Stop beyond the sweep extreme (1.1005), take profit at 2R.
    assert d.stop_loss < 1.1005
    assert d.stop_loss < d.price < d.take_profit
    assert (d.take_profit - d.price) == pytest.approx(
        2.0 * (d.price - d.stop_loss), abs=2e-5)  # 5-dp rounding
    assert "bullish" in d.reasoning


def test_bearish_sweep_fvg_proposes_sell():
    d = decide(_snap(_bearish_m15(), _bearish_h1()), None)
    assert d.action == "propose"
    assert d.direction == "sell"
    assert d.entry_type == "limit"
    assert d.playbook == PLAYBOOK_NAME
    assert d.price == pytest.approx(1.1010, abs=1e-4)
    assert d.stop_loss > 1.1035  # beyond the sweep extreme (high)
    assert d.take_profit < d.price < d.stop_loss
    assert (d.price - d.take_profit) == pytest.approx(
        2.0 * (d.stop_loss - d.price), abs=2e-5)  # 5-dp rounding
    assert "bearish" in d.reasoning


def test_sl_never_closer_than_one_atr():
    d = decide(_snap(_bullish_m15(), _bullish_h1()), None)
    snap = _snap(_bullish_m15(), _bullish_h1())
    atr = snap["timeframes"]["M15"]["atr14"]
    assert (d.price - d.stop_loss) >= atr * 1.0 - 1e-9


# ---------------------------------------------------------------------------
# Missing pieces -> no_trade
# ---------------------------------------------------------------------------


def test_no_sweep_no_trade():
    flat = _df([(1.1020, 1.1025, 1.1015, 1.1020)] * 105)
    d = decide(_snap(flat, _bullish_h1()), None)
    assert d.action == "no_trade"
    assert "sweep" in d.reasoning


def test_sweep_without_fvg_no_trade():
    rows = [(1.1020, 1.1025, 1.1015, 1.1020)] * 100
    rows += [
        (1.1020, 1.1022, 1.1005, 1.1018),  # sweep
        (1.1018, 1.1020, 1.1014, 1.1017),
        (1.1017, 1.1021, 1.1013, 1.1019),  # chop, no displacement
        (1.1019, 1.1022, 1.1015, 1.1020),
        (1.1020, 1.1023, 1.1016, 1.1021),
    ]
    d = decide(_snap(_df(rows), _bullish_h1()), None)
    assert d.action == "no_trade"
    assert "FVG" in d.reasoning


def test_bias_against_sweep_no_trade():
    # Bearish H1 bias with a bullish sweep setup -> stand down.
    d = decide(_snap(_bullish_m15(), _bearish_h1()), None)
    assert d.action == "no_trade"


def test_ranging_h1_no_trade():
    flat_h1 = _df([(1.1000, 1.1005, 1.0995, 1.1000)] * 70, step=3600)
    d = decide(_snap(_bullish_m15(), flat_h1), None)
    assert d.action == "no_trade"
    assert "bias" in d.reasoning


def test_filled_fvg_no_trade():
    rows = [(1.1020, 1.1025, 1.1015, 1.1020)] * 100
    rows += [
        (1.1020, 1.1022, 1.1005, 1.1018),  # sweep
        (1.1018, 1.1025, 1.1016, 1.1023),
        (1.1023, 1.1040, 1.1030, 1.1038),  # bullish FVG
        (1.1038, 1.1042, 1.1010, 1.1012),  # crashes back through the gap
        (1.1012, 1.1018, 1.1008, 1.1015),  # last bar
    ]
    d = decide(_snap(_df(rows), _bullish_h1()), None)
    assert d.action == "no_trade"


def test_wide_spread_no_trade():
    snap = _snap(_bullish_m15(), _bullish_h1())
    snap["spread"] = 999
    d = decide(snap, None)
    assert d.action == "no_trade"
    assert "spread" in d.reasoning


def test_missing_timeframes_no_trade():
    assert decide({"symbol": "EURUSD", "timeframes": {}}, None).action == "no_trade"
    snap = _snap(_bullish_m15(), _bullish_h1())
    del snap["timeframes"]["H1"]
    assert decide(snap, None).action == "no_trade"


# ---------------------------------------------------------------------------
# Session filter
# ---------------------------------------------------------------------------


def _session_cfg():
    return SimpleNamespace(
        session_hours=SimpleNamespace(start="08:00", end="20:00",
                                      timezone="UTC", skip_weekends=True))


def test_inside_session_trades():
    # Last M15 bar: T0 + 104*900 = 2024-01-01 02:00 UTC -> outside; shift t0.
    m15 = _df([(1.1020, 1.1025, 1.1015, 1.1020)] * 100 + [
        (1.1020, 1.1022, 1.1005, 1.1018),
        (1.1018, 1.1025, 1.1016, 1.1023),
        (1.1023, 1.1040, 1.1030, 1.1038),
        (1.1038, 1.1045, 1.1035, 1.1042),
        (1.1042, 1.1048, 1.1038, 1.1045),
    ], t0=T0 + 10 * 3600)  # last bar 12:00 UTC, Monday
    d = decide(_snap(m15, _bullish_h1()), _session_cfg())
    assert d.action == "propose"


def test_outside_session_no_trade():
    m15 = _bullish_m15()  # last bar 2024-01-01 02:00 UTC
    d = decide(_snap(m15, _bullish_h1()), _session_cfg())
    assert d.action == "no_trade"
    assert "session" in d.reasoning


# ---------------------------------------------------------------------------
# Kill-zone filter
# ---------------------------------------------------------------------------


def _killzone_cfg():
    return SimpleNamespace(
        session_hours=SimpleNamespace(start="08:00", end="20:00",
                                      timezone="UTC", skip_weekends=True),
        kill_zones=["07:00-10:00", "12:00-15:00"])


def _setup_m15(t0):
    # Same firing setup as test_inside_session_trades; 105 bars so the
    # last bar lands at t0 + 104*900 = t0 + 26h.
    return _df([(1.1020, 1.1025, 1.1015, 1.1020)] * 100 + [
        (1.1020, 1.1022, 1.1005, 1.1018),
        (1.1018, 1.1025, 1.1016, 1.1023),
        (1.1023, 1.1040, 1.1030, 1.1038),
        (1.1038, 1.1045, 1.1035, 1.1042),
        (1.1042, 1.1048, 1.1038, 1.1045),
    ], t0=t0)


def test_inside_kill_zone_trades():
    # Last bar 2024-01-01 09:00 UTC: inside the London kill zone.
    m15 = _setup_m15(T0 - 17 * 3600)
    d = decide(_snap(m15, _bullish_h1()), _killzone_cfg())
    assert d.action == "propose"


def test_outside_kill_zone_no_trade():
    # Last bar 2024-01-01 11:00 UTC: inside session, outside kill zones.
    m15 = _setup_m15(T0 - 15 * 3600)
    d = decide(_snap(m15, _bullish_h1()), _killzone_cfg())
    assert d.action == "no_trade"
    assert "kill zone" in d.reasoning


def test_no_kill_zones_configured_no_filter():
    # Same 11:00 UTC bar, but the session-only config has no kill zones.
    m15 = _setup_m15(T0 - 15 * 3600)
    d = decide(_snap(m15, _bullish_h1()), _session_cfg())
    assert d.action == "propose"


# ---------------------------------------------------------------------------
# Sanity on suggestion fields
# ---------------------------------------------------------------------------


def test_suggestion_fields_sane():
    d = decide(_snap(_bullish_m15(), _bullish_h1()), None)
    assert 0.0 < d.confidence <= 1.0
    assert d.risk_pct > 0
    assert d.symbol == "EURUSD"
