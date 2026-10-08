"""Tests for app/backtest.py.

Uses synthetic M15 candles with an engineered sweep+FVG round-trip so the
full scanner -> strategy -> risk -> paper-execution path is exercised.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from app.backtest import load_candles, resample_h1, run_backtest
from app.config import load_config

# Signal bar (index 304) lands on 2023-11-15 12:00 UTC (Wednesday),
# inside the default 08:00-20:00 UTC session.
T0 = 1700049600 - 304 * 900


def _engineered_rows():
    rows = []
    # Bars 0-279: gentle uptrend + noise -> H1 bias stays bullish.
    for i in range(280):
        base = 1.1000 + i * 0.00003 + 0.0004 * math.sin(i * 0.35)
        o = base
        c = base + 0.0001 * math.sin(i * 1.7)
        rows.append((o, max(o, c) + 0.0002, min(o, c) - 0.0002, c))
    P = rows[-1][3]
    # Bars 280-299: flat shelf at P.
    for _ in range(20):
        rows.append((P, P + 0.0005, P - 0.0005, P))
    # Bars 300-304: liquidity sweep of the shelf low + bullish FVG + rally.
    rows += [
        (P, P + 0.0002, P - 0.0015, P - 0.0002),   # 300: sweep
        (P - 0.0002, P + 0.0005, P - 0.0004, P + 0.0003),
        (P + 0.0003, P + 0.0020, P + 0.0010, P + 0.0018),  # 302: FVG
        (P + 0.0018, P + 0.0025, P + 0.0015, P + 0.0022),
        (P + 0.0022, P + 0.0028, P + 0.0018, P + 0.0025),  # 304: signal bar
    ]
    # Bars 305-307: pullback into the FVG -> fills the limit order.
    rows += [
        (P + 0.0025, P + 0.0026, P + 0.0008, P + 0.0010),
        (P + 0.0010, P + 0.0012, P + 0.0005, P + 0.0007),
        (P + 0.0007, P + 0.0010, P + 0.0004, P + 0.0008),
    ]
    # Bars 308-319: rally through the 2R target.
    for i in range(12):
        c = P + 0.0008 + i * 0.0006
        o = c - 0.0003
        rows.append((o, c + 0.0003, o - 0.0002, c))
    return rows


def _write_csv(tmp_path: Path, rows) -> Path:
    df = pd.DataFrame([
        {"time": T0 + i * 900, "open": o, "high": h, "low": l,
         "close": c, "volume": 100}
        for i, (o, h, l, c) in enumerate(rows)
    ])
    p = tmp_path / "xauusd_m15.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def cfg(tmp_path):
    return load_config("config.yaml", env_path=tmp_path / ".env")


def test_backtest_takes_the_engineered_trade(tmp_path, cfg):
    csv = _write_csv(tmp_path, _engineered_rows())
    s = run_backtest(csv, symbol="XAUUSD", timeframe="M15", config=cfg)
    assert s["bars"] == 320
    assert s["trades"] >= 1
    assert s["wins"] >= 1
    # Winners exit at the 2R target.
    assert s["avg_r"] > 0
    assert s["final_equity"] > 10000.0
    assert 0.0 <= s["max_drawdown_pct"] < 100.0


def test_backtest_flat_market_takes_no_trades(tmp_path, cfg):
    rows = [(1.1000, 1.1005, 1.0995, 1.1000)] * 200
    csv = _write_csv(tmp_path, rows)
    s = run_backtest(csv, symbol="XAUUSD", timeframe="M15", config=cfg)
    assert s["trades"] == 0
    assert s["final_equity"] == pytest.approx(10000.0)
    assert s["profit_factor"] == 0.0


def test_backtest_rejects_short_csv(tmp_path, cfg):
    rows = [(1.1000, 1.1005, 1.0995, 1.1000)] * 10
    csv = _write_csv(tmp_path, rows)
    with pytest.raises(ValueError, match="at least"):
        run_backtest(csv, symbol="XAUUSD", timeframe="M15", config=cfg)


def test_backtest_rejects_bad_columns(tmp_path, cfg):
    p = tmp_path / "bad.csv"
    pd.DataFrame([{"a": 1}]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        run_backtest(p, symbol="XAUUSD", timeframe="M15", config=cfg)


def test_resample_h1_shape():
    rows = _engineered_rows()
    df = pd.DataFrame([
        {"time": T0 + i * 900, "open": o, "high": h, "low": l,
         "close": c, "volume": 100}
        for i, (o, h, l, c) in enumerate(rows)
    ])
    h1 = resample_h1(df)
    assert list(h1.columns) == ["time", "open", "high", "low", "close", "volume"]
    assert len(h1) == pytest.approx(len(df) / 4, abs=2)
    # H1 bars are chronologically ordered and non-overlapping.
    assert (h1["time"].diff().dropna() == 3600).all()


def test_load_candles_sorts_and_coerces(tmp_path):
    p = tmp_path / "unsorted.csv"
    pd.DataFrame([
        {"time": 3, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0,
         "volume": 5},
        {"time": 1, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0,
         "volume": 5},
        {"time": 2, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0},
    ]).to_csv(p, index=False)
    df = load_candles(p)
    assert list(df["time"]) == [1, 2, 3]
    assert "volume" in df.columns
