"""Tests for app/scanner.py: indicator primitives and build_snapshot."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import pytest

from app.scanner import (
    atr,
    build_snapshot,
    ema,
    rsi,
    scan_watchlist,
)
from tests.fixtures_candles import (
    fixture_to_rates_dict,
    make_constant_fixture,
    make_downtrend_fixture,
    make_flat_then_pop_fixture,
    make_uptrend_fixture,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeAccount:
    balance: float = 10000.0
    equity: float = 10000.0
    trade_mode: str = "demo"


@dataclass
class FakePos:
    ticket: int = 0
    symbol: str = "EURUSD"
    direction: str = "buy"
    volume: float = 0.10
    price_open: float = 1.1000
    price_current: float = 1.1050
    sl: float = 1.0950
    tp: float = 1.1100
    profit: float = 50.0


def rates_dict_to_rates_obj(d: list[dict], symbol: str = "EURUSD",
                             tf: str = "M15"):
    """Wrap a list of rate dicts in the same shape as app.mt5_client.Rates."""

    @dataclass
    class _Rates:
        rates: list[dict]
        symbol: str
        timeframe: str

    return _Rates(rates=d, symbol=symbol, timeframe=tf)


# ---------------------------------------------------------------------------
# Primitive indicator tests
# ---------------------------------------------------------------------------


def test_ema_constant_series():
    s = pd.Series([1.1] * 50)
    out = ema(s, 20)
    # For a constant series, EMA == constant to floating-point precision.
    for v in out.iloc[19:]:
        assert math.isclose(v, 1.1, abs_tol=1e-12)


def test_ema_uptrend_lags_price():
    s = pd.Series([1.0 + i * 0.001 for i in range(250)])
    e = ema(s, 50)
    last = float(s.iloc[-1])
    e_last = float(e.iloc[-1])
    # EMA must lag the uptrend.
    assert e_last < last
    assert e_last > float(s.iloc[0])


def test_ema_invalid_period():
    with pytest.raises(ValueError):
        ema(pd.Series([1.0, 2.0]), 0)


def test_rsi_constant_is_50():
    s = pd.Series([1.1000] * 30)
    out = rsi(s, 14)
    # No gains, no losses => RSI falls back to 50.
    for v in out.iloc[13:]:
        assert math.isclose(v, 50.0, abs_tol=1e-9)


def test_rsi_pure_uptrend_is_high():
    s = pd.Series([1.0 + i * 0.001 for i in range(50)])
    out = rsi(s, 14)
    last = float(out.iloc[-1])
    assert last > 70.0  # strong uptrend


def test_rsi_pure_downtrend_is_low():
    s = pd.Series([1.3 - i * 0.001 for i in range(50)])
    out = rsi(s, 14)
    last = float(out.iloc[-1])
    assert last < 30.0  # strong downtrend


def test_rsi_invalid_period():
    with pytest.raises(ValueError):
        rsi(pd.Series([1.0, 2.0]), 0)


def test_atr_constant_series_known_value():
    """For a flat series with constant high-low = 2*point, ATR(14) = 2*point."""
    n = 50
    point = 0.0001
    close = pd.Series([1.1000] * n)
    high = pd.Series([1.1000 + point] * n)
    low = pd.Series([1.1000 - point] * n)
    out = atr(high, low, close, 14)
    last = float(out.iloc[-1])
    assert math.isclose(last, 2 * point, abs_tol=1e-6)


def test_atr_zero_when_no_range():
    n = 30
    close = pd.Series([1.1] * n)
    high = pd.Series([1.1] * n)
    low = pd.Series([1.1] * n)
    out = atr(high, low, close, 14)
    assert math.isclose(float(out.iloc[-1]), 0.0, abs_tol=1e-12)


def test_atr_invalid_period():
    with pytest.raises(ValueError):
        atr(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]), 0)


# ---------------------------------------------------------------------------
# build_snapshot tests (deterministic fixtures)
# ---------------------------------------------------------------------------


def test_snapshot_constant_series_indicators():
    fx = make_constant_fixture(n=250, price=1.1000, point=0.0001)
    rates = fixture_to_rates_dict(fx)
    snap = build_snapshot("eurusd", {"M15": rates}, FakeAccount(), [])

    assert snap["symbol"] == "EURUSD"
    tf_block = snap["timeframes"]["M15"]
    assert tf_block["bars"] == 250
    assert math.isclose(tf_block["last_close"], 1.1, abs_tol=1e-5)
    # EMAs converge to 1.1; RSI 50; ATR = 2*point.
    assert math.isclose(tf_block["ema20"], 1.1, abs_tol=1e-4)
    assert math.isclose(tf_block["ema50"], 1.1, abs_tol=1e-4)
    assert math.isclose(tf_block["ema200"], 1.1, abs_tol=1e-4)
    assert math.isclose(tf_block["rsi14"], 50.0, abs_tol=1e-3)
    assert math.isclose(tf_block["atr14"], 2 * 0.0001, abs_tol=1e-5)


def test_snapshot_uptrend_signals():
    fx = make_uptrend_fixture(n=250, start=1.0, step=0.001)
    rates = fixture_to_rates_dict(fx)
    snap = build_snapshot("EURUSD", {"M15": rates}, FakeAccount(), [])
    tf_block = snap["timeframes"]["M15"]
    # Last close known analytically.
    assert math.isclose(tf_block["last_close"], 1.0 + 249 * 0.001, abs_tol=1e-4)
    # EMA50 lags; last_close > EMA50 > EMA200 (trend).
    assert tf_block["last_close"] > tf_block["ema50"] > tf_block["ema200"]
    # Strong RSI.
    assert tf_block["rsi14"] > 70
    # 50-bar candle list at the tail.
    assert len(tf_block["candles"]) == 50
    # All OHLC values rounded to 5 dp.
    for c in tf_block["candles"]:
        for k in ("o", "h", "l", "c"):
            # The serialized value should be either None or a float.
            assert c[k] is None or isinstance(c[k], float)


def test_snapshot_downtrend_signals():
    fx = make_downtrend_fixture(n=250, start=1.3, step=0.001)
    rates = fixture_to_rates_dict(fx)
    snap = build_snapshot("EURUSD", {"M15": rates}, FakeAccount(), [])
    tf_block = snap["timeframes"]["M15"]
    assert tf_block["last_close"] < tf_block["ema50"] < tf_block["ema200"]
    assert tf_block["rsi14"] < 30


def test_snapshot_swing_high_low_uptrend():
    """Last-20 swing in an uptrend = max(high.tail(20)), min(low.tail(20))."""
    fx = make_uptrend_fixture(n=250, start=1.0, step=0.001)
    rates = fixture_to_rates_dict(fx)
    snap = build_snapshot("EURUSD", {"M15": rates}, FakeAccount(), [])
    last20_high = max(fx.high[-20:])
    last20_low = min(fx.low[-20:])
    assert math.isclose(snap["swing_high"], round(last20_high, 5), abs_tol=1e-5)
    assert math.isclose(snap["swing_low"], round(last20_low, 5), abs_tol=1e-5)


def test_snapshot_session_bounds_constant_series():
    fx = make_constant_fixture(n=250)
    rates = fixture_to_rates_dict(fx)
    snap = build_snapshot("EURUSD", {"M15": rates}, FakeAccount(), [])
    assert snap["session"]["start_ts"] is not None
    assert snap["session"]["end_ts"] is not None
    assert snap["session"]["end_ts"] > snap["session"]["start_ts"]
    # Session high/low for constant series = price +/- point.
    assert snap["session"]["high"] is not None
    assert snap["session"]["low"] is not None


def test_snapshot_open_positions_included():
    fx = make_constant_fixture(n=250)
    pos = FakePos(ticket=12345, symbol="EURUSD", direction="buy",
                  volume=0.20, price_open=1.1, price_current=1.105,
                  sl=1.09, tp=1.12, profit=50.0)
    snap = build_snapshot("EURUSD", {"M15": fixture_to_rates_dict(fx)},
                            FakeAccount(), [pos])
    assert len(snap["open_positions"]) == 1
    p = snap["open_positions"][0]
    assert p["ticket"] == 12345
    assert p["direction"] == "buy"
    assert math.isclose(p["volume"], 0.20, abs_tol=1e-5)


def test_snapshot_multiple_timeframes():
    fx_long = make_constant_fixture(n=250, price=1.1)
    fx_short = make_uptrend_fixture(n=80, start=1.0, step=0.001)
    snap = build_snapshot(
        "EURUSD",
        {
            "M15": fixture_to_rates_dict(fx_long),
            "H1": fixture_to_rates_dict(fx_short),
        },
        FakeAccount(),
        [],
    )
    assert set(snap["timeframes"].keys()) == {"M15", "H1"}
    # H1 has 80 bars (< 200), so EMA200 is None but EMA50 must be set.
    assert snap["timeframes"]["H1"]["ema200"] is None
    assert snap["timeframes"]["H1"]["ema50"] is not None


def test_snapshot_short_series_indicators_none():
    """A 10-bar series should produce None for EMA50/EMA200/RSI14/ATR14."""
    fx = make_constant_fixture(n=10, price=1.1)
    snap = build_snapshot("EURUSD", {"M15": fixture_to_rates_dict(fx)},
                            FakeAccount(), [])
    tf_block = snap["timeframes"]["M15"]
    assert math.isclose(tf_block["last_close"], 1.1, abs_tol=1e-5)
    # None because we don't have enough bars to seed the indicator.
    assert tf_block["ema50"] is None
    assert tf_block["ema200"] is None
    assert tf_block["rsi14"] is None
    assert tf_block["atr14"] is None
    # All 10 bars in the candle tail (less than 50).
    assert len(tf_block["candles"]) == 10


def test_snapshot_empty_rates():
    snap = build_snapshot("EURUSD", {"M15": []}, FakeAccount(), [])
    tf_block = snap["timeframes"]["M15"]
    assert tf_block["bars"] == 0
    assert tf_block["last_close"] is None
    assert tf_block["candles"] == []
    # No swing/session because no data.
    assert snap["swing_high"] is None
    assert snap["swing_low"] is None


def test_snapshot_spread_from_primary_tf():
    fx = make_constant_fixture(n=250)
    snap = build_snapshot("EURUSD", {"M15": fixture_to_rates_dict(fx)},
                            FakeAccount(), [])
    assert snap["spread"] == 5  # from the last bar's spread column


def test_snapshot_account_block():
    fx = make_constant_fixture(n=250)
    snap = build_snapshot("EURUSD", {"M15": fixture_to_rates_dict(fx)},
                            FakeAccount(balance=12345.6789, equity=12000.0), [])
    assert snap["account"]["balance"] == 12345.6789
    assert snap["account"]["equity"] == 12000.0
    assert snap["account"]["trade_mode"] == "demo"


def test_snapshot_output_is_stable_json():
    """The snapshot must be JSON-serialisable with no NaN/Inf leaks."""
    import json

    fx = make_flat_then_pop_fixture()
    snap = build_snapshot("EURUSD", {"M15": fixture_to_rates_dict(fx)},
                            FakeAccount(), [])
    encoded = json.dumps(snap)  # must not raise
    decoded = json.loads(encoded)
    assert decoded["symbol"] == "EURUSD"
    # No NaN/Inf in serialized output.
    assert "NaN" not in encoded
    assert "Infinity" not in encoded


def test_snapshot_rounds_to_5dp():
    fx = make_uptrend_fixture(n=250, start=1.0, step=0.001)
    snap = build_snapshot("EURUSD", {"M15": fixture_to_rates_dict(fx)},
                            FakeAccount(), [])
    for c in snap["timeframes"]["M15"]["candles"]:
        for k in ("o", "h", "l", "c"):
            v = c[k]
            assert v is None or round(v, 5) == v


# ---------------------------------------------------------------------------
# scan_watchlist integration (with a fake client)
# ---------------------------------------------------------------------------


class FakeClient:
    def __init__(self, rates_by_symbol_tf: dict):
        self.rates_by_symbol_tf = rates_by_symbol_tf
        self.calls: list[tuple] = []

    def account_info(self):
        return FakeAccount()

    def copy_rates(self, symbol, tf, count):
        self.calls.append(("copy_rates", symbol, tf))
        return self.rates_by_symbol_tf.get((symbol, tf), [])

    def symbol_tick(self, symbol):
        self.calls.append(("symbol_tick", symbol))
        from app.mt5_client import Tick
        return Tick(symbol=symbol, bid=1.0, ask=1.0001, spread=0.0001, time=1)

    def positions(self, symbol=None):
        self.calls.append(("positions", symbol))
        return []


@dataclass
class Cfg:
    symbols: list
    timeframes: list


def test_scan_watchlist_one_snapshot_per_symbol():
    fx = make_constant_fixture(n=250)
    rates_dict = fixture_to_rates_dict(fx)
    rates_obj = rates_dict_to_rates_obj(rates_dict)
    client = FakeClient({("EURUSD", "M15"): rates_obj, ("GBPUSD", "M15"): rates_obj})
    cfg = Cfg(symbols=["EURUSD", "GBPUSD"], timeframes=["M15"])
    snaps = scan_watchlist(client, cfg)
    assert len(snaps) == 2
    assert {s["symbol"] for s in snaps} == {"EURUSD", "GBPUSD"}


def test_scan_watchlist_survives_symbol_failure():
    """A failing symbol must not abort the scan."""
    client = FakeClient({})  # empty -> copy_rates returns []

    class FlakyClient(FakeClient):
        def copy_rates(self, symbol, tf, count):
            if symbol == "GBPUSD":
                raise RuntimeError("boom")
            return super().copy_rates(symbol, tf, count)

    client = FlakyClient({("EURUSD", "M15"): rates_dict_to_rates_obj(
        fixture_to_rates_dict(make_constant_fixture(n=250)))})
    cfg = Cfg(symbols=["EURUSD", "GBPUSD"], timeframes=["M15"])
    snaps = scan_watchlist(client, cfg)
    assert len(snaps) == 2
    symbols = {s["symbol"] for s in snaps}
    assert symbols == {"EURUSD", "GBPUSD"}


def test_scan_watchlist_sets_spread_from_tick():
    fx = make_constant_fixture(n=250)
    rates_obj = rates_dict_to_rates_obj(fixture_to_rates_dict(fx))
    client = FakeClient({("EURUSD", "M15"): rates_obj})
    cfg = Cfg(symbols=["EURUSD"], timeframes=["M15"])
    snaps = scan_watchlist(client, cfg)
    # spread comes from the tick (bid=1.0, ask=1.0001 -> 0.0001)
    assert snaps[0]["spread"] is not None
    assert snaps[0]["spread"] == pytest.approx(0.0001, abs=1e-9)