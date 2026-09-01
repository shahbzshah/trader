"""Deterministic synthetic-candle fixtures for scanner tests."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CandleFixture:
    """A candle series with a known, deterministic pattern."""

    bars: int
    close: list[float]
    high: list[float]
    low: list[float]
    open: list[float]
    time: list[int]
    spread: list[int] | None = None


def _const_series(n: int, value: float) -> list[float]:
    return [float(value)] * n


def make_constant_fixture(n: int = 250, price: float = 1.1000,
                          point: float = 0.0001) -> CandleFixture:
    """All closes = price, all highs = price+point, all lows = price-point.

    For a perfectly constant series:
      EMA(close, N) == price
      ATR(14) == 2 * point  (true range = high - low)
      RSI(14) == 50  (no gains, no losses)
    """
    time = list(range(1700000000, 1700000000 + n * 900, 900))  # M15 spacing
    return CandleFixture(
        bars=n,
        close=_const_series(n, price),
        high=[float(price) + point] * n,
        low=[float(price) - point] * n,
        open=_const_series(n, price),
        time=time,
        spread=[5] * n,
    )


def make_uptrend_fixture(n: int = 250, start: float = 1.0000,
                          step: float = 0.0010) -> CandleFixture:
    """Linear uptrend: close[i] = start + i * step.

    For this series:
      last_close = start + (n-1) * step
      EMA(N) < last_close for any N (it lags)
      RSI(14) > 70 (all gains)
      ATR(14) > 0 (constant up move = constant true range of 2*step - ish,
        because prev_close = close[i-1] so |high-low|=2*step and
        |high-prev_close|=|step-step|=0, |low-prev_close|=|step-step|=0)
    """
    time = list(range(1700000000, 1700000000 + n * 900, 900))
    close = [start + i * step for i in range(n)]
    return CandleFixture(
        bars=n,
        close=close,
        high=[c + step for c in close],
        low=[c - step for c in close],
        open=list(close),
        time=time,
        spread=[5] * n,
    )


def make_downtrend_fixture(n: int = 250, start: float = 1.3000,
                            step: float = 0.0010) -> CandleFixture:
    """Linear downtrend: close[i] = start - i * step."""
    time = list(range(1700000000, 1700000000 + n * 900, 900))
    close = [start - i * step for i in range(n)]
    return CandleFixture(
        bars=n,
        close=close,
        high=[c + step for c in close],
        low=[c - step for c in close],
        open=list(close),
        time=time,
        spread=[5] * n,
    )


def fixture_to_rates_dict(fx: CandleFixture) -> list[dict]:
    """Convert a fixture into the dict shape Rates expects."""
    spread = fx.spread if fx.spread is not None else [0] * fx.bars
    return [
        {
            "time": int(fx.time[i]),
            "open": float(fx.open[i]),
            "high": float(fx.high[i]),
            "low": float(fx.low[i]),
            "close": float(fx.close[i]),
            "tick_volume": 100,
            "spread": int(spread[i]),
            "real_volume": 0,
        }
        for i in range(fx.bars)
    ]


def make_flat_then_pop_fixture(n_flat: int = 60, pop_size: int = 5,
                                flat_price: float = 1.2000,
                                pop_size_value: float = 0.0100) -> CandleFixture:
    """60 bars at flat_price then 5 bars trending up by pop_size_value each.

    Used to test swing high/low and ATR behaviour around a regime change.
    """
    n = n_flat + pop_size
    time = list(range(1700000000, 1700000000 + n * 900, 900))
    close = [flat_price] * n_flat + [flat_price + (i + 1) * pop_size_value
                                      for i in range(pop_size)]
    return CandleFixture(
        bars=n,
        close=close,
        high=[c + 0.0005 for c in close],
        low=[c - 0.0005 for c in close],
        open=list(close),
        time=time,
        spread=[5] * n,
    )


# Approximate equality for floats in tests.
def approx(x: float, *, rel: float = 1e-6, abs_: float = 1e-9) -> float:
    return x


def close(x: float, y: float, tol: float = 1e-4) -> bool:
    return math.isclose(x, y, rel_tol=tol, abs_tol=tol)