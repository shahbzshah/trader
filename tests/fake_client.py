"""Fake market-data client for unit tests (no network, no MT5).

Serves canned candles per (symbol, timeframe) and a canned tick, and
records every call so tests can assert the scanner/runner asked for the
right data. Implements the :class:`app.market_data.MarketDataClient`
surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.market_data import (
    MarketDataError,
    PaperAccountInfo,
    Rates,
    Tick,
    XAUUSD_SPEC,
)


def rates_from_rows(rows, t0: int, step: int, *, spread: int | None = None) -> Rates:
    """Build a :class:`Rates` from (open, high, low, close) tuples."""
    return Rates(rates=[
        {
            "time": t0 + i * step,
            "open": o, "high": h, "low": low, "close": c,
            "tick_volume": 100,
            "spread": spread,
            "real_volume": 0,
        }
        for i, (o, h, low, c) in enumerate(rows)
    ])


@dataclass
class FakeClient:
    rates_by_tf: dict = field(default_factory=dict)
    tick: Tick | None = None
    balance: float = 10000.0
    connect_raises: Exception | None = None
    name: str = "fake"

    def __post_init__(self) -> None:
        self.calls: list[tuple] = []
        self._connected = False

    def connect(self):
        self.calls.append(("connect",))
        if self.connect_raises is not None:
            raise self.connect_raises
        self._connected = True

    def shutdown(self):
        self.calls.append(("shutdown",))
        self._connected = False

    def copy_rates(self, symbol, timeframe, count):
        self.calls.append(("copy_rates", symbol, timeframe, count))
        rates = self.rates_by_tf.get((symbol, timeframe))
        if rates is None:
            return Rates(rates=[])
        if isinstance(rates, Rates):
            return rates
        return Rates(rates=list(rates))

    def symbol_tick(self, symbol):
        self.calls.append(("symbol_tick", symbol))
        return self.tick

    def symbol_info(self, symbol):
        return XAUUSD_SPEC

    def select_symbol(self, symbol):
        self.calls.append(("select_symbol", symbol))
        return True

    def positions(self, symbol=None):
        self.calls.append(("positions", symbol))
        return []

    def account_info(self):
        return PaperAccountInfo(
            balance=self.balance, equity=self.balance, trade_mode="paper")

    def feed_status(self):
        return {
            "name": self.name,
            "connected": self._connected,
            "last_tick_epoch": self.tick.time if self.tick else None,
            "last_error": None,
        }


__all__ = ["FakeClient", "rates_from_rows"]
