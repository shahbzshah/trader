"""Shared market-data types and the client interface (signals-only).

This module is the single boundary between the rules engine
(:mod:`app.scanner`, :mod:`app.strategy`, :mod:`app.risk`) and the
market-data providers. It replaces ``app.mt5_client`` — there is no
MetaTrader5 import anywhere in the signals-only codebase.

Two providers implement :class:`MarketDataClient`:

* :mod:`app.oanda_client` — OANDA v20 REST (Practice), the MAIN source.
* :mod:`app.deriv_client` — Deriv WebSocket v3, the FALLBACK / cross-check.

Both expose the same surface the scanner already uses:

* ``copy_rates(symbol, timeframe, count)`` -> :class:`Rates`
* ``symbol_tick(symbol)`` -> :class:`Tick` | None
* ``symbol_info(symbol)`` -> :class:`SymbolInfo`
* ``select_symbol(symbol)`` -> bool (best-effort no-op for REST/WS feeds)
* ``positions(symbol=None)`` -> always ``[]`` (signals-only: no positions
  exist, so breakeven / open-trade logic has nothing to manage)
* ``account_info()`` -> :class:`PaperAccountInfo` built from the configured
  hypothetical balance (the scanner reads balance/equity best-effort)

Symbol naming: the bot's config uses ``XAUUSD``. OANDA calls the
instrument ``XAU_USD``; Deriv calls it ``frxXAUUSD``. Each client maps
between the two. ``XAUUSD`` is the only supported symbol for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# --- symbol mapping ----------------------------------------------------


#: Canonical bot symbol.
BOT_SYMBOL = "XAUUSD"
#: OANDA v20 instrument name for gold.
OANDA_INSTRUMENT = "XAU_USD"
#: Deriv symbol for gold (spot, USD).
DERIV_SYMBOL = "frxXAUUSD"


class MarketDataError(Exception):
    """Raised when a feed cannot serve a request (network, auth, bad data)."""


# --- shared data types ---------------------------------------------------


@dataclass(frozen=True)
class Rates:
    """Candles for one symbol/timeframe.

    Each entry is a dict with keys ``time`` (epoch seconds, UTC),
    ``open``, ``high``, ``low``, ``close``, ``tick_volume``,
    ``spread`` (points, or ``None`` when the feed does not report one)
    and ``real_volume``. The scanner accepts this object, a plain list
    of dicts, or a DataFrame.
    """

    rates: list[dict]


@dataclass(frozen=True)
class Tick:
    """Latest quote for a symbol.

    ``spread`` is in *points* (price units / ``point``), matching
    ``strategy.MAX_SPREAD_POINTS`` — e.g. a $0.35 bid/ask gap on gold
    (point 0.01) is 35 points. ``None`` when the feed reports no spread.
    """

    bid: float
    ask: float
    spread: float | None
    time: int  # epoch seconds, UTC


@dataclass(frozen=True)
class SymbolInfo:
    """Static contract facts the sizing math needs."""

    digits: int
    point: float
    volume_min: float
    volume_max: float
    volume_step: float
    trade_contract_size: float
    description: str = ""


#: Gold contract facts (OANDA XAU_USD / Deriv frxXAUUSD spot, per oz).
#: 1 lot = 100 oz, so $1 of price move = $100 per lot.
XAUUSD_SPEC = SymbolInfo(
    digits=2,
    point=0.01,
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
    trade_contract_size=100.0,
    description="Gold / US Dollar (spot, per oz)",
)


@dataclass(frozen=True)
class PaperAccountInfo:
    """Hypothetical account state for signals-only sizing.

    There is no broker account: the bot sizes every signal against the
    *configured* hypothetical balance (``signals.hypothetical_balance_usd``,
    default $10,000). ``trade_mode`` is ``"paper"`` — risk.py's demo check
    is satisfied by the signals layer passing ``trade_mode="demo"`` to
    ``AccountState`` explicitly; this object is only what the scanner
    reads best-effort.
    """

    balance: float
    equity: float
    trade_mode: str = "paper"
    currency: str = "USD"
    login: int = 0
    server: str = ""


# --- client interface ------------------------------------------------------


@runtime_checkable
class MarketDataClient(Protocol):
    """Data-only market feed. No order, position or account-mutation API."""

    name: str

    def connect(self) -> None:
        """Verify the feed is reachable (probe request). Raises
        :class:`MarketDataError` when it is not."""
        ...

    def shutdown(self) -> None:
        """Release any connections / background threads."""
        ...

    def copy_rates(self, symbol: str, timeframe: str, count: int) -> Rates:
        """Return up to ``count`` candles, oldest first. ``timeframe``
        is one of the config names (``M15``, ``H1``)."""
        ...

    def symbol_tick(self, symbol: str) -> Tick | None:
        """Return the latest quote, or ``None`` when unavailable."""
        ...

    def symbol_info(self, symbol: str) -> SymbolInfo:
        """Return static contract facts for ``symbol``."""
        ...

    def select_symbol(self, symbol: str) -> bool:
        """Best-effort subscribe/visibility no-op. Always True for
        REST/WebSocket feeds (symbols are always available)."""
        ...

    def positions(self, symbol: str | None = None) -> list:
        """Always ``[]`` in signals-only mode — there are no positions."""
        ...

    def account_info(self) -> PaperAccountInfo:
        """Return the hypothetical paper account state."""
        ...

    def feed_status(self) -> dict:
        """Small dict for /health and /api/state: name, connected flag,
        last tick epoch, last error (if any)."""
        ...


__all__ = [
    "BOT_SYMBOL",
    "DERIV_SYMBOL",
    "OANDA_INSTRUMENT",
    "MarketDataClient",
    "MarketDataError",
    "PaperAccountInfo",
    "Rates",
    "SymbolInfo",
    "Tick",
    "XAUUSD_SPEC",
]
