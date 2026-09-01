"""The ONLY module that imports MetaTrader5.

This thin, typed wrapper around the MT5 Python API converts MT5 enums to
plain strings at the boundary so the rest of the codebase never touches
``MetaTrader5`` types directly. Every method raises a clear exception on
failure and logs at the appropriate level.

The import of ``MetaTrader5`` is performed lazily so unit tests can swap in
a fake module via ``app.mt5_client._mt5`` before any client is constructed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Lazy-loaded module reference. Tests can monkeypatch this with a fake.
_mt5: Any = None


def _get_mt5():
    """Return the MetaTrader5 module, importing on first use."""
    global _mt5
    if _mt5 is None:
        import MetaTrader5 as _real_mt5  # noqa: WPS433 (deliberate lazy import)

        _mt5 = _real_mt5
    return _mt5


class MT5Error(RuntimeError):
    """Base class for MT5 client errors."""


class MT5NotInstalledError(MT5Error):
    """MetaTrader5 package import failed."""


class MT5ConnectionError(MT5Error):
    """Terminal connection / initialize failed."""


class MT5LoginError(MT5Error):
    """Login to the MT5 account failed."""


class LiveAccountError(MT5Error):
    """Refusing to operate on a non-demo account (safety invariant)."""


class MT5RequestError(MT5Error):
    """A request to MT5 (rates, order, ...) failed."""


@dataclass(frozen=True)
class AccountInfo:
    login: int
    name: str
    server: str
    currency: str
    trade_mode: str  # "demo" or "real" or "unknown"
    balance: float
    equity: float
    margin: float
    free_margin: float
    leverage: int

    @classmethod
    def from_mt5(cls, info: Any) -> "AccountInfo":
        mt5 = _get_mt5()
        # trade_mode enum: ACCOUNT_TRADE_MODE_DEMO=0 -> demo, else real/contest
        raw_mode = int(getattr(info, "trade_mode", -1))
        if raw_mode == int(getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)):
            mode = "demo"
        elif raw_mode == int(getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 1)):
            mode = "real"
        elif raw_mode == int(getattr(mt5, "ACCOUNT_TRADE_MODE_CONTEST", 2)):
            mode = "contest"
        else:
            mode = "unknown"
        return cls(
            login=int(info.login),
            name=str(getattr(info, "name", "")),
            server=str(getattr(info, "server", "")),
            currency=str(getattr(info, "currency", "")),
            trade_mode=mode,
            balance=float(info.balance),
            equity=float(info.equity),
            margin=float(info.margin),
            free_margin=float(info.margin_free),
            leverage=int(info.leverage),
        )


@dataclass(frozen=True)
class Position:
    ticket: int
    symbol: str
    direction: str  # "buy" | "sell"
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    comment: str

    @classmethod
    def from_mt5(cls, p: Any) -> "Position":
        mt5 = _get_mt5()
        direction = "buy" if int(p.type) == int(mt5.ORDER_TYPE_BUY) else "sell"
        return cls(
            ticket=int(p.ticket),
            symbol=str(p.symbol),
            direction=direction,
            volume=float(p.volume),
            price_open=float(p.price_open),
            price_current=float(p.price_current),
            sl=float(p.sl),
            tp=float(p.tp),
            profit=float(p.profit),
            comment=str(getattr(p, "comment", "")),
        )


@dataclass(frozen=True)
class Rates:
    """OHLC rates for one symbol/timeframe."""

    symbol: str
    timeframe: str
    rates: list[dict]

    def to_dataframe(self):
        """Convert to pandas DataFrame. Imports pandas lazily to avoid hard dep."""
        import pandas as pd

        return pd.DataFrame(self.rates)


@dataclass(frozen=True)
class Tick:
    symbol: str
    bid: float
    ask: float
    spread: float
    time: int

    @classmethod
    def from_mt5(cls, t: Any, symbol: str) -> "Tick":
        spread = float(t.ask - t.bid)
        return cls(
            symbol=symbol,
            bid=float(t.bid),
            ask=float(t.ask),
            spread=spread,
            time=int(t.time),
        )


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    digits: int
    point: float
    volume_min: float
    volume_max: float
    volume_step: float
    trade_contract_size: float


@dataclass(frozen=True)
class OrderResult:
    retcode: int
    order_id: int
    price: float
    comment: str


# Map friendly timeframe strings to MT5 constants.
_TIMEFRAME_MAP: dict[str, str] = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
    "MN1": "TIMEFRAME_MN1",
}

_ORDER_TYPE_MAP: dict[str, str] = {
    "buy": "ORDER_TYPE_BUY",
    "sell": "ORDER_TYPE_SELL",
}

_FILLING_MAP: dict[str, int] = {
    "FOK": 0,
    "IOC": 1,
    "RETURN": 2,
}


class MT5Client:
    """Thin, typed wrapper around the MetaTrader5 package."""

    def __init__(self) -> None:
        self._connected = False
        self._login: int = 0
        self._server: str = ""

    # --- lifecycle -------------------------------------------------------

    def connect(
        self,
        login: int,
        password: str,
        server: str,
        path: str = "",
    ) -> None:
        """Initialize and log in. Retries once on transient failure."""
        mt5 = _get_mt5()
        attempts = 0
        last_exc: Exception | None = None
        while attempts < 2:
            attempts += 1
            try:
                init_args: dict[str, Any] = {}
                if path:
                    init_args["path"] = path
                ok = mt5.initialize(**init_args) if init_args else mt5.initialize()
                if not ok:
                    err = mt5.last_error()
                    raise MT5ConnectionError(f"initialize() failed: {err}")
                authorized = mt5.login(
                    login=int(login),
                    password=password,
                    server=server,
                )
                if not authorized:
                    err = mt5.last_error()
                    raise MT5LoginError(f"login() failed: {err}")
                self._connected = True
                self._login = int(login)
                self._server = server
                logger.info("MT5 connected login=%s server=%s", login, server)
                return
            except (MT5ConnectionError, MT5LoginError) as exc:
                last_exc = exc
                logger.warning("MT5 connect attempt %s failed: %s", attempts, exc)
                try:
                    mt5.shutdown()
                except Exception:  # noqa: BLE001 - best effort cleanup
                    pass
                time.sleep(0.5)
        raise last_exc if last_exc else MT5ConnectionError("connect failed")

    def shutdown(self) -> None:
        mt5 = _get_mt5()
        try:
            mt5.shutdown()
        finally:
            self._connected = False

    def ensure_demo(self) -> AccountInfo:
        """Refuse to operate on a non-demo account."""
        info = self.account_info()
        if info.trade_mode != "demo":
            raise LiveAccountError(
                f"refusing to trade on non-demo account (trade_mode={info.trade_mode})"
            )
        return info

    # --- queries ---------------------------------------------------------

    def account_info(self) -> AccountInfo:
        self._require_connected()
        mt5 = _get_mt5()
        info = mt5.account_info()
        if info is None:
            raise MT5RequestError(f"account_info() failed: {mt5.last_error()}")
        return AccountInfo.from_mt5(info)

    def positions(self, symbol: str | None = None) -> list[Position]:
        self._require_connected()
        mt5 = _get_mt5()
        if symbol is not None:
            raw = mt5.positions_get(symbol=symbol)
        else:
            raw = mt5.positions_get()
        if raw is None:
            return []
        return [Position.from_mt5(p) for p in raw]

    def symbol_info(self, symbol: str) -> SymbolInfo:
        self._require_connected()
        mt5 = _get_mt5()
        s = mt5.symbol_info(symbol)
        if s is None:
            raise MT5RequestError(f"symbol_info({symbol}) failed: {mt5.last_error()}")
        return SymbolInfo(
            symbol=str(s.name),
            digits=int(s.digits),
            point=float(s.point),
            volume_min=float(s.volume_min),
            volume_max=float(s.volume_max),
            volume_step=float(s.volume_step),
            trade_contract_size=float(getattr(s, "trade_contract_size", 1.0) or 1.0),
        )

    def copy_rates(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> Rates:
        self._require_connected()
        mt5 = _get_mt5()
        tf_attr = _TIMEFRAME_MAP.get(timeframe.upper())
        if tf_attr is None:
            raise ValueError(f"unsupported timeframe: {timeframe}")
        tf = getattr(mt5, tf_attr)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(count))
        if rates is None:
            raise MT5RequestError(
                f"copy_rates_from_pos({symbol}, {timeframe}, {count}) failed: "
                f"{mt5.last_error()}"
            )
        # Convert structured numpy array to list[dict] with safe field coercion.
        records: list[dict] = []
        for r in rates:
            records.append(
                {
                    "time": int(r["time"]),
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "tick_volume": int(r["tick_volume"]),
                    "spread": int(r["spread"]),
                    "real_volume": int(r["real_volume"]),
                }
            )
        return Rates(symbol=symbol, timeframe=timeframe, rates=records)

    def symbol_tick(self, symbol: str) -> Tick:
        self._require_connected()
        mt5 = _get_mt5()
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            raise MT5RequestError(f"symbol_info_tick({symbol}) failed: {mt5.last_error()}")
        return Tick.from_mt5(t, symbol)

    # --- market-watch helpers --------------------------------------------

    def select_symbol(self, symbol: str, enable: bool = True) -> bool:
        """Make ``symbol`` visible in the terminal's Market Watch.

        Real brokers require a symbol to be selected before its rates/ticks
        can be read, so this is best-effort and idempotent. Returns True when
        the symbol is (now) available, False otherwise.
        """
        self._require_connected()
        mt5 = _get_mt5()
        # Already visible -> fast path.
        try:
            if mt5.symbol_info(symbol) is not None:
                return True
        except Exception:  # noqa: BLE001 - defensive
            pass
        try:
            return bool(mt5.symbol_select(symbol, enable))
        except Exception:  # noqa: BLE001 - some fakes/experts lack symbol_select
            return False

    def symbol_names(self, pattern: str = "*") -> list[str]:
        """All symbols available in the terminal (for broker-suffix discovery).

        Returns the names of every symbol whose name matches ``pattern``
        according to the terminal's ``symbols_get`` (default: all). This lets
        the doctor discover e.g. ``EURUSD.a`` / ``EURUSDm`` variants without
        importing ``MetaTrader5`` elsewhere.
        """
        self._require_connected()
        mt5 = _get_mt5()
        names: list[str] = []
        try:
            rows = mt5.symbols_get(pattern) or []
        except Exception:  # noqa: BLE001
            rows = []
        for r in rows:
            name = getattr(r, "name", None)
            if name:
                names.append(str(name))
        # De-duplicate while preserving terminal order.
        seen: set[str] = set()
        return [n for n in names if not (n in seen or seen.add(n))]

    def _type_filling(self, symbol: str) -> int:
        """Pick a supported order filling mode for ``symbol``.

        ``symbol_info.filling_mode`` is a bitmask of allowed modes
        (SYMBOL_FILLING_MODE_FOK/IOC/RETURN). We prefer IOC (the historical
        default), then FOK, then RETURN. If the symbol's allowed modes can't
        be determined, fall back to IOC which has the broadest support.
        """
        mt5 = _get_mt5()
        mode = 0
        try:
            info = mt5.symbol_info(symbol)
            if info is not None:
                mode = int(getattr(info, "filling_mode", 0) or 0)
        except Exception:  # noqa: BLE001
            mode = 0
        bits = {
            "FOK": int(getattr(mt5, "SYMBOL_FILLING_MODE_FOK", 1)),
            "IOC": int(getattr(mt5, "SYMBOL_FILLING_MODE_IOC", 2)),
            "RETURN": int(getattr(mt5, "SYMBOL_FILLING_MODE_RETURN", 4)),
        }
        for label in ("IOC", "FOK", "RETURN"):
            if mode == 0 or (mode & bits[label]):
                return int(getattr(mt5, f"ORDER_FILLING_{label}", 1))
        return int(getattr(mt5, "ORDER_FILLING_IOC", 1))

    # --- orders ----------------------------------------------------------

    def market_order(
        self,
        symbol: str,
        direction: str,
        lots: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> OrderResult:
        self._require_connected()
        # Re-check demo at order time (safety invariant).
        self.ensure_demo()
        mt5 = _get_mt5()
        type_attr = _ORDER_TYPE_MAP.get(direction.lower())
        if type_attr is None:
            raise ValueError(f"direction must be 'buy' or 'sell', got {direction!r}")
        order_type = getattr(mt5, type_attr)
        tick = self.symbol_tick(symbol)
        price = tick.ask if direction.lower() == "buy" else tick.bid
        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": 99001,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._type_filling(symbol),
        }
        result = mt5.order_send(request)
        if result is None:
            raise MT5RequestError(f"order_send returned None: {mt5.last_error()}")
        retcode = int(result.retcode)
        if retcode != int(mt5.TRADE_RETCODE_DONE):
            raise MT5RequestError(
                f"order rejected retcode={retcode} comment={result.comment!r}"
            )
        return OrderResult(
            retcode=retcode,
            order_id=int(result.order),
            price=float(result.price),
            comment=str(result.comment),
        )

    def close_position(self, ticket: int) -> OrderResult:
        self._require_connected()
        self.ensure_demo()
        mt5 = _get_mt5()
        pos_list: Iterable[Any] = mt5.positions_get(ticket=int(ticket)) or []
        pos = next(iter(pos_list), None)
        if pos is None:
            raise MT5RequestError(f"position {ticket} not found")
        direction = "sell" if int(pos.type) == int(mt5.ORDER_TYPE_BUY) else "buy"
        tick = self.symbol_tick(pos.symbol)
        price = tick.bid if direction == "sell" else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": float(pos.volume),
            "type": getattr(mt5, f"ORDER_TYPE_{direction.upper()}"),
            "position": int(ticket),
            "price": price,
            "deviation": 20,
            "magic": 99001,
            "comment": "close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._type_filling(pos.symbol),
        }
        result = mt5.order_send(request)
        if result is None:
            raise MT5RequestError(f"order_send returned None: {mt5.last_error()}")
        retcode = int(result.retcode)
        if retcode != int(mt5.TRADE_RETCODE_DONE):
            raise MT5RequestError(
                f"close rejected retcode={retcode} comment={result.comment!r}"
            )
        return OrderResult(
            retcode=retcode,
            order_id=int(result.order),
            price=float(result.price),
            comment=str(result.comment),
        )

    def modify_position(self, ticket: int, sl: float, tp: float) -> bool:
        self._require_connected()
        mt5 = _get_mt5()
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "sl": float(sl),
            "tp": float(tp),
        }
        result = mt5.order_send(request)
        if result is None:
            raise MT5RequestError(f"order_send returned None: {mt5.last_error()}")
        return int(result.retcode) == int(mt5.TRADE_RETCODE_DONE)

    # --- helpers ---------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected:
            raise MT5ConnectionError("MT5 not connected; call connect() first")


__all__ = [
    "AccountInfo",
    "LiveAccountError",
    "MT5Client",
    "MT5ConnectionError",
    "MT5Error",
    "MT5LoginError",
    "MT5NotInstalledError",
    "MT5RequestError",
    "OrderResult",
    "Position",
    "Rates",
    "SymbolInfo",
    "Tick",
]