"""Fake MetaTrader5 module used for unit tests.

This fake emulates the subset of the MetaTrader5 API surface that
``app.mt5_client.py`` calls. Tests can install it via
``app.mt5_client._mt5 = fake`` before constructing an MT5Client.

All call sequences are recorded on ``fake.calls`` so tests can assert
the wrapper invoked the API correctly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# --- enums / constants (mirror MetaTrader5) ---------------------------

# Account trade_mode (ACCOUNT_TRADE_MODE_* family)
ACCOUNT_TRADE_MODE_DEMO = 0
ACCOUNT_TRADE_MODE_REAL = 1
ACCOUNT_TRADE_MODE_CONTEST = 2

# Margin mode constants (kept for compatibility with the broader MT5 API)
ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 1
ACCOUNT_MARGIN_MODE_RETAIL_NETTING = 2
ACCOUNT_MARGIN_MODE_EXCHANGE = 3

TIMEFRAME_M1 = 1
TIMEFRAME_M5 = 5
TIMEFRAME_M15 = 15
TIMEFRAME_M30 = 30
TIMEFRAME_H1 = 16385
TIMEFRAME_H4 = 16388
TIMEFRAME_D1 = 16408
TIMEFRAME_W1 = 32769
TIMEFRAME_MN1 = 49153

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
ORDER_TYPE_BUY_STOP = 4
ORDER_TYPE_SELL_STOP = 5

TRADE_ACTION_DEAL = 1
TRADE_ACTION_SLTP = 2
TRADE_ACTION_PENDING = 5

ORDER_TIME_GTC = 0
ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2

TRADE_RETCODE_DONE = 10009


# --- data classes used by the fake -------------------------------------


@dataclass
class FakeAccountInfo:
    login: int = 100200
    name: str = "Demo Trader"
    server: str = "DemoBroker-Server"
    currency: str = "USD"
    trade_mode: int = ACCOUNT_TRADE_MODE_DEMO
    balance: float = 10000.0
    equity: float = 10000.0
    margin: float = 0.0
    margin_free: float = 10000.0
    leverage: int = 100


@dataclass
class FakePosition:
    ticket: int = 0
    symbol: str = "EURUSD"
    type: int = ORDER_TYPE_BUY
    volume: float = 0.10
    price_open: float = 1.1000
    price_current: float = 1.1050
    sl: float = 1.0950
    tp: float = 1.1100
    profit: float = 50.0
    comment: str = "test"


@dataclass
class FakeSymbolInfo:
    name: str = "EURUSD"
    digits: int = 5
    point: float = 0.00001
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    trade_contract_size: float = 100000.0
    filling_mode: int = 0  # bitmask of allowed fill modes (0 => unknown)


@dataclass
class FakeTick:
    bid: float = 1.1000
    ask: float = 1.10005
    time: int = 0


@dataclass
class FakeRatesRow:
    time: int = 0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    tick_volume: int = 0
    spread: int = 0
    real_volume: int = 0

    def __getitem__(self, key: str):
        return getattr(self, key)


@dataclass
class FakeOrderResult:
    retcode: int = TRADE_RETCODE_DONE
    order: int = 0
    price: float = 0.0
    comment: str = "ok"


# --- the fake module --------------------------------------------------


class FakeMT5:
    """Replacement for the MetaTrader5 module.

    Mirrors the constants and methods exercised by ``app.mt5_client.py``.
    Constants are exposed both as class attributes (so ``fake.ORDER_TYPE_BUY``
    works) and as instance attributes for compatibility.
    """

    # Constants (class-level so getattr(fake, 'NAME') works)
    ACCOUNT_TRADE_MODE_DEMO = ACCOUNT_TRADE_MODE_DEMO
    ACCOUNT_TRADE_MODE_REAL = ACCOUNT_TRADE_MODE_REAL
    ACCOUNT_TRADE_MODE_CONTEST = ACCOUNT_TRADE_MODE_CONTEST

    ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = ACCOUNT_MARGIN_MODE_RETAIL_HEDGING
    ACCOUNT_MARGIN_MODE_RETAIL_NETTING = ACCOUNT_MARGIN_MODE_RETAIL_NETTING
    ACCOUNT_MARGIN_MODE_EXCHANGE = ACCOUNT_MARGIN_MODE_EXCHANGE

    TIMEFRAME_M1 = TIMEFRAME_M1
    TIMEFRAME_M5 = TIMEFRAME_M5
    TIMEFRAME_M15 = TIMEFRAME_M15
    TIMEFRAME_M30 = TIMEFRAME_M30
    TIMEFRAME_H1 = TIMEFRAME_H1
    TIMEFRAME_H4 = TIMEFRAME_H4
    TIMEFRAME_D1 = TIMEFRAME_D1
    TIMEFRAME_W1 = TIMEFRAME_W1
    TIMEFRAME_MN1 = TIMEFRAME_MN1

    ORDER_TYPE_BUY = ORDER_TYPE_BUY
    ORDER_TYPE_SELL = ORDER_TYPE_SELL
    ORDER_TYPE_BUY_LIMIT = ORDER_TYPE_BUY_LIMIT
    ORDER_TYPE_SELL_LIMIT = ORDER_TYPE_SELL_LIMIT
    ORDER_TYPE_BUY_STOP = ORDER_TYPE_BUY_STOP
    ORDER_TYPE_SELL_STOP = ORDER_TYPE_SELL_STOP

    TRADE_ACTION_DEAL = TRADE_ACTION_DEAL
    TRADE_ACTION_SLTP = TRADE_ACTION_SLTP
    TRADE_ACTION_PENDING = TRADE_ACTION_PENDING

    ORDER_TIME_GTC = ORDER_TIME_GTC
    ORDER_FILLING_FOK = ORDER_FILLING_FOK
    ORDER_FILLING_IOC = ORDER_FILLING_IOC
    ORDER_FILLING_RETURN = ORDER_FILLING_RETURN

    # Symbol filling_mode bitmask bits (SYMBOL_FILLING_MODE_*).
    SYMBOL_FILLING_MODE_FOK = 1
    SYMBOL_FILLING_MODE_IOC = 2
    SYMBOL_FILLING_MODE_RETURN = 4

    TRADE_RETCODE_DONE = TRADE_RETCODE_DONE

    def __init__(
        self,
        initialize_ok: bool = True,
        login_ok: bool = True,
        trade_mode: int = ACCOUNT_TRADE_MODE_DEMO,
        account: FakeAccountInfo | None = None,
    ) -> None:
        self.initialize_ok = initialize_ok
        self.login_ok = login_ok
        self.trade_mode = trade_mode
        self.account = account or FakeAccountInfo()
        self.account.trade_mode = trade_mode
        self.positions_data: list[FakePosition] = []
        self.symbol_data: dict[str, FakeSymbolInfo] = {}
        self.tick_data: dict[str, FakeTick] = {}
        self.rates_data: list[FakeRatesRow] = []
        self.rates_by_tf: dict[int, list[FakeRatesRow]] = {}
        self.next_order_id = 12345
        self.initialize_fail_message = "fake init fail"
        self.login_fail_message = "fake login fail"
        self.order_fail_retcode = TRADE_RETCODE_DONE
        self.last_error_value: tuple = (0, "no error")
        self.calls: list[tuple] = []
        self.symbol_select_ok = True
        self.symbols_get_result: list = []

    # --- API surface --------------------------------------------------

    def last_error(self):
        self.calls.append(("last_error",))
        return self.last_error_value

    def initialize(self, path: str = ""):
        self.calls.append(("initialize", path))
        return self.initialize_ok

    def shutdown(self):
        self.calls.append(("shutdown",))

    def login(self, login: int, password: str, server: str):
        self.calls.append(("login", login, password, server))
        if self.login_ok:
            self.account.login = login
            self.account.server = server
            return True
        self.last_error_value = (1, self.login_fail_message)
        return False

    def account_info(self):
        self.calls.append(("account_info",))
        self.account.trade_mode = self.trade_mode
        return self.account

    def positions_get(self, symbol: str = "", ticket: int = 0):
        self.calls.append(("positions_get", symbol, ticket))
        if ticket:
            return [p for p in self.positions_data if p.ticket == ticket]
        if symbol:
            return [p for p in self.positions_data if p.symbol == symbol]
        return list(self.positions_data)

    def symbol_info(self, symbol: str):
        self.calls.append(("symbol_info", symbol))
        return self.symbol_data.get(symbol)

    def symbol_select(self, symbol: str, enable: bool = True):
        self.calls.append(("symbol_select", symbol, enable))
        return self.symbol_select_ok

    def symbols_get(self, *args, **kwargs):
        self.calls.append(("symbols_get", args, kwargs))
        if self.symbols_get_result:
            return list(self.symbols_get_result)
        return list(self.symbol_data.values())

    def symbol_info_tick(self, symbol: str):
        self.calls.append(("symbol_info_tick", symbol))
        return self.tick_data.get(symbol)

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start: int, count: int):
        self.calls.append(("copy_rates_from_pos", symbol, timeframe, start, count))
        if timeframe in self.rates_by_tf:
            return list(self.rates_by_tf[timeframe])
        return list(self.rates_data)

    def order_send(self, request: dict):
        self.calls.append(("order_send", dict(request)))
        if self.order_fail_retcode != TRADE_RETCODE_DONE:
            return FakeOrderResult(
                retcode=self.order_fail_retcode,
                order=0,
                price=0.0,
                comment="rejected",
            )
        self.next_order_id += 1
        price = float(request.get("price", 0.0))
        return FakeOrderResult(
            retcode=TRADE_RETCODE_DONE,
            order=self.next_order_id,
            price=price,
            comment="ok",
        )


def make_fake(
    trade_mode: int = ACCOUNT_TRADE_MODE_DEMO,
    account: FakeAccountInfo | None = None,
) -> FakeMT5:
    """Build a default fake MT5 module."""
    fake = FakeMT5(trade_mode=trade_mode, account=account or FakeAccountInfo())
    fake.symbol_data["EURUSD"] = FakeSymbolInfo(name="EURUSD")
    fake.tick_data["EURUSD"] = FakeTick(bid=1.1000, ask=1.10010, time=int(time.time()))
    return fake