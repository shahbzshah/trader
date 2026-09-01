"""Tests for app/executor.py: re-validate, place, journal."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from app.brain import Decision
from app.executor import Executor
from app.journal import Journal


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeAccountInfo:
    balance: float = 10000.0
    equity: float = 10000.0
    margin: float = 0.0
    margin_free: float = 10000.0
    leverage: int = 100
    login: int = 12345
    name: str = "demo"
    server: str = "x"
    currency: str = "USD"
    trade_mode: str = "demo"


@dataclass
class FakeSymbolInfo:
    name: str = "EURUSD"
    digits: int = 5
    point: float = 0.0001
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    trade_contract_size: float = 100000.0


@dataclass
class FakeTick:
    bid: float = 1.1000
    ask: float = 1.1001
    spread: float = 0.0001
    time: int = 1
    symbol: str = "EURUSD"


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


@dataclass
class FakeOrderResult:
    retcode: int = 10009
    order: int = 0
    price: float = 1.1001
    comment: str = "ok"


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.connect_called = 0
        self._positions: list[FakePos] = []
        self._next_order_id = 99999
        self._raise_on_order: Exception | None = None
        self._account = FakeAccountInfo()

    def ensure_demo(self):
        if self._account.trade_mode != "demo":
            from app.mt5_client import LiveAccountError
            raise LiveAccountError("not demo")

    def account_info(self):
        return self._account

    def positions(self, symbol=None):
        if symbol is None:
            return list(self._positions)
        return [p for p in self._positions if p.symbol == symbol]

    def symbol_info(self, symbol):
        return FakeSymbolInfo(name=symbol)

    def symbol_tick(self, symbol):
        return FakeTick(symbol=symbol)

    def market_order(self, symbol, direction, lots, sl, tp, comment=""):
        if self._raise_on_order is not None:
            raise self._raise_on_order
        self._next_order_id += 1
        return FakeOrderResult(
            retcode=10009, order=self._next_order_id,
            price=1.1001, comment="ok",
        )

    def close_position(self, ticket):
        self._next_order_id += 1
        return FakeOrderResult(
            retcode=10009, order=self._next_order_id,
            price=1.1050, comment="closed",
        )

    def modify_position(self, ticket, sl, tp):
        return True

    def shutdown(self):
        pass


# ---------------------------------------------------------------------------
# Config / state fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeRisk:
    max_risk_pct_per_trade: float = 1.0
    max_daily_loss_pct: float = 3.0
    max_open_trades: int = 3
    max_trades_per_day: int = 5
    min_confidence: float = 0.6
    default_playbook: str = "breakout_retest"


@dataclass
class FakePaths:
    data_dir: str = "data"
    kill_switch_path: str = "data/KILL_SWITCH"


@dataclass
class FakeConfig:
    symbols: list = field(default_factory=lambda: ["EURUSD"])
    risk: FakeRisk = field(default_factory=FakeRisk)
    paths: FakePaths = field(default_factory=FakePaths)
    mode: str = "demo"


@pytest.fixture
def env(tmp_path: Path) -> tuple[FakeClient, Journal, FakeConfig]:
    cfg = FakeConfig()
    cfg.paths = FakePaths(
        data_dir=str(tmp_path), kill_switch_path=str(tmp_path / "KILL_SWITCH"),
    )
    j = Journal(Path(cfg.paths.data_dir) / "journal.db")
    c = FakeClient()
    return c, j, cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_executor_propose_approved_places_order(env):
    client, j, cfg = env
    ex = Executor(client, j, cfg)
    d = Decision(
        action="propose", symbol="EURUSD", direction="buy",
        price=1.1000, stop_loss=1.0950, take_profit=1.1100,
        risk_pct=0.5, confidence=0.8, playbook="breakout_retest",
        reasoning="ok",
    )
    res = ex.execute(d)
    assert res.status == "filled"
    assert res.ticket is not None
    # Decision + order rows present.
    decs = j.list_decisions(limit=10)
    assert any(x["approved"] for x in decs)
    orders = j.list_orders(limit=10)
    assert any(o["status"] == "filled" for o in orders)


def test_executor_propose_rejected_when_risk_blocks(env):
    client, j, cfg = env
    ex = Executor(client, j, cfg)
    d = Decision(
        action="propose", symbol="EURUSD", direction="buy",
        price=1.1000, stop_loss=1.0950, take_profit=1.1100,
        risk_pct=0.5, confidence=0.1, playbook="breakout_retest",
        reasoning="x",
    )
    res = ex.execute(d)
    assert res.status == "rejected"
    assert any("confidence_below_min" in r for r in res.reject_reasons)
    decs = j.list_decisions(limit=10)
    assert any(not x["approved"] for x in decs)
    orders = j.list_orders(limit=10)
    assert not any(o["status"] == "filled" for o in orders)


def test_executor_propose_blocked_on_live_account(env):
    client, j, cfg = env
    client._account.trade_mode = "real"
    ex = Executor(client, j, cfg)
    d = Decision(
        action="propose", symbol="EURUSD", direction="buy",
        price=1.1000, stop_loss=1.0950, take_profit=1.1100,
        risk_pct=0.5, confidence=0.8, reasoning="x",
    )
    res = ex.execute(d)
    assert res.status == "rejected"
    assert "not_demo_account" in res.reject_reasons


def test_executor_propose_blocked_by_kill_switch(env):
    client, j, cfg = env
    Path(cfg.paths.kill_switch_path).touch()
    ex = Executor(client, j, cfg)
    d = Decision(
        action="propose", symbol="EURUSD", direction="buy",
        price=1.1000, stop_loss=1.0950, take_profit=1.1100,
        risk_pct=0.5, confidence=0.8, reasoning="x",
    )
    res = ex.execute(d)
    assert res.status == "rejected"
    assert "kill_switch_engaged" in res.reject_reasons


def test_executor_propose_sizing_failure(env):
    client, j, cfg = env
    ex = Executor(client, j, cfg)
    # SL must be on the correct side of entry to pass check_decision.
    # Use a tiny SL distance (0.0001 = 1 pip) and very high risk
    # to overflow the max lot (> 100.0). Actually we need a
    # value that yields lots < vmin. 1% of 10000 = $100; 1 pip
    # at 10 USD/pip = 10 lots. 10 > 100.0? No, 10 < 100 -> OK.
    # The cleanest way: make the risk very small (0.0001%) so
    # sizing rounds down to below vmin (0.01).
    d = Decision(
        action="propose", symbol="EURUSD", direction="buy",
        price=1.1000, stop_loss=1.0999,  # 1 pip
        take_profit=1.1100, risk_pct=0.001, confidence=0.8,
        reasoning="x",
    )
    res = ex.execute(d)
    # With 1 pip SL and 0.001% risk on 10k equity:
    # $0.10 risk, 1 pip = $10/lot -> 0.01 lots -> exactly vmin.
    # Bump to 0 to make it < vmin.
    d.risk_pct = 0.0001
    res = ex.execute(d)
    assert res.status == "rejected"
    assert "position_size_unable" in res.reject_reasons


def test_executor_no_trade_journaled_only(env):
    client, j, cfg = env
    ex = Executor(client, j, cfg)
    d = Decision(action="no_trade", symbol="EURUSD", confidence=0.8)
    res = ex.execute(d)
    assert res.status == "skipped"
    decs = j.list_decisions(limit=10)
    assert any(x["action"] == "no_trade" and x["approved"] for x in decs)
    orders = j.list_orders(limit=10)
    assert orders == []


def test_executor_order_send_exception_recorded(env):
    client, j, cfg = env
    client._raise_on_order = RuntimeError("broker down")
    ex = Executor(client, j, cfg)
    d = Decision(
        action="propose", symbol="EURUSD", direction="buy",
        price=1.1000, stop_loss=1.0950, take_profit=1.1100,
        risk_pct=0.5, confidence=0.8, reasoning="x",
    )
    res = ex.execute(d)
    assert res.status == "rejected"
    orders = j.list_orders(limit=10)
    assert any(o["status"] == "rejected" and "broker down" in (o["error"] or "")
               for o in orders)


def test_executor_close_action(env):
    client, j, cfg = env
    client._positions = [FakePos(ticket=42, symbol="EURUSD")]
    ex = Executor(client, j, cfg)
    d = Decision(action="close", symbol="EURUSD", price=42.0, confidence=1.0)
    res = ex.execute(d)
    assert res.status == "closed"
    assert res.ticket is not None


def test_executor_modify_action(env):
    client, j, cfg = env
    ex = Executor(client, j, cfg)
    d = Decision(
        action="modify", symbol="EURUSD", price=42.0,
        stop_loss=1.0900, take_profit=1.1200, confidence=1.0,
    )
    res = ex.execute(d)
    assert res.status == "modified"


def test_executor_modify_missing_sl_or_tp(env):
    client, j, cfg = env
    ex = Executor(client, j, cfg)
    d = Decision(action="modify", symbol="EURUSD", price=42.0,
                  stop_loss=None, take_profit=1.12, confidence=1.0)
    res = ex.execute(d)
    assert res.status == "rejected"
    assert "missing_sl_or_tp" in res.reject_reasons


def test_executor_max_open_trades_blocks(env):
    client, j, cfg = env
    # 3 open positions (== max_open_trades)
    client._positions = [
        FakePos(ticket=i, symbol="EURUSD") for i in range(1, 4)
    ]
    ex = Executor(client, j, cfg)
    d = Decision(
        action="propose", symbol="EURUSD", direction="buy",
        price=1.1000, stop_loss=1.0950, take_profit=1.1100,
        risk_pct=0.5, confidence=0.8, reasoning="x",
    )
    res = ex.execute(d)
    assert res.status == "rejected"
    assert any("max_open_trades_reached" in r for r in res.reject_reasons)


def test_executor_clamps_risk_pct_to_config(env):
    client, j, cfg = env
    ex = Executor(client, j, cfg)
    d = Decision(
        action="propose", symbol="EURUSD", direction="buy",
        price=1.1000, stop_loss=1.0950, take_profit=1.1100,
        risk_pct=10.0,  # LLM wants 10%
        confidence=0.8, reasoning="x",
    )
    res = ex.execute(d)
    assert res.status == "rejected"
    assert any("risk_pct_above_max" in r for r in res.reject_reasons)