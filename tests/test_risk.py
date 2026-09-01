"""Tests for app/risk.py: position sizing + veto checks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.risk import (
    AccountState,
    RiskVerdict,
    check_decision,
    daily_realized_pl,
    position_size,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class SymInfo:
    point: float = 0.0001
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    trade_contract_size: float = 100000.0
    digits: int = 5


@dataclass
class Decision:
    action: str
    symbol: str = "EURUSD"
    direction: str = "buy"
    entry_type: str = "market"
    price: float | None = 1.1000
    stop_loss: float | None = 1.0950
    take_profit: float | None = 1.1100
    risk_pct: float = 1.0
    confidence: float = 0.8
    playbook: str = "breakout_retest"
    reasoning: str = "test"


@dataclass
class Cfg:
    symbols: list = field(default_factory=lambda: ["EURUSD", "GBPUSD"])
    risk: Any = None


@dataclass
class RiskCfg:
    max_risk_pct_per_trade: float = 1.0
    max_daily_loss_pct: float = 3.0
    max_open_trades: int = 3
    max_trades_per_day: int = 5
    min_confidence: float = 0.6
    default_playbook: str = "breakout_retest"


def _default_cfg() -> Cfg:
    return Cfg(risk=RiskCfg())


def _open_positions(*, n: int = 0, symbols: list[str] | None = None) -> list:
    out = []
    for i in range(n):
        out.append(
            type("P", (), {
                "symbol": (symbols or ["EURUSD"])[i % len(symbols or ["EURUSD"])],
                "ticket": i + 1,
                "direction": "buy",
                "volume": 0.10,
            })()
        )
    return out


# ---------------------------------------------------------------------------
# position_size tests
# ---------------------------------------------------------------------------


def test_position_size_basic_1pct_on_eurusd():
    sym = SymInfo()
    lots = position_size(
        equity=10000.0,
        risk_pct=1.0,         # 1% of 10,000 = $100
        entry=1.1000,
        stop_loss=1.0950,     # 50 pips distance (in 5-digit pricing = 0.0050)
        symbol_info=sym,
    )
    # pip_value_per_lot = 0.0001 * 100000 = 10 USD per pip
    # 50 pips = 0.0050 -> 50 pips * 10 USD/pip/lot = 500 USD risk per lot
    # $100 / $500 = 0.20 lots
    assert math.isclose(lots, 0.20, abs_tol=1e-6)


def test_position_size_rounds_down_to_step():
    sym = SymInfo(volume_step=0.05, volume_max=200.0)
    lots = position_size(
        equity=10000.0, risk_pct=1.0, entry=1.100, stop_loss=1.050,
        symbol_info=sym,
    )
    # 500 pips (0.0500) at 0.0001/contract*100000 = 10 USD/pip per lot.
    # 500 pips * 10 = 5000 USD risk per lot. $100 risk = 0.02 lots,
    # floored to 0.05 step = 0.00 lots -> below min 0.10 -> 0.
    # Try smaller step: with volume_step=0.05, 500 pips, $100 risk:
    # actually compute: 100 / (500 * 10) = 0.02 lots -> < vmin -> 0.
    assert lots == 0.0

    # Now try a setup where the result IS a clean 0.05-step multiple.
    sym2 = SymInfo(volume_step=0.05, volume_max=200.0)
    lots2 = position_size(
        equity=100000.0, risk_pct=1.0, entry=1.100, stop_loss=1.095,
        symbol_info=sym2,
    )
    # 50 pips, 10 USD/pip/lot, $1000 risk -> 2.00 lots -> multiple of 0.05.
    assert math.isclose(lots2, 2.00, abs_tol=1e-6)
    assert math.isclose(lots2 / 0.05, round(lots2 / 0.05))


def test_position_size_respects_volume_min():
    sym = SymInfo(volume_min=0.10, volume_step=0.01)
    lots = position_size(
        equity=100.0,  # tiny equity
        risk_pct=0.01,  # 1 cent of risk
        entry=1.1, stop_loss=1.095,
        symbol_info=sym,
    )
    # 0.01 lots, but min is 0.10 -> return 0 (can't size this trade)
    assert lots == 0.0


def test_position_size_respects_volume_max():
    sym = SymInfo(volume_max=1.0, volume_step=0.01)
    lots = position_size(
        equity=1_000_000.0, risk_pct=10.0, entry=1.1, stop_loss=1.095,
        symbol_info=sym,
    )
    assert lots <= 1.0


def test_position_size_zero_equity_or_risk():
    sym = SymInfo()
    assert position_size(0.0, 1.0, 1.1, 1.09, sym) == 0.0
    assert position_size(10000.0, 0.0, 1.1, 1.09, sym) == 0.0
    assert position_size(10000.0, -1.0, 1.1, 1.09, sym) == 0.0


def test_position_size_invalid_entry_or_sl():
    sym = SymInfo()
    assert position_size(10000.0, 1.0, 0.0, 1.09, sym) == 0.0
    assert position_size(10000.0, 1.0, 1.1, 0.0, sym) == 0.0
    assert position_size(10000.0, 1.0, 1.1, 1.1, sym) == 0.0  # SL == entry


def test_position_size_invalid_symbol_info():
    sym = SymInfo(point=0.0)
    assert position_size(10000.0, 1.0, 1.1, 1.09, sym) == 0.0


# ---------------------------------------------------------------------------
# daily_realized_pl
# ---------------------------------------------------------------------------


def _row(kind: str, ts: float, profit: float) -> dict:
    return {"kind": kind, "ts": ts, "profit": profit}


def test_daily_realized_pl_filters_today_utc(monkeypatch):
    import datetime as _dt
    import app.risk as risk_mod
    fixed = _dt.datetime(2026, 8, 31, 12, 0, 0, tzinfo=_dt.timezone.utc)
    monkeypatch.setattr(risk_mod, "_now_utc", lambda: fixed)
    rows = [
        _row("order_closed", fixed.timestamp() - 3600, 50.0),
        _row("order_closed", fixed.timestamp() - 86400 * 2, 999.0),
        _row("noise", fixed.timestamp(), 1000.0),
    ]
    assert daily_realized_pl(rows) == 50.0


def test_daily_realized_pl_handles_bad_rows(monkeypatch):
    import datetime as _dt
    import app.risk as risk_mod
    fixed = _dt.datetime(2026, 8, 31, 12, 0, 0, tzinfo=_dt.timezone.utc)
    monkeypatch.setattr(risk_mod, "_now_utc", lambda: fixed)
    rows = [
        _row("order_closed", fixed.timestamp(), "not-a-number"),
        {"kind": "order_closed"},
        _row("order_closed", fixed.timestamp(), 10.0),
    ]
    assert daily_realized_pl(rows) == 10.0


# ---------------------------------------------------------------------------
# check_decision tests
# ---------------------------------------------------------------------------


def _account(trade_mode: str = "demo", kill: bool = False,
              starting_balance: float = 10000.0) -> AccountState:
    return AccountState(
        balance=10000.0, equity=10000.0,
        starting_balance_today=starting_balance,
        trade_mode=trade_mode, kill_switch_engaged=kill,
    )


def test_check_decision_kill_switch_blocks():
    d = Decision(action="no_trade")
    v = check_decision(d, _account(kill=True), [], [], _default_cfg())
    assert v.approved is False
    assert "kill_switch_engaged" in v.reasons


def test_check_decision_live_account_blocks():
    d = Decision(action="propose")
    v = check_decision(d, _account(trade_mode="real"), [], [], _default_cfg())
    assert v.approved is False
    assert "not_demo_account" in v.reasons


def test_check_decision_invalid_action_blocks():
    d = Decision(action="sideways")
    v = check_decision(d, _account(), [], [], _default_cfg())
    assert v.approved is False
    assert any("invalid_action" in r for r in v.reasons)


def test_check_decision_no_trade_approved_in_demo():
    d = Decision(action="no_trade")
    v = check_decision(d, _account(trade_mode="demo"), [], [], _default_cfg())
    assert v.approved is True


def test_check_decision_no_trade_blocked_on_live():
    d = Decision(action="no_trade")
    v = check_decision(d, _account(trade_mode="real"), [], [], _default_cfg())
    assert v.approved is False
    assert "not_demo_account" in v.reasons


def test_check_decision_close_and_modify_approved():
    cfg = _default_cfg()
    for action in ("close", "modify"):
        d = Decision(action=action)
        v = check_decision(d, _account(), _open_positions(n=cfg.risk.max_open_trades),
                           [], cfg)
        assert v.approved is True, action


def test_check_decision_symbol_not_in_watchlist():
    d = Decision(action="propose", symbol="AUDJPY")
    v = check_decision(d, _account(), [], [], _default_cfg())
    assert v.approved is False
    assert any("symbol_not_in_watchlist" in r for r in v.reasons)


def test_check_decision_sl_wrong_side_for_buy():
    d = Decision(action="propose", direction="buy", price=1.1000, stop_loss=1.1050)
    v = check_decision(d, _account(), [], [], _default_cfg())
    assert v.approved is False
    assert "sl_wrong_side_for_buy" in v.reasons


def test_check_decision_sl_wrong_side_for_sell():
    d = Decision(action="propose", direction="sell", price=1.1000, stop_loss=1.0950)
    v = check_decision(d, _account(), [], [], _default_cfg())
    assert v.approved is False
    assert "sl_wrong_side_for_sell" in v.reasons


def test_check_decision_missing_sl_or_entry():
    d = Decision(action="propose", price=None, stop_loss=None)
    v = check_decision(d, _account(), [], [], _default_cfg())
    assert v.approved is False
    assert "missing_entry_or_sl" in v.reasons

    d = Decision(action="propose", price=1.1, stop_loss=None, take_profit=None)
    v = check_decision(d, _account(), [], [], _default_cfg())
    assert v.approved is False
    assert "missing_tp" in v.reasons


def test_check_decision_confidence_below_min():
    d = Decision(action="propose", confidence=0.50)
    v = check_decision(d, _account(), [], [], _default_cfg())
    assert v.approved is False
    assert any("confidence_below_min" in r for r in v.reasons)


def test_check_decision_risk_pct_above_max():
    d = Decision(action="propose", risk_pct=5.0)
    cfg = _default_cfg()  # max 1.0
    v = check_decision(d, _account(), [], [], cfg)
    assert v.approved is False
    assert any("risk_pct_above_max" in r for r in v.reasons)


def test_check_decision_max_open_trades_reached():
    cfg = _default_cfg()
    d = Decision(action="propose")
    v = check_decision(d, _account(),
                       _open_positions(n=cfg.risk.max_open_trades), [], cfg)
    assert v.approved is False
    assert any("max_open_trades_reached" in r for r in v.reasons)


def test_check_decision_max_trades_per_day_reached(monkeypatch):
    cfg = _default_cfg()
    d = Decision(action="propose")
    import datetime as _dt
    import app.risk as risk_mod
    fixed = _dt.datetime(2026, 8, 31, 12, 0, 0, tzinfo=_dt.timezone.utc)
    monkeypatch.setattr(risk_mod, "_now_utc", lambda: fixed)
    rows = [
        {"kind": "order_placed", "ts": fixed.timestamp() - 3600, "profit": 0.0}
        for _ in range(cfg.risk.max_trades_per_day)
    ]
    v = check_decision(d, _account(), [], rows, cfg)
    assert v.approved is False
    assert any("max_trades_per_day_reached" in r for r in v.reasons)


def test_check_decision_daily_loss_cap_exceeded(monkeypatch):
    cfg = _default_cfg()
    d = Decision(action="propose", risk_pct=1.0)
    import datetime as _dt
    import app.risk as risk_mod
    fixed = _dt.datetime(2026, 8, 31, 12, 0, 0, tzinfo=_dt.timezone.utc)
    monkeypatch.setattr(risk_mod, "_now_utc", lambda: fixed)
    # -4% realised today (cap is -3%). Even with this trade's 1% risk
    # we are already over.
    rows = [
        {"kind": "order_closed", "ts": fixed.timestamp() - 3600,
         "profit": -400.0},  # 4% of 10,000
    ]
    v = check_decision(
        d,
        _account(starting_balance=10000.0),
        [], rows, cfg,
    )
    assert v.approved is False
    assert any("daily_loss_cap_exceeded" in r for r in v.reasons)


def test_check_decision_approved_when_all_pass():
    cfg = _default_cfg()
    d = Decision(action="propose", confidence=0.8, risk_pct=0.5)
    v = check_decision(d, _account(), _open_positions(n=1), [], cfg)
    assert v.approved is True
    assert v.reasons == []


def test_check_decision_invalid_direction_blocks():
    d = Decision(action="propose", direction="sideways")
    v = check_decision(d, _account(), [], [], _default_cfg())
    assert v.approved is False
    assert any("invalid_direction" in r for r in v.reasons)