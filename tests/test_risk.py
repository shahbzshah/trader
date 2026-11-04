"""Tests for app/risk.py: position sizing + daily P/L accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.risk import (
    AccountState,
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


