"""Tests for app/manage.py: breakeven stop moves."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from app.manage import breakeven_moves


@dataclass
class Position:
    ticket: int = 1
    symbol: str = "XAUUSD"
    direction: str = "buy"
    volume: float = 0.01
    price_open: float = 2000.00
    price_current: float = 2000.00
    sl: float = 1995.00      # 5.00 risk distance
    tp: float = 2010.00
    profit: float = 0.0
    comment: str = ""


def _cfg(**kw):
    risk = dict(breakeven_enabled=True, breakeven_at_r=1.0)
    risk.update(kw.pop("risk", {}))
    return SimpleNamespace(
        symbols=["XAUUSD"],
        risk=SimpleNamespace(**risk),
        **kw,
    )


def test_buy_at_1r_moves_sl_to_entry():
    # +6.00 on 5.00 risk = 1.2R -> qualifies.
    p = Position(price_current=2006.00)
    moves = breakeven_moves([p], _cfg())
    assert len(moves) == 1
    m = moves[0]
    assert m.action == "modify"
    assert m.price == 1.0            # ticket carried in .price
    assert m.stop_loss == 2000.00   # -> entry
    assert m.take_profit == 2010.00
    assert m.playbook == "breakeven"


def test_buy_below_1r_no_move():
    p = Position(price_current=2002.00)  # 0.4R
    assert breakeven_moves([p], _cfg()) == []


def test_buy_already_at_breakeven_no_move():
    p = Position(price_current=2006.00, sl=2000.00)
    assert breakeven_moves([p], _cfg()) == []


def test_sell_side_mirror():
    # Short from 2000, SL 2005 (5.00 risk), now 1994 -> 1.2R.
    p = Position(direction="sell", price_current=1994.00, sl=2005.00,
                 tp=1990.00)
    moves = breakeven_moves([p], _cfg())
    assert len(moves) == 1
    assert moves[0].stop_loss == 2000.00


def test_sell_below_1r_no_move():
    p = Position(direction="sell", price_current=1998.00, sl=2005.00)
    assert breakeven_moves([p], _cfg()) == []


def test_disabled_via_config():
    p = Position(price_current=2006.00)
    assert breakeven_moves([p], _cfg(risk={"breakeven_enabled": False})) == []


def test_custom_threshold():
    p = Position(price_current=2006.00)  # 1.2R
    assert breakeven_moves([p], _cfg(risk={"breakeven_at_r": 1.5})) == []
    p2 = Position(price_current=2008.00)  # 1.6R
    assert len(breakeven_moves([p2], _cfg(risk={"breakeven_at_r": 1.5}))) == 1


def test_non_watchlist_symbol_skipped():
    p = Position(symbol="EURUSD", price_current=2006.00)
    assert breakeven_moves([p], _cfg()) == []


def test_invalid_position_skipped():
    assert breakeven_moves([Position(sl=0.0)], _cfg()) == []
    assert breakeven_moves([Position(ticket=0)], _cfg()) == []
    assert breakeven_moves([Position(direction="hold")], _cfg()) == []


def test_mixed_positions_only_qualifiers():
    ps = [
        Position(ticket=1, price_current=2006.00),   # qualifies
        Position(ticket=2, price_current=2001.00),   # 0.2R, no
        Position(ticket=3, direction="sell", price_current=1994.00,
                 sl=2005.00, tp=1990.00),            # qualifies
    ]
    moves = breakeven_moves(ps, _cfg())
    assert sorted(m.price for m in moves) == [1.0, 3.0]
