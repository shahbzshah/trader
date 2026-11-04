"""Pure risk functions: position sizing + daily P/L accounting.

No I/O. All inputs passed in. 100% unit-testable.

The strategy was removed on 2026-11-04 (Megah's order), so the
decision veto gate (``check_decision``) is gone with it. What remains
is the plumbing the next strategy will plug into: position sizing
(``position_size``) and realised-P/L accounting (``daily_realized_pl``).
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass
class AccountState:
    """Minimal account snapshot for risk checks."""

    balance: float
    equity: float
    starting_balance_today: float
    trade_mode: str  # "demo" or anything else
    kill_switch_engaged: bool


def _now_utc() -> _dt.datetime:
    """Return the current UTC time. Indirected so tests can monkeypatch it."""
    return _dt.datetime.now(_dt.timezone.utc)


def _round_down_to_step(value: float, step: float) -> float:
    """Round value DOWN to the nearest multiple of step.

    Step must be positive. Returns 0.0 if value is below one step.
    A small epsilon avoids the floating-point edge case where
    ``math.floor(0.20 / 0.01)`` evaluates to 19 instead of 20.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    if value <= 0:
        return 0.0
    eps = step * 1e-9
    return math.floor((value + eps) / step) * step


def position_size(
    equity: float,
    risk_pct: float,
    entry: float,
    stop_loss: float,
    symbol_info: Any,
) -> float:
    """Compute lot size given a per-trade risk percentage.

    Args:
        equity: account equity in account currency.
        risk_pct: percentage of equity to risk (e.g. 1.0 for 1%).
        entry: planned entry price.
        stop_loss: planned stop-loss price (must be on the correct side of
            entry; this function does NOT enforce that — the caller does).
        symbol_info: an object with attributes ``volume_min``,
            ``volume_max``, ``volume_step``, ``trade_contract_size`` and
            ``point``. The pip value is computed as
            ``point * contract_size`` per 1.0 lot.

    Returns:
        Lot size rounded DOWN to ``symbol_info.volume_step``, clipped to
        ``[volume_min, volume_max]``. Returns 0.0 if the risk is too small
        to clear the minimum lot.
    """
    if equity <= 0:
        return 0.0
    if risk_pct <= 0:
        return 0.0
    if entry <= 0 or stop_loss <= 0:
        return 0.0

    point = float(getattr(symbol_info, "point", 0.0))
    contract = float(getattr(symbol_info, "trade_contract_size", 1.0) or 1.0)
    if point <= 0 or contract <= 0:
        return 0.0

    sl_distance = abs(float(entry) - float(stop_loss))
    if sl_distance <= 0:
        return 0.0

    risk_amount = float(equity) * (float(risk_pct) / 100.0)
    pip_value_per_lot = point * contract
    if pip_value_per_lot <= 0:
        return 0.0

    raw_lots = risk_amount / (sl_distance / point * pip_value_per_lot)
    # Equivalently: raw_lots = risk_amount / (sl_distance * contract).

    step = float(getattr(symbol_info, "volume_step", 0.01))
    lots = _round_down_to_step(raw_lots, step)

    vmin = float(getattr(symbol_info, "volume_min", step))
    vmax = float(getattr(symbol_info, "volume_max", 100.0))
    if vmin <= 0:
        vmin = step
    lots = min(max(lots, 0.0), vmax)
    if lots < vmin:
        # Below minimum lot — caller should treat as "can't size this trade".
        return 0.0
    return lots


def daily_realized_pl(journal_rows: Iterable[Mapping]) -> float:
    """Sum realized P/L for the current UTC day from journal rows.

    Each row is expected to be a mapping (or sqlite3.Row) that contains
    a ``kind`` (e.g. "order_closed") and ``profit`` field, plus a
    ``ts``/``time`` field. Rows whose ``ts`` is on the current UTC day
    are summed.
    """
    total = 0.0
    today = _now_utc().date()
    for row in journal_rows:
        if hasattr(row, "__getitem__"):
            kind = row.get("kind", "") if hasattr(row, "get") else row["kind"]
            ts_raw = row.get("ts") if hasattr(row, "get") else row["ts"]
            profit_raw = row.get("profit") if hasattr(row, "get") else row["profit"]
        else:
            kind = getattr(row, "kind", "")
            ts_raw = getattr(row, "ts", None)
            profit_raw = getattr(row, "profit", 0.0)
        if kind not in ("order_closed", "trade_closed", "order_filled"):
            continue
        if ts_raw is None:
            continue
        try:
            ts = _dt.datetime.fromtimestamp(float(ts_raw), tz=_dt.timezone.utc)
        except (TypeError, ValueError):
            continue
        if ts.date() != today:
            continue
        try:
            total += float(profit_raw)
        except (TypeError, ValueError):
            continue
    return total


__all__ = [
    "AccountState",
    "daily_realized_pl",
    "position_size",
]