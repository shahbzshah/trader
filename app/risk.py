"""Pure risk functions: position sizing + veto checks.

No I/O. All inputs passed in. 100% unit-testable.

This module is the ONLY authority on position sizing and limits. The LLM
proposes; code disposes. Any decision failing a check is rejected and
journaled with the rejection reasons.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass
class RiskVerdict:
    approved: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"approved": self.approved, "reasons": list(self.reasons)}


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


def _count_open_trades(open_positions: Iterable[Any], symbol: str | None = None) -> int:
    if symbol is None:
        return sum(1 for _ in open_positions)
    return sum(1 for p in open_positions if getattr(p, "symbol", "") == symbol)


def _count_trades_today(journal_rows: Iterable[Mapping]) -> int:
    """Count order rows placed today (UTC).

    Mirrors daily_realized_pl's day filter but counts rather than sums.
    """
    today = _now_utc().date()
    count = 0
    for row in journal_rows:
        if hasattr(row, "__getitem__"):
            kind = row.get("kind", "") if hasattr(row, "get") else row["kind"]
            ts_raw = row.get("ts") if hasattr(row, "get") else row["ts"]
        else:
            kind = getattr(row, "kind", "")
            ts_raw = getattr(row, "ts", None)
        if kind not in ("order_placed", "order_filled", "order_submitted"):
            continue
        if ts_raw is None:
            continue
        try:
            ts = _dt.datetime.fromtimestamp(float(ts_raw), tz=_dt.timezone.utc)
        except (TypeError, ValueError):
            continue
        if ts.date() == today:
            count += 1
    return count


def check_decision(
    decision: Any,
    account_state: AccountState,
    open_positions: Iterable[Any],
    journal: Iterable[Mapping],
    config: Any,
) -> RiskVerdict:
    """Veto gate for an LLM decision. Returns approved=True only if all
    checks pass.

    Checks performed (in order):
      1. Kill switch must be off.
      2. Account must be in demo mode.
      3. ``decision.action`` must be one of: "no_trade", "propose", "close",
         "modify". (This module is also the gate for action type; the
         executor enforces action-specific rules in M4.)
      4. For "propose" / "modify": symbol must be on the watchlist; SL must
         be on the correct side of (proposed) entry; confidence must be
         >= ``config.risk.min_confidence``; ``risk_pct`` is clamped to the
         config maximum.
      5. Open-trade cap.
      6. Trades-today cap.
      7. Daily loss cap: realised P/L + (this trade's projected risk on
         equity) must not exceed ``config.risk.max_daily_loss_pct``.

    A ``"no_trade"`` decision is always approved (it carries no risk).
    """
    reasons: list[str] = []
    if account_state.kill_switch_engaged:
        reasons.append("kill_switch_engaged")
        return RiskVerdict(False, reasons)

    if account_state.trade_mode != "demo":
        reasons.append("not_demo_account")
        return RiskVerdict(False, reasons)

    action = getattr(decision, "action", None)
    if action not in ("no_trade", "propose", "close", "modify"):
        reasons.append(f"invalid_action:{action!r}")
        return RiskVerdict(False, reasons)

    if action == "no_trade":
        return RiskVerdict(True, [])

    # close / modify do not open new exposure.
    if action in ("close", "modify"):
        return RiskVerdict(True, [])

    # --- "propose": enforce full set of rules ---
    symbol = getattr(decision, "symbol", "")
    watchlist = list(getattr(config, "symbols", []) or [])
    if symbol and watchlist and symbol.upper() not in {s.upper() for s in watchlist}:
        reasons.append(f"symbol_not_in_watchlist:{symbol}")

    entry = getattr(decision, "price", None)
    sl = getattr(decision, "stop_loss", None)
    tp = getattr(decision, "take_profit", None)
    direction = getattr(decision, "direction", None)
    if direction not in ("buy", "sell"):
        reasons.append(f"invalid_direction:{direction!r}")
    if entry is None or sl is None:
        reasons.append("missing_entry_or_sl")
    else:
        try:
            entry_f = float(entry)
            sl_f = float(sl)
            if direction == "buy" and sl_f >= entry_f:
                reasons.append("sl_wrong_side_for_buy")
            elif direction == "sell" and sl_f <= entry_f:
                reasons.append("sl_wrong_side_for_sell")
        except (TypeError, ValueError):
            reasons.append("non_numeric_entry_or_sl")
    if tp is None:
        reasons.append("missing_tp")

    # Confidence floor.
    min_conf = float(getattr(getattr(config, "risk", None), "min_confidence", 0.6))
    confidence = float(getattr(decision, "confidence", 0.0) or 0.0)
    if confidence < min_conf:
        reasons.append(f"confidence_below_min:{confidence:.2f}<{min_conf:.2f}")

    # risk_pct cap (use config if lower than LLM's).
    max_risk = float(getattr(getattr(config, "risk", None), "max_risk_pct_per_trade", 1.0))
    llm_risk = float(getattr(decision, "risk_pct", 0.0) or 0.0)
    if llm_risk > max_risk:
        reasons.append(f"risk_pct_above_max:{llm_risk:.2f}>{max_risk:.2f}")

    # Open-trade cap.
    open_count = _count_open_trades(open_positions)
    max_open = int(getattr(getattr(config, "risk", None), "max_open_trades", 3))
    if open_count >= max_open:
        reasons.append(f"max_open_trades_reached:{open_count}>={max_open}")

    # Trades-today cap.
    today_count = _count_trades_today(journal)
    max_today = int(getattr(getattr(config, "risk", None), "max_trades_per_day", 5))
    if today_count >= max_today:
        reasons.append(f"max_trades_per_day_reached:{today_count}>={max_today}")

    # Daily loss cap.
    max_daily_loss = float(getattr(getattr(config, "risk", None), "max_daily_loss_pct", 3.0))
    realised = daily_realized_pl(journal)
    starting = float(getattr(account_state, "starting_balance_today", account_state.balance) or account_state.balance)
    if starting > 0:
        # Convert realised (account ccy) to a percentage of starting equity.
        realised_pct = (realised / starting) * 100.0
        projected_risk_pct = float(llm_risk) if llm_risk > 0 else 0.0
        if -realised_pct + projected_risk_pct > max_daily_loss:
            reasons.append(
                f"daily_loss_cap_exceeded:realised={realised_pct:.2f}% "
                f"+trade_risk={projected_risk_pct:.2f}%>{max_daily_loss:.2f}%"
            )

    if reasons:
        return RiskVerdict(False, reasons)
    return RiskVerdict(True, [])


__all__ = [
    "AccountState",
    "RiskVerdict",
    "check_decision",
    "daily_realized_pl",
    "position_size",
]