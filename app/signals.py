"""Signal record shape + paper-balance helper.

The strategy was removed on 2026-11-04 (Megah's order): the sweep+FVG
rules engine, the decision pipeline (risk gate -> paper sizing ->
dedupe -> alert) and the backtest/breakeven helpers are gone. What
remains here is the plumbing the next strategy will plug into:

* :class:`Signal` — the journaled/alertable record shape (entry/SL/TP
  + paper sizing), still used by the Telegram formatter (duck-typed)
  and the signal ledger;
* :func:`hypothetical_balance` — the configured paper balance
  (``signals.hypothetical_balance_usd``, default $10,000).

No decision is produced anywhere in this codebase until the new
strategy lands. Nothing here can open, modify or close a position.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any

import logging

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """One alertable setup, fully priced and sized (paper)."""

    symbol: str
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    risk_pct: float
    confidence: float
    playbook: str
    reasoning: str
    source_feed: str
    ts: int
    lots: float
    risk_usd: float
    per_dollar_usd: float

    @property
    def ts_label(self) -> str:
        return _dt.datetime.fromtimestamp(
            self.ts, tz=_dt.timezone.utc
        ).strftime("%Y-%m-%d %H:%M")


def hypothetical_balance(cfg: Any) -> float:
    """Configured paper balance for sizing (default $10,000)."""
    sig = getattr(cfg, "signals", None)
    try:
        return float(getattr(sig, "hypothetical_balance_usd", 10000.0))
    except (TypeError, ValueError):
        return 10000.0


__all__ = [
    "Signal",
    "hypothetical_balance",
]
