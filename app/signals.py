"""Signals-only pipeline: risk gate -> paper sizing -> dedupe -> alert.

This module replaces ``app.executor``. There is no order placement
anywhere in the signals-only codebase: a validated setup becomes a
:func:`Signal` (entry/SL/TP + paper sizing) that is journaled and
posted to Telegram. Nothing here can open, modify or close a position.

What survives from the old executor, unchanged in intent:
* re-validation against :mod:`app.risk` immediately before alerting
  (state may have changed since the strategy decided);
* limit-price validation against the current tick (a buy limit must sit
  below the ask, a sell limit above the bid) — a signal-quality check;
* sizing with :func:`app.risk.position_size`.

What changes:
* the account is the configured *hypothetical* balance
  (``signals.hypothetical_balance_usd``, default $10,000), not a broker
  account — so ``risk$ = balance x risk%`` and ``lots = risk$ / (SL$ x
  100)`` for gold (1 lot = 100 oz, so $1 of move = $100 per lot);
* no order rows are written — only the decision row, so the journal's
  ``orders`` table stays empty by design;
* open-trade / trades-today / daily-loss caps have no meaning without
  positions, so they are not enforced here (the risk gate is called
  with empty positions and journal rows — the structural checks still
  apply: kill switch, confidence floor, SL side, TP present, symbol in
  watchlist, risk% cap);
* de-duplication: an identical signal (same symbol, direction, entry,
  SL, TP) already alerted today is not re-sent — without this, the
  strategy would re-fire the same setup every 15-minute cycle while
  the FVG is still fresh;
* a daily cap (``signals.max_signals_per_day``, default 5) bounds
  alert spam.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Any

from app.brain import Decision
from app.journal import Journal
from app.market_data import MarketDataClient
from app.risk import AccountState, check_decision, position_size
from app.telegram import TelegramAlerter, format_signal_text

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


@dataclass
class SignalResult:
    status: str  # "signaled" | "rejected" | "duplicate" | "skipped" | "capped"
    signal: Signal | None = None
    decision_id: int | None = None
    telegram_sent: bool = False
    reject_reasons: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.reject_reasons is None:
            self.reject_reasons = []


def hypothetical_balance(cfg: Any) -> float:
    """Configured paper balance for sizing (default $10,000)."""
    sig = getattr(cfg, "signals", None)
    try:
        return float(getattr(sig, "hypothetical_balance_usd", 10000.0))
    except (TypeError, ValueError):
        return 10000.0


def size_signal(
    cfg: Any,
    client: MarketDataClient,
    decision: Decision,
) -> tuple[float, float, float]:
    """Paper-size one decision.

    Returns ``(lots, risk_usd, per_dollar_usd)`` where ``lots`` follows
    the report's formula ``lots = risk$ / (SL$ x 100)`` (via
    :func:`app.risk.position_size` with the feed's contract spec).
    ``per_dollar_usd`` is the P/L per $1 of gold move at that size
    (``lots x 100``). Raises ``ValueError`` when sizing is impossible
    (no SL, zero risk distance, risk% <= 0).
    """
    balance = hypothetical_balance(cfg)
    max_risk = float(getattr(getattr(cfg, "risk", None),
                             "max_risk_pct_per_trade", 1.0))
    risk_pct = min(float(decision.risk_pct or 0.0), max_risk)
    if risk_pct <= 0:
        raise ValueError("risk_pct_non_positive")
    if decision.stop_loss is None or decision.price is None:
        raise ValueError("missing_entry_or_sl")
    spec = client.symbol_info(decision.symbol)
    lots = position_size(
        equity=balance,
        risk_pct=risk_pct,
        entry=float(decision.price),
        stop_loss=float(decision.stop_loss),
        symbol_info=spec,
    )
    if lots <= 0:
        raise ValueError("position_size_unable")
    risk_usd = balance * risk_pct / 100.0
    per_dollar = lots * float(spec.trade_contract_size)
    return lots, risk_usd, per_dollar


class SignalEngine:
    """Turns approved ``propose`` decisions into journaled Telegram signals."""

    def __init__(
        self,
        client: MarketDataClient,
        journal: Journal,
        config: Any,
        telegram: TelegramAlerter | None = None,
    ) -> None:
        self.client = client
        self.journal = journal
        self.config = config
        self.telegram = telegram

    # --- helpers ---------------------------------------------------------

    def _paper_state(self) -> AccountState:
        balance = hypothetical_balance(self.config)
        kill = self.journal.is_kill_switch_engaged(
            self.config.paths.kill_switch_path
        )
        return AccountState(
            balance=balance,
            equity=balance,
            starting_balance_today=balance,
            trade_mode="demo",
            kill_switch_engaged=bool(kill),
        )

    def _record(self, decision: Decision, approved: bool,
                reasons: list[str], raw_llm_json: str = "") -> int:
        return self.journal.record_decision(
            symbol=decision.symbol,
            action=decision.action,
            playbook=decision.playbook or "",
            confidence=float(decision.confidence),
            risk_pct=float(decision.risk_pct),
            reasoning=decision.reasoning,
            approved=approved,
            reject_reasons=reasons,
            raw_llm_json=raw_llm_json,
        )

    def _already_signaled_today(self, decision: Decision) -> bool:
        """True when an identical approved propose was already journaled
        today (same symbol, direction, entry, SL, TP)."""
        start_ts, _ = self.journal.today_utc_window()
        rows = self.journal.list_decisions(since_ts=start_ts, limit=500)
        for row in rows:
            if row.get("action") != "propose" or not row.get("approved"):
                continue
            if row.get("symbol") != decision.symbol:
                continue
            raw = row.get("raw_llm_json") or ""
            if not raw:
                continue
            try:
                import json

                prev = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if (
                prev.get("direction") == decision.direction
                and float(prev.get("price") or 0) == float(decision.price or 0)
                and float(prev.get("stop_loss") or 0)
                == float(decision.stop_loss or 0)
                and float(prev.get("take_profit") or 0)
                == float(decision.take_profit or 0)
            ):
                return True
        return False

    def _signals_today(self) -> int:
        start_ts, _ = self.journal.today_utc_window()
        rows = self.journal.list_decisions(since_ts=start_ts, limit=1000)
        return sum(
            1 for r in rows if r.get("action") == "propose" and r.get("approved")
        )

    # --- public API ------------------------------------------------------

    def process(
        self,
        decision: Decision,
        *,
        raw_llm_json: str = "",
        dry_run: bool = False,
    ) -> SignalResult:
        """Validate, size, dedupe, journal and (unless ``dry_run``) alert.

        ``dry_run`` is the analyze path: full validation + sizing, but
        nothing is journaled and no Telegram message is sent.
        """
        if decision.action == "no_trade":
            decision_id = None if dry_run else self._record(
                decision, True, [], raw_llm_json)
            return SignalResult(
                status="skipped", decision_id=decision_id, reject_reasons=[]
            )
        if decision.action != "propose":
            decision_id = None if dry_run else self._record(
                decision, False, [f"unhandled_action:{decision.action}"],
                raw_llm_json)
            return SignalResult(
                status="rejected", decision_id=decision_id,
                reject_reasons=[f"unhandled_action:{decision.action}"],
            )

        # 1. Risk gate (structural checks; paper account, no positions).
        verdict = check_decision(
            decision, self._paper_state(), [], [], self.config
        )
        if not verdict.approved:
            decision_id = None if dry_run else self._record(
                decision, False, verdict.reasons, raw_llm_json)
            return SignalResult(
                status="rejected", decision_id=decision_id,
                reject_reasons=list(verdict.reasons),
            )

        # 2. Limit-price sanity vs the current tick (signal quality).
        entry_type = str(getattr(decision, "entry_type", "market") or "market").lower()
        entry = decision.price
        if entry_type == "limit":
            if entry is None:
                decision_id = None if dry_run else self._record(
                    decision, False, ["limit_requires_price"], raw_llm_json)
                return SignalResult(
                    status="rejected", decision_id=decision_id,
                    reject_reasons=["limit_requires_price"],
                )
            try:
                tick = self.client.symbol_tick(decision.symbol)
            except Exception as exc:  # noqa: BLE001
                decision_id = None if dry_run else self._record(
                    decision, False, [f"tick_fetch_failed:{exc}"], raw_llm_json)
                return SignalResult(
                    status="rejected", decision_id=decision_id,
                    reject_reasons=[f"tick_fetch_failed:{exc}"],
                )
            if tick is not None:
                if decision.direction == "buy" and not float(entry) < float(tick.ask):
                    reasons = [f"limit_price_not_below_market:entry={entry}>=ask={tick.ask}"]
                    decision_id = None if dry_run else self._record(
                        decision, False, reasons, raw_llm_json)
                    return SignalResult(
                        status="rejected", decision_id=decision_id,
                        reject_reasons=reasons)
                if decision.direction == "sell" and not float(entry) > float(tick.bid):
                    reasons = [f"limit_price_not_above_market:entry={entry}<=bid={tick.bid}"]
                    decision_id = None if dry_run else self._record(
                        decision, False, reasons, raw_llm_json)
                    return SignalResult(
                        status="rejected", decision_id=decision_id,
                        reject_reasons=reasons)
        elif entry is None:
            try:
                tick = self.client.symbol_tick(decision.symbol)
            except Exception as exc:  # noqa: BLE001
                decision_id = None if dry_run else self._record(
                    decision, False, [f"tick_fetch_failed:{exc}"], raw_llm_json)
                return SignalResult(
                    status="rejected", decision_id=decision_id,
                    reject_reasons=[f"tick_fetch_failed:{exc}"],
                )
            entry = (tick.ask if decision.direction == "buy" else tick.bid) \
                if tick is not None else None
            if entry is None:
                decision_id = None if dry_run else self._record(
                    decision, False, ["tick_fetch_failed:no_tick"], raw_llm_json)
                return SignalResult(
                    status="rejected", decision_id=decision_id,
                    reject_reasons=["tick_fetch_failed:no_tick"],
                )

        # 3. Paper sizing.
        sized = Decision(
            action=decision.action,
            symbol=decision.symbol,
            direction=decision.direction,
            price=float(entry),
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            risk_pct=decision.risk_pct,
            confidence=decision.confidence,
            playbook=decision.playbook,
            reasoning=decision.reasoning,
            entry_type=decision.entry_type,
        )
        try:
            lots, risk_usd, per_dollar = size_signal(self.config, self.client, sized)
        except ValueError as exc:
            reasons = [str(exc)]
            decision_id = None if dry_run else self._record(
                decision, False, reasons, raw_llm_json)
            return SignalResult(
                status="rejected", decision_id=decision_id, reject_reasons=reasons)

        signal = Signal(
            symbol=sized.symbol,
            direction=sized.direction,
            entry=float(entry),
            stop_loss=float(sized.stop_loss),
            take_profit=float(sized.take_profit),
            risk_pct=min(float(sized.risk_pct or 0.0),
                         float(getattr(getattr(self.config, "risk", None),
                                       "max_risk_pct_per_trade", 1.0))),
            confidence=float(sized.confidence),
            playbook=sized.playbook or "",
            reasoning=sized.reasoning,
            source_feed=self.client.name,
            ts=int(_dt.datetime.now(_dt.timezone.utc).timestamp()),
            lots=lots,
            risk_usd=risk_usd,
            per_dollar_usd=per_dollar,
        )

        if dry_run:
            return SignalResult(status="signaled", signal=signal)

        # 4. Dedupe: same setup already alerted today?
        if self._already_signaled_today(sized):
            logger.info("duplicate signal suppressed: %s %s @ %s",
                        signal.symbol, signal.direction, signal.entry)
            return SignalResult(status="duplicate", signal=signal)

        # 5. Daily cap.
        max_per_day = int(getattr(getattr(self.config, "signals", None),
                                  "max_signals_per_day", 5) or 0)
        if max_per_day > 0 and self._signals_today() >= max_per_day:
            decision_id = self._record(
                sized, False, ["max_signals_per_day_reached"],
                sized.model_dump_json())
            return SignalResult(
                status="capped", decision_id=decision_id,
                reject_reasons=["max_signals_per_day_reached"])

        # 6. Journal + alert. The journaled row is the *sized* decision
        # (resolved entry), so the dedupe check compares real prices.
        decision_id = self._record(sized, True, [], sized.model_dump_json())
        telegram_sent = False
        if self.telegram is not None and self.telegram.enabled:
            telegram_sent = self.telegram.send_message(format_signal_text(signal))
            if not telegram_sent:
                logger.warning("signal journaled (id=%s) but Telegram send failed",
                               decision_id)
        elif self.telegram is not None:
            logger.warning("Telegram not configured; signal journaled only")
        return SignalResult(
            status="signaled", signal=signal, decision_id=decision_id,
            telegram_sent=telegram_sent,
        )


__all__ = [
    "Signal",
    "SignalEngine",
    "SignalResult",
    "hypothetical_balance",
    "size_signal",
]
