"""Journal-verified order placement.

The executor takes an approved Decision, re-validates against
``app.risk.py`` immediately before placement (state may have changed
since the brain decided), sizes the position, places the order via
``app.mt5_client``, and records the order row linked to the decision id.

It also handles close/modify actions for open positions and reports
clear status strings for partial failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from app.brain import Decision
from app.journal import Journal
from app.mt5_client import MT5Client
from app.risk import AccountState, RiskVerdict, check_decision, position_size

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    status: str            # "filled" | "rejected" | "closed" | "modified" | "skipped"
    decision_id: int | None
    order_id: int | None
    ticket: int | None
    message: str
    reject_reasons: list[str]


class Executor:
    def __init__(
        self,
        client: MT5Client,
        journal: Journal,
        config: Any,
    ) -> None:
        self.client = client
        self.journal = journal
        self.config = config

    # --- internal helpers ----------------------------------------------

    def _account_state(self) -> AccountState:
        info = self.client.account_info()
        # For "starting balance today" we read the first equity row of
        # the current UTC day, falling back to balance if not present.
        start_ts, _ = self.journal.today_utc_window()
        first_equity = self.journal.list_equity(since_ts=start_ts, limit=1)
        if first_equity:
            starting = float(first_equity[0]["balance"])
        else:
            starting = float(info.balance)
        kill = self.journal.is_kill_switch_engaged(self.config.paths.kill_switch_path)
        return AccountState(
            balance=float(info.balance),
            equity=float(info.equity),
            starting_balance_today=starting,
            trade_mode=str(info.trade_mode),
            kill_switch_engaged=bool(kill),
        )

    def _journal_rows(self) -> list[Mapping]:
        start_ts, _ = self.journal.today_utc_window()
        # Combine today's orders + closed positions so risk.py sees the
        # full day of activity.
        orders = self.journal.list_orders(since_ts=start_ts, limit=1000)
        rows: list[Mapping] = []
        for o in orders:
            rows.append({
                "kind": "order_placed" if o["status"] in ("filled", "submitted")
                        else "order_rejected",
                "ts": o["ts"],
                "profit": 0.0,
            })
        return rows

    def _record_decision(
        self,
        decision: Decision,
        approved: bool,
        reasons: list[str],
        raw_llm_json: str = "",
    ) -> int:
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

    # --- public API -----------------------------------------------------

    def execute(self, decision: Decision, *, raw_llm_json: str = "") -> ExecutionResult:
        """Run the full execute pipeline: re-validate, size, place, journal.

        For ``action == "no_trade"`` the decision is journaled but no
        order is placed. For ``"close"``/``"modify"`` the matching
        open position is acted on.
        """
        account = self._account_state()
        positions = self.client.positions()
        journal_rows = self._journal_rows()
        verdict: RiskVerdict = check_decision(
            decision, account, positions, journal_rows, self.config
        )

        if not verdict.approved:
            decision_id = self._record_decision(decision, False, verdict.reasons,
                                                 raw_llm_json=raw_llm_json)
            return ExecutionResult(
                status="rejected",
                decision_id=decision_id,
                order_id=None,
                ticket=None,
                message="risk check failed",
                reject_reasons=list(verdict.reasons),
            )

        if decision.action == "no_trade":
            decision_id = self._record_decision(decision, True, [],
                                                 raw_llm_json=raw_llm_json)
            return ExecutionResult(
                status="skipped",
                decision_id=decision_id,
                order_id=None,
                ticket=None,
                message="no_trade decision acknowledged",
                reject_reasons=[],
            )

        if decision.action == "close":
            return self._execute_close(decision, raw_llm_json=raw_llm_json)
        if decision.action == "modify":
            return self._execute_modify(decision, raw_llm_json=raw_llm_json)
        if decision.action == "propose":
            return self._execute_propose(decision, raw_llm_json=raw_llm_json)

        # Unreachable: check_decision would have rejected unknown actions.
        decision_id = self._record_decision(decision, False,
                                             [f"unhandled_action:{decision.action}"],
                                             raw_llm_json=raw_llm_json)
        return ExecutionResult(
            status="rejected",
            decision_id=decision_id,
            order_id=None,
            ticket=None,
            message="unhandled action",
            reject_reasons=[f"unhandled_action:{decision.action}"],
        )

    # --- action handlers -----------------------------------------------

    def _execute_propose(self, decision: Decision, *, raw_llm_json: str) -> ExecutionResult:
        try:
            sym_info = self.client.symbol_info(decision.symbol)
        except Exception as exc:  # noqa: BLE001
            return self._reject_with_reason(decision, raw_llm_json,
                                             [f"symbol_info_failed:{exc}"])

        # Use the more conservative of LLM-suggested and config risk.
        max_risk = float(self.config.risk.max_risk_pct_per_trade)
        risk_pct = min(float(decision.risk_pct or 0.0), max_risk)
        if risk_pct <= 0:
            return self._reject_with_reason(decision, raw_llm_json,
                                             ["risk_pct_non_positive"])

        # Use proposed entry if given, else fetch a tick price.
        entry = decision.price
        if entry is None:
            try:
                tick = self.client.symbol_tick(decision.symbol)
                entry = tick.ask if decision.direction == "buy" else tick.bid
            except Exception as exc:  # noqa: BLE001
                return self._reject_with_reason(decision, raw_llm_json,
                                                 [f"tick_fetch_failed:{exc}"])

        lots = position_size(
            equity=float(self.client.account_info().equity),
            risk_pct=risk_pct,
            entry=float(entry),
            stop_loss=float(decision.stop_loss),
            symbol_info=sym_info,
        )
        if lots <= 0:
            return self._reject_with_reason(
                decision, raw_llm_json,
                ["position_size_unable", f"requested_risk_pct={risk_pct}"],
            )

        # Record the approved decision.
        decision_id = self._record_decision(decision, True, [],
                                             raw_llm_json=raw_llm_json)

        # Place the order.
        order_id = self.journal.record_order(
            decision_id=decision_id,
            symbol=decision.symbol,
            direction=decision.direction,
            lots=lots,
            status="submitted",
            entry=float(entry),
            sl=float(decision.stop_loss),
            tp=(None if decision.take_profit is None else float(decision.take_profit)),
        )
        try:
            res = self.client.market_order(
                decision.symbol,
                decision.direction,
                lots,
                float(decision.stop_loss),
                float(decision.take_profit) if decision.take_profit is not None else 0.0,
                comment=f"agent:{decision.playbook}"[:32],
            )
        except Exception as exc:  # noqa: BLE001
            self.journal.update_order_status(order_id, status="rejected",
                                              error=str(exc))
            return ExecutionResult(
                status="rejected",
                decision_id=decision_id,
                order_id=order_id,
                ticket=None,
                message=f"order_send failed: {exc}",
                reject_reasons=[f"order_send_failed:{exc}"],
            )

        ticket = int(getattr(res, "order_id", None) or getattr(res, "order", 0) or 0)
        self.journal.update_order_status(order_id, status="filled",
                                          ticket=ticket)
        return ExecutionResult(
            status="filled",
            decision_id=decision_id,
            order_id=order_id,
            ticket=ticket,
            message=f"filled {lots} lots at {getattr(res, 'price', 0.0)}",
            reject_reasons=[],
        )

    def _execute_close(self, decision: Decision, *, raw_llm_json: str) -> ExecutionResult:
        ticket = int(getattr(decision, "price", 0) or 0)
        # The Decision schema uses ``price`` for the requested close ticket
        # in the close action; if it's not set, fall back to first matching
        # open position.
        if ticket <= 0:
            for p in self.client.positions(decision.symbol):
                ticket = int(p.ticket)
                break
        if ticket <= 0:
            return self._reject_with_reason(decision, raw_llm_json,
                                             ["no_open_position_for_close"])

        decision_id = self._record_decision(decision, True, [],
                                             raw_llm_json=raw_llm_json)
        order_id = self.journal.record_order(
            decision_id=decision_id,
            symbol=decision.symbol,
            direction="close",
            lots=0.0,
            status="submitted",
        )
        try:
            res = self.client.close_position(ticket)
        except Exception as exc:  # noqa: BLE001
            self.journal.update_order_status(order_id, status="rejected",
                                              error=str(exc))
            return ExecutionResult(
                status="rejected",
                decision_id=decision_id,
                order_id=order_id,
                ticket=ticket,
                message=f"close failed: {exc}",
                reject_reasons=[f"close_failed:{exc}"],
            )
        self.journal.update_order_status(order_id, status="filled",
                                          ticket=int(getattr(res, "order_id", 0) or getattr(res, "order", 0) or 0))
        return ExecutionResult(
            status="closed",
            decision_id=decision_id,
            order_id=order_id,
            ticket=int(getattr(res, "order_id", 0) or getattr(res, "order", 0) or 0),
            message=f"closed ticket {ticket} at {getattr(res, 'price', 0.0)}",
            reject_reasons=[],
        )

    def _execute_modify(self, decision: Decision, *, raw_llm_json: str) -> ExecutionResult:
        ticket = int(getattr(decision, "price", 0) or 0)
        if ticket <= 0:
            return self._reject_with_reason(decision, raw_llm_json,
                                             ["missing_position_ticket"])
        if decision.stop_loss is None or decision.take_profit is None:
            return self._reject_with_reason(decision, raw_llm_json,
                                             ["missing_sl_or_tp"])

        decision_id = self._record_decision(decision, True, [],
                                             raw_llm_json=raw_llm_json)
        order_id = self.journal.record_order(
            decision_id=decision_id,
            symbol=decision.symbol,
            direction="modify",
            lots=0.0,
            status="submitted",
            sl=float(decision.stop_loss),
            tp=float(decision.take_profit),
        )
        try:
            ok = self.client.modify_position(
                ticket,
                float(decision.stop_loss),
                float(decision.take_profit),
            )
        except Exception as exc:  # noqa: BLE001
            self.journal.update_order_status(order_id, status="rejected",
                                              error=str(exc))
            return ExecutionResult(
                status="rejected",
                decision_id=decision_id,
                order_id=order_id,
                ticket=ticket,
                message=f"modify failed: {exc}",
                reject_reasons=[f"modify_failed:{exc}"],
            )
        if not ok:
            self.journal.update_order_status(order_id, status="rejected",
                                              error="retcode != done")
            return ExecutionResult(
                status="rejected",
                decision_id=decision_id,
                order_id=order_id,
                ticket=ticket,
                message="modify rejected by broker",
                reject_reasons=["modify_rejected"],
            )
        self.journal.update_order_status(order_id, status="modified")
        return ExecutionResult(
            status="modified",
            decision_id=decision_id,
            order_id=order_id,
            ticket=ticket,
            message="modified",
            reject_reasons=[],
        )

    def _reject_with_reason(self, decision: Decision, raw_llm_json: str,
                              reasons: list[str]) -> ExecutionResult:
        decision_id = self._record_decision(decision, False, reasons,
                                             raw_llm_json=raw_llm_json)
        return ExecutionResult(
            status="rejected",
            decision_id=decision_id,
            order_id=None,
            ticket=None,
            message="; ".join(reasons),
            reject_reasons=reasons,
        )


__all__ = ["ExecutionResult", "Executor"]