# Risk Rules

<!--
Filled out in milestone M3. Each rule uses a `## Rule` heading and a one-line
machine-readable summary starting with `RULE:` that brain.py can parse.
-->

RULE: max_risk_pct: 1.0
RULE: max_daily_loss_pct: 3.0
RULE: max_open_trades: 3
RULE: max_trades_per_day: 5
RULE: min_confidence: 0.6

## Rule: Per-trade risk cap

Never risk more than 1.0% of current equity on a single trade. Use stop-loss
distance from entry to compute lots via app/risk.py.

## Rule: Daily loss cap

Halt trading once realized + floating P/L for the day drops below -3.0% of
starting equity.

## Rule: Demo only

This agent must only connect to a MetaTrader 5 DEMO account. Any connection
to a live account is rejected by app/mt5_client.py:ensure_demo().

## Rule: Kill switch

If `data/KILL_SWITCH` exists, no orders are sent and no scan cycles run.

## Rule: No LLM overrides

The LLM proposes; code disposes. Any decision that fails a risk check is
rejected and journaled with the rejection reasons.