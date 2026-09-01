# Breakout & Retest (placeholder)

> **Placeholder playbook.** Replace this file with your real rules.

## Name
Breakout & Retest

## Market / Session
Liquid forex majors during London / NY overlap. Avoid rollovers.

## Trigger
- Price consolidates below a key resistance level for at least 15 candles on M15.
- ATR(14) on M15 compressed under the 50th percentile of last 50 bars.

## Confirmation
- Strong M15 close above the level with body > 50% of candle range.
- H1 trend agrees (close > EMA50).
- RSI14 H1 > 50 but < 80.

## Stop loss logic
Below the most recent swing low (last 20 bars on M15), minimum 1× ATR(14) M15.

## Take profit logic
- TP1: 1R.
- TP2: 2R.
- Trail stop to breakeven after TP1.

## Invalidation (stand down when)
- Spread > 2× median spread.
- Within 30 minutes of a high-impact news release.
- H1 RSI(14) > 80 (overextended).

## Notes
This file is a TEMPLATE. The LLM must NOT treat these as your live rules.