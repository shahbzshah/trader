# Trend Pullback (placeholder)

> **Placeholder playbook.** Replace this file with your real rules.

## Name
Trend Pullback

## Market / Session
Liquid forex majors and XAUUSD, London or NY session.

## Trigger
- H1 trend up: close > EMA50 > EMA200.
- M15 pulls back to EMA20 then prints a bullish engulfing or pin bar.

## Confirmation
- RSI(14) on M15 between 40 and 55 (pullback zone).
- Volume (tick_volume) on trigger candle above 20-bar average.

## Stop loss logic
Below the M15 swing low that formed the pullback, plus 0.5× ATR(14) M15.

## Take profit logic
- TP1: previous H1 swing high.
- TP2: 2R from entry.

## Invalidation (stand down when)
- H1 trend reverses (close back below EMA200).
- Spread > 2× median spread.
- News within 30 minutes.

## Notes
This file is a TEMPLATE. The LLM must NOT treat these as your live rules.