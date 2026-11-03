# Sweep + FVG Reclaim

> **Live playbook.** These are the exact rules encoded in `app/strategy.py`.
> Thresholds are module-level constants there -- tune the constants, and
> update this file to match.

## Name
Sweep + FVG Reclaim

## Market / Session
XAUUSD on M15 triggers with H1 bias. Since 2026-11-03 the bot trades
every session (00:00-23:59 UTC, weekends skipped — gold does not trade
weekends on Deriv). Kill zones were removed the same day on Megah's
order, so entries are no longer restricted to London/New York windows;
the most reliable windows remain the London open and the London/New
York overlap; gold moves best in the New York session.

## Position management
- **Breakeven:** once a position is up **1.0R**, the stop is moved to the
  entry price (`risk.breakeven_at_r`, `risk.breakeven_enabled` in
  config.yaml). Winners can't turn into losers.
- **Daily loss halt:** if realised P/L for the UTC day reaches
  **-3%** (`risk.max_daily_loss_pct`), no new entries for the rest of the
  day. A single trade whose own risk would breach the cap is also blocked.

## Bias (H1)
- Bullish when H1 market structure trend is **up** (higher highs + higher
  lows, or a bullish BOS/CHoCH) **and** H1 close is above EMA50.
- Bearish when H1 structure trend is **down** **and** H1 close is below EMA50.
- Otherwise: no bias, no trade.

## Trigger (M15 liquidity sweep)
- Bullish: a bar within the last **5** M15 bars wicked **below** the recent
  swing low (20-bar window, excluding the last 5 bars) and closed back
  **above** it -- sell-side liquidity swept.
- Bearish: mirror -- wick above the recent swing high, close back below it.
- No sweep in the window: no trade.

## Confirmation / Entry (M15 fair value gap)
- A **fresh, unfilled** FVG of the bias direction, formed **at or after**
  the sweep bar and within the last **8** bars.
  - Bullish FVG: 3-candle displacement up, low[i] > high[i-2], gap at
    least **0.25 x ATR14**.
  - A gap is *filled* (invalid) once a later bar trades fully through it.
- Entry: **limit order at the FVG midpoint**. The midpoint must sit on the
  correct side of the market (below last close for longs, above for shorts).

## Stop loss logic
- Beyond the sweep extreme (below the sweep low for longs, above the sweep
  high for shorts), plus a **0.1 x ATR14** buffer.
- Never closer than **1.0 x ATR14** from entry -- if the sweep extreme is
  nearer, the stop is widened to 1.0 x ATR.

## Take profit logic
- Single target at **2R** (twice the entry-to-stop distance).

## Invalidation (stand down when)
- Spread wider than **30 points**.
- Outside the configured session hours.
- H1 bias unclear (structure trend disagrees with EMA50).
- The FVG fills before the limit order triggers.
- A fresh opposite-direction sweep prints (bias/trend may be flipping).

## Notes
- The engine proposes; `app/risk.py` disposes. Suggested risk is 1% and
  confidence 0.75 on every setup -- risk.py clamps these to its configured
  maxima and can veto for any reason (daily loss cap, open-trade cap, kill
  switch, non-demo account).
- One setup, traded mechanically. Do not stack discretionary filters on top.
