# Playbook: phil-rudy-v2 (Phil v2 + RUDY trend filter)

**Status:** live 2026-11-11 (Megah's order: "combine the 2").
**Playbook string:** `phil-rudy-v2`. **Source of truth:** `app/strategy.py`
(port verified exactly against the frozen backtest data — see below).

## The system, in one line

Take a Phil v2 triangle **only when RUDY's internal trend agrees** with it.
Phil is the entry; RUDY's trend is the direction check. The filter throws
out the 39 of 190 Phil signals (21%) that fire against structure.

## Exact rules

- **Timeframe:** M15 only. **Bars:** closed bars only (the forming bar is
  dropped: a bar counts once `now >= bar_time + 900s`). Each cycle
  fetches 400 M15 bars (warmup 300 + margin) and evaluates only the
  last closed bar.
- **Phil v2** (EMAs 8/13/21/55 on close, pandas `ewm(span, adjust=False)`):
  - Buy = 21 crosses above 55 (prev diff <= 0, curr > 0) AND ma8 > ma13
    AND |close − ma55| <= $20.00 AND |ma8 − ma13| >= $0.75.
  - Sell = mirror.
- **RUDY itrend:** literal port of RUDYINDICATOR-V1 `structure()` (Manual
  mode, internal lookback 5, swing 50). First-match pivot recording
  (iH, iL, sL, sH — one pivot per bar), `.l` arrays never cleared,
  crossover against the previous bar's level, swing blocks included.
  Fire only when `itrend` after processing the signal bar equals the
  Phil direction (+1 buy, −1 sell).
- **Entry:** signal-bar close.
- **Stop loss:** entry −/+ $10.00 — Megah's standing rule, always $10.
- **Take profit: NONE.** The backtest has no TP. Winners exit at the
  opposite filtered signal (flip). Never invent a TP price, a fixed
  R target, or an average-win target.
- **Position state:** one open position max. Same-direction signal
  while open is ignored. Opposite signal closes the open position at
  the new signal's entry (flip) and opens the new one. If the same bar
  triggers both the stop and a flip, the stop takes precedence for the
  exit reason and the flip still opens.
- **Sizing:** 1% of the $10,000 hypothetical balance via
  `position_size()` with XAUUSD_SPEC — with a $10 SL that is
  **0.10 lots**. $0.50 round-trip spread on every close.
- **Confidence 0.48:** the backtested win rate, labelled as such in the
  reasoning. NOT a prediction, NOT 1.0.
- **Gates, in this order:** kill switch → news blackout (High-impact USD
  ±30 min, fail-open) → max 5 signals/day → daily-loss halt (−3% of
  the hypothetical balance). A gated signal is journaled as a rejected
  `no_trade` decision.
- **No breakeven manager** (`manage.py` was deleted 2026-11-04;
  breakeven was never backtested with this system and would change the
  numbers). The $10 stop stays where the signal put it.

## Backtested numbers (frozen Deriv frxXAUUSD M15, 16,577 bars, 24 Feb–10 Nov 2026)

| | No stop | With $10 stop (live) |
|---|---|---|
| Signals | 151 (78 Buy / 73 Sell) | 151 — same |
| Trades | 100 | 139 |
| Win rate | 48.0% | 20.1% |
| Avg win / avg loss | +$73.56 / −$37.23 | +$78.53 / −$10.29 |
| Expectancy | +$15.95/trade | +$7.60/trade |
| Total | +$1,595.14 | +$1,056.61 |
| Profit factor | 1.82 | 1.93 |
| Max drawdown | $346.17 | $233.73 |
| Avg hold | 160.8 bars (40.2h) | 55.8 bars (14.0h) |
| Halves | +$11.18 / +$20.35 | +$3.49 / +$11.43 |
| Stop-outs | — | 108 (77.7%) |

For comparison on the same data: Phil v2 alone 41.1% / +$9.74
(+$6.10 with the stop); RUDY breaks alone 37.3% / +$0.90 (+$0.26
with the stop). The V2 pair is the best system tested, with or
without the stop.

**The $10 stop hurts expectancy** (+$15.95 → +$7.60) because ~78% of
trades end at the stop before the win can develop. What it buys is a
smoother ride: profit factor up, drawdown cut by a third. The low
20% win rate with the stop is expected — the edge is winners ~8x
the stop. Do not "fix" the win rate by moving the stop; that is a
different system and needs its own backtest.

## Caveats (honest limits)

- **Feed:** backtest is Deriv candles; Megah's chart is IC. Clear
  signals agree; knife-edge signals (e.g. a cross where 8/13 are
  $0.70 apart instead of $0.75) can differ either way.
- **Slippage:** 23% of stop-outs (25 of 108) had the triggering bar
  run more than $5 past the stop price, so the stop-variant numbers
  are likely slightly optimistic. Mean stop fill −$10.51.
- **No TP, long holds:** winners run to the flip — average hold with
  the stop is ~14h, without it ~40h. This is not a scalping system:
  on 5m with $4–8 targets every tested cell lost money.
- **One position max, flip exits** — the ledger's FLIP closes are
  real trades, not a display artefact.

## What is NOT in this system

Smart Entry, ORB, CHoCH, Volume Profile, any other indicator or
filter. Do not add one without a fresh backtest and Megah's word.
