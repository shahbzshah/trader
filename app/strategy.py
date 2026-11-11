"""Phil v2 + RUDY internal-trend filter ("V2 pair") — the gold signals
strategy (live since 2026-11-11, Megah's order "combine the 2").

Rules (exactly as backtested; do not tune, do not invent):

* M15 only, closed bars only, warmup 300 bars.
* Phil v2 at the signal bar — EMAs 8/13/21/55 on close, pandas
  ``ewm(span, adjust=False)``:
  Buy  = 21 crosses above 55 (prev diff <= 0, curr > 0) AND ma8 > ma13
         AND |close - ma55| <= $20.00 AND |ma8 - ma13| >= $0.75.
  Sell = mirror.
* RUDY ``itrend`` — a literal port of RUDYINDICATOR-V1 ``structure()``
  (Manual mode, internal lookback 5, swing 50): first-match pivot
  recording (iH, iL, sL, sH), ``.l`` arrays NEVER cleared, crossover
  against the previous bar's level, swing blocks included (they share
  state). A Phil triangle fires only when ``itrend`` AFTER processing
  the signal bar equals the Phil direction (+1 buy, -1 sell).
* Entry = signal-bar close. Stop loss = entry -/+ $10.00 (Megah's
  standing rule). Take profit = NONE — the backtest has no TP;
  winners exit at the opposite filtered signal (flip). No TP price
  is ever invented here.

Backtested on frozen Deriv frxXAUUSD M15 (16,577 bars, 24 Feb–
10 Nov 2026), $0.50 round-trip spread:
  V2 pair, no stop:  151 signals, 100 trades, 48.0% win,
                     +$15.95/trade, total +$1,595.14.
  V2 pair, $10 stop: 151 signals, 139 trades, 20.1% win,
                     +$7.60/trade, total +$1,056.61, PF 1.93,
                     max DD $233.73, 77.7% stop-outs.
  Slippage warning: 23% of stop-outs had the triggering bar run
  more than $5 past the stop, so the stop-variant numbers are
  likely slightly optimistic.

This module is pure (no I/O, no clock): it takes OHLC lists and
returns signals. ``scan_all`` / ``simulate_stop10`` exist so the
port can be verified against the frozen backtest data exactly.
"""

from __future__ import annotations

import pandas as pd

WARMUP_BARS = 300
HISTORY_BARS = 400
MAX_DIST_USD = 20.0
MIN_SEP_813_USD = 0.75
STOP_USD = 10.0
SPREAD_RT = 0.50
CONFIDENCE = 0.48  # backtested win rate, labelled as such, NOT a prediction
PLAYBOOK = "phil-rudy-v2"

# RUDY structure() constants (Manual mode defaults).
ILEN = 5
SLEN = 50


# ---------------------------------------------------------------------------
# Phil v2
# ---------------------------------------------------------------------------

def _emas(closes: list[float]) -> dict[int, list[float]]:
    """EMAs 8/13/21/55 on close, pandas ewm(span, adjust=False) —
    the exact convention the backtest used. (Do NOT use
    app.scanner.ema: it uses adjust=True and gives different values.)"""
    series = pd.Series([float(c) for c in closes], dtype=float)
    return {
        m: series.ewm(span=m, adjust=False).mean().tolist()
        for m in (8, 13, 21, 55)
    }


def phil_v2_signals(closes: list[float], warmup: int = WARMUP_BARS,
                    max_dist: float = MAX_DIST_USD,
                    min_sep: float = MIN_SEP_813_USD) -> list[dict]:
    """Phil v2 triangles over the whole series.

    Returns one dict per signal: bar index, direction (+1 buy / -1 sell),
    close and the four MA values at the signal bar.
    """
    n = len(closes)
    emas = _emas(closes)
    ma8, ma13, ma21, ma55 = emas[8], emas[13], emas[21], emas[55]
    out: list[dict] = []
    for i in range(max(warmup, 1), n):
        d_prev = ma21[i - 1] - ma55[i - 1]
        d = ma21[i] - ma55[i]
        cross_up = d_prev <= 0 and d > 0
        cross_dn = d_prev >= 0 and d < 0
        if cross_up and ma8[i] > ma13[i] \
                and abs(closes[i] - ma55[i]) <= max_dist \
                and abs(ma8[i] - ma13[i]) >= min_sep:
            out.append({"bar": i, "dir": 1, "close": float(closes[i]),
                        "ma8": ma8[i], "ma13": ma13[i],
                        "ma21": ma21[i], "ma55": ma55[i]})
        elif cross_dn and ma8[i] < ma13[i] \
                and abs(closes[i] - ma55[i]) <= max_dist \
                and abs(ma8[i] - ma13[i]) >= min_sep:
            out.append({"bar": i, "dir": -1, "close": float(closes[i]),
                        "ma8": ma8[i], "ma13": ma13[i],
                        "ma21": ma21[i], "ma55": ma55[i]})
    return out


# ---------------------------------------------------------------------------
# RUDY structure() — literal port
# ---------------------------------------------------------------------------

def _pivot_high(high_v: list[float], i: int, look: int) -> float | None:
    """Pivot high confirmed at bar i (pivot bar j = i - look), or None.

    Strictly greater than the ``look`` bars to the left, >= the
    ``look`` bars to the right (Pine ta.pivothigh semantics)."""
    if i < 2 * look:
        return None
    j = i - look
    if high_v[j] > max(high_v[j - look:j]) and \
            high_v[j] >= max(high_v[j + 1:i + 1]):
        return high_v[j]
    return None


def _pivot_low(low_v: list[float], i: int, look: int) -> float | None:
    if i < 2 * look:
        return None
    j = i - look
    if low_v[j] < min(low_v[j - look:j]) and \
            low_v[j] <= min(low_v[j + 1:i + 1]):
        return low_v[j]
    return None


def run_structure(close_v: list[float], high_v: list[float],
                  low_v: list[float]) -> tuple[list[tuple], list[int]]:
    """Literal port of RUDYINDICATOR-V1 ``structure()`` (lines 609–959).

    Returns ``(events, itrend_log)`` where events are
    ``(bar_i, dir, kind, scope)`` with kind in BOS/ChoCH/CHoCH+ and
    scope in internal/swing, and ``itrend_log[i]`` is the internal
    trend AFTER bar ``i`` was processed.

    Faithful details (do not "simplify"):
    * pivot recording per bar: FIRST matching arm only, in order
      iH, iL, sL, sH — at most one pivot recorded per bar;
    * crossover uses the level as of the END of the previous bar
      (``prev_*`` snapshots) against the current close;
    * on a break the crossed side's ``.n``/``.p`` arrays are cleared;
      the ``.l`` arrays are NEVER cleared anywhere;
    * swing blocks run in the same pass and share the same recording
      step (they affect ``trend``, not ``itrend``, but their pivots
      consume the one-recording-per-bar slot).
    """
    m = len(close_v)
    up_p: list[float] = []
    up_l: list[float] = []
    up_n: list[int] = []
    dn_p: list[float] = []
    dn_l: list[float] = []
    dn_n: list[int] = []
    sup_p: list[float] = []
    sup_l: list[float] = []
    sup_n: list[int] = []
    sdn_p: list[float] = []
    sdn_l: list[float] = []
    sdn_n: list[int] = []
    itrend = 0
    trend = 0
    prev_up: float | None = None
    prev_dn: float | None = None
    prev_sup: float | None = None
    prev_sdn: float | None = None
    events: list[tuple] = []
    itrend_log: list[int] = []

    for i in range(m):
        # --- pivot detection (confirmed on this bar) ---
        iH = iL = sH = sL = None
        ji = js = None
        if i >= 2 * ILEN:
            ji = i - ILEN
            iH = _pivot_high(high_v, i, ILEN)
            iL = _pivot_low(low_v, i, ILEN)
        if i >= 2 * SLEN:
            js = i - SLEN
            sL = _pivot_low(low_v, i, SLEN)
            sH = _pivot_high(high_v, i, SLEN)

        # --- recording: first matching arm only, order iH, iL, sL, sH ---
        if iH is not None:
            up_p.insert(0, iH)
            up_l.insert(0, iH)
            up_n.insert(0, ji)
        elif iL is not None:
            dn_p.insert(0, iL)
            dn_l.insert(0, iL)
            dn_n.insert(0, ji)
        elif sL is not None:
            sdn_p.insert(0, sL)
            sdn_l.insert(0, sL)
            sdn_n.insert(0, js)
        elif sH is not None:
            sup_p.insert(0, sH)
            sup_l.insert(0, sH)
            sup_n.insert(0, js)

        # --- internal bullish block ---
        if up_p and len(dn_l) > 1 and prev_up is not None and i > 0:
            if close_v[i - 1] <= prev_up and close_v[i] > up_p[0]:
                if itrend < 0:
                    kind = "CHoCH+" if dn_l[0] > dn_l[1] else "CHoCH"
                else:
                    kind = "BOS"
                events.append((i, 1, kind, "internal"))
                itrend = 1
                up_n.clear()
                up_p.clear()

        # --- internal bearish block ---
        if dn_p and len(up_l) > 1 and prev_dn is not None and i > 0:
            if close_v[i - 1] >= prev_dn and close_v[i] < dn_p[0]:
                if itrend > 0:
                    kind = "CHoCH+" if up_l[0] < up_l[1] else "CHoCH"
                else:
                    kind = "BOS"
                events.append((i, -1, kind, "internal"))
                itrend = -1
                dn_n.clear()
                dn_p.clear()

        # --- swing bullish block ---
        if sup_p and len(sdn_l) > 1 and prev_sup is not None and i > 0:
            if close_v[i - 1] <= prev_sup and close_v[i] > sup_p[0]:
                if trend < 0:
                    kind = "CHoCH+" if sdn_l[0] > sdn_l[1] else "CHoCH"
                else:
                    kind = "BOS"
                events.append((i, 1, kind, "swing"))
                trend = 1
                sup_n.clear()
                sup_p.clear()

        # --- swing bearish block ---
        if sdn_p and len(sup_l) > 1 and prev_sdn is not None and i > 0:
            if close_v[i - 1] >= prev_sdn and close_v[i] < sdn_p[0]:
                if trend > 0:
                    kind = "CHoCH+" if sup_l[0] < sup_l[1] else "CHoCH"
                else:
                    kind = "BOS"
                events.append((i, -1, kind, "swing"))
                trend = -1
                sdn_n.clear()
                sdn_p.clear()

        # --- snapshot levels at end of bar (for next bar's crossover) ---
        prev_up = up_p[0] if up_p else None
        prev_dn = dn_p[0] if dn_p else None
        prev_sup = sup_p[0] if sup_p else None
        prev_sdn = sdn_p[0] if sdn_p else None
        itrend_log.append(itrend)
    return events, itrend_log


# ---------------------------------------------------------------------------
# Combined scan (live path + verification path share this code)
# ---------------------------------------------------------------------------

def scan_all(closes: list[float], highs: list[float], lows: list[float],
             warmup: int = WARMUP_BARS,
             max_dist: float = MAX_DIST_USD,
             min_sep: float = MIN_SEP_813_USD) -> dict:
    """Scan a full OHLC series. Returns phil signals, RUDY internal
    signals and the V2 filtered signals (each a list of signal dicts
    with ``bar`` / ``dir``), plus the itrend log."""
    phil = phil_v2_signals(closes, warmup=warmup,
                           max_dist=max_dist, min_sep=min_sep)
    events, itrend_log = run_structure(closes, highs, lows)
    rudy = [{"bar": e[0], "dir": e[1], "kind": e[2], "scope": e[3]}
            for e in events
            if e[3] == "internal" and e[0] >= warmup]
    v2 = [dict(s) for s in phil if itrend_log[s["bar"]] == s["dir"]]
    for s in v2:
        s["itrend"] = itrend_log[s["bar"]]
    return {"phil": phil, "rudy": rudy, "v2": v2, "itrend": itrend_log}


def evaluate_last_bar(bars: list[dict], warmup: int = WARMUP_BARS,
                      max_dist: float = MAX_DIST_USD,
                      min_sep: float = MIN_SEP_813_USD) -> dict | None:
    """Evaluate only the LAST bar of a closed-bar list (the live path).

    ``bars`` must be closed bars only, oldest first, each with
    ``time``/``open``/``high``/``low``/``close``. Returns the V2
    signal dict for the last bar (with ``epoch`` and the trigger
    values), or None when the last bar is not a V2 signal. Returns
    None when too few bars are available to clear the warmup.
    """
    n = len(bars)
    if n <= warmup:
        return None
    closes = [float(b["close"]) for b in bars]
    highs = [float(b["high"]) for b in bars]
    lows = [float(b["low"]) for b in bars]
    scan = scan_all(closes, highs, lows, warmup=warmup,
                    max_dist=max_dist, min_sep=min_sep)
    last = n - 1
    for s in scan["v2"]:
        if s["bar"] == last:
            out = dict(s)
            out["epoch"] = int(bars[last]["time"])
            out["dist55"] = abs(out["close"] - out["ma55"])
            out["sep813"] = abs(out["ma8"] - out["ma13"])
            return out
    return None


def last_bar_values(bars: list[dict]) -> dict | None:
    """Indicator values at the last bar (for reasoning text), even
    when the bar is not a signal. None when too few bars."""
    n = len(bars)
    if n <= WARMUP_BARS:
        return None
    closes = [float(b["close"]) for b in bars]
    emas = _emas(closes)
    i = n - 1
    _, itrend_log = run_structure(
        closes,
        [float(b["high"]) for b in bars],
        [float(b["low"]) for b in bars],
    )
    return {
        "bar": i, "epoch": int(bars[i]["time"]),
        "close": closes[i], "ma8": emas[8][i], "ma13": emas[13][i],
        "ma21": emas[21][i], "ma55": emas[55][i],
        "itrend": itrend_log[i],
        "dist55": abs(closes[i] - emas[55][i]),
        "sep813": abs(emas[8][i] - emas[13][i]),
    }


# ---------------------------------------------------------------------------
# $10-stop trade engine (verification — mirrors stop10_backtest.py)
# ---------------------------------------------------------------------------

def simulate_stop10(sig_idx: list[int], sig_dir: list[int],
                    closes: list[float], highs: list[float],
                    lows: list[float], opens: list[float],
                    stop: float = STOP_USD,
                    warmup: int = WARMUP_BARS) -> tuple[list[tuple], int]:
    """Trade engine with the $10 stop, exactly as backtested.

    Entry at the signal bar's close (+/- $0.25 spread). Winners exit
    at the opposite signal's close (flip); the stop exits at the stop
    fill (or the bar's open on a gap through it). Same-direction
    signals while a position is open are ignored; once flat, the next
    signal of either direction opens a position. If the same bar
    triggers both the stop and the flip, the stop takes precedence
    for the exit reason and the flip still opens the next position.

    Returns ``(trades, n_breach)`` where each trade is
    ``(entry_bar, exit_bar, dir, pnl, exit_reason)`` with exit_reason
    in stop/flip/eod, pnl in price-$ per lot (spread included as
    +/- $0.25 at entry and exit), and ``n_breach`` counts
    stop-triggering bars whose extreme went more than $5 past the
    stop price.
    """
    n = len(closes)
    sig = {i: s for i, s in zip(sig_idx, sig_dir)}
    trades: list[tuple] = []
    n_breach = 0
    pos = None  # (entry_bar, dir, entry_px, stop_px)
    for i in range(warmup, n):
        if pos is not None and i > pos[0]:
            ebar, pdir, epx, spx = pos
            if pdir == 1:  # long
                if lows[i] <= spx:
                    if lows[i] <= spx - 5.0:
                        n_breach += 1
                    fill = opens[i] if opens[i] < spx else spx
                    exit_px = fill - 0.25
                    trades.append((ebar, i, pdir, exit_px - epx, "stop"))
                    pos = None
            else:  # short
                if highs[i] >= spx:
                    if highs[i] >= spx + 5.0:
                        n_breach += 1
                    fill = opens[i] if opens[i] > spx else spx
                    exit_px = fill + 0.25
                    trades.append((ebar, i, pdir, (exit_px - epx) * -1, "stop"))
                    pos = None
        if i in sig:
            s = sig[i]
            if pos is None:
                epx = closes[i] + 0.25 * s
                pos = (i, s, epx, closes[i] - stop * s)
            elif s != pos[1]:
                exit_px = closes[i] - 0.25 * pos[1]
                trades.append((pos[0], i, pos[1],
                               (exit_px - pos[2]) * pos[1], "flip"))
                epx = closes[i] + 0.25 * s
                pos = (i, s, epx, closes[i] - stop * s)
            # same direction while holding: ignore
    if pos is not None:
        exit_px = closes[-1] - 0.25 * pos[1]
        trades.append((pos[0], n - 1, pos[1],
                       (exit_px - pos[2]) * pos[1], "eod"))
    return trades, n_breach


def trade_stats(trades: list[tuple]) -> dict:
    """Win rate / expectancy / total from ``simulate_stop10`` trades."""
    pnls = [t[3] for t in trades]
    wins = [p for p in pnls if p > 0]
    return {
        "trades": len(trades),
        "wins": len(wins),
        "win_rate": 100.0 * len(wins) / max(len(pnls), 1),
        "expectancy": sum(pnls) / max(len(pnls), 1),
        "total": sum(pnls),
        "stop_outs": sum(1 for t in trades if t[4] == "stop"),
        "flip_exits": sum(1 for t in trades if t[4] == "flip"),
    }


__all__ = [
    "CONFIDENCE",
    "HISTORY_BARS",
    "MAX_DIST_USD",
    "MIN_SEP_813_USD",
    "PLAYBOOK",
    "STOP_USD",
    "WARMUP_BARS",
    "evaluate_last_bar",
    "last_bar_values",
    "phil_v2_signals",
    "run_structure",
    "scan_all",
    "simulate_stop10",
    "trade_stats",
]
