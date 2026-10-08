"""Deterministic rules engine: the "sweep + FVG reclaim" setup.

No LLM, no network, no I/O. Pure functions over the snapshot dicts produced
by :mod:`app.scanner` (which now include per-timeframe ``smc`` blocks with
FVGs, sweeps and market-structure data).

The engine encodes ONE setup first:

    "sweep + FVG reclaim"
        H1 bias  : market-structure trend (BOS/CHoCH) aligned with EMA50.
        Trigger  : M15 liquidity sweep of a recent swing high/low
                   (wick beyond the level, close back inside).
        Entry    : M15 fair value gap formed at/after the sweep,
                   limit order at the FVG midpoint.
        Stop     : beyond the sweep extreme, minimum 1x ATR14 (M15) from entry.
        Target   : 2R, single target.

This module *proposes*; :mod:`app.risk` remains the ONLY authority on
position sizing and vetoes. Strategy suggestions (risk_pct, confidence)
never bypass risk.py.

All thresholds are module-level constants so the owner can tune them in
one place. The matching human-readable spec lives in
``knowledge/playbooks/sweep_fvg_reclaim.md`` -- keep the two in sync.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Mapping

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Decision schema (canonical home; app.brain re-exports it)
# ---------------------------------------------------------------------------

VALID_ACTIONS = ("no_trade", "propose", "close", "modify")
VALID_DIRECTIONS = ("buy", "sell")
VALID_ENTRY_TYPES = ("market", "limit")


class Decision(BaseModel):
    action: str
    symbol: str
    direction: str | None = None
    entry_type: str = "market"
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_pct: float = 0.0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    playbook: str = ""
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Tunable thresholds -- change these, not the logic below
# ---------------------------------------------------------------------------

PLAYBOOK_NAME = "sweep_fvg_reclaim"

BIAS_TIMEFRAME = "H1"
TRIGGER_TIMEFRAME = "M15"

# FVG / sweep freshness (in TRIGGER_TIMEFRAME bars, counted from the last bar)
SWEEP_LOOKBACK_BARS = 5        # sweep must occur within the last N bars
FVG_MAX_AGE_BARS = 8           # FVG must have formed within the last N bars
FVG_MIN_SIZE_ATR = 0.25        # ignore gaps smaller than 0.25x ATR14

# Trade construction
SL_ATR_MIN_MULT = 1.0          # stop is never closer than 1x ATR14 from entry
SL_BUFFER_ATR = 0.1            # stop sits this far beyond the sweep extreme
REWARD_MULT = 2.0              # single take-profit at 2R

# Filters
MAX_SPREAD_POINTS = 30         # skip when spread (points) is wider than this
                               # (XAUUSD: 30 points ≈ $0.30 at point=0.01 —
                               #  sane for gold, but verify against YOUR broker)

# Suggestion fields (advisory only -- risk.py clamps/enforces)
SETUP_CONFIDENCE = 0.75        # fixed confidence when the setup fires
SETUP_RISK_PCT = 1.0           # suggested risk % (risk.py caps at its maximum)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round5(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), 5)


def _no_trade(symbol: str, reason: str) -> Decision:
    return Decision(
        action="no_trade",
        symbol=symbol,
        playbook=PLAYBOOK_NAME,
        confidence=0.0,
        reasoning=f"no_trade: {reason}",
    )


def _last_bar_time_utc(m15: Mapping) -> _dt.datetime | None:
    candles = m15.get("candles") or []
    if not candles:
        return None
    try:
        return _dt.datetime.fromtimestamp(int(candles[-1]["t"]), tz=_dt.timezone.utc)
    except (KeyError, TypeError, ValueError, OSError):
        return None


def _in_session(m15: Mapping, config: Any) -> bool:
    """True when the last M15 bar falls inside the configured session."""
    sh = getattr(config, "session_hours", None)
    start = getattr(sh, "start", None)
    end = getattr(sh, "end", None)
    if not start or not end:
        return True  # no session configured -> no filter
    dt = _last_bar_time_utc(m15)
    if dt is None:
        return True
    if bool(getattr(sh, "skip_weekends", False)) and dt.weekday() >= 5:
        return False
    hm = dt.strftime("%H:%M")
    # Overnight sessions (e.g. 20:00-04:00) wrap around midnight.
    if start <= end:
        return start <= hm < end
    return hm >= start or hm < end


def _in_kill_zone(m15: Mapping, config: Any) -> bool:
    """True when the last M15 bar falls inside a configured kill zone.

    Kill zones are a stricter entry filter layered on top of the session:
    entries only during high-liquidity windows (e.g. London / New York).
    No kill zones configured -> no filter.
    """
    zones = list(getattr(config, "kill_zones", None) or [])
    if not zones:
        return True
    dt = _last_bar_time_utc(m15)
    if dt is None:
        return True
    hm = dt.strftime("%H:%M")
    for z in zones:
        try:
            start, end = str(z).split("-")
        except ValueError:
            continue
        if start <= end:
            if start <= hm < end:
                return True
        elif hm >= start or hm < end:
            return True
    return False


def _htf_bias(h1: Mapping) -> str | None:
    """Bullish/bearish/None from H1 structure trend + EMA50 alignment."""
    smc = h1.get("smc") or {}
    structure = smc.get("structure") or {}
    trend = structure.get("trend")
    last_close = h1.get("last_close")
    ema50 = h1.get("ema50")
    if trend not in ("up", "down") or last_close is None or ema50 is None:
        return None
    if trend == "up" and last_close > ema50:
        return "bullish"
    if trend == "down" and last_close < ema50:
        return "bearish"
    return None


def _pick_sweep(sweeps: list, bias: str) -> dict | None:
    """Most recent sweep on the side matching the bias.

    Bullish bias wants a sweep of the swing LOW (sell-side liquidity taken
    before the move up); bearish bias wants a sweep of the swing HIGH.
    """
    want = "low" if bias == "bullish" else "high"
    cands = [s for s in sweeps if s.get("side") == want]
    if not cands:
        return None
    return max(cands, key=lambda s: int(s.get("bar_index", -1)))


def _pick_fvg(fvgs: list, sweep: dict, bias: str, atr: float | None,
              last_index: int) -> dict | None:
    """Freshest unfilled FVG of the right type, formed at/after the sweep."""
    want = "bullish" if bias == "bullish" else "bearish"
    sweep_idx = int(sweep.get("bar_index", -1))
    cands = []
    for f in fvgs:
        if f.get("type") != want:
            continue
        if bool(f.get("filled")):
            continue
        fidx = int(f.get("formed_index", -1))
        if fidx < sweep_idx:
            continue
        if last_index - fidx > FVG_MAX_AGE_BARS:
            continue
        if atr and atr > 0:
            size = abs(float(f.get("top", 0.0)) - float(f.get("bottom", 0.0)))
            if size < FVG_MIN_SIZE_ATR * atr:
                continue
        cands.append(f)
    if not cands:
        return None
    return max(cands, key=lambda f: int(f.get("formed_index", -1)))


def _build_trade(symbol: str, bias: str, sweep: dict, fvg: dict,
                 atr: float | None, last_close: float) -> Decision | None:
    """Construct entry/SL/TP for the setup. Returns None if invalid."""
    direction = "buy" if bias == "bullish" else "sell"
    entry = _round5(float(fvg.get("midpoint")))
    extreme = float(sweep.get("extreme"))
    if entry is None or entry <= 0 or atr is None or atr <= 0:
        return None

    # A limit entry must sit on the correct side of the market.
    if direction == "buy" and not entry < last_close:
        return None
    if direction == "sell" and not entry > last_close:
        return None

    if direction == "buy":
        sl = extreme - SL_BUFFER_ATR * atr
        min_sl_dist = SL_ATR_MIN_MULT * atr
        if entry - sl < min_sl_dist:
            sl = entry - min_sl_dist
        risk_dist = entry - sl
        tp = entry + REWARD_MULT * risk_dist
        valid = sl < entry < tp
    else:
        sl = extreme + SL_BUFFER_ATR * atr
        min_sl_dist = SL_ATR_MIN_MULT * atr
        if sl - entry < min_sl_dist:
            sl = entry + min_sl_dist
        risk_dist = sl - entry
        tp = entry - REWARD_MULT * risk_dist
        valid = tp < entry < sl
    if not valid or risk_dist <= 0:
        return None

    r_multiple = REWARD_MULT
    reasoning = (
        f"{bias} {PLAYBOOK_NAME}: H1 {bias} bias (structure + EMA50); "
        f"M15 swept {sweep.get('side')} {sweep.get('level')} "
        f"(extreme {extreme}, bar {sweep.get('bar_index')}); "
        f"{fvg.get('type')} FVG {fvg.get('bottom')}-{fvg.get('top')} "
        f"(formed bar {fvg.get('formed_index')}); "
        f"limit {entry}, SL {round(sl, 5)} (beyond sweep extreme, "
        f">= {SL_ATR_MIN_MULT}x ATR), TP {round(tp, 5)} ({r_multiple}R)."
    )
    return Decision(
        action="propose",
        symbol=symbol,
        direction=direction,
        entry_type="limit",
        price=entry,
        stop_loss=_round5(sl),
        take_profit=_round5(tp),
        risk_pct=SETUP_RISK_PCT,
        confidence=SETUP_CONFIDENCE,
        playbook=PLAYBOOK_NAME,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decide(snapshot: Mapping, config: Any = None) -> Decision:
    """Evaluate the snapshot against the rules and return a Decision.

    Deterministic: the same snapshot always yields the same Decision.
    No network, no randomness. Returns ``action="no_trade"`` unless every
    condition of the sweep+FVG setup is met.
    """
    symbol = str(snapshot.get("symbol", "")).upper()
    tfs = snapshot.get("timeframes") or {}
    h1 = tfs.get(BIAS_TIMEFRAME) or {}
    m15 = tfs.get(TRIGGER_TIMEFRAME) or {}

    if not h1.get("candles") or not m15.get("candles"):
        return _no_trade(symbol, f"missing {BIAS_TIMEFRAME}/{TRIGGER_TIMEFRAME} candles")

    if config is not None and not _in_session(m15, config):
        return _no_trade(symbol, "outside configured session hours")

    if config is not None and not _in_kill_zone(m15, config):
        return _no_trade(symbol, "outside kill zones")

    spread = snapshot.get("spread")
    if spread is not None:
        try:
            if float(spread) > MAX_SPREAD_POINTS:
                return _no_trade(symbol, f"spread {spread} > max {MAX_SPREAD_POINTS} points")
        except (TypeError, ValueError):
            pass

    bias = _htf_bias(h1)
    if bias is None:
        return _no_trade(symbol, "no clear H1 bias (structure trend + EMA50)")

    smc = m15.get("smc") or {}
    sweeps = smc.get("sweeps") or []
    # Positional indices are relative to the scanner's full frame; the
    # timeframe block ships the full bar count as "bars", so the last
    # positional index is bars - 1. The scanner already limits sweeps to
    # the recent lookback window, so any reported sweep is fresh.
    last_index = int(m15.get("bars") or len(m15.get("candles") or [])) - 1
    sweep = _pick_sweep(sweeps, bias)
    if sweep is None:
        return _no_trade(symbol, f"no recent liquidity sweep for {bias} bias")

    atr = m15.get("atr14")
    fvgs = smc.get("fvgs") or []
    fvg = _pick_fvg(fvgs, sweep, bias, atr, last_index)
    if fvg is None:
        return _no_trade(symbol, "no fresh unfilled FVG at/after the sweep")

    last_close = m15.get("last_close")
    if last_close is None:
        return _no_trade(symbol, "no last close on M15")
    trade = _build_trade(symbol, bias, sweep, fvg, atr, float(last_close))
    if trade is None:
        return _no_trade(symbol, "trade construction failed sanity checks")
    return trade


__all__ = [
    "BIAS_TIMEFRAME",
    "TRIGGER_TIMEFRAME",
    "SWEEP_LOOKBACK_BARS",
    "FVG_MAX_AGE_BARS",
    "FVG_MIN_SIZE_ATR",
    "SL_ATR_MIN_MULT",
    "SL_BUFFER_ATR",
    "REWARD_MULT",
    "MAX_SPREAD_POINTS",
    "SETUP_CONFIDENCE",
    "SETUP_RISK_PCT",
    "PLAYBOOK_NAME",
    "VALID_ACTIONS",
    "VALID_DIRECTIONS",
    "VALID_ENTRY_TYPES",
    "Decision",
    "decide",
]
