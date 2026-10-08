"""Scanner: convert raw OHLC rates into rules-engine-ready snapshot dicts.

Public surface:
    build_snapshot(symbol, rates_by_tf, account, positions) -> dict
    scan_watchlist(client, config) -> list[dict]
    detect_fvg(df, ...) -> list[dict]
    detect_sweep(df, swing_high, swing_low, ...) -> list[dict]
    detect_structure(df, ...) -> dict

Snapshot per symbol:
- last close per timeframe
- EMA20, EMA50, EMA200 (per timeframe)
- RSI14, ATR14 (per timeframe)
- session high/low (current session = today's calendar day UTC by default)
- current spread
- recent swing high/low (last 20 bars on the primary timeframe)
- per-timeframe ``smc`` block: fair value gaps, liquidity sweeps,
  market-structure (BOS/CHoCH) trend and swing reference levels
- last 50 closed candles per timeframe (OHLC rounded)
- open positions for that symbol

All indicator math is hand-rolled with pandas so we don't pull in TA-Lib.
Functions are pure: no I/O, no datetime.now() reads. The caller passes
everything in.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import pandas as pd


# ---------------------------------------------------------------------------
# Indicator primitives
# ---------------------------------------------------------------------------


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average with standard adjust=True."""
    if period <= 0:
        raise ValueError("period must be positive")
    return series.ewm(span=period, adjust=True, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI (EMA of gains/losses).

    Convention:
      - pure uptrend  (avg_loss == 0) -> 100
      - pure downtrend (avg_gain == 0) -> 0
      - perfectly flat (both == 0)   -> 50
    """
    if period <= 0:
        raise ValueError("period must be positive")
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing: alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    # Default value for any position is 50 (neutral).
    out = pd.Series(50.0, index=series.index)
    # Pure uptrend (avg_loss == 0) -> 100.
    up_mask = (avg_loss == 0) & (avg_gain > 0)
    out = out.where(~up_mask, 100.0)
    # Pure downtrend (avg_gain == 0) -> 0.
    down_mask = (avg_gain == 0) & (avg_loss > 0)
    out = out.where(~down_mask, 0.0)
    # Otherwise compute the standard RSI.
    both_mask = (avg_gain > 0) & (avg_loss > 0)
    rs = avg_gain[both_mask] / avg_loss[both_mask]
    out.loc[both_mask] = 100.0 - (100.0 / (1.0 + rs))
    return out


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    if period <= 0:
        raise ValueError("period must be positive")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Rate-frame helpers
# ---------------------------------------------------------------------------


def _rates_to_df(rates: Any) -> pd.DataFrame:
    """Convert a Rates dataclass or list[dict] into a DataFrame sorted by time."""
    if rates is None:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close",
                                     "tick_volume", "spread", "real_volume"])
    rows: list[dict]
    if hasattr(rates, "rates"):
        rows = list(rates.rates)
    elif isinstance(rates, pd.DataFrame):
        rows_df = rates
        if rows_df.empty:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close",
                                         "tick_volume", "spread", "real_volume"])
        rows_df = rows_df.sort_values("time").reset_index(drop=True)
        for col in ("open", "high", "low", "close"):
            if col in rows_df.columns:
                rows_df[col] = rows_df[col].astype(float)
        return rows_df
    elif isinstance(rates, list):
        # Each element should be a dict; bail out to empty if not.
        if not rates or not isinstance(rates[0], dict):
            return pd.DataFrame(columns=["time", "open", "high", "low", "close",
                                         "tick_volume", "spread", "real_volume"])
        rows = rates
    else:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close",
                                     "tick_volume", "spread", "real_volume"])

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("time").reset_index(drop=True)
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    return df


def _round(value: float | None, ndigits: int = 5) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), ndigits)


def _session_bounds(times: pd.Series, session_start_hour_utc: int = 0) -> tuple[int | None, int | None]:
    """Return (start_ts, end_ts) of the current session.

    A "session" is the current UTC calendar day by default. Returns
    (None, None) if no timestamps are available.
    """
    if times is None or len(times) == 0:
        return None, None
    last_ts = int(times.iloc[-1])
    # Build the UTC day boundaries in seconds relative to last_ts.
    last_dt = pd.to_datetime(last_ts, unit="s", utc=True)
    day_start = pd.Timestamp(year=last_dt.year, month=last_dt.month,
                              day=last_dt.day, tz="UTC")
    if session_start_hour_utc:
        # If session_start_hour_utc > 0, the session starts that hour today;
        # otherwise it started at midnight UTC.
        day_start = day_start.replace(hour=session_start_hour_utc)
    day_end = day_start + pd.Timedelta(hours=24)
    return int(day_start.timestamp()), int(day_end.timestamp())


def _swing_highs_lows(close: pd.Series, high: pd.Series, low: pd.Series,
                       lookback: int = 20, exclude_last: int = 0) -> tuple[float | None, float | None]:
    """Return (swing_high, swing_low) over the last ``lookback`` bars.

    "Swing" = the max(high) / min(low) over that window. Last-bar-anchored
    so the caller gets a recent reference, not the all-time extreme.
    ``exclude_last`` drops that many of the most recent bars first, which
    is what sweep detection needs: the reference level must pre-date the
    bars being tested for a sweep.
    """
    if len(close) == 0:
        return None, None
    if exclude_last:
        if exclude_last >= len(high):
            return None, None
        high = high.iloc[:-exclude_last]
        low = low.iloc[:-exclude_last]
    window = min(lookback, len(high))
    tail_h = high.tail(window)
    tail_l = low.tail(window)
    return _round(float(tail_h.max())), _round(float(tail_l.min()))


# ---------------------------------------------------------------------------
# Smart Money Concepts detectors (pure, unit-testable)
# ---------------------------------------------------------------------------
#
# Conventions used throughout:
#   - ``bar_index`` / ``formed_index`` are POSITIONAL indices into the
#     DataFrame that was passed in (0 == first row). Callers comparing
#     indices must use values from the same frame.
#   - A bullish FVG at bar i means low[i] > high[i-2] (displacement up,
#     leaving an unfilled gap between candle i-2's high and candle i's low).


def detect_fvg(df: pd.DataFrame, min_size_atr: float = 0.25,
               max_age_bars: int = 40) -> list[dict]:
    """Detect fair value gaps (3-candle displacement gaps).

    Returns a list of dicts, oldest first, each with:
        type: "bullish" | "bearish"
        top / bottom / midpoint: gap boundaries (float)
        formed_index: positional index of the displacing (3rd) candle
        filled: True when a later bar traded fully through the gap
        size_atr: gap height expressed in ATR14 units

    Only gaps formed within ``max_age_bars`` of the last bar and at least
    ``min_size_atr`` x ATR14 tall are returned, so the snapshot stays lean.
    """
    n = len(df)
    if n < 3:
        return []
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    atr_s = atr(high, low, df["close"].astype(float), 14)

    # Vectorised gap detection: bullish gap at i <-> low[i] > high[i-2].
    prev2_high = high.shift(2)
    prev2_low = low.shift(2)
    bull_mask = low > prev2_high
    bear_mask = high < prev2_low

    out: list[dict] = []
    for i in range(2, n):
        a = float(atr_s.iloc[i])
        if not (a > 0):
            continue
        if bool(bull_mask.iloc[i]):
            top = float(low.iloc[i])
            bottom = float(prev2_high.iloc[i])
            size = top - bottom
            if size < min_size_atr * a:
                continue
            filled = bool((low.iloc[i + 1:] <= bottom).any()) if i + 1 < n else False
            out.append({
                "type": "bullish",
                "top": _round(top),
                "bottom": _round(bottom),
                "midpoint": _round((top + bottom) / 2.0),
                "formed_index": i,
                "filled": filled,
                "size_atr": _round(size / a, 3),
            })
        elif bool(bear_mask.iloc[i]):
            top = float(prev2_low.iloc[i])
            bottom = float(high.iloc[i])
            size = top - bottom
            if size < min_size_atr * a:
                continue
            filled = bool((high.iloc[i + 1:] >= top).any()) if i + 1 < n else False
            out.append({
                "type": "bearish",
                "top": _round(top),
                "bottom": _round(bottom),
                "midpoint": _round((top + bottom) / 2.0),
                "formed_index": i,
                "filled": filled,
                "size_atr": _round(size / a, 3),
            })

    cutoff = (n - 1) - max_age_bars
    return [f for f in out if f["formed_index"] >= cutoff]


def detect_sweep(df: pd.DataFrame, swing_high: float | None,
                 swing_low: float | None, lookback: int = 5) -> list[dict]:
    """Detect liquidity sweeps within the last ``lookback`` bars.

    A sweep of the swing HIGH: a bar whose wick traded above the level but
    whose close finished back below it (stop-hunt of buy-side liquidity).
    Mirror logic applies to the swing LOW.

    Returns a list of dicts, oldest first:
        side: "high" | "low"   (which level was swept)
        level: the swept reference level
        bar_index: positional index of the sweeping bar
        extreme: the wick extreme beyond the level
    """
    n = len(df)
    if n == 0 or (swing_high is None and swing_low is None):
        return []
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    out: list[dict] = []
    for i in range(max(0, n - lookback), n):
        h, l, c = float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i])
        if swing_high is not None and h > swing_high and c < swing_high:
            out.append({
                "side": "high",
                "level": _round(swing_high),
                "bar_index": i,
                "extreme": _round(h),
            })
        if swing_low is not None and l < swing_low and c > swing_low:
            out.append({
                "side": "low",
                "level": _round(swing_low),
                "bar_index": i,
                "extreme": _round(l),
            })
    return out


def _fractal_swings(df: pd.DataFrame, left: int = 2, right: int = 2,
                    lookback: int = 60) -> list[dict]:
    """Fractal swing points: bar i is a swing high when high[i] is the max
    of high[i-left .. i+right] (mirror for lows). Returns dicts sorted by
    positional index: {type: "high"|"low", price, index}."""
    n = len(df)
    if n < left + right + 1:
        return []
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    start = max(0, n - lookback)
    swings: list[dict] = []
    for i in range(max(start, left), n - right):
        h, l = float(high.iloc[i]), float(low.iloc[i])
        window_h = high.iloc[i - left:i + right + 1]
        window_l = low.iloc[i - left:i + right + 1]
        if h >= float(window_h.max()) and (window_h == h).sum() == 1:
            swings.append({"type": "high", "price": _round(h), "index": i})
        if l <= float(window_l.min()) and (window_l == l).sum() == 1:
            swings.append({"type": "low", "price": _round(l), "index": i})
    swings.sort(key=lambda s: s["index"])
    return swings


def detect_structure(df: pd.DataFrame, left: int = 2, right: int = 2,
                     lookback: int = 60) -> dict:
    """Market structure: trend + most recent BOS/CHoCH.

    Trend comes from the swing sequence (higher highs + higher lows = up,
    lower highs + lower lows = down, otherwise range), then refined by
    walking closes through the confirmed swings: a close beyond the last
    swing extreme in the trend's direction is a BOS (break of structure);
    against it, a CHoCH (change of character), which flips the trend.

    Returns:
        trend: "up" | "down" | "range"
        swings: the fractal swing list used
        last_break: {type: "BOS"|"CHoCH", direction: "bullish"|"bearish",
                     index, price} | None
    """
    n = len(df)
    empty = {"trend": "range", "swings": [], "last_break": None}
    if n < left + right + 3:
        return empty
    close = df["close"].astype(float)
    swings = _fractal_swings(df, left, right, lookback)
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]

    def _seq_trend() -> str:
        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1]["price"] > highs[-2]["price"]
            hl = lows[-1]["price"] > lows[-2]["price"]
            lh = highs[-1]["price"] < highs[-2]["price"]
            ll = lows[-1]["price"] < lows[-2]["price"]
            if hh and hl:
                return "up"
            if lh and ll:
                return "down"
        return "range"

    trend = _seq_trend()
    last_break: dict | None = None
    start = max(1, n - lookback)
    for i in range(start, n):
        ref_h = next((s for s in reversed(highs) if s["index"] < i), None)
        ref_l = next((s for s in reversed(lows) if s["index"] < i), None)
        c_now, c_prev = float(close.iloc[i]), float(close.iloc[i - 1])
        direction = None
        if ref_h is not None and c_now > ref_h["price"] >= c_prev:
            direction = "bullish"
        elif ref_l is not None and c_now < ref_l["price"] <= c_prev:
            direction = "bearish"
        if direction is None:
            continue
        if trend == "range":
            kind = "BOS"  # break out of range establishes the new trend
        else:
            with_trend = (direction == "bullish") == (trend == "up")
            kind = "BOS" if with_trend else "CHoCH"
        last_break = {
            "type": kind,
            "direction": direction,
            "index": i,
            "price": _round(c_now),
        }
        trend = "up" if direction == "bullish" else "down"
    return {"trend": trend, "swings": swings, "last_break": last_break}


# ---------------------------------------------------------------------------
# Per-timeframe summary
# ---------------------------------------------------------------------------


def _summarize_timeframe(df: pd.DataFrame) -> dict:
    """Build the per-timeframe block: indicators + 50-bar tail + last close."""
    block: dict[str, Any] = {
        "bars": int(len(df)),
        "last_close": None,
        "ema20": None,
        "ema50": None,
        "ema200": None,
        "rsi14": None,
        "atr14": None,
        "candles": [],
    }
    if df.empty:
        return block

    close = df["close"]
    high = df["high"]
    low = df["low"]

    block["last_close"] = _round(float(close.iloc[-1]))
    block["ema20"] = _round(float(ema(close, 20).iloc[-1])) if len(close) >= 20 else None
    block["ema50"] = _round(float(ema(close, 50).iloc[-1])) if len(close) >= 50 else None
    block["ema200"] = _round(float(ema(close, 200).iloc[-1])) if len(close) >= 200 else None
    block["rsi14"] = _round(float(rsi(close, 14).iloc[-1])) if len(close) >= 15 else None
    block["atr14"] = _round(float(atr(high, low, close, 14).iloc[-1])) if len(close) >= 15 else None

    tail = df.tail(50)
    block["candles"] = [
        {
            "t": int(r["time"]),
            "o": _round(float(r["open"])),
            "h": _round(float(r["high"])),
            "l": _round(float(r["low"])),
            "c": _round(float(r["close"])),
        }
        for _, r in tail.iterrows()
    ]

    # Smart Money Concepts block: FVGs, liquidity sweeps and structure.
    # Indices are positional within the full frame passed in.
    ref_high, ref_low = _swing_highs_lows(close, high, low,
                                          lookback=20, exclude_last=5)
    block["smc"] = {
        "fvgs": detect_fvg(df),
        "sweeps": detect_sweep(df, ref_high, ref_low, lookback=5),
        "structure": detect_structure(df),
        "swing_high_ref": ref_high,
        "swing_low_ref": ref_low,
    }
    return block


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_snapshot(
    symbol: str,
    rates_by_tf: Mapping[str, Any],
    account: Any,
    positions: Iterable[Any],
) -> dict:
    """Build a deterministic snapshot dict for one symbol.

    Args:
        symbol: the symbol name (uppercased internally).
        rates_by_tf: mapping of timeframe label (e.g. "M15", "H1") to either
            a ``Rates`` object or a list of OHLC dicts.
        account: an ``AccountInfo``-like object with ``balance`` and ``equity``
            (or ``None`` if unknown).
        positions: any iterable of ``Position``-like objects for this symbol.

    Returns:
        A plain dict for the rules engine. Floating-point values are
        rounded to 5 decimal places so the output is stable.
    """
    symbol = symbol.upper()
    snap: dict[str, Any] = {
        "symbol": symbol,
        "timeframes": {},
        "session": {"start_ts": None, "end_ts": None, "high": None, "low": None},
        "spread": None,
        "swing_high": None,
        "swing_low": None,
        "open_positions": [],
        "account": {
            "balance": _round(getattr(account, "balance", None)),
            "equity": _round(getattr(account, "equity", None)),
            "trade_mode": getattr(account, "trade_mode", None),
        },
    }

    primary_df: pd.DataFrame | None = None
    primary_tf: str | None = None

    for tf in sorted(rates_by_tf.keys()):
        df = _rates_to_df(rates_by_tf[tf])
        snap["timeframes"][tf] = _summarize_timeframe(df)
        # The first timeframe alphabetically is treated as primary for swing.
        # We use the first one the caller supplied with enough data.
        if primary_df is None and not df.empty:
            primary_df = df
            primary_tf = tf

    if primary_df is not None:
        times = primary_df["time"]
        start_ts, end_ts = _session_bounds(times)
        snap["session"]["start_ts"] = start_ts
        snap["session"]["end_ts"] = end_ts
        if start_ts is not None:
            mask = (times >= start_ts) & (times < end_ts)
            if mask.any():
                session_df = primary_df[mask]
                snap["session"]["high"] = _round(float(session_df["high"].max()))
                snap["session"]["low"] = _round(float(session_df["low"].min()))
        sh, sl = _swing_highs_lows(
            primary_df["close"], primary_df["high"], primary_df["low"], lookback=20
        )
        snap["swing_high"] = sh
        snap["swing_low"] = sl
        # spread from the primary timeframe's last bar
        if "spread" in primary_df.columns and not primary_df.empty:
            snap["spread"] = int(primary_df["spread"].iloc[-1])
        snap["primary_timeframe"] = primary_tf

    for pos in positions:
        snap["open_positions"].append(
            {
                "ticket": int(getattr(pos, "ticket")),
                "direction": getattr(pos, "direction", None),
                "volume": _round(float(getattr(pos, "volume", 0.0))),
                "price_open": _round(float(getattr(pos, "price_open", 0.0))),
                "price_current": _round(float(getattr(pos, "price_current", 0.0))),
                "sl": _round(float(getattr(pos, "sl", 0.0))),
                "tp": _round(float(getattr(pos, "tp", 0.0))),
                "profit": _round(float(getattr(pos, "profit", 0.0))),
            }
        )

    return snap


def scan_watchlist(client: Any, config: Any) -> list[dict]:
    """Scan every symbol in ``config.symbols`` over every timeframe in
    ``config.timeframes`` and return a list of snapshot dicts.

    This is the only function in this module that touches I/O (the MT5
    client). Indicators stay pure inside ``build_snapshot``.

    Returns an empty list (no exception) for any symbol whose rates fetch
    fails, so one broken symbol doesn't kill the whole scan.
    """
    history_count = 250  # enough for EMA200 with margin
    snapshots: list[dict] = []

    account = None
    try:
        account = client.account_info()
    except Exception:  # noqa: BLE001 - account info is best-effort
        account = None

    for symbol in config.symbols:
        # Best-effort: ensure the symbol is visible in Market Watch on a real
        # terminal so live rates/ticks can be read. Fakes may lack this method.
        if hasattr(client, "select_symbol"):
            try:
                client.select_symbol(symbol)
            except Exception:  # noqa: BLE001
                pass
        rates_by_tf: dict[str, Any] = {}
        for tf in config.timeframes:
            try:
                rates_by_tf[tf] = client.copy_rates(symbol, tf, history_count)
            except Exception:  # noqa: BLE001 - per-symbol resilience
                rates_by_tf[tf] = []

        spread = None
        try:
            tick = client.symbol_tick(symbol)
            spread = _round(tick.spread)
        except Exception:  # noqa: BLE001
            spread = None

        try:
            positions = client.positions(symbol=symbol)
        except Exception:  # noqa: BLE001
            positions = []

        snap = build_snapshot(symbol, rates_by_tf, account, positions)
        if spread is not None:
            snap["spread"] = spread
        snapshots.append(snap)

    return snapshots


__all__ = [
    "atr",
    "build_snapshot",
    "detect_fvg",
    "detect_structure",
    "detect_sweep",
    "ema",
    "rsi",
    "scan_watchlist",
]