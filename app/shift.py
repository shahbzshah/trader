"""Market-shift (bell) detector for the Gold Watch app.

Mirrors ``~/workspace/trading/shift-signal-5m.pine`` and the backtest
in ``~/workspace/trading/shift-signal/shift_backtest.py`` exactly —
the Pine script is the source of truth; do not redesign the rule
here.

    RSI(14) Wilder on 5m closes of frxXAUUSD.
    ARM   RSI >= 70 (sell side) or RSI <= 30 (buy side).
    FIRE  while armed, the first CLOSED bar with RSI back inside
          (< 70 / > 30) AND closing in the reversal direction
          (down close = SELL shift, up close = BUY shift).
    RESET after a fire, or when RSI drifts to 50 (sell arm clears
          when RSI < 50, buy arm when RSI > 50).

A shift event is identified by its firing bar's epoch:
``{"epoch", "direction", "price", "rsi"}``.

Pure stdlib — no numpy/pandas needed for 300 bars.
"""

from __future__ import annotations

RSI_LEN = 14
OB_LEVEL = 70.0
OS_LEVEL = 30.0
MID_LEVEL = 50.0
# Bars skipped before signals are counted: the RSI seed needs 14
# bars; 100 matches the backtest's warmup exactly (the anchor test
# replays the frozen file with this value and must reproduce its
# 492/262/230 counts).
WARMUP_BARS = 100


def wilder_rsi(closes: list[float], length: int = RSI_LEN) -> list[float]:
    """Wilder RSI, seeded exactly like the backtest (and Pine's
    ta.rsi): the first ``length`` bars read 100.0 — no gains/losses
    averaged yet — then the average gain/loss of bars 1..length
    seeds the Wilder smoothing from bar ``length + 1`` on."""
    n = len(closes)
    if n == 0:
        return []
    if n <= length:
        return [100.0] * n
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        d = closes[i] - closes[i - 1]
        if d > 0:
            gains[i] = d
        elif d < 0:
            losses[i] = -d
    avg_gain = sum(gains[1:length + 1]) / length
    avg_loss = sum(losses[1:length + 1]) / length
    rsi = [100.0] * n
    for i in range(length, n):
        if i > length:
            avg_gain = (avg_gain * (length - 1) + gains[i]) / length
            avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def detect_shifts(candles: list[dict],
                  warmup: int = WARMUP_BARS) -> tuple[list[dict], dict]:
    """Replay the shift rule over ``candles`` (dicts with ``time`` or
    ``epoch`` plus ``open``/``close``, oldest first).

    Returns ``(events, state)``: events oldest-first, each
    ``{"epoch", "direction", "price", "rsi"}``; state is
    ``{"rsi", "armed", "last_close", "last_bar_epoch"}`` with
    ``armed`` one of ``"none"`` / ``"sell"`` / ``"buy"`` and ``rsi``
    the RSI of the last bar (None when there are no candles).
    """
    epochs = [int(c.get("time", c.get("epoch", 0))) for c in candles]
    opens = [float(c["open"]) for c in candles]
    closes = [float(c["close"]) for c in candles]
    n = len(candles)
    if n == 0:
        return [], {"rsi": None, "armed": "none", "last_close": None,
                    "last_bar_epoch": None}
    rsi = wilder_rsi(closes)
    armed_s = False
    armed_l = False
    events: list[dict] = []
    for i in range(max(warmup, 0), n):
        if rsi[i] >= OB_LEVEL:
            armed_s = True
        if rsi[i] <= OS_LEVEL:
            armed_l = True
        if rsi[i] < MID_LEVEL:
            armed_s = False
        if rsi[i] > MID_LEVEL:
            armed_l = False
        if armed_s and rsi[i] < OB_LEVEL and closes[i] < opens[i]:
            events.append({"epoch": epochs[i], "direction": "SELL",
                           "price": round(closes[i], 2),
                           "rsi": round(rsi[i], 2)})
            armed_s = False
        elif armed_l and rsi[i] > OS_LEVEL and closes[i] > opens[i]:
            events.append({"epoch": epochs[i], "direction": "BUY",
                           "price": round(closes[i], 2),
                           "rsi": round(rsi[i], 2)})
            armed_l = False
    state = {
        "rsi": round(rsi[-1], 2),
        "armed": "sell" if armed_s else ("buy" if armed_l else "none"),
        "last_close": round(closes[-1], 2),
        "last_bar_epoch": epochs[-1],
    }
    return events, state


__all__ = ["detect_shifts", "wilder_rsi", "RSI_LEN", "OB_LEVEL",
           "OS_LEVEL", "WARMUP_BARS"]
