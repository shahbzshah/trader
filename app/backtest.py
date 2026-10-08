"""Minimal event-driven backtester: scanner -> strategy -> risk, no MT5.

Replays historical candles through the real production code path
(:mod:`app.scanner`, :mod:`app.strategy`, :mod:`app.risk`) with paper
sizing. No lookahead: a signal computed at the close of bar ``i`` can
only fill on bars ``i+1`` and later. Limit fills are conservative
(filled at the limit price when touched); when a single bar touches both
stop and target, the stop is assumed hit first.

Paper specs are approximations for FX/metals (5-digit FX, 2-digit gold).
P/L is tracked in R multiples converted via the risked amount, which is
the honest unit for a fixed-fractional system.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from app import risk as risk_mod
from app import scanner as scanner_mod
from app.strategy import decide


WARMUP_BARS = 60          # enough for EMA50 + structure lookback
MAX_PENDING_BARS = 5      # cancel unfilled limit orders after N bars
PAPER_START_EQUITY = 10000.0
# Rolling window fed to the snapshot builder each bar. The strategy only
# needs recent bars (FVG scan 40, sweep lookback 5, structure lookback 60,
# EMA50 needs 50); truncating keeps the backtest O(n) instead of O(n^2).
# Positional indices stay consistent because the strategy only compares
# them relatively within one snapshot.
SNAP_M15_BARS = 150
SNAP_H1_BARS = 80

# point / contract size / volume steps used for paper sizing.
PAPER_SPECS = {
    "EURUSD": {"point": 0.00001, "trade_contract_size": 100000.0,
               "volume_min": 0.01, "volume_step": 0.01, "volume_max": 100.0},
    "GBPUSD": {"point": 0.00001, "trade_contract_size": 100000.0,
               "volume_min": 0.01, "volume_step": 0.01, "volume_max": 100.0},
    "XAUUSD": {"point": 0.01, "trade_contract_size": 100.0,
               "volume_min": 0.01, "volume_step": 0.01, "volume_max": 100.0},
}
DEFAULT_SPEC = PAPER_SPECS["XAUUSD"]


def load_candles(csv_path: str | Path) -> pd.DataFrame:
    """Load candles from CSV with columns time,open,high,low,close,volume.

    ``time`` is epoch seconds (UTC), matching MT5 ``copy_rates`` output.
    """
    df = pd.read_csv(csv_path)
    need = {"time", "open", "high", "low", "close"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    if "volume" not in df.columns:
        df["volume"] = 0
    df = df.sort_values("time").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["time"] = df["time"].astype(int)
    return df[["time", "open", "high", "low", "close", "volume"]]


def resample_h1(df_m15: pd.DataFrame) -> pd.DataFrame:
    """Resample M15 candles to H1 (used for the strategy's bias timeframe)."""
    work = df_m15.set_index(
        pd.to_datetime(df_m15["time"], unit="s", utc=True)
    ).sort_index()
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    h1 = work.resample("1h").agg(agg).dropna().reset_index()
    h1 = h1.rename(columns={"index": "time"})
    # Resolution-independent datetime -> epoch seconds.
    ts = pd.to_datetime(h1["time"], utc=True)
    h1["time"] = ((ts - pd.Timestamp("1970-01-01", tz="UTC"))
                  // pd.Timedelta("1s")).astype(int)
    return h1[["time", "open", "high", "low", "close", "volume"]]


def _settle_bar(bar: pd.Series, pending: list, open_pos: list,
                journal_rows: list, trades: list) -> float:
    """Settle pending limit orders and open positions against one bar.

    Returns the realised P/L delta from positions closed on this bar.
    """
    pnl_delta = 0.0
    ts = int(bar["time"])

    # Pending limit fills (conservative: filled at the limit price).
    still_pending = []
    for o in pending:
        filled = (bar["low"] <= o["entry"] if o["direction"] == "buy"
                  else bar["high"] >= o["entry"])
        if filled:
            open_pos.append(o)
        elif ts - o["placed_ts"] > o["max_wait_secs"]:
            pass  # expired unfilled
        else:
            still_pending.append(o)
    pending[:] = still_pending

    # Open positions: SL / TP (both touched in one bar -> SL, conservative).
    still_open = []
    for p in open_pos:
        entry, sl, tp = p["entry"], p["sl"], p["tp"]
        if p["direction"] == "buy":
            sl_hit = bar["low"] <= sl
            tp_hit = bar["high"] >= tp
            exit_px = sl if sl_hit else (tp if tp_hit else None)
            risk_dist = entry - sl
            r = (exit_px - entry) / risk_dist if exit_px is not None else None
        else:
            sl_hit = bar["high"] >= sl
            tp_hit = bar["low"] <= tp
            exit_px = sl if sl_hit else (tp if tp_hit else None)
            risk_dist = sl - entry
            r = (entry - exit_px) / risk_dist if exit_px is not None else None
        if exit_px is None:
            still_open.append(p)
            continue
        pnl = r * p["risk_amount"]
        pnl_delta += pnl
        trades.append({"direction": p["direction"], "r": r, "pnl": pnl,
                       "ts": ts})
        journal_rows.append({"kind": "order_closed", "ts": ts, "profit": pnl})
    open_pos[:] = still_open
    return pnl_delta


def run_backtest(csv_path: str | Path, *, symbol: str = "XAUUSD",
                 timeframe: str = "M15", config: Any = None) -> dict:
    """Replay ``csv_path`` through scanner -> strategy -> risk.

    Returns a summary dict with trades, win rate, avg R, profit factor,
    max drawdown and equity stats.
    """
    symbol = symbol.upper()
    df = load_candles(csv_path)
    if len(df) < WARMUP_BARS + 10:
        raise ValueError(
            f"need at least {WARMUP_BARS + 10} candles, got {len(df)}")
    h1_full = resample_h1(df)
    spec = SimpleNamespace(**PAPER_SPECS.get(symbol, DEFAULT_SPEC))

    equity = PAPER_START_EQUITY
    peak = equity
    max_dd = 0.0
    day_start_equity = equity
    current_day: int | None = None

    pending: list = []
    open_pos: list = []
    journal_rows: list = []
    trades: list = []

    tf_minutes = 15 if timeframe.upper() == "M15" else 60

    for i in range(WARMUP_BARS, len(df)):
        bar = df.iloc[i]
        ts = int(bar["time"])
        day = ts // 86400
        if day != current_day:
            current_day = day
            day_start_equity = equity

        # 1. Settle with the new bar (fills / SL / TP).
        equity += _settle_bar(bar, pending, open_pos, journal_rows, trades)

        # Drawdown on the equity curve.
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)

        # 2. Signal from closed history only (bars 0..i), truncated to a
        # rolling window for speed. No lookahead: bar i is the latest close.
        hist = df.iloc[max(0, i + 1 - SNAP_M15_BARS):i + 1]
        h1_hist = h1_full[h1_full["time"] + 3600 <= ts].tail(SNAP_H1_BARS)
        snap = scanner_mod.build_snapshot(
            symbol,
            {timeframe.upper(): hist, "H1": h1_hist},
            account=None,
            positions=[],
        )
        decision = decide(snap, config)
        if decision.action != "propose":
            continue

        # 3. Real risk gate, paper account state.
        acct = risk_mod.AccountState(
            balance=equity,
            equity=equity,
            starting_balance_today=day_start_equity,
            trade_mode="demo",
            kill_switch_engaged=False,
        )
        positions_ns = [SimpleNamespace(symbol=symbol) for _ in open_pos]
        verdict = risk_mod.check_decision(
            decision, acct, positions_ns, journal_rows, config)
        if not verdict.approved:
            continue

        max_risk = float(getattr(getattr(config, "risk", None),
                                 "max_risk_pct_per_trade", 1.0)) if config else 1.0
        risk_pct = min(float(decision.risk_pct or 0.0), max_risk)
        lots = risk_mod.position_size(
            equity, risk_pct, decision.price, decision.stop_loss, spec)
        if lots <= 0:
            continue

        pending.append({
            "direction": decision.direction,
            "entry": float(decision.price),
            "sl": float(decision.stop_loss),
            "tp": float(decision.take_profit),
            "risk_amount": equity * risk_pct / 100.0,
            "lots": lots,
            "placed_ts": ts,
            "max_wait_secs": MAX_PENDING_BARS * tf_minutes * 60,
            "symbol": symbol,
        })
        journal_rows.append({"kind": "order_placed", "ts": ts, "profit": 0.0})

    wins = sum(1 for t in trades if t["r"] > 0)
    losses = sum(1 for t in trades if t["r"] <= 0)
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    avg_r = (sum(t["r"] for t in trades) / len(trades)) if trades else 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    return {
        "symbol": symbol,
        "timeframe": timeframe.upper(),
        "bars": len(df),
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": (wins / len(trades) * 100.0) if trades else 0.0,
        "avg_r": avg_r,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd,
        "final_equity": equity,
        "return_pct": (equity - PAPER_START_EQUITY) / PAPER_START_EQUITY * 100.0,
    }


__all__ = ["run_backtest", "load_candles", "resample_h1",
           "WARMUP_BARS", "PAPER_START_EQUITY"]
