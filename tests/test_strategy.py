"""Tests for app/strategy.py — Phil v2 + RUDY trend filter ("V2 pair").

The frozen-data test is the hard gate: the bot module, run over the
frozen Deriv frxXAUUSD M15 file, must regenerate the backtested
counts exactly (Phil 190, RUDY 772, V2 151, same bar epochs) and the
$10-stop trade engine must reproduce the published stop-variant
numbers (139 trades, 20.1% win, +$7.60/trade, +$1,056.61 total).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.strategy import (
    evaluate_last_bar,
    phil_v2_signals,
    run_structure,
    scan_all,
    simulate_stop10,
    trade_stats,
)

FROZEN = Path.home() / "workspace/trading/phil-backtest/gold-m15.json"
RUDY_RESULTS = (
    Path.home() / "workspace/trading/rudy-backtest/rudy_results.json"
)


def _load_frozen():
    if not FROZEN.exists():
        pytest.skip(f"frozen data not present: {FROZEN}")
    bars = json.loads(FROZEN.read_text(encoding="utf-8"))
    return (
        [float(b["close"]) for b in bars],
        [float(b["high"]) for b in bars],
        [float(b["low"]) for b in bars],
        [float(b["open"]) for b in bars],
        [int(b["epoch"]) for b in bars],
    )


def test_frozen_regeneration_exact():
    """Port check: Phil 190, RUDY 772, V2 151 — same bar epochs as
    the backtest — and the $10-stop engine reproduces 139 trades,
    20.1% win, +$7.60/trade, +$1,056.61 total."""
    closes, highs, lows, opens, epochs = _load_frozen()
    scan = scan_all(closes, highs, lows)

    assert len(scan["phil"]) == 190
    assert len(scan["rudy"]) == 772
    assert len(scan["v2"]) == 151
    assert sum(1 for s in scan["v2"] if s["dir"] == 1) == 78
    assert sum(1 for s in scan["v2"] if s["dir"] == -1) == 73

    # Epoch-exactness against the backtest's own RUDY signal list.
    ref = json.loads(RUDY_RESULTS.read_text(encoding="utf-8"))
    ref_bars = [s["bar"] for s in ref["signals"] if s["scope"] == "internal"]
    assert [s["bar"] for s in scan["rudy"]] == ref_bars

    trades, n_breach = simulate_stop10(
        [s["bar"] for s in scan["v2"]],
        [s["dir"] for s in scan["v2"]],
        closes, highs, lows, opens,
    )
    stats = trade_stats(trades)
    assert stats["trades"] == 139
    assert stats["win_rate"] == pytest.approx(20.1, abs=0.05)
    assert stats["expectancy"] == pytest.approx(7.60, abs=0.01)
    assert stats["total"] == pytest.approx(1056.61, abs=1.0)
    assert stats["stop_outs"] == 108
    assert stats["flip_exits"] == 31
    assert n_breach == 25


def test_evaluate_last_bar_on_frozen_prefixes():
    """The live path (last closed bar only) over frozen prefixes."""
    closes, highs, lows, opens, epochs = _load_frozen()

    def _bars(upto: int) -> list[dict]:
        return [
            {"time": epochs[i], "open": opens[i], "high": highs[i],
             "low": lows[i], "close": closes[i]}
            for i in range(upto)
        ]

    # First V2 buy (bar 499, epoch 1772598600, close 5161.77).
    sig = evaluate_last_bar(_bars(500))
    assert sig is not None
    assert sig["dir"] == 1
    assert sig["epoch"] == 1772598600
    assert sig["close"] == pytest.approx(5161.77)
    assert sig["itrend"] == 1

    # A Phil triangle whose RUDY trend disagrees (bar 702, sell with
    # itrend +1) must NOT fire.
    assert evaluate_last_bar(_bars(703)) is None

    # An ordinary bar (500) is not a signal.
    assert evaluate_last_bar(_bars(501)) is None

    # Warmup: 300 bars or fewer can never signal.
    assert evaluate_last_bar(_bars(300)) is None


def test_phil_v2_cross_on_synthetic_series():
    """Flat history then a $10 step: exactly one buy cross (and one
    sell on the mirror). The step puts 8/13 $0.79 apart on the cross
    bar itself — a smooth ramp would not, because the cross happens
    before the fast pair has separated, which is the gate working."""
    closes = [4000.0] * 350 + [4010.0] * 60
    sigs = phil_v2_signals(closes)
    assert len(sigs) == 1
    assert sigs[0]["dir"] == 1
    assert sigs[0]["bar"] == 350
    closes_down = [4000.0] * 350 + [3990.0] * 60
    sigs_down = phil_v2_signals(closes_down)
    assert len(sigs_down) == 1
    assert sigs_down[0]["dir"] == -1
    assert sigs_down[0]["bar"] == 350


def test_run_structure_synthetic_bullish_break():
    """Spike high, double bottom, rally through the spike high: an
    internal bullish break must fire and itrend must become +1.

    Construction notes (Pine pivot semantics): a pivot high needs a
    strictly higher bar than its 5 left neighbours, so the spike must
    stand above a flat top; the bullish block also needs TWO recorded
    internal lows, hence the double bottom."""
    closes = (
        [100.0] * 10
        + [101.0]  # spike high (pivot high 101.5)
        + [99.0, 97.0, 95.0, 93.0, 91.0, 90.0]  # decline
        + [90.0] * 7  # first bottom flattens -> pivot low 89.5
        + [92.0, 94.0, 96.0, 95.0, 93.0, 92.0, 91.0]  # bounce, 2nd dip
        + [91.0] * 7  # second bottom flattens -> pivot low 90.5
        + [93.0, 95.0, 97.0, 99.0, 100.0, 101.0, 102.0, 103.0]  # rally
        + [103.0] * 8
    )
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    events, itrend_log = run_structure(closes, highs, lows)
    internal_buys = [e for e in events
                     if e[3] == "internal" and e[1] == 1]
    assert internal_buys, "expected an internal bullish break"
    assert itrend_log[internal_buys[-1][0]] == 1
    assert itrend_log[-1] == 1


def test_simulate_stop10_synthetic_stop_and_flip():
    """Stop fill at the stop price (-$10.50 with spread); a flip exit
    at the opposite signal's close; same-direction ignored."""
    closes = [100.0, 105.0, 90.0, 95.0, 96.0]
    highs = [100.5, 105.5, 90.5, 95.5, 96.5]
    lows = [99.5, 104.5, 89.5, 94.5, 95.5]
    opens = [100.0, 104.0, 105.0, 90.0, 95.0]
    # Buy bar 0, sell bar 3: bar 2 stops the long out first.
    trades, _ = simulate_stop10([0, 3], [1, -1], closes, highs, lows,
                                opens, warmup=0)
    assert len(trades) == 2
    assert trades[0][4] == "stop"
    assert trades[0][3] == pytest.approx(-10.50)
    # The short opened at bar 3 is never stopped (bar 4 tops at 96.5,
    # below its $105 stop) and closes at end of data.
    assert trades[1][4] == "eod"
    assert trades[1][3] == pytest.approx(-1.50)

    # No stop in the way: buy bar 0, sell bar 1 -> flip exit.
    closes2 = [100.0, 102.0, 103.0]
    highs2 = [100.5, 102.5, 103.5]
    lows2 = [99.5, 101.5, 102.5]
    opens2 = [100.0, 101.0, 102.0]
    trades2, _ = simulate_stop10([0, 1], [1, -1], closes2, highs2,
                                 lows2, opens2, warmup=0)
    assert trades2[0][4] == "flip"
    # exit 102 - 0.25, entry 100 + 0.25 -> +1.50
    assert trades2[0][3] == pytest.approx(1.50)


def test_strategy_config_defaults(tmp_path):
    from app.config import load_config

    cfg = load_config("config.yaml", env_path=tmp_path / ".env")
    assert cfg.strategy.name == "phil-rudy-v2"
    assert cfg.strategy.phil_rudy.max_dist_usd == pytest.approx(20.0)
    assert cfg.strategy.phil_rudy.min_sep_813_usd == pytest.approx(0.75)
    assert cfg.strategy.phil_rudy.stop_usd == pytest.approx(10.0)
    assert cfg.strategy.phil_rudy.warmup_bars == 300
    assert cfg.strategy.phil_rudy.history_bars == 400
    assert cfg.risk.breakeven_enabled is False


def test_format_signal_text_no_take_profit():
    from app.signals import Signal
    from app.telegram import format_signal_text

    signal = Signal(
        symbol="XAUUSD", direction="sell", entry=5161.77,
        stop_loss=5171.77, take_profit=None, risk_pct=1.0,
        confidence=0.48, playbook="phil-rudy-v2",
        reasoning="Phil v2 SELL test", source_feed="deriv",
        ts=1772598600, lots=0.10, risk_usd=100.0, per_dollar_usd=10.0,
    )
    text = format_signal_text(signal)
    assert "GOLD SIGNAL — SELL XAUUSD" in text
    assert "Stop loss: 5171.77" in text
    assert "Take profit: none — exit at opposite signal (flip)" in text
    assert "0.10 lots" in text
    # No invented R:R anywhere.
    assert "R)" not in text
