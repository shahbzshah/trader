"""Tests for app/signals.py — the remaining plumbing.

The decision pipeline (SignalEngine, size_signal, risk gate, dedupe,
daily cap) was removed with the strategy on 2026-11-04. What remains
is the Signal record shape and the hypothetical-balance helper.
"""

from __future__ import annotations

from app.config import load_config
from app.signals import Signal, hypothetical_balance


def _signal(**overrides):
    base = dict(
        symbol="XAUUSD",
        direction="buy",
        entry=3899.0,
        stop_loss=3884.0,
        take_profit=3929.0,
        risk_pct=1.0,
        confidence=0.85,
        playbook="",
        reasoning="",
        source_feed="deriv",
        ts=1793000000,
        lots=0.06,
        risk_usd=100.0,
        per_dollar_usd=6.0,
    )
    base.update(overrides)
    return Signal(**base)


def test_signal_ts_label():
    sig = _signal()
    assert sig.ts_label == "2026-10-26 07:33"


def test_signal_fields_round_trip():
    sig = _signal(direction="sell", entry=3900.0)
    assert sig.direction == "sell"
    assert sig.entry == 3900.0
    assert sig.lots == 0.06
    assert sig.risk_usd == 100.0


def test_hypothetical_balance_default(tmp_path):
    cfg = load_config("config.yaml", env_path=tmp_path / ".env")
    assert hypothetical_balance(cfg) == 10000.0


def test_hypothetical_balance_configured(tmp_path):
    cfg = load_config("config.yaml", env_path=tmp_path / ".env")
    cfg.signals = cfg.signals.model_copy(
        update={"hypothetical_balance_usd": 5000.0})
    assert hypothetical_balance(cfg) == 5000.0


def test_hypothetical_balance_bad_value_falls_back():
    assert hypothetical_balance(object()) == 10000.0
