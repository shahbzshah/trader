"""Tests for app/signals.py — the signals-only pipeline.

Covers the report's sizing formula (lots = risk$ / (SL$ x 100) on gold),
the risk gate, de-duplication, the daily cap, the kill switch, and the
Telegram hand-off. No network, no orders, no MT5.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.brain import Decision
from app.config import load_config
from app.journal import Journal
from app.market_data import Tick
from app.signals import SignalEngine, size_signal
from tests.fake_client import FakeClient


def _cfg(tmp_path, **signals_overrides):
    cfg = load_config("config.yaml", env_path=tmp_path / ".env")
    cfg.paths.data_dir = str(tmp_path)
    cfg.paths.kill_switch_path = str(tmp_path / "KILL_SWITCH")
    if signals_overrides:
        cfg.signals = cfg.signals.model_copy(update=signals_overrides)
    return cfg


def _engine(tmp_path, *, telegram=None, **signals_overrides):
    cfg = _cfg(tmp_path, **signals_overrides)
    journal = Journal(Path(cfg.paths.data_dir) / "journal.db")
    client = FakeClient()
    # Tick at a realistic gold price; the buy-limit entry must sit
    # below the ask for the limit check to pass.
    client.tick = Tick(bid=3900.0, ask=3900.5, spread=50.0, time=1700049600)
    return SignalEngine(client, journal, cfg, telegram=telegram), cfg, journal


def _buy_decision(**overrides):
    base = dict(
        action="propose",
        symbol="XAUUSD",
        direction="buy",
        price=3899.0,          # limit entry below the ask (3900.5)
        stop_loss=3884.0,      # $15 SL
        take_profit=3929.0,    # 2R
        risk_pct=1.0,
        confidence=0.85,
        playbook="sweep_fvg_reclaim",
        reasoning="sweep + FVG reclaim",
        entry_type="limit",
    )
    base.update(overrides)
    return Decision(**base)


class _FakeTelegram:
    """Records sent messages instead of calling Telegram."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.messages: list[str] = []

    def send_message(self, text: str) -> bool:
        self.messages.append(text)
        return True


# --- sizing ----------------------------------------------------------------

def test_size_signal_report_example(tmp_path):
    """The report's example: $10k, 1% = $100 risk, $15 SL -> 0.06 lots."""
    cfg = _cfg(tmp_path)
    client = FakeClient()
    decision = _buy_decision()
    lots, risk_usd, per_dollar = size_signal(cfg, client, decision)
    assert lots == pytest.approx(0.06, abs=0.005)
    assert risk_usd == pytest.approx(100.0)
    # 0.06 lots x 100 oz/lot = $6.00 per $1 of gold move.
    assert per_dollar == pytest.approx(6.0, abs=0.5)


def test_size_signal_respects_risk_cap(tmp_path):
    cfg = _cfg(tmp_path)
    client = FakeClient()
    # 5% asked, 1% cap -> sized at 1%.
    decision = _buy_decision(risk_pct=5.0)
    lots, risk_usd, _ = size_signal(cfg, client, decision)
    assert risk_usd == pytest.approx(100.0)
    assert lots == pytest.approx(0.06, abs=0.005)


def test_size_signal_rejects_missing_sl(tmp_path):
    cfg = _cfg(tmp_path)
    client = FakeClient()
    decision = _buy_decision(stop_loss=None)
    with pytest.raises(ValueError, match="missing_entry_or_sl"):
        size_signal(cfg, client, decision)


# --- engine ----------------------------------------------------------------

def test_process_signals_and_alerts(tmp_path):
    telegram = _FakeTelegram()
    engine, cfg, journal = _engine(tmp_path, telegram=telegram)
    result = engine.process(_buy_decision(),
                            raw_llm_json=_buy_decision().model_dump_json())
    assert result.status == "signaled"
    assert result.signal is not None
    assert result.signal.lots == pytest.approx(0.06, abs=0.005)
    assert result.signal.risk_usd == pytest.approx(100.0)
    assert result.telegram_sent is True
    assert len(telegram.messages) == 1
    assert "GOLD SIGNAL — BUY XAUUSD" in telegram.messages[0]
    assert "3899.00" in telegram.messages[0]
    assert "3884.00" in telegram.messages[0]
    # Journaled as an approved propose decision; no order rows.
    decs = journal.list_decisions(limit=10)
    assert len(decs) == 1
    assert decs[0]["action"] == "propose"
    assert decs[0]["approved"] is True
    assert journal.list_orders(limit=10) == []


def test_process_duplicate_suppressed(tmp_path):
    telegram = _FakeTelegram()
    engine, cfg, journal = _engine(tmp_path, telegram=telegram)
    first = engine.process(_buy_decision())
    assert first.status == "signaled"
    second = engine.process(_buy_decision())
    assert second.status == "duplicate"
    assert len(telegram.messages) == 1
    # A different entry is a new signal, not a duplicate.
    third = engine.process(_buy_decision(price=3895.0))
    assert third.status == "signaled"
    assert len(telegram.messages) == 2


def test_process_kill_switch_rejects(tmp_path):
    telegram = _FakeTelegram()
    engine, cfg, journal = _engine(tmp_path, telegram=telegram)
    Path(cfg.paths.kill_switch_path).touch()
    result = engine.process(_buy_decision())
    assert result.status == "rejected"
    assert "kill_switch_engaged" in result.reject_reasons
    assert telegram.messages == []


def test_process_low_confidence_rejects(tmp_path):
    engine, cfg, journal = _engine(tmp_path)
    result = engine.process(_buy_decision(confidence=0.5))
    assert result.status == "rejected"
    assert any("confidence" in r for r in result.reject_reasons)


def test_process_daily_cap(tmp_path):
    telegram = _FakeTelegram()
    engine, cfg, journal = _engine(
        tmp_path, telegram=telegram, max_signals_per_day=1)
    first = engine.process(_buy_decision())
    assert first.status == "signaled"
    second = engine.process(_buy_decision(price=3895.0))
    assert second.status == "capped"
    assert "max_signals_per_day_reached" in second.reject_reasons
    assert len(telegram.messages) == 1


def test_process_no_trade_skipped(tmp_path):
    telegram = _FakeTelegram()
    engine, cfg, journal = _engine(tmp_path, telegram=telegram)
    decision = Decision(
        action="no_trade", symbol="XAUUSD", direction="buy",
        reasoning="no setup",
    )
    result = engine.process(decision)
    assert result.status == "skipped"
    assert telegram.messages == []
    decs = journal.list_decisions(limit=10)
    assert len(decs) == 1
    assert decs[0]["action"] == "no_trade"


def test_process_dry_run_journals_nothing(tmp_path):
    telegram = _FakeTelegram()
    engine, cfg, journal = _engine(tmp_path, telegram=telegram)
    result = engine.process(_buy_decision(), dry_run=True)
    assert result.status == "signaled"
    assert result.signal is not None
    assert telegram.messages == []
    assert journal.list_decisions(limit=10) == []


def test_process_limit_above_market_rejects(tmp_path):
    engine, cfg, journal = _engine(tmp_path)
    # Buy limit entry ABOVE the ask is not a valid limit.
    result = engine.process(_buy_decision(price=3901.0))
    assert result.status == "rejected"
    assert any("limit_price_not_below_market" in r for r in result.reject_reasons)


def test_process_without_telegram_journals_only(tmp_path):
    engine, cfg, journal = _engine(tmp_path, telegram=None)
    result = engine.process(_buy_decision())
    assert result.status == "signaled"
    assert result.telegram_sent is False
    assert journal.list_decisions(limit=10)[0]["approved"] is True
