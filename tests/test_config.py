"""Tests for app/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (
    AppConfig,
    EnvSecrets,
    RiskConfig,
    SessionHoursConfig,
    load_config,
)


def test_load_config_from_real_yaml(config_yaml_path, env_path):
    cfg = load_config(config_yaml_path, env_path)
    assert isinstance(cfg, AppConfig)
    assert cfg.mode == "demo"
    assert cfg.symbols == ["XAUUSD"]
    assert cfg.timeframes == ["M15", "H1"]
    assert cfg.schedule.scan_interval_minutes == 15
    assert cfg.session_hours.start == "08:00"
    assert cfg.session_hours.end == "20:00"
    assert cfg.risk.max_risk_pct_per_trade == 1.0
    assert cfg.risk.max_daily_loss_pct == 3.0
    assert cfg.risk.max_open_trades == 3
    assert cfg.risk.max_trades_per_day == 5
    assert cfg.dashboard.port == 8765
    assert cfg.signals.hypothetical_balance_usd == 10000.0
    assert cfg.signals.feed == "auto"
    assert cfg.signals.max_signals_per_day == 5
    assert not hasattr(cfg, "llm")
    

def test_load_config_rejects_non_demo(tmp_path, env_path):
    bad = tmp_path / "config.yaml"
    bad.write_text(
        "mode: live\n"
        "symbols: [EURUSD]\n"
        "timeframes: [M15]\n"
        "schedule: {scan_interval_minutes: 15}\n"
        "session_hours: {start: '08:00', end: '20:00'}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be 'demo'"):
        load_config(bad, env_path)


def test_load_config_missing_file(tmp_path, env_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml", env_path)


def test_load_config_empty_symbols(tmp_path, env_path):
    bad = tmp_path / "config.yaml"
    bad.write_text(
        "mode: demo\n"
        "symbols: []\n"
        "timeframes: [M15]\n"
        "schedule: {scan_interval_minutes: 15}\n"
        "session_hours: {start: '08:00', end: '20:00'}\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception) as exc:
        load_config(bad, env_path)
    assert "symbol" in str(exc.value).lower()


def test_env_secrets_parsing(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "OANDA_API_TOKEN=tok-1\n"
        "OANDA_ACCOUNT_ID=001-011-1234567-001\n"
        "TELEGRAM_BOT_TOKEN=bot-1\n"
        "TELEGRAM_CHAT_ID=987654321\n",
        encoding="utf-8",
    )
    s = EnvSecrets.from_env(p)
    assert s.oanda_api_token == "tok-1"
    assert s.oanda_account_id == "001-011-1234567-001"
    assert s.telegram_bot_token == "bot-1"
    assert s.telegram_chat_id == "987654321"


def test_env_secrets_missing_file_ok(tmp_path):
    s = EnvSecrets.from_env(tmp_path / "missing.env")
    assert s.oanda_api_token == ""
    assert s.oanda_account_id == ""
    assert s.telegram_bot_token == ""
    assert s.telegram_chat_id == ""


def test_env_secrets_process_env_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("OANDA_API_TOKEN", "proc-tok")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "proc-acct")
    s = EnvSecrets.from_env(tmp_path / "missing.env")
    assert s.oanda_api_token == "proc-tok"
    assert s.oanda_account_id == "proc-acct"


def test_signals_config_defaults():
    from app.config import SignalsConfig

    s = SignalsConfig()
    assert s.hypothetical_balance_usd == 10000.0
    assert s.feed == "auto"
    assert s.telegram_enabled is True
    assert s.max_signals_per_day == 5


def test_session_hours_normalized():
    s = SessionHoursConfig(start="8:5", end="20:00")
    assert s.start == "08:05"
    with pytest.raises(Exception):
        SessionHoursConfig(start="25:00", end="20:00")
    with pytest.raises(Exception):
        SessionHoursConfig(start="bad", end="20:00")


def test_risk_defaults():
    r = RiskConfig()
    assert r.max_risk_pct_per_trade == 1.0
    assert r.max_daily_loss_pct == 3.0
    assert r.max_open_trades == 3
    assert r.max_trades_per_day == 5
    assert r.min_confidence == 0.6
    assert r.default_playbook == "sweep_fvg_reclaim"
    assert r.breakeven_enabled is True
    assert r.breakeven_at_r == 1.0


def test_kill_zones_normalized_and_validated():
    cfg = AppConfig(
        schedule={"scan_interval_minutes": 15},
        session_hours={"start": "08:00", "end": "20:00"},
        symbols=["XAUUSD"],
        timeframes=["M15"],
        kill_zones=["7:5-10:00", "12:00-15:00"],
    )
    assert cfg.kill_zones == ["07:05-10:00", "12:00-15:00"]
    with pytest.raises(Exception):
        AppConfig(
            schedule={"scan_interval_minutes": 15},
            session_hours={"start": "08:00", "end": "20:00"},
            symbols=["XAUUSD"],
            timeframes=["M15"],
            kill_zones=["not-a-zone"],
        )
    with pytest.raises(Exception):
        AppConfig(
            schedule={"scan_interval_minutes": 15},
            session_hours={"start": "08:00", "end": "20:00"},
            symbols=["XAUUSD"],
            timeframes=["M15"],
            kill_zones=["25:00-26:00"],
        )


def test_kill_zones_default_empty():
    cfg = AppConfig(
        schedule={"scan_interval_minutes": 15},
        session_hours={"start": "08:00", "end": "20:00"},
        symbols=["XAUUSD"],
        timeframes=["M15"],
    )
    assert cfg.kill_zones == []


def test_load_config_uses_overrides(tmp_path, env_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "mode: demo\n"
        "symbols: [USDJPY]\n"
        "timeframes: [H1]\n"
        "schedule: {scan_interval_minutes: 5}\n"
        "session_hours: {start: '07:30', end: '16:00', timezone: 'UTC', skip_weekends: true}\n"
        "risk: {max_risk_pct_per_trade: 0.5, max_daily_loss_pct: 2.0, "
        "max_open_trades: 1, max_trades_per_day: 2, min_confidence: 0.7, "
        "default_playbook: trend_pullback}\n"
        "dashboard: {host: '0.0.0.0', port: 9000}\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path, env_path)
    assert cfg.symbols == ["USDJPY"]
    assert cfg.schedule.scan_interval_minutes == 5
    assert cfg.risk.max_risk_pct_per_trade == 0.5
    assert cfg.risk.min_confidence == 0.7
    assert cfg.risk.default_playbook == "trend_pullback"
    assert cfg.dashboard.port == 9000

def test_load_config_ignores_legacy_llm_section(tmp_path, env_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "mode: demo\n"
        "symbols: [EURUSD]\n"
        "timeframes: [M15]\n"
        "schedule: {scan_interval_minutes: 15}\n"
        "session_hours: {start: '08:00', end: '20:00'}\n"
        "llm: {base_url: 'http://x', model: 'm'}\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path, env_path)
    assert cfg.symbols == ["EURUSD"]
    assert not hasattr(cfg, "llm")
