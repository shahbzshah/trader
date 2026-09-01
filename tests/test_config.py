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
    assert cfg.symbols == ["EURUSD", "GBPUSD", "XAUUSD"]
    assert cfg.timeframes == ["M15", "H1"]
    assert cfg.schedule.scan_interval_minutes == 15
    assert cfg.session_hours.start == "08:00"
    assert cfg.session_hours.end == "20:00"
    assert cfg.risk.max_risk_pct_per_trade == 1.0
    assert cfg.risk.max_daily_loss_pct == 3.0
    assert cfg.risk.max_open_trades == 3
    assert cfg.risk.max_trades_per_day == 5
    assert cfg.llm.model == "poolside/laguna-s-2.1"
    assert cfg.llm.base_url == "https://inference.poolside.ai/v1"
    assert cfg.dashboard.port == 8765
    assert cfg.llm_api_key == "sk-test-xyz"


def test_load_config_rejects_non_demo(tmp_path, env_path):
    bad = tmp_path / "config.yaml"
    bad.write_text(
        "mode: live\n"
        "symbols: [EURUSD]\n"
        "timeframes: [M15]\n"
        "schedule: {scan_interval_minutes: 15}\n"
        "session_hours: {start: '08:00', end: '20:00'}\n"
        "llm: {base_url: 'http://x', model: 'm'}\n",
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
        "session_hours: {start: '08:00', end: '20:00'}\n"
        "llm: {base_url: 'http://x', model: 'm'}\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception) as exc:
        load_config(bad, env_path)
    assert "symbol" in str(exc.value).lower()


def test_env_secrets_parsing(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "MT5_LOGIN=42\n"
        "MT5_PASSWORD=pw\n"
        "MT5_SERVER=S\n"
        "LLM_API_KEY=k\n",
        encoding="utf-8",
    )
    s = EnvSecrets.from_env(p)
    assert s.mt5_login == 42
    assert s.mt5_password == "pw"
    assert s.mt5_server == "S"
    assert s.llm_api_key == "k"


def test_env_secrets_missing_file_ok(tmp_path):
    s = EnvSecrets.from_env(tmp_path / "missing.env")
    assert s.mt5_login == 0
    assert s.mt5_password == ""


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
    assert r.default_playbook == "breakout_retest"


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
        "llm: {base_url: 'http://localhost:11434/v1', model: 'llama3', "
        "temperature: 0.0, timeout_seconds: 30}\n"
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