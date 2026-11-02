"""Tests for .env-backed secret resolution in app/main.py.

OANDA + Telegram secrets come from the .env beside config.yaml (or the
process environment). Nothing is passed on the CLI.
"""

from __future__ import annotations

import argparse


def _args(tmp_path) -> argparse.Namespace:
    return argparse.Namespace(config=str(tmp_path / "config.yaml"))


def test_env_file_fills_secrets(tmp_path, env_path):
    from app.main import _resolve_secrets

    got = _resolve_secrets(_args(tmp_path))
    assert got.oanda_api_token == "tok-abc"
    assert got.oanda_account_id == "001-011-1234567-001"
    assert got.telegram_bot_token == "bot-xyz"
    assert got.telegram_chat_id == "123456789"


def test_missing_env_file_yields_empty_secrets(tmp_path, monkeypatch):
    from app.main import _resolve_secrets

    for key in ("OANDA_API_TOKEN", "OANDA_ACCOUNT_ID",
                "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(key, raising=False)
    got = _resolve_secrets(_args(tmp_path / "nowhere"))
    assert got.oanda_api_token == ""
    assert got.oanda_account_id == ""
    assert got.telegram_bot_token == ""
    assert got.telegram_chat_id == ""


def test_process_env_fills_secrets(tmp_path, monkeypatch):
    from app.main import _resolve_secrets

    monkeypatch.setenv("OANDA_API_TOKEN", "proc-tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "555")
    got = _resolve_secrets(_args(tmp_path / "nowhere"))
    assert got.oanda_api_token == "proc-tok"
    assert got.telegram_chat_id == "555"


def test_env_path_prefers_config_adjacent_dotenv(tmp_path):
    from app.main import _env_path_for, _resolve_secrets

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / ".env").write_text("OANDA_API_TOKEN=adjacent-tok\n", encoding="utf-8")
    resolved = _env_path_for(_args(cfg_dir))
    assert resolved.name == ".env"
    assert resolved.parent == cfg_dir.resolve()
    assert _resolve_secrets(_args(cfg_dir)).oanda_api_token == "adjacent-tok"
