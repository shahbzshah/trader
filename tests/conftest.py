"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure no stray env vars leak into config tests.
for key in ("OANDA_API_TOKEN", "OANDA_ACCOUNT_ID",
            "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
    os.environ.pop(key, None)


@pytest.fixture
def fake_client():
    """Provide a fresh fake market-data client (no network, no MT5)."""
    from tests.fake_client import FakeClient

    return FakeClient()


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def config_yaml_path(repo_root: Path) -> Path:
    return repo_root / "config.yaml"


@pytest.fixture
def env_path(repo_root: Path, tmp_path: Path) -> Path:
    """Create a temp .env with OANDA + Telegram secrets."""
    p = tmp_path / ".env"
    p.write_text(
        "OANDA_API_TOKEN=tok-abc\n"
        "OANDA_ACCOUNT_ID=001-011-1234567-001\n"
        "TELEGRAM_BOT_TOKEN=bot-xyz\n"
        "TELEGRAM_CHAT_ID=123456789\n",
        encoding="utf-8",
    )
    return p
