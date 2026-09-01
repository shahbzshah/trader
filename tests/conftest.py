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
for key in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_PATH", "LLM_API_KEY"):
    os.environ.pop(key, None)


@pytest.fixture
def fake_mt5():
    """Provide a fresh FakeMT5 and install it as app.mt5_client._mt5."""
    from tests.fake_mt5 import FakeMT5, make_fake

    fake = make_fake()
    import app.mt5_client as client_module

    prev = client_module._mt5
    client_module._mt5 = fake
    yield fake
    client_module._mt5 = prev


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def config_yaml_path(repo_root: Path) -> Path:
    return repo_root / "config.yaml"


@pytest.fixture
def env_path(repo_root: Path, tmp_path: Path) -> Path:
    """Create a temp .env with all required MT5 creds."""
    p = tmp_path / ".env"
    p.write_text(
        "MT5_LOGIN=12345\n"
        "MT5_PASSWORD=secret\n"
        "MT5_SERVER=Broker-Demo\n"
        "MT5_PATH=\n"
        "LLM_API_KEY=sk-test-xyz\n",
        encoding="utf-8",
    )
    return p