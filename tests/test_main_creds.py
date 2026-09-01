"""Tests for .env-backed credential resolution in app/main.py.

The CLI flags must win when provided; otherwise the ``.env`` secrets
(beside config.yaml) fill the gaps so ``python -m app.main run`` works
without repeating credentials on the command line.
"""

from __future__ import annotations

import argparse


def _args(tmp_path, *, login: int = 0, password: str = "", server: str = "",
          path: str = "") -> argparse.Namespace:
    return argparse.Namespace(
        config=str(tmp_path / "config.yaml"),
        login=login,
        password=password,
        server=server,
        path=path,
    )


def test_env_file_fills_missing_cli_flags(tmp_path, env_path):
    from app.main import _resolve_creds

    got = _resolve_creds(_args(tmp_path))
    assert got == (12345, "secret", "Broker-Demo", "")


def test_cli_flags_override_env_file(tmp_path, env_path):
    from app.main import _resolve_creds

    got = _resolve_creds(
        _args(tmp_path, login=999, password="cli", server="CLI-Srv", path="C:/mt5")
    )
    assert got == (999, "cli", "CLI-Srv", "C:/mt5")


def test_missing_env_file_yields_zero_creds(tmp_path, monkeypatch):
    from app.main import _resolve_creds

    for key in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_PATH"):
        monkeypatch.delenv(key, raising=False)
    got = _resolve_creds(_args(tmp_path / "nowhere"))
    assert got == (0, "", "", "")


def test_env_path_prefers_config_adjacent_dotenv(tmp_path):
    from app.main import _env_path_for, _resolve_creds

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / ".env").write_text("MT5_LOGIN=777\n", encoding="utf-8")
    resolved = _env_path_for(_args(cfg_dir))
    assert resolved.name == ".env"
    assert resolved.parent == cfg_dir.resolve()
    assert _resolve_creds(_args(cfg_dir))[0] == 777
