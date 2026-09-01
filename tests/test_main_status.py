"""Tests for app/main.py CLI status behaviour."""

from __future__ import annotations

from app.main import build_parser


def test_build_parser_has_required_subcommands():
    parser = build_parser()
    # ensure no exception
    help_text = parser.format_help()
    for cmd in ("status", "kill", "resume", "run", "once", "analyze", "doctor"):
        assert cmd in help_text


def test_status_does_not_crash_when_mt5_unavailable(capsys, monkeypatch):
    """`python -m app.main status` must print a clean error, not crash."""
    from app.main import main

    # Force MT5 import to fail by clearing _mt5 and ensuring real pkg absent.
    import app.mt5_client as client_module

    client_module._mt5 = None

    class _Boom:
        def __getattr__(self, name):
            raise ImportError("MetaTrader5 not installed")

    monkeypatch.setattr(client_module, "_get_mt5", lambda: (_ for _ in ()).throw(ImportError("not installed")))

    rc = main(["--config", "config.yaml", "status",
               "--login", "0", "--password", "", "--server", "", "--path", ""])
    out = capsys.readouterr().out
    # Acceptable clean messages per acceptance criteria.
    assert rc == 0
    assert ("MT5 terminal not running" in out) or ("not installed" in out.lower())