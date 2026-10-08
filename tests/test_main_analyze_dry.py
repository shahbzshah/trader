"""End-to-end test of `python -m app.main analyze --dry`.

Exercises the analyze path end-to-end: it must run with the deterministic
rules engine and print a Decision without touching orders.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_main_analyze_dry_prints_decision(capsys, monkeypatch, tmp_path):
    from app import main as main_module
    from app.config import load_config

    cfg = load_config("config.yaml", env_path=tmp_path / ".env")

    # Force no-MT5 path: stub out MT5Client to return empty data.
    class _NoopClient:
        def connect(self, *args, **kwargs):
            return True

        def account_info(self):
            class A:
                balance = 10000.0
                equity = 10000.0
                trade_mode = "demo"
            return A()

        def copy_rates(self, symbol, tf, count):
            return []

        def symbol_tick(self, symbol):
            return None

        def positions(self, symbol=None):
            return []

        def shutdown(self):
            return None

    monkeypatch.setattr(main_module, "MT5Client", _NoopClient)

    # Make load_config deterministic and fast.
    monkeypatch.setattr(main_module, "load_config", lambda *a, **k: cfg)

    rc = main_module.main(["--config", "config.yaml", "analyze", "--dry"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no_trade" in out
    assert "Traceback" not in out