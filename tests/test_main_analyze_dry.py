"""End-to-end test of `python -m app.main analyze --dry`.

Exercises the analyze path end-to-end: it must run with the deterministic
rules engine and print a Decision without journaling anything and
without sending any alert.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_main_analyze_dry_prints_decision(capsys, monkeypatch, tmp_path):
    from app import main as main_module
    from app.config import load_config
    from tests.fake_client import FakeClient

    cfg = load_config("config.yaml", env_path=tmp_path / ".env")
    cfg.paths.data_dir = str(tmp_path)
    cfg.paths.kill_switch_path = str(tmp_path / "KILL_SWITCH")

    # Empty feed: no candles -> the rules engine says no_trade.
    fake = FakeClient()
    monkeypatch.setattr(
        main_module, "build_market_client", lambda *a, **k: (fake, "fake"))
    monkeypatch.setattr(main_module, "load_config", lambda *a, **k: cfg)

    rc = main_module.main(["--config", "config.yaml", "analyze", "--dry"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no_trade" in out
    assert "Traceback" not in out
    # Analyze never journals.
    assert not (Path(cfg.paths.data_dir) / "journal.db").exists()


def test_main_analyze_dry_with_setup_prints_sizing(
        capsys, monkeypatch, tmp_path):
    """A setup in analyze mode prints the paper sizing preview."""
    from app import main as main_module
    from app.config import load_config
    from tests.fake_client import FakeClient, rates_from_rows
    from tests.test_main_once import (
        _bullish_m15_rows,
        _setup_fake_with_setup,
        _uptrend_h1_rows,
        T0_H1,
        T0_M15,
    )

    cfg = load_config("config.yaml", env_path=tmp_path / ".env")
    cfg.paths.data_dir = str(tmp_path)
    cfg.paths.kill_switch_path = str(tmp_path / "KILL_SWITCH")

    fake = FakeClient()
    _setup_fake_with_setup(fake)
    monkeypatch.setattr(
        main_module, "build_market_client", lambda *a, **k: (fake, "fake"))
    monkeypatch.setattr(main_module, "load_config", lambda *a, **k: cfg)

    rc = main_module.main(["--config", "config.yaml", "analyze", "--dry"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "propose" in out
    assert "sizing:" in out
    assert "lots" in out
    assert "Traceback" not in out
