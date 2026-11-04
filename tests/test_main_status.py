"""Tests for app/main.py CLI status behaviour."""

from __future__ import annotations

from app.main import build_parser
from app.market_data import MarketDataError, Rates, Tick
from tests.fake_client import FakeClient


def test_build_parser_has_required_subcommands():
    parser = build_parser()
    help_text = parser.format_help()
    for cmd in ("status", "kill", "resume", "run", "serve", "once",
                "doctor", "review"):
        assert cmd in help_text


def _install_fake_feed(monkeypatch, fake):
    from app import main as main_module

    monkeypatch.setattr(
        main_module, "build_market_client", lambda *a, **k: (fake, "fake"))


def test_status_prints_feed_and_tick(capsys, monkeypatch):
    """`python -m app.main status` prints the feed + latest tick."""
    from app.main import main

    fake = FakeClient()
    fake.tick = Tick(bid=3897.56, ask=3898.10, spread=54.0, time=1700000000)
    fake.rates_by_tf[("XAUUSD", "M15")] = Rates(rates=[
        {"time": 1700000000, "open": 3897.0, "high": 3899.0,
         "low": 3896.0, "close": 3898.0, "tick_volume": 0,
         "spread": None, "real_volume": 0},
    ])
    _install_fake_feed(monkeypatch, fake)

    rc = main(["--config", "config.yaml", "status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "feed=fake" in out
    assert "3897.56" in out
    assert "3898.10" in out
    assert "M15 candles: 1" in out


def test_status_does_not_crash_when_feed_unavailable(capsys, monkeypatch):
    """`python -m app.main status` must print a clean error, not crash."""
    from app import main as main_module

    monkeypatch.setattr(
        main_module, "build_market_client",
        lambda *a, **k: (_ for _ in ()).throw(
            MarketDataError("all feeds down")))

    rc = main_module.main(["--config", "config.yaml", "status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no market feed available" in out
