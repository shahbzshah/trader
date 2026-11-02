"""Tests for the `doctor` pre-flight command in app/main.py.

The doctor runs against a fake market feed so every branch (feed down,
symbol with no data, kill switch) can be exercised deterministically.
"""

from __future__ import annotations

from app.market_data import MarketDataError, Rates, Tick
from tests.fake_client import FakeClient, rates_from_rows


def _install_cfg(monkeypatch, tmp_path, *, symbols, timeframes=("M15",)):
    """Point app.main.load_config at a controlled demo config."""
    from app.config import load_config

    cfg = load_config("config.yaml", env_path=tmp_path / ".env")
    cfg = cfg.model_copy(
        update={"symbols": list(symbols), "timeframes": list(timeframes)}
    )
    cfg.paths.data_dir = str(tmp_path)
    cfg.paths.kill_switch_path = str(tmp_path / "KILL_SWITCH")
    import app.main as main_module

    monkeypatch.setattr(main_module, "load_config", lambda *a, **k: cfg)
    return cfg


def _install_feed(monkeypatch, fake):
    from app import main as main_module

    monkeypatch.setattr(
        main_module, "build_market_client", lambda *a, **k: (fake, "fake"))


def _add_rates(fake, n: int = 10) -> None:
    base = 1_700_000_000
    fake.rates_by_tf[("XAUUSD", "M15")] = Rates(rates=[
        {"time": base + i * 900, "open": 1900.0, "high": 1901.0,
         "low": 1899.0, "close": 1900.5, "tick_volume": 0,
         "spread": None, "real_volume": 0}
        for i in range(n)
    ])
    fake.tick = Tick(bid=1900.4, ask=1900.5, spread=10.0, time=base)


def _run_doctor() -> int:
    from app.main import main

    return main(["--config", "config.yaml", "doctor"])


def test_doctor_all_green(capsys, monkeypatch, tmp_path):
    _install_cfg(monkeypatch, tmp_path, symbols=["XAUUSD"])
    fake = FakeClient()
    _add_rates(fake)
    _install_feed(monkeypatch, fake)
    rc = _run_doctor()
    out = capsys.readouterr().out
    assert rc == 0
    assert "[OK]" in out
    assert "All checks passed." in out
    assert "[FAIL]" not in out
    assert "market feed" in out
    assert "symbol XAUUSD live data" in out
    # No OANDA creds / Telegram in the test env: warnings, not failures.
    assert "OANDA credentials" in out
    assert "Telegram alerts" in out


def test_doctor_fails_when_feed_down(capsys, monkeypatch, tmp_path):
    from app import main as main_module

    _install_cfg(monkeypatch, tmp_path, symbols=["XAUUSD"])
    monkeypatch.setattr(
        main_module, "build_market_client",
        lambda *a, **k: (_ for _ in ()).throw(
            MarketDataError("all feeds down")))
    rc = _run_doctor()
    out = capsys.readouterr().out
    assert rc == 1
    assert "market feed" in out
    assert "[FAIL]" in out


def test_doctor_fails_when_symbol_has_no_data(capsys, monkeypatch, tmp_path):
    _install_cfg(monkeypatch, tmp_path, symbols=["XAUUSD"])
    fake = FakeClient()  # no rates, no tick
    _install_feed(monkeypatch, fake)
    rc = _run_doctor()
    out = capsys.readouterr().out
    assert rc == 1
    assert "symbol XAUUSD live data" in out
    assert "1 check(s) FAILED" in out


def test_doctor_warns_on_kill_switch(capsys, monkeypatch, tmp_path):
    from pathlib import Path

    cfg = _install_cfg(monkeypatch, tmp_path, symbols=["XAUUSD"])
    fake = FakeClient()
    _add_rates(fake)
    _install_feed(monkeypatch, fake)
    Path(cfg.paths.kill_switch_path).touch()
    rc = _run_doctor()
    out = capsys.readouterr().out
    assert rc == 0
    assert "kill switch" in out
    assert "ENGAGED" in out
