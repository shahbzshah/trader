"""Tests for the `doctor` pre-flight command in app/main.py.

The doctor runs against a FakeMT5 so every branch (package missing,
terminal unreachable, live account, missing symbol, broker-suffix
suggestion) can be exercised deterministically.
"""

from __future__ import annotations

from tests.fake_mt5 import (
    ACCOUNT_TRADE_MODE_REAL,
    FakeRatesRow,
    FakeSymbolInfo,
    FakeTick,
)


def _install_cfg(monkeypatch, tmp_path, *, symbols, timeframes=("M15",)):
    """Point app.main.load_config at a controlled demo config."""
    from app.config import load_config

    cfg = load_config("config.yaml", env_path=tmp_path / ".env")
    cfg = cfg.model_copy(
        update={"symbols": list(symbols), "timeframes": list(timeframes)}
    )
    import app.main as main_module

    monkeypatch.setattr(main_module, "load_config", lambda *a, **k: cfg)


def _add_rates(fake, n: int = 10) -> None:
    base = 1_700_000_000
    fake.rates_data = [
        FakeRatesRow(
            time=base + i * 900,
            open=1.10,
            high=1.11,
            low=1.09,
            close=1.105,
            tick_volume=100,
            spread=2,
            real_volume=0,
        )
        for i in range(n)
    ]


def _run_doctor() -> int:
    from app.main import main

    argv = [
        "--config", "config.yaml",
        "doctor",
        "--login", "0",
        "--password", "",
        "--server", "",
        "--path", "",
    ]
    return main(argv)


def test_doctor_all_green(capsys, monkeypatch, tmp_path, fake_mt5):
    _install_cfg(monkeypatch, tmp_path, symbols=["EURUSD"])
    _add_rates(fake_mt5)
    rc = _run_doctor()
    out = capsys.readouterr().out
    assert rc == 0
    assert "[OK]" in out
    assert "All checks passed." in out
    assert "[FAIL]" not in out
    assert "demo account" in out
    assert "symbol EURUSD live data" in out


def test_doctor_suggests_broker_suffix_match(capsys, monkeypatch, tmp_path, fake_mt5):
    """EURUSD configured but only EURUSD.m in the terminal -> warn + suggest."""
    _install_cfg(monkeypatch, tmp_path, symbols=["EURUSD"])
    _add_rates(fake_mt5)
    fake_mt5.symbols_get_result = [FakeSymbolInfo(name="EURUSD.m")]
    fake_mt5.tick_data["EURUSD.m"] = FakeTick(bid=1.1, ask=1.1001, time=1_700_000_000)
    rc = _run_doctor()
    out = capsys.readouterr().out
    assert rc == 0
    assert "possible broker-suffix matches" in out
    assert "EURUSD.m" in out


def test_doctor_fails_when_symbol_absent(capsys, monkeypatch, tmp_path, fake_mt5):
    _install_cfg(monkeypatch, tmp_path, symbols=["ZZZUSD"])
    _add_rates(fake_mt5)
    rc = _run_doctor()
    out = capsys.readouterr().out
    assert rc == 1
    assert "not present in terminal" in out
    assert "1 check(s) FAILED" in out


def test_doctor_fails_on_real_account(capsys, monkeypatch, tmp_path, fake_mt5):
    """Safety invariant: a live account must make doctor fail loudly."""
    _install_cfg(monkeypatch, tmp_path, symbols=["EURUSD"])
    _add_rates(fake_mt5)
    fake_mt5.trade_mode = ACCOUNT_TRADE_MODE_REAL
    rc = _run_doctor()
    out = capsys.readouterr().out
    assert rc == 1
    assert "non-demo account" in out


def test_doctor_reports_missing_metatrader5(capsys, monkeypatch, tmp_path):
    import app.mt5_client as client_module

    _install_cfg(monkeypatch, tmp_path, symbols=["EURUSD"])
    monkeypatch.setattr(client_module, "_mt5", None)
    monkeypatch.setattr(
        client_module,
        "_get_mt5",
        lambda: (_ for _ in ()).throw(ImportError("MetaTrader5 not installed")),
    )
    rc = _run_doctor()
    out = capsys.readouterr().out
    assert rc == 1
    assert "MetaTrader5 package import" in out
    assert "Install with" in out
