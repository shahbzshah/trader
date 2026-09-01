"""End-to-end test of `python -m app.main once` with a mocked MT5.

Exercises the M4 acceptance criterion: a full --once cycle on a mocked
environment must write decision + order rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tests.fake_mt5 import (
    ACCOUNT_TRADE_MODE_DEMO,
    FakeAccountInfo,
    FakeMT5,
    FakePosition,
    FakeRatesRow,
    FakeSymbolInfo,
    FakeTick,
    make_fake,
)


def _install_fake_mt5(fake: FakeMT5) -> None:
    import app.mt5_client as client_module
    client_module._mt5 = fake


def _build_rates(n: int = 250, price: float = 1.1000, point: float = 0.0001):
    return [
        FakeRatesRow(
            time=1700000000 + i * 900,
            open=price, high=price + point, low=price - point,
            close=price, tick_volume=100, spread=5, real_volume=0,
        )
        for i in range(n)
    ]


def test_once_cycle_writes_decision_and_no_order_for_no_trade(tmp_path, monkeypatch):
    from app import main as main_module
    from app.config import load_config
    import app.journal as journal_module

    cfg = load_config("config.yaml", env_path=tmp_path / ".env")
    cfg.paths.data_dir = str(tmp_path)
    cfg.paths.kill_switch_path = str(tmp_path / "KILL_SWITCH")

    # Install fake MT5.
    fake = make_fake()
    fake.symbol_data["EURUSD"] = FakeSymbolInfo(name="EURUSD")
    fake.symbol_data["GBPUSD"] = FakeSymbolInfo(name="GBPUSD")
    fake.symbol_data["XAUUSD"] = FakeSymbolInfo(name="XAUUSD", point=0.01,
                                                 digits=2, volume_step=0.01,
                                                 trade_contract_size=100.0)
    fake.tick_data["EURUSD"] = FakeTick(bid=1.1, ask=1.1001, time=1)
    fake.tick_data["GBPUSD"] = FakeTick(bid=1.25, ask=1.2501, time=1)
    fake.tick_data["XAUUSD"] = FakeTick(bid=1900.0, ask=1900.1, time=1)
    fake.rates_data = _build_rates(250, 1.1)
    _install_fake_mt5(fake)

    # LLM responds with a no_trade (one per snapshot).
    from app.brain import LLMClient as RealLLMClient
    from tests.fake_http import make_chat_response, recording_transport

    no_trade = make_chat_response(json.dumps({
        "action": "no_trade", "symbol": "EURUSD", "confidence": 0.7,
        "reasoning": "no setup"
    }))
    transport, _ = recording_transport([no_trade for _ in range(10)])

    class MockedLLMClient(RealLLMClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(main_module, "LLMClient", MockedLLMClient)
    monkeypatch.setattr(main_module, "load_config", lambda *a, **k: cfg)

    rc = main_module.main([
        "--config", "config.yaml", "once",
        "--login", "12345", "--password", "pw", "--server", "x",
    ])
    assert rc == 0

    # Validate journal rows.
    from app.journal import Journal
    j = Journal(Path(cfg.paths.data_dir) / "journal.db")
    decs = j.list_decisions(limit=20)
    assert len(decs) >= 1
    # All decisions are no_trade + approved.
    assert all(d["action"] == "no_trade" for d in decs)
    assert all(d["approved"] for d in decs)
    # No orders.
    assert j.list_orders(limit=20) == []
    # Equity snapshot.
    eq = j.list_equity(limit=5)
    assert len(eq) == 1


def test_once_cycle_propose_writes_decision_and_order(tmp_path, monkeypatch):
    from app import main as main_module
    from app.config import load_config

    cfg = load_config("config.yaml", env_path=tmp_path / ".env")
    cfg.paths.data_dir = str(tmp_path)
    cfg.paths.kill_switch_path = str(tmp_path / "KILL_SWITCH")

    fake = make_fake()
    fake.symbol_data["EURUSD"] = FakeSymbolInfo(name="EURUSD")
    fake.symbol_data["GBPUSD"] = FakeSymbolInfo(name="GBPUSD")
    fake.symbol_data["XAUUSD"] = FakeSymbolInfo(name="XAUUSD", point=0.01,
                                                 digits=2, volume_step=0.01,
                                                 trade_contract_size=100.0)
    fake.tick_data["EURUSD"] = FakeTick(bid=1.1, ask=1.1001, time=1)
    fake.tick_data["GBPUSD"] = FakeTick(bid=1.25, ask=1.2501, time=1)
    fake.tick_data["XAUUSD"] = FakeTick(bid=1900.0, ask=1900.1, time=1)
    fake.rates_data = _build_rates(250, 1.1)
    _install_fake_mt5(fake)

    from app.brain import LLMClient as RealLLMClient
    from tests.fake_http import make_chat_response, recording_transport

    # 3 propose calls. The brain's prompt mentions the symbol in the user
    # message; the LLM doesn't know the symbol a priori, but we always
    # say "EURUSD" in the canned responses. (Each decide() call gets
    # its own snapshot so a no_trade on the first two + a propose on
    # the third is also valid. We use three identical propose
    # responses and let risk check the first two — but they'll be
    # approved because demo + no kill switch + low open trades. The
    # executor will then place orders for all 3. We assert the totals.)
    propose_payload = {
        "action": "propose", "symbol": "EURUSD", "direction": "buy",
        "entry_type": "market", "price": 1.1000,
        "stop_loss": 1.0950, "take_profit": 1.1200,
        "risk_pct": 0.5, "confidence": 0.8,
        "playbook": "breakout_retest",
        "reasoning": "trend confirmed",
    }
    transport, _ = recording_transport([
        make_chat_response(json.dumps(propose_payload))
        for _ in range(10)
    ])

    class MockedLLMClient(RealLLMClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(main_module, "LLMClient", MockedLLMClient)
    monkeypatch.setattr(main_module, "load_config", lambda *a, **k: cfg)

    rc = main_module.main([
        "--config", "config.yaml", "once",
        "--login", "12345", "--password", "pw", "--server", "x",
    ])
    assert rc == 0

    from app.journal import Journal
    j = Journal(Path(cfg.paths.data_dir) / "journal.db")
    decs = j.list_decisions(limit=20)
    orders = j.list_orders(limit=20)

    # We should have at least 1 propose decision + 1 filled order.
    propose_decs = [d for d in decs if d["action"] == "propose"]
    filled = [o for o in orders if o["status"] == "filled"]
    assert len(propose_decs) >= 1
    assert len(filled) >= 1
    # The first order's decision_id should link to a propose decision.
    assert filled[0]["decision_id"] in {d["id"] for d in propose_decs}


def test_once_cycle_kill_switch_blocks(tmp_path, monkeypatch):
    from app import main as main_module
    from app.config import load_config

    cfg = load_config("config.yaml", env_path=tmp_path / ".env")
    cfg.paths.data_dir = str(tmp_path)
    cfg.paths.kill_switch_path = str(tmp_path / "KILL_SWITCH")
    Path(cfg.paths.kill_switch_path).touch()

    fake = make_fake()
    fake.symbol_data["EURUSD"] = FakeSymbolInfo(name="EURUSD")
    fake.tick_data["EURUSD"] = FakeTick(bid=1.1, ask=1.1001, time=1)
    fake.rates_data = _build_rates(250, 1.1)
    _install_fake_mt5(fake)

    from app.brain import LLMClient as RealLLMClient
    from tests.fake_http import make_chat_response, recording_transport

    transport, _ = recording_transport([
        make_chat_response(json.dumps({
            "action": "propose", "symbol": "EURUSD", "direction": "buy",
            "entry_type": "market", "price": 1.1, "stop_loss": 1.095,
            "take_profit": 1.12, "risk_pct": 0.5, "confidence": 0.8,
            "playbook": "breakout_retest",
            "reasoning": "ok"
        }))
    ])

    class MockedLLMClient(RealLLMClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(main_module, "LLMClient", MockedLLMClient)
    monkeypatch.setattr(main_module, "load_config", lambda *a, **k: cfg)

    rc = main_module.main([
        "--config", "config.yaml", "once",
        "--login", "12345", "--password", "pw", "--server", "x",
    ])
    assert rc == 0
    from app.journal import Journal
    j = Journal(Path(cfg.paths.data_dir) / "journal.db")
    orders = j.list_orders(limit=20)
    # Kill switch prevents any orders; but decisions still get journaled
    # as rejected by the executor's re-validation.
    assert all(o["status"] != "filled" for o in orders)
    decs = j.list_decisions(limit=20)
    assert any("kill_switch_engaged" in d["reject_reasons"] for d in decs)