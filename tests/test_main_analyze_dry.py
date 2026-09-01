"""End-to-end test of `python -m app.main analyze --dry` with a mocked LLM.

Exercises M3 acceptance criterion: the analyze path must run end-to-end
with a mocked LLM and print a Decision without touching orders.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import httpx


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

    # Intercept the LLMClient constructor so it uses a MockTransport.
    from app.brain import LLMClient as RealLLMClient
    from tests.fake_http import make_chat_response, recording_transport

    valid = {"action": "no_trade", "symbol": "EURUSD", "confidence": 0.7,
             "reasoning": "no clear setup"}
    transport, _ = recording_transport([make_chat_response(json.dumps(valid))])

    class MockedLLMClient(RealLLMClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(main_module, "LLMClient", MockedLLMClient)

    # Make load_config deterministic and fast.
    monkeypatch.setattr(main_module, "load_config", lambda *a, **k: cfg)

    rc = main_module.main(["--config", "config.yaml", "analyze", "--dry"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no_trade" in out
    assert "Traceback" not in out