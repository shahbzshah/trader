"""Tests for app/telegram.py with mocked HTTP (no real Telegram calls)."""

from __future__ import annotations

import httpx
import pytest

from app.signals import Signal
from app.telegram import TelegramAlerter, format_signal_text

TOKEN = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
CHAT_ID = "123456789"


def _alerter_with(handler) -> TelegramAlerter:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return TelegramAlerter(TOKEN, CHAT_ID, http_client=http)


def test_enabled_only_when_both_set():
    assert TelegramAlerter(TOKEN, CHAT_ID).enabled is True
    assert TelegramAlerter("", CHAT_ID).enabled is False
    assert TelegramAlerter(TOKEN, "").enabled is False
    assert TelegramAlerter("", "").enabled is False


def test_send_message_success():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen["url"] = str(request.url)
        seen["json"] = _json.loads(request.content)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}},
                              request=request)

    alerter = _alerter_with(handler)
    assert alerter.send_message("hello") is True
    assert "api.telegram.org" in seen["url"]
    assert f"bot{TOKEN}" in seen["url"]
    assert "/sendMessage" in seen["url"]
    assert seen["json"]["chat_id"] == CHAT_ID
    assert seen["json"]["text"] == "hello"
    # The token must never appear in a log line — and the URL carries it
    # by Telegram's design; the client never logs it (asserted by code
    # review: send_message logs only the description, not the URL).


def test_send_message_telegram_error_returns_false():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={
            "ok": False, "error_code": 403,
            "description": "Forbidden: bot was blocked by the user"},
            request=request)

    alerter = _alerter_with(handler)
    assert alerter.send_message("hello") is False


def test_send_message_disabled_returns_false():
    alerter = TelegramAlerter("", "")
    assert alerter.send_message("hello") is False


def test_validate_token_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/getMe" in str(request.url)
        return httpx.Response(200, json={
            "ok": True, "result": {"username": "gold_signals_bot"}},
            request=request)

    alerter = _alerter_with(handler)
    ok, detail = alerter.validate_token()
    assert ok is True
    assert detail == "@gold_signals_bot"


def test_validate_token_bad_token():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={
            "ok": False, "error_code": 401,
            "description": "Unauthorized: invalid token"},
            request=request)

    alerter = _alerter_with(handler)
    ok, detail = alerter.validate_token()
    assert ok is False
    assert "Unauthorized" in detail
    # The detail must not contain the token.
    assert TOKEN not in detail


def test_validate_token_missing_token():
    alerter = TelegramAlerter("", CHAT_ID)
    ok, detail = alerter.validate_token()
    assert ok is False
    assert "not set" in detail


def test_format_signal_text():
    signal = Signal(
        symbol="XAUUSD", direction="buy", entry=3900.0, stop_loss=3885.0,
        take_profit=3930.0, risk_pct=1.0, confidence=0.85,
        playbook="sweep_fvg_reclaim", reasoning="sweep + FVG reclaim",
        source_feed="oanda", ts=1700049600, lots=0.06, risk_usd=100.0,
        per_dollar_usd=6.0,
    )
    text = format_signal_text(signal)
    assert "GOLD SIGNAL — BUY XAUUSD" in text
    assert "Entry (limit): 3900.00" in text
    assert "Stop loss: 3885.00" in text
    assert "Take profit: 3930.00" in text
    assert "2.0R" in text  # (3930-3900) / (3900-3885) = 2R
    assert "Risk: $100.00" in text
    assert "0.06 lots" in text
    assert "$ per $1 move: $6.00" in text
    assert "Feed: oanda" in text
