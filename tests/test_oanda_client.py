"""Tests for app/oanda_client.py with mocked HTTP responses.

No network, no token: httpx.MockTransport serves canned v20 payloads.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.market_data import MarketDataError
from app.oanda_client import OandaClient

TOKEN = "test-token"
ACCOUNT = "001-011-1234567-001"

CANDLES_PAYLOAD = {
    "instrument": "XAU_USD",
    "granularity": "M15",
    "candles": [
        {
            "complete": True,
            "volume": 123,
            "time": "2023-11-15T12:00:00.000000000Z",
            "mid": {"o": "1900.00", "h": "1901.50", "l": "1899.00", "c": "1900.80"},
        },
        {
            "complete": False,
            "volume": 45,
            "time": "2023-11-15T12:15:00.000000000Z",
            "mid": {"o": "1900.80", "h": "1902.00", "l": "1900.10", "c": "1901.20"},
        },
    ],
}

PRICING_PAYLOAD = {
    "prices": [
        {
            "type": "PRICE",
            "instrument": "XAU_USD",
            "time": "2023-11-15T12:16:30.123456789Z",
            "bids": [{"price": "1901.10", "liquidity": 100}],
            "asks": [{"price": "1901.45", "liquidity": 100}],
            "closeoutBid": "1901.05",
            "closeoutAsk": "1901.50",
        }
    ]
}


def _client_with_routes(routes: dict, *, status: int = 200) -> OandaClient:
    """Build an OandaClient whose httpx transport answers from ``routes``.

    ``routes`` maps a URL substring to a JSON payload (or a
    ``(status, payload)`` tuple).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        for needle, payload in routes.items():
            if needle in str(request.url):
                if isinstance(payload, tuple):
                    code, body = payload
                else:
                    code, body = status, payload
                return httpx.Response(code, json=body, request=request)
        return httpx.Response(404, json={"errorMessage": "not found"},
                              request=request)

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="https://api-fxpractice.oanda.com")
    return OandaClient(TOKEN, ACCOUNT, http_client=http)


def _std_client() -> OandaClient:
    return _client_with_routes({
        "/candles": CANDLES_PAYLOAD,
        "/pricing": PRICING_PAYLOAD,
    })


def test_copy_rates_parses_candles():
    client = _std_client()
    rates = client.copy_rates("XAUUSD", "M15", 200)
    assert len(rates.rates) == 2
    first = rates.rates[0]
    assert first["time"] == 1700049600
    assert first["open"] == 1900.00
    assert first["high"] == 1901.50
    assert first["low"] == 1899.00
    assert first["close"] == 1900.80
    assert first["tick_volume"] == 123
    assert first["spread"] is None  # OANDA candles carry no spread
    # Candles are sorted oldest-first.
    assert rates.rates[0]["time"] < rates.rates[1]["time"]


def test_copy_rates_h1_granularity():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=CANDLES_PAYLOAD, request=request)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = OandaClient(TOKEN, ACCOUNT, http_client=http)
    client.copy_rates("XAUUSD", "H1", 80)
    assert "granularity=H1" in seen["url"]
    assert "XAU_USD" in seen["url"]


def test_symbol_tick_parses_pricing():
    client = _std_client()
    tick = client.symbol_tick("XAUUSD")
    assert tick is not None
    assert tick.bid == 1901.10
    assert tick.ask == 1901.45
    # Spread is in POINTS: $0.35 / 0.01 = 35 points.
    assert tick.spread == pytest.approx(35.0)
    assert tick.time == 1700050590  # 2023-11-15 12:16:30 UTC


def test_connect_probes_pricing():
    client = _std_client()
    client.connect()
    assert client.feed_status()["connected"] is True


def test_unauthorized_token_raises_market_data_error():
    client = _client_with_routes({
        "/pricing": (401, {"errorMessage": "Insufficient authorization"}),
        "/candles": (401, {"errorMessage": "Insufficient authorization"}),
    })
    with pytest.raises(MarketDataError, match="401/403"):
        client.connect()
    with pytest.raises(MarketDataError, match="401/403"):
        client.copy_rates("XAUUSD", "M15", 10)


def test_unknown_account_raises_on_pricing():
    client = _client_with_routes({
        "/pricing": (404, {"errorMessage": "Account not found"}),
    })
    with pytest.raises(MarketDataError, match="account not found"):
        client.connect()


def test_unsupported_symbol_and_timeframe():
    client = _std_client()
    with pytest.raises(MarketDataError, match="unsupported symbol"):
        client.copy_rates("EURUSD", "M15", 10)
    with pytest.raises(MarketDataError, match="unsupported timeframe"):
        client.copy_rates("XAUUSD", "W1", 10)


def test_symbol_info_and_no_positions():
    client = _std_client()
    info = client.symbol_info("XAUUSD")
    assert info.point == 0.01
    assert info.trade_contract_size == 100.0
    assert info.volume_min == 0.01
    assert client.positions() == []
    assert client.select_symbol("XAUUSD") is True
    acct = client.account_info()
    assert acct.balance == 10000.0
    assert acct.trade_mode == "paper"


def test_missing_token_raises_at_construction():
    with pytest.raises(MarketDataError, match="OANDA_API_TOKEN"):
        OandaClient("", ACCOUNT)
    with pytest.raises(MarketDataError, match="OANDA_ACCOUNT_ID"):
        OandaClient(TOKEN, "")


def test_auth_header_sent():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=PRICING_PAYLOAD, request=request)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = OandaClient(TOKEN, ACCOUNT, http_client=http)
    client.symbol_tick("XAUUSD")
    assert seen["auth"] == f"Bearer {TOKEN}"
