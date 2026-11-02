"""Tests for app/deriv_client.py.

Most tests mock the transport (``DerivClient._request``) so the suite
runs offline. The live test at the bottom hits the real Deriv public
gateway — it is skipped (not failed) when the network is unavailable,
so the suite stays green offline, but it genuinely verifies the feed
when run with network access.
"""

from __future__ import annotations

import pytest

from app.deriv_client import DerivClient
from app.market_data import MarketDataError

CANDLES_RESP = {
    "echo_req": {"ticks_history": "frxXAUUSD"},
    "msg_type": "candles",
    "candles": [
        {"close": 1900.80, "epoch": 1700049600, "high": 1901.50,
         "low": 1899.00, "open": 1900.00},
        {"close": 1901.20, "epoch": 1700050500, "high": 1902.00,
         "low": 1900.10, "open": 1900.80},
    ],
}

TICK_RESP = {
    "echo_req": {"ticks": "frxXAUUSD"},
    "msg_type": "tick",
    "tick": {"ask": 1901.45, "bid": 1901.10, "epoch": 1700050590,
             "pip_size": 2, "quote": 1901.28, "symbol": "frxXAUUSD"},
}


class _StubClient(DerivClient):
    """DerivClient with a canned _request (no sockets)."""

    def __init__(self, responses, **kwargs):
        super().__init__(**kwargs)
        self._responses = list(responses)
        self.requests: list[dict] = []

    def _request(self, payload):
        self.requests.append(payload)
        if not self._responses:
            raise AssertionError("no more canned responses")
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def test_copy_rates_parses_candles():
    client = _StubClient([CANDLES_RESP])
    rates = client.copy_rates("XAUUSD", "M15", 200)
    assert len(rates.rates) == 2
    first = rates.rates[0]
    assert first["time"] == 1700049600
    assert first["open"] == 1900.00
    assert first["high"] == 1901.50
    assert first["low"] == 1899.00
    assert first["close"] == 1900.80
    assert first["spread"] is None  # Deriv candles carry no spread
    assert first["tick_volume"] == 0
    req = client.requests[0]
    assert req["ticks_history"] == "frxXAUUSD"
    assert req["granularity"] == 900
    assert req["style"] == "candles"


def test_copy_rates_h1_granularity():
    client = _StubClient([CANDLES_RESP])
    client.copy_rates("XAUUSD", "H1", 80)
    assert client.requests[0]["granularity"] == 3600


def test_copy_rates_paginates_over_1000():
    """count > 1000 pages backwards (Deriv caps one request at 1000)."""
    batch1 = {"msg_type": "candles", "candles": [
        {"close": 1900.0 + i, "epoch": 1700000000 + i * 900,
         "high": 1901.0, "low": 1899.0, "open": 1900.0}
        for i in range(1000)
    ]}
    batch2 = {"msg_type": "candles", "candles": [
        {"close": 1890.0 + i, "epoch": 1699000000 + i * 900,
         "high": 1891.0, "low": 1889.0, "open": 1890.0}
        for i in range(500)
    ]}
    client = _StubClient([batch1, batch2])
    rates = client.copy_rates("XAUUSD", "M15", 1500)
    assert len(rates.rates) == 1500
    assert len(client.requests) == 2
    assert client.requests[1]["end"] == 1700000000 - 1
    # Oldest first.
    assert rates.rates[0]["time"] < rates.rates[-1]["time"]


def test_symbol_tick_parses_quote():
    client = _StubClient([TICK_RESP])
    tick = client.symbol_tick("XAUUSD")
    assert tick is not None
    assert tick.bid == 1901.10
    assert tick.ask == 1901.45
    # Spread in points: $0.35 / 0.01 = 35 points.
    assert tick.spread == pytest.approx(35.0)
    assert tick.time == 1700050590


def test_symbol_tick_none_when_feed_down():
    client = _StubClient([MarketDataError("all endpoints down")])
    assert client.symbol_tick("XAUUSD") is None


def test_connect_uses_tick_probe():
    client = _StubClient([TICK_RESP])
    client.connect()
    assert client.feed_status()["connected"] is True


def test_connect_fails_when_no_tick():
    client = _StubClient([MarketDataError("all endpoints down")])
    with pytest.raises(MarketDataError, match="connect failed"):
        client.connect()


def test_deriv_error_object_raises(monkeypatch):
    """A Deriv {"error": ...} payload surfaces as MarketDataError."""
    import json

    import app.deriv_client as dc

    class FakeWS:
        def send(self, payload):
            pass

        def recv(self):
            return json.dumps({
                "error": {"code": "InvalidSymbol", "message": "bad symbol"},
                "msg_type": "error",
            })

        def close(self):
            pass

    monkeypatch.setattr(dc, "create_connection", lambda *a, **k: FakeWS())
    client = DerivClient()
    with pytest.raises(MarketDataError, match="InvalidSymbol"):
        client.copy_rates("XAUUSD", "M15", 10)


def test_unsupported_symbol_and_timeframe():
    client = _StubClient([CANDLES_RESP])
    with pytest.raises(MarketDataError, match="unsupported symbol"):
        client.copy_rates("EURUSD", "M15", 10)
    with pytest.raises(MarketDataError, match="unsupported timeframe"):
        client.copy_rates("XAUUSD", "W1", 10)


def test_symbol_info_and_no_positions():
    client = _StubClient([CANDLES_RESP])
    info = client.symbol_info("XAUUSD")
    assert info.point == 0.01
    assert info.trade_contract_size == 100.0
    assert client.positions() == []
    assert client.select_symbol("XAUUSD") is True
    acct = client.account_info()
    assert acct.balance == 10000.0
    assert acct.trade_mode == "paper"


# ---------------------------------------------------------------------------
# Live test (real Deriv public gateway; skipped when unreachable)
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_live_tick_and_candles():
    """Live verification against Deriv's public gateway (no auth)."""
    client = DerivClient()
    try:
        client.connect()
        tick = client.symbol_tick("XAUUSD")
        rates = client.copy_rates("XAUUSD", "M15", 5)
    except MarketDataError as exc:
        pytest.skip(f"Deriv gateway unreachable from this environment: {exc}")
    finally:
        client.shutdown()
    assert tick is not None
    assert tick.bid > 0 and tick.ask >= tick.bid
    assert tick.time > 1_700_000_000
    assert len(rates.rates) == 5
    assert rates.rates[-1]["close"] > 0
    # Candles are 15 minutes apart and oldest-first.
    times = [r["time"] for r in rates.rates]
    assert times == sorted(times)
    assert all(b - a == 900 for a, b in zip(times, times[1:]))
