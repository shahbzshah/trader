"""OANDA v20 REST client — the MAIN market-data source (signals-only).

Practice environment only: ``https://api-fxpractice.oanda.com``.
Auth is a personal access token + v20 account ID, both from the
environment (``OANDA_API_TOKEN``, ``OANDA_ACCOUNT_ID``) — never
hardcoded, never logged, never committed. No token exists yet; this
client is built against the documented v20 API and tested with mocked
HTTP responses (see ``tests/test_oanda_client.py``).

Endpoints used (read-only — the bot never places orders):

* ``GET /v3/instruments/{instrument}/candles`` — M15/H1 candles
  (``price=MBA``, granularity ``M15``/``H1``).
* ``GET /v3/accounts/{accountID}/pricing`` — latest bid/ask tick.

Notes:
* OANDA candle ``time`` is RFC 3339; converted to epoch seconds (UTC).
* OANDA candles carry ``volume`` (tick volume) but no spread; the
  spread in :class:`Rates` is therefore ``None`` and the strategy's
  spread filter falls back to the live tick spread (in points).
* The last candle returned is the *forming* bar, same as MT5
  ``copy_rates`` — the strategy evaluates it as the latest close.
* ``count`` is clamped to OANDA's 5000-candle limit per request.
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from typing import Any

import httpx

from app.market_data import (
    BOT_SYMBOL,
    OANDA_INSTRUMENT,
    MarketDataClient,
    MarketDataError,
    PaperAccountInfo,
    Rates,
    SymbolInfo,
    Tick,
    XAUUSD_SPEC,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api-fxpractice.oanda.com"
_MAX_COUNT = 5000

_TF_TO_GRANULARITY = {
    "M1": "M1",
    "M5": "M5",
    "M15": "M15",
    "M30": "M30",
    "H1": "H1",
    "H2": "H2",
    "H4": "H4",
    "D1": "D",
}


def _to_epoch(rfc3339: str) -> int:
    dt = _dt.datetime.fromisoformat(rfc3339.replace("Z", "+00:00"))
    return int(dt.timestamp())


class OandaClient(MarketDataClient):
    """OANDA v20 Practice feed. ``http_client`` is injectable for tests."""

    name = "oanda"

    def __init__(
        self,
        token: str,
        account_id: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        hypothetical_balance: float = 10000.0,
        http_client: httpx.Client | None = None,
        timeout: float = 20.0,
    ) -> None:
        if not token:
            raise MarketDataError("OANDA_API_TOKEN is not set")
        if not account_id:
            raise MarketDataError("OANDA_ACCOUNT_ID is not set")
        self._token = token
        self._account_id = account_id
        self._base_url = base_url.rstrip("/")
        self._balance = float(hypothetical_balance)
        from app.http_client import make_http_client

        self._http = http_client or make_http_client(timeout=timeout)
        self._owns_http = http_client is None
        self._connected = False
        self._last_tick_epoch: int | None = None
        self._last_error: str | None = None

    # --- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        """Probe: fetch the account's pricing for XAU_USD.

        Raises :class:`MarketDataError` on 401/403 (bad token), 404 (bad
        account id) or any transport failure.
        """
        try:
            self._pricing()
        except MarketDataError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MarketDataError(f"OANDA connect failed: {exc}") from exc
        self._connected = True
        self._last_error = None
        logger.info("OANDA feed connected (account %s)", _mask(self._account_id))

    def shutdown(self) -> None:
        self._connected = False
        if self._owns_http:
            try:
                self._http.close()
            except Exception:  # noqa: BLE001 - best-effort
                pass

    # --- MarketDataClient interface ----------------------------------------

    def copy_rates(self, symbol: str, timeframe: str, count: int) -> Rates:
        instrument = self._instrument(symbol)
        granularity = _TF_TO_GRANULARITY.get(timeframe.upper())
        if granularity is None:
            raise MarketDataError(f"unsupported timeframe for OANDA: {timeframe!r}")
        count = max(1, min(int(count), _MAX_COUNT))
        try:
            resp = self._http.get(
                f"{self._base_url}/v3/instruments/{instrument}/candles",
                params={"price": "MBA", "granularity": granularity, "count": count},
                headers=self._headers(),
            )
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            raise MarketDataError(f"OANDA candles request failed: {exc}") from exc
        if resp.status_code in (401, 403):
            raise MarketDataError(
                "OANDA auth rejected the token (401/403) — check OANDA_API_TOKEN"
            )
        if resp.status_code != 200:
            raise MarketDataError(
                f"OANDA candles HTTP {resp.status_code}: {resp.text[:200]}"
            )
        rows: list[dict] = []
        for c in resp.json().get("candles", []):
            mid = c.get("mid") or {}
            try:
                rows.append(
                    {
                        "time": _to_epoch(str(c["time"])),
                        "open": float(mid["o"]),
                        "high": float(mid["h"]),
                        "low": float(mid["l"]),
                        "close": float(mid["c"]),
                        "tick_volume": int(c.get("volume", 0) or 0),
                        "spread": None,
                        "real_volume": 0,
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        rows.sort(key=lambda r: r["time"])
        return Rates(rates=rows)

    def symbol_tick(self, symbol: str) -> Tick | None:
        try:
            prices = self._pricing(instruments=[self._instrument(symbol)])
        except MarketDataError as exc:
            self._last_error = str(exc)
            return None
        if not prices:
            return None
        p = prices[0]
        try:
            bid = float(p["bids"][0]["price"])
            ask = float(p["asks"][0]["price"])
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        epoch = _to_epoch(str(p["time"])) if p.get("time") else int(time.time())
        self._last_tick_epoch = epoch
        point = XAUUSD_SPEC.point
        return Tick(bid=bid, ask=ask, spread=(ask - bid) / point, time=epoch)

    def symbol_info(self, symbol: str) -> SymbolInfo:
        return XAUUSD_SPEC

    def select_symbol(self, symbol: str) -> bool:
        return True

    def positions(self, symbol: str | None = None) -> list:
        return []

    def account_info(self) -> PaperAccountInfo:
        return PaperAccountInfo(
            balance=self._balance,
            equity=self._balance,
            trade_mode="paper",
            server="oanda-practice",
        )

    def feed_status(self) -> dict:
        return {
            "name": self.name,
            "connected": self._connected,
            "last_tick_epoch": self._last_tick_epoch,
            "last_error": self._last_error,
        }

    # --- internals -----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def _instrument(self, symbol: str) -> str:
        if symbol.upper() in (BOT_SYMBOL, OANDA_INSTRUMENT):
            return OANDA_INSTRUMENT
        raise MarketDataError(f"unsupported symbol for OANDA: {symbol!r}")

    def _pricing(self, instruments: list[str] | None = None) -> list[dict]:
        names = instruments or [OANDA_INSTRUMENT]
        try:
            resp = self._http.get(
                f"{self._base_url}/v3/accounts/{self._account_id}/pricing",
                params={"instruments": ",".join(names)},
                headers=self._headers(),
            )
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            raise MarketDataError(f"OANDA pricing request failed: {exc}") from exc
        if resp.status_code in (401, 403):
            raise MarketDataError(
                "OANDA auth rejected the token (401/403) — check OANDA_API_TOKEN"
            )
        if resp.status_code == 404:
            raise MarketDataError(
                f"OANDA account not found (404) — check OANDA_ACCOUNT_ID "
                f"({_mask(self._account_id)})"
            )
        if resp.status_code != 200:
            raise MarketDataError(
                f"OANDA pricing HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json().get("prices", [])


def _mask(value: str) -> str:
    """Show only the tail of an account id for logs (never the token)."""
    if len(value) <= 4:
        return "****"
    return f"****{value[-4:]}"


__all__ = ["DEFAULT_BASE_URL", "OandaClient"]
