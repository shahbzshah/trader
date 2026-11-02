"""Deriv WebSocket v3 client — the FALLBACK / cross-check feed.

No auth: Deriv's public API needs only an ``app_id`` (1089 is Deriv's
public demo app, the same one the Crash 1000 bot uses). No token, no
account, nothing secret — there is nothing to leak.

Endpoints are tried in order (same failover list as the Crash bot;
the public gateway is first because it is the one that answers from
this environment — ``ws.derivws.com`` currently returns Cloudflare 520s):

* ``wss://api.derivws.com/trading/v1/options/ws/public``
* ``wss://ws.derivws.com/websockets/v3``
* ``wss://green.derivws.com/websockets/v3``
* ``wss://blue.derivws.com/websockets/v3``
* ``wss://red.derivws.com/websockets/v3``

Requests used (read-only):

* ``ticks_history`` with ``style: "candles"`` — native M15 (900s) and
  H1 (3600s) candles for ``frxXAUUSD``. Deriv caps one request at
  1000 candles, so larger histories are paginated backwards.
* ``ticks`` — one-shot latest quote (``symbol_tick``).
* ``ticks`` with ``subscribe: 1`` — the live tick stream
  (:meth:`DerivClient.start_tick_stream`), which the dashboard's
  /health endpoint uses to report feed staleness.

Notes:
* Deriv candles carry no volume and no spread; the scanner's spread
  filter therefore falls back to the live tick spread (in points).
* Tick rate on frxXAUUSD is roughly 1 tick/second.
* The client is synchronous (``websocket-client``) to match the rest of
  the codebase; each request opens a short-lived connection. The tick
  stream runs on a background thread with auto-reconnect.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from websocket import WebSocketApp, create_connection

from app.market_data import (
    BOT_SYMBOL,
    DERIV_SYMBOL,
    MarketDataClient,
    MarketDataError,
    PaperAccountInfo,
    Rates,
    SymbolInfo,
    Tick,
    XAUUSD_SPEC,
)

logger = logging.getLogger(__name__)

APP_ID = 1089  # Deriv's public demo app id (no auth, no secret)

ENDPOINTS: tuple[str, ...] = (
    "wss://api.derivws.com/trading/v1/options/ws/public",
    "wss://ws.derivws.com/websockets/v3",
    "wss://green.derivws.com/websockets/v3",
    "wss://blue.derivws.com/websockets/v3",
    "wss://red.derivws.com/websockets/v3",
)

_TF_TO_GRANULARITY = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800,
                      "H1": 3600, "H4": 14400, "D1": 86400}
_MAX_PER_REQUEST = 1000
_REQUEST_TIMEOUT = 20.0


def _ws_url(endpoint: str) -> str:
    return f"{endpoint}?app_id={APP_ID}"


class DerivClient(MarketDataClient):
    """Deriv v3 feed. No credentials of any kind."""

    name = "deriv"

    def __init__(
        self,
        *,
        hypothetical_balance: float = 10000.0,
        endpoints: tuple[str, ...] = ENDPOINTS,
        timeout: float = _REQUEST_TIMEOUT,
    ) -> None:
        self._balance = float(hypothetical_balance)
        self._endpoints = endpoints
        self._timeout = timeout
        self._connected = False
        self._last_tick_epoch: int | None = None
        self._last_tick: Tick | None = None
        self._last_error: str | None = None
        self._stream_thread: threading.Thread | None = None
        self._stream_stop = threading.Event()
        self._stream_lock = threading.Lock()
        self._tick_count = 0

    # --- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        """Probe: one-shot tick request. Raises :class:`MarketDataError`
        when every endpoint fails."""
        tick = self._one_shot_tick(DERIV_SYMBOL)
        if tick is None:
            raise MarketDataError("Deriv connect failed: no tick from any endpoint")
        self._connected = True
        self._last_error = None
        logger.info("Deriv feed connected (tick %s)", tick.time)

    def shutdown(self) -> None:
        self.stop_tick_stream()
        self._connected = False

    # --- MarketDataClient interface ----------------------------------------

    def copy_rates(self, symbol: str, timeframe: str, count: int) -> Rates:
        deriv_symbol = self._deriv_symbol(symbol)
        granularity = _TF_TO_GRANULARITY.get(timeframe.upper())
        if granularity is None:
            raise MarketDataError(f"unsupported timeframe for Deriv: {timeframe!r}")
        count = max(1, int(count))
        rows: list[dict] = []
        end: str | int = "latest"
        remaining = count
        # Deriv caps one ticks_history request at 1000 candles; page
        # backwards from the earliest bar we already have.
        while remaining > 0:
            batch = min(remaining, _MAX_PER_REQUEST)
            payload: dict[str, Any] = {
                "ticks_history": deriv_symbol,
                "end": end,
                "count": batch,
                "granularity": granularity,
                "style": "candles",
            }
            resp = self._request(payload)
            candles = resp.get("candles") or []
            if not candles:
                break
            batch_rows = [self._candle_row(c) for c in candles]
            rows = batch_rows + rows
            remaining -= len(candles)
            if len(candles) < batch:
                break  # no more history available
            end = int(candles[0]["epoch"]) - 1
        rows.sort(key=lambda r: r["time"])
        return Rates(rates=rows[-count:] if len(rows) > count else rows)

    def symbol_tick(self, symbol: str) -> Tick | None:
        tick = self._one_shot_tick(self._deriv_symbol(symbol))
        if tick is not None:
            self._last_tick_epoch = tick.time
        return tick

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
            server="deriv-public",
        )

    def feed_status(self) -> dict:
        with self._stream_lock:
            tick_count = self._tick_count
        return {
            "name": self.name,
            "connected": self._connected,
            "last_tick_epoch": self._last_tick_epoch,
            "tick_count": tick_count,
            "stream_running": self._stream_thread is not None
            and self._stream_thread.is_alive(),
            "last_error": self._last_error,
        }

    # --- live tick stream ----------------------------------------------------

    def start_tick_stream(self, symbol: str = BOT_SYMBOL) -> None:
        """Start the background tick subscription (idempotent).

        Updates ``last_tick`` / ``tick_count`` as ticks arrive and
        reconnects automatically. Used by the dashboard's /health
        endpoint to report feed staleness while the bot runs.
        """
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return
        self._stream_stop.clear()
        deriv_symbol = self._deriv_symbol(symbol)
        thread = threading.Thread(
            target=self._stream_loop,
            args=(deriv_symbol,),
            name="deriv-tick-stream",
            daemon=True,
        )
        self._stream_thread = thread
        thread.start()
        logger.info("Deriv tick stream started (%s)", deriv_symbol)

    def stop_tick_stream(self) -> None:
        self._stream_stop.set()
        thread = self._stream_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._stream_thread = None

    @property
    def last_tick(self) -> Tick | None:
        with self._stream_lock:
            return self._last_tick

    # --- internals -----------------------------------------------------------

    def _deriv_symbol(self, symbol: str) -> str:
        if symbol.upper() in (BOT_SYMBOL, DERIV_SYMBOL):
            return DERIV_SYMBOL
        raise MarketDataError(f"unsupported symbol for Deriv: {symbol!r}")

    @staticmethod
    def _candle_row(c: dict[str, Any]) -> dict:
        return {
            "time": int(c["epoch"]),
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
            "tick_volume": 0,
            "spread": None,
            "real_volume": 0,
        }

    @staticmethod
    def _tick_from_msg(msg: dict[str, Any]) -> Tick | None:
        tick = msg.get("tick") or {}
        try:
            epoch = int(tick["epoch"])
            quote = float(tick["quote"])
        except (KeyError, TypeError, ValueError):
            return None
        bid = tick.get("bid")
        ask = tick.get("ask")
        try:
            bid_f = float(bid) if bid is not None else quote
            ask_f = float(ask) if ask is not None else quote
        except (TypeError, ValueError):
            bid_f = ask_f = quote
        point = XAUUSD_SPEC.point
        spread = (ask_f - bid_f) / point if bid is not None and ask is not None else None
        return Tick(bid=bid_f, ask=ask_f, spread=spread, time=epoch)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One-shot request/response, trying each endpoint in order.

        Raises :class:`MarketDataError` when every endpoint fails or
        the response carries a Deriv ``error`` object.
        """
        last_exc: Exception | None = None
        for endpoint in self._endpoints:
            try:
                ws = create_connection(_ws_url(endpoint), timeout=self._timeout)
                try:
                    ws.send(json.dumps(payload))
                    raw = ws.recv()
                finally:
                    ws.close()
                resp = json.loads(raw)
            except Exception as exc:  # noqa: BLE001 - try next endpoint
                last_exc = exc
                logger.debug("Deriv endpoint %s failed: %s", endpoint, exc)
                continue
            if isinstance(resp, dict) and resp.get("error"):
                err = resp["error"]
                raise MarketDataError(
                    f"Deriv error {err.get('code')}: {err.get('message')}"
                )
            return resp if isinstance(resp, dict) else {}
        raise MarketDataError(
            f"Deriv request failed on all endpoints: {last_exc}"
        )

    def _one_shot_tick(self, deriv_symbol: str) -> Tick | None:
        try:
            resp = self._request({"ticks": deriv_symbol})
        except MarketDataError as exc:
            self._last_error = str(exc)
            return None
        return self._tick_from_msg(resp)

    def _stream_loop(self, deriv_symbol: str) -> None:
        """Reconnecting subscription loop (background thread).

        Rotates across endpoints on failure so one dead host cannot
        stall the stream forever.
        """
        backoff = 1.0
        endpoint_idx = 0
        while not self._stream_stop.is_set():
            endpoint = self._endpoints[endpoint_idx % len(self._endpoints)]
            try:
                self._stream_once(deriv_symbol, endpoint)
                backoff = 1.0  # clean run resets the backoff
            except Exception as exc:  # noqa: BLE001 - reconnect, don't die
                self._last_error = str(exc)
                endpoint_idx += 1
                logger.warning("Deriv tick stream dropped (%s); retrying", exc)
            if self._stream_stop.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def _stream_once(self, deriv_symbol: str, endpoint: str) -> None:
        """One subscription session on ``endpoint``; returns when the
        socket closes (the loop then reconnects)."""
        app = WebSocketApp(
            _ws_url(endpoint),
            on_open=lambda ws, _s=deriv_symbol: ws.send(
                json.dumps({"ticks": _s, "subscribe": 1})
            ),
            on_message=self._on_stream_message,
            on_error=lambda ws, err: logger.debug("Deriv stream error: %s", err),
        )
        # run_forever blocks until the socket closes or an error occurs.
        app.run_forever(ping_interval=20, ping_timeout=10)

    def _on_stream_message(self, ws: Any, message: str) -> None:  # noqa: ARG002
        try:
            msg = json.loads(message)
        except (TypeError, ValueError):
            return
        if msg.get("msg_type") != "tick":
            return
        tick = self._tick_from_msg(msg)
        if tick is None:
            return
        with self._stream_lock:
            self._last_tick = tick
            self._last_tick_epoch = tick.time
            self._tick_count += 1


__all__ = ["APP_ID", "DerivClient", "ENDPOINTS"]
