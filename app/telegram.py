"""Telegram alerts for signals (send-only, no polling, no trading).

The bot posts each signal to Megah's chat via his own Telegram bot.
Credentials come from the environment only — ``TELEGRAM_BOT_TOKEN``
and ``TELEGRAM_CHAT_ID`` — never hardcoded, never printed, never
committed. The token value is never written to a log line.

Messages are plain text (no Markdown/HTML parse mode) so prices and
symbols can never break formatting.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org"


class TelegramAlerter:
    """Minimal Telegram Bot API sender (sendMessage + getMe)."""

    def __init__(
        self,
        token: str,
        chat_id: str,
        *,
        http_client: httpx.Client | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._token = token or ""
        self._chat_id = str(chat_id or "")
        from app.http_client import make_http_client

        self._http = http_client or make_http_client(timeout=timeout)
        self._owns_http = http_client is None

    @property
    def enabled(self) -> bool:
        """True only when both the token and the chat id are configured."""
        return bool(self._token and self._chat_id)

    def _url(self, method: str) -> str:
        return f"{_API_BASE}/bot{self._token}/{method}"

    def validate_token(self) -> tuple[bool, str]:
        """Check the token with ``getMe`` — validates without sending
        any message. Returns ``(ok, detail)``; detail is the bot's
        username on success, the Telegram error description otherwise.
        Never includes the token in the detail.
        """
        if not self._token:
            return False, "TELEGRAM_BOT_TOKEN is not set"
        try:
            resp = self._http.get(self._url("getMe"))
        except Exception as exc:  # noqa: BLE001
            return False, f"network error: {exc}"
        try:
            body = resp.json()
        except ValueError:
            return False, f"HTTP {resp.status_code} (non-JSON reply)"
        if resp.status_code == 200 and body.get("ok"):
            username = (body.get("result") or {}).get("username", "?")
            return True, f"@{username}"
        return False, str(body.get("description", f"HTTP {resp.status_code}"))

    def send_message(self, text: str) -> bool:
        """Post ``text`` to the configured chat. Returns True on
        Telegram ``ok``. Failures are logged (without the token) and
        return False — a failed alert never crashes a scan cycle."""
        if not self.enabled:
            logger.warning("Telegram not configured; alert not sent")
            return False
        try:
            resp = self._http.post(
                self._url("sendMessage"),
                json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                },
            )
            body = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram send failed (network): %s", exc)
            return False
        if resp.status_code == 200 and body.get("ok"):
            return True
        logger.warning(
            "Telegram send failed: HTTP %s %s",
            resp.status_code,
            str(body.get("description", ""))[:200],
        )
        return False

    def close(self) -> None:
        if self._owns_http:
            try:
                self._http.close()
            except Exception:  # noqa: BLE001 - best-effort
                pass


def format_signal_text(signal: Any) -> str:
    """Render one signal as a plain-text Telegram message.

    ``signal`` is a :class:`app.signals.Signal` (duck-typed here to
    avoid a circular import).
    """
    arrow = "BUY" if signal.direction == "buy" else "SELL"
    rr = signal.take_profit - signal.entry
    risk = abs(signal.entry - signal.stop_loss)
    rr_ratio = abs(rr) / risk if risk > 0 else 0.0
    lines = [
        f"GOLD SIGNAL — {arrow} {signal.symbol}",
        f"Entry (limit): {signal.entry:.2f}",
        f"Stop loss: {signal.stop_loss:.2f}",
        f"Take profit: {signal.take_profit:.2f}  ({rr_ratio:.1f}R)",
        "",
        f"Risk: ${signal.risk_usd:.2f}  •  Size: {signal.lots:.2f} lots",
        f"$ per $1 move: ${signal.per_dollar_usd:.2f}",
        f"Confidence: {signal.confidence:.0%}  •  {signal.playbook}",
        "",
        signal.reasoning,
        "",
        f"Feed: {signal.source_feed}  •  {signal.ts_label} UTC",
    ]
    return "\n".join(lines)


__all__ = ["TelegramAlerter", "format_signal_text"]
