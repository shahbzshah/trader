"""FastAPI dashboard backend (signals-only).

Endpoints:
    GET  /health        process alive + feed status (Render keep-alive)
    GET  /api/state     feed, kill switch, today's decisions/signals, last signal
    GET  /api/signals   recent signaled setups (approved propose decisions)
    GET  /api/decisions?limit=
    GET  /api/candles   recent M15 candles (epoch/open/high/low/close)
    GET  /api/config    read-only connection + strategy params (no secrets)
    POST /explain       store an AI explanation on a signal record
    POST /api/kill
    POST /api/resume
    GET  /              serves the static dashboard page

Signal history survives deploys: outcomes and explanations are kept
in ``data/history_snapshot.json`` (latest 100 records) plus an
append-only ``data/signal_history.jsonl`` archive (see app/history.py),
loaded at startup and re-checked against live candles/ticks on every
poll, so still-open signals resume TP/SL tracking after a restart.

/api/state, /api/candles, /api/signals, /api/config and the dashboard
page are gzip-compressed when the client sends Accept-Encoding: gzip
and the body is >= 1024 bytes (same approach as the Crash bot's
health.py); clients that never ask get plain JSON unchanged.

The dashboard page lives at app/web/static/index.html and is served as a
single-file vanilla-JS app. It polls /api/state every 2s and
/api/signals + /api/decisions every 15s.

No account, no positions, no equity, no orders — the signals-only bot
has none of those. The feed client is best-effort: when the feed is
down the API returns feed_ok=false and the page says so.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator, model_validator

from app.config import load_config
from app.history import (
    append_archive,
    archive_path,
    load_snapshot,
    snapshot_path,
    write_snapshot,
)
from app.journal import Journal
from app.market_data import XAUUSD_SPEC
from app.news import get_events, in_blackout
from app.risk import position_size

logger = logging.getLogger(__name__)

_DEFAULT_LIMIT = 50
_DAY_CACHE_TTL_S = 120.0
_NEWS_CACHE_TTL_S = 120.0
_CANDLES_CACHE_TTL_S = 60.0


class ExplainRequest(BaseModel):
    """AI explanation for one signal. ``signal_id`` (journal id) or
    ``epoch`` (signal timestamp) locates the signal; ``explanation``
    is stored verbatim with the signal record. Never touches signals,
    entries, SL/TP or outcomes.

    Module-level (not inside create_app): with
    ``from __future__ import annotations`` FastAPI can only resolve a
    body-model annotation that lives in module globals.
    """

    signal_id: int | None = None
    epoch: int | None = None
    explanation: str
    model: str | None = None

    @model_validator(mode="after")
    def _needs_locator(self) -> "ExplainRequest":
        if self.signal_id is None and self.epoch is None:
            raise ValueError("signal_id or epoch is required")
        return self

    @field_validator("explanation")
    @classmethod
    def _explanation_nonempty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("explanation must not be empty")
        if len(v) > 10000:
            raise ValueError("explanation too long (max 10000 chars)")
        return v

    @field_validator("model")
    @classmethod
    def _model_capped(cls, v: str | None) -> str | None:
        if v is not None and len(v) > 200:
            raise ValueError("model too long (max 200 chars)")
        return v


def create_app(
    config_path: Path | str = "config.yaml",
    *,
    client_factory: Any = None,
    components: dict | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``client_factory`` is optional and used by tests. ``components`` is
    the dict built by ``app.main._build_components`` (serve mode) —
    when given, the dashboard shares its journal + feed client.
    """
    cfg = load_config(config_path)
    if components is not None:
        journal = components["journal"]
    else:
        journal = Journal(Path(cfg.paths.data_dir) / "journal.db")

    # Durable signal history (survives deploys): snapshot + archive in
    # the data dir. A missing/corrupt snapshot loads as empty — startup
    # must never fail on it (see app/history.py).
    _data_dir = Path(cfg.paths.data_dir)
    _history_snapshot = snapshot_path(_data_dir)
    _history_archive = archive_path(_data_dir)
    _history_records = load_snapshot(_history_snapshot)
    _history: dict[int, dict] = {r["id"]: r for r in _history_records}

    state: dict[str, Any] = {
        "client": components["client"] if components else None,
        "client_factory": client_factory,
        "components": components,
        "kill_switch_path": Path(cfg.paths.kill_switch_path),
        "config_path": str(config_path),
        "started_at": time.time(),
        # Signal id -> durable history record (enriched signal +
        # outcome + optional AI explanation). Journal rows are the
        # source of truth for signals; history restores records the
        # journal lost on a redeploy and carries outcomes/explanations
        # the journal schema has no columns for.
        "history": _history,
        "history_snapshot_path": _history_snapshot,
        "history_archive_path": _history_archive,
        # TTL caches for the dashboard's live blocks (each entry is
        # {"ts": float, "data": ...}); a failed fetch caches its
        # failure briefly so one dead feed cannot stall every poll.
        "day_cache": {"ts": 0.0, "data": None},
        "news_cache": {"ts": 0.0, "data": None},
        "candles_cache": {"ts": 0.0, "data": None},
        # Signal id -> resolved outcome {"status", "close_ts",
        # "close_price", "r"}; once a signal hits TP/SL it never changes.
        # Seeded from the restored snapshot so resolved signals stay
        # resolved across a restart.
        "resolved_outcomes": {
            r["id"]: {"status": r["status"], "close_ts": r.get("close_ts"),
                      "close_price": r.get("close_price"), "r": r.get("r")}
            for r in _history_records if r["status"] in ("TP", "SL")
        },
        # Last time a feed build/connect failed; retries are throttled so
        # a dead feed cannot make /health (polled by a pinger) hang.
        "client_failed_at": 0.0,
    }

    def _get_client() -> Any:
        """Return the feed client, or None if no feed is reachable.

        A failed build/connect is retried at most once a minute.
        """
        if state["client"] is not None:
            return state["client"]
        if time.time() - state["client_failed_at"] < 60.0:
            return None
        factory = state["client_factory"]
        if factory is not None:
            try:
                c = factory()
            except Exception:  # noqa: BLE001
                state["client_failed_at"] = time.time()
                return None
            try:
                c.connect()
            except Exception:  # noqa: BLE001
                state["client_failed_at"] = time.time()
                return None
            state["client"] = c
            return c
        # Production default: build the configured feed (auto).
        try:
            from app.config import EnvSecrets
            from app.main import build_market_client

            secrets = EnvSecrets.from_env(
                Path(state["config_path"]).resolve().parent / ".env")
            c, _source = build_market_client(cfg, secrets)
        except Exception:  # noqa: BLE001
            state["client_failed_at"] = time.time()
            return None
        state["client"] = c
        return c

    def _disconnect_client() -> None:
        if state["components"] is not None:
            return  # serve mode owns the client lifecycle
        c = state.get("client")
        if c is None:
            return
        try:
            c.shutdown()
        except Exception:  # noqa: BLE001
            pass
        state["client"] = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        _disconnect_client()

    app = FastAPI(title="Gold Signals Dashboard", version="0.2.0", lifespan=lifespan)

    # --- helpers ------------------------------------------------------

    def _kill_engaged() -> bool:
        try:
            return state["kill_switch_path"].exists()
        except OSError:
            return False

    def _feed_block() -> dict:
        client = _get_client()
        if client is None:
            return {"name": None, "connected": False, "feed_ok": False,
                    "last_tick_epoch": None, "last_tick_age_s": None,
                    "last_error": "no feed available"}
        try:
            status = client.feed_status()
        except Exception:  # noqa: BLE001
            return {"name": getattr(client, "name", None), "connected": False,
                    "feed_ok": False, "last_tick_epoch": None,
                    "last_tick_age_s": None, "last_error": "feed_status failed"}
        last_tick = status.get("last_tick_epoch")
        age = int(time.time()) - int(last_tick) if last_tick else None
        # A feed is "ok" when it connected and its last tick is fresh
        # (under 10 minutes old — the scan cycle runs every 15 min, and
        # the Deriv stream updates the tick continuously when running).
        feed_ok = bool(status.get("connected")) and (age is None or age < 600)
        return {
            "name": status.get("name"),
            "connected": bool(status.get("connected")),
            "feed_ok": feed_ok,
            "last_tick_epoch": last_tick,
            "last_tick_age_s": age,
            "tick_count": status.get("tick_count"),
            "last_error": status.get("last_error"),
        }

    def _last_signal() -> dict | None:
        rows = journal.list_decisions(limit=50)
        for row in rows:
            if row.get("action") == "propose" and row.get("approved"):
                sig = {"ts": row.get("ts"), "symbol": row.get("symbol"),
                       "confidence": row.get("confidence"),
                       "reasoning": row.get("reasoning")}
                raw = row.get("raw_llm_json") or ""
                try:
                    prev = json.loads(raw) if raw else {}
                except ValueError:
                    prev = {}
                sig.update({
                    "direction": prev.get("direction"),
                    "entry": prev.get("price"),
                    "stop_loss": prev.get("stop_loss"),
                    "take_profit": prev.get("take_profit"),
                    "risk_pct": prev.get("risk_pct"),
                })
                return sig
        return None

    # --- live dashboard blocks (additive; all fail-soft) ---------------

    def _hypothetical_balance() -> float:
        try:
            return float(
                getattr(getattr(cfg, "signals", None),
                        "hypothetical_balance_usd", 10000.0) or 10000.0
            )
        except (TypeError, ValueError):
            return 10000.0

    def _tick_block() -> dict:
        """Latest quote: prefer the feed's live tick stream when one is
        running (Deriv in serve mode), else a one-shot tick request."""
        client = _get_client()
        tick = None
        if client is not None:
            tick = getattr(client, "last_tick", None)
            if tick is None:
                try:
                    tick = client.symbol_tick("XAUUSD")
                except Exception:  # noqa: BLE001
                    tick = None
        if tick is None:
            return {"bid": None, "ask": None, "mid": None,
                    "spread_points": None, "spread_pips": None,
                    "ts": None, "age_s": None}
        spread = tick.spread
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "mid": (tick.bid + tick.ask) / 2.0,
            "spread_points": spread,
            "spread_pips": (spread / 10.0 if spread is not None else None),
            "ts": tick.time,
            "age_s": int(time.time()) - int(tick.time),
        }

    def _utc_midnight_ts() -> int:
        now = _dt.datetime.now(_dt.timezone.utc)
        return int(_dt.datetime(now.year, now.month, now.day,
                                tzinfo=_dt.timezone.utc).timestamp())

    def _day_block() -> dict | None:
        """Today's (UTC) day stats from M15 candles, cached 2 minutes."""
        now = time.time()
        cache = state["day_cache"]
        if now - cache["ts"] < _DAY_CACHE_TTL_S and cache["data"] is not None:
            return cache["data"]
        data: dict | None = None
        client = _get_client()
        if client is not None:
            try:
                rates = client.copy_rates("XAUUSD", "M15", 200)
                rows = [r for r in rates.rates
                        if r["time"] >= _utc_midnight_ts()]
                if rows:
                    day_open = rows[0]["open"]
                    last_close = rows[-1]["close"]
                    change = last_close - day_open
                    data = {
                        "open": day_open,
                        "low": min(r["low"] for r in rows),
                        "high": max(r["high"] for r in rows),
                        "last_close": last_close,
                        "change": change,
                        "change_pct": (change / day_open * 100.0
                                       if day_open else None),
                    }
            except Exception:  # noqa: BLE001
                data = None
        state["day_cache"] = {"ts": now, "data": data}
        return data

    def _event_dict(event: Any) -> dict:
        return {
            "title": event.title,
            "currency": event.currency,
            "time_utc": event.time_utc.isoformat(),
            "impact": event.impact,
        }

    def _news_block() -> dict:
        """News blackout state from the ForexFactory calendar (the same
        feed the bot's blackout uses). Cached 2 minutes; fail-open:
        when the calendar cannot be read, ``available`` is False and no
        blackout is claimed."""
        now = time.time()
        cache = state["news_cache"]
        if now - cache["ts"] < _NEWS_CACHE_TTL_S and cache["data"] is not None:
            return cache["data"]
        window = 30.0
        try:
            window = float(
                getattr(getattr(cfg, "news", None), "window_minutes", 30)
                or 30
            )
        except (TypeError, ValueError):
            window = 30.0
        enabled = bool(getattr(getattr(cfg, "news", None), "enabled", True))
        data: dict[str, Any] = {
            "enabled": enabled,
            "window_minutes": window,
            "available": False,
            "blackout_active": False,
            "active_event": None,
            "next_event": None,
        }
        try:
            events = get_events(cfg)
            now_utc = _dt.datetime.now(_dt.timezone.utc)
            # Staleness guard: get_events is fail-open by design — on a
            # fetch failure it silently returns the disk cache, however
            # old. A cache older than the TTL is not a live calendar, so
            # do not claim the week is clear on its say-so; report the
            # calendar as unavailable instead. (The bot's own blackout
            # check is separate and stays fail-open.)
            fresh = False
            try:
                cache_path = Path(getattr(
                    getattr(cfg, "news", None), "cache_path",
                    "data/news_cache.json"))
                payload = json.loads(
                    cache_path.read_text(encoding="utf-8"))
                fetched_at = _dt.datetime.fromisoformat(
                    str(payload.get("fetched_at", "")))
                if fetched_at.tzinfo is None:
                    fetched_at = fetched_at.replace(
                        tzinfo=_dt.timezone.utc)
                ttl = float(getattr(
                    getattr(cfg, "news", None), "cache_ttl_hours", 12)
                    or 12)
                fresh = (_dt.datetime.now(_dt.timezone.utc) - fetched_at) \
                    < _dt.timedelta(hours=ttl)
            except Exception:  # noqa: BLE001 - unreadable cache = stale
                fresh = False
            data["available"] = fresh
            if not fresh:
                state["news_cache"] = {"ts": now, "data": data}
                return data
            if enabled:
                active = in_blackout(now_utc, events, window)
                if active is not None:
                    data["blackout_active"] = True
                    data["active_event"] = _event_dict(active)
            upcoming = [e for e in events if e.time_utc > now_utc]
            if upcoming:
                nxt = upcoming[0]
                entry = _event_dict(nxt)
                entry["minutes_until"] = max(
                    0, int((nxt.time_utc - now_utc).total_seconds() // 60))
                data["next_event"] = entry
        except Exception:  # noqa: BLE001 - fail-open, never break /api/state
            data["available"] = False
        state["news_cache"] = {"ts": now, "data": data}
        return data

    def _enrich_signal(row: dict) -> dict:
        """One journaled approved proposal -> dashboard signal dict,
        with paper sizing recomputed from the stored entry/SL/risk%
        (sizing is not journaled; the formula is the same one
        signals.size_signal uses: lots = risk$ / (SL$ x 100))."""
        try:
            prev = json.loads(row.get("raw_llm_json") or "{}")
        except ValueError:
            prev = {}
        if not isinstance(prev, dict):
            prev = {}
        entry = prev.get("price")
        stop_loss = prev.get("stop_loss")
        take_profit = prev.get("take_profit")
        risk_pct = prev.get("risk_pct")
        balance = _hypothetical_balance()
        risk_usd = None
        lots = None
        per_dollar = None
        rr = None
        try:
            risk_usd = (balance * float(risk_pct) / 100.0
                        if risk_pct is not None else None)
        except (TypeError, ValueError):
            risk_usd = None
        sl_dist = (abs(float(entry) - float(stop_loss))
                   if entry is not None and stop_loss is not None else None)
        if risk_usd is not None and sl_dist:
            # Same sizing as signals.size_signal: position_size floors
            # lots down to the 0.01 volume step (0.0667 -> 0.06), so the
            # dashboard never shows a size the bot would not send.
            lots = position_size(
                equity=balance, risk_pct=float(risk_pct),
                entry=float(entry), stop_loss=float(stop_loss),
                symbol_info=XAUUSD_SPEC)
            per_dollar = lots * 100.0
        if sl_dist and entry is not None and take_profit is not None:
            rr = abs(float(take_profit) - float(entry)) / sl_dist
        return {
            "id": row.get("id"),
            "ts": row.get("ts"),
            "symbol": row.get("symbol"),
            "direction": prev.get("direction"),
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_pct": risk_pct,
            "lots": lots,
            "risk_usd": risk_usd,
            "per_dollar_usd": per_dollar,
            "rr": rr,
            "confidence": row.get("confidence"),
            "playbook": row.get("playbook"),
            "reasoning": row.get("reasoning"),
            # Outcome + explanation live in the durable history, not
            # the journal; overlaid by _merged_signals when present.
            "status": "OPEN",
            "close_ts": None,
            "close_price": None,
            "r": None,
            "explanation": None,
            "explanation_model": None,
            "explanation_ts": None,
        }

    def _candles_m15() -> list[dict]:
        """M15 candles (up to 1000), cached 60s, oldest first."""
        now = time.time()
        cache = state["candles_cache"]
        if now - cache["ts"] < _CANDLES_CACHE_TTL_S and cache["data"] is not None:
            return cache["data"]
        rows: list[dict] = []
        client = _get_client()
        if client is not None:
            try:
                rates = client.copy_rates("XAUUSD", "M15", 1000)
                rows = sorted(rates.rates, key=lambda r: r["time"])
            except Exception:  # noqa: BLE001
                rows = []
        state["candles_cache"] = {"ts": now, "data": rows}
        return rows

    def _resolve_one(sig: dict, candles: list[dict]) -> dict:
        """Outcome of one signal against real M15 candles after its
        timestamp: the first candle whose range touches TP or SL decides.
        When both are touched in one candle, the level nearer the candle
        open is taken as hit first. A signal with no touch in the
        available history stays OPEN (history only reaches back days)."""
        entry, sl, tp = sig["entry"], sig["stop_loss"], sig["take_profit"]
        empty = {"status": "OPEN", "close_ts": None,
                 "close_price": None, "r": None}
        if entry is None or sl is None or tp is None:
            return empty
        direction = (sig.get("direction") or "").lower()
        if direction not in ("buy", "sell"):
            return empty
        sl_dist = abs(float(entry) - float(sl))
        rr = sig.get("rr")
        for c in candles:
            if c["time"] <= sig["ts"]:
                continue
            if direction == "buy":
                hit_sl = c["low"] <= sl
                hit_tp = c["high"] >= tp
            else:
                hit_sl = c["high"] >= sl
                hit_tp = c["low"] <= tp
            if hit_sl and hit_tp:
                if abs(c["open"] - sl) <= abs(c["open"] - tp):
                    hit_tp = False
                else:
                    hit_sl = False
            if hit_sl:
                return {"status": "SL", "close_ts": c["time"],
                        "close_price": float(sl), "r": -1.0}
            if hit_tp:
                return {"status": "TP", "close_ts": c["time"],
                        "close_price": float(tp), "r": rr}
        return empty

    def _evaluate_outcomes(signals: list[dict]) -> dict[int, dict]:
        """Signal id -> outcome dict. Resolved (TP/SL) outcomes are
        memoized forever; open ones are re-checked against fresh
        candles on each call."""
        resolved: dict = state["resolved_outcomes"]
        out: dict[int, dict] = {}
        pending = [s for s in signals if s["id"] not in resolved]
        if pending:
            candles = _candles_m15()
            for s in pending:
                out[s["id"]] = _resolve_one(s, candles)
                if out[s["id"]]["status"] in ("TP", "SL"):
                    resolved[s["id"]] = out[s["id"]]
        for s in signals:
            if s["id"] in resolved:
                out[s["id"]] = resolved[s["id"]]
            elif s["id"] not in out:
                out[s["id"]] = {"status": "OPEN", "close_ts": None,
                                "close_price": None, "r": None}
        return out

    # --- gzip (same approach as the Crash bot's health.py) --------------

    _GZIP_MIN_BYTES = 1024

    def _client_accepts_gzip(request: Request) -> bool:
        """True only when the client explicitly asks for gzip. Browsers
        and the dashboard WebView send Accept-Encoding: gzip and
        decompress transparently; clients that never ask keep getting
        plain JSON exactly as before."""
        accept = request.headers.get("accept-encoding", "") or ""
        for token in accept.split(","):
            token = token.strip().lower()
            if token == "gzip" or token.startswith("gzip;"):
                if "q=0" not in token:
                    return True
        return False

    def _gzip_response(request: Request, payload: Any,
                       status_code: int = 200) -> Response:
        """JSON response, gzip-compressed when the client asks and the
        body is >= 1024 bytes; plain JSON otherwise."""
        body = json.dumps(payload).encode("utf-8")
        headers: dict[str, str] = {}
        if _client_accepts_gzip(request) and len(body) >= _GZIP_MIN_BYTES:
            body = gzip.compress(body, compresslevel=6)
            headers["Content-Encoding"] = "gzip"
        return Response(content=body, media_type="application/json",
                        headers=headers, status_code=status_code)

    def _gzip_bytes_response(request: Request, body: bytes,
                             media_type: str) -> Response:
        headers: dict[str, str] = {}
        if _client_accepts_gzip(request) and len(body) >= _GZIP_MIN_BYTES:
            body = gzip.compress(body, compresslevel=6)
            headers["Content-Encoding"] = "gzip"
        return Response(content=body, media_type=media_type, headers=headers)

    # --- durable history --------------------------------------------------

    def _persist_history(changed: list[dict]) -> None:
        """Append changed records to the archive and rewrite the
        snapshot. Best-effort: persistence must never break a request."""
        if not changed:
            return
        for rec in changed:
            append_archive(state["history_archive_path"], rec)
        write_snapshot(state["history_snapshot_path"],
                       list(state["history"].values()))

    def _merged_signals(limit: int = 1000) -> list[dict]:
        """All known signals, newest first: journal rows (source of
        truth) plus history records the journal lost on a redeploy.
        A journal row wins over a restored record with the same id;
        outcome + explanation are overlaid from history either way."""
        history: dict[int, dict] = state["history"]
        rows = journal.list_decisions(limit=1000)
        signals: list[dict] = []
        seen_ids: set[int] = set()
        for row in rows:
            if row.get("action") != "propose" or not row.get("approved"):
                continue
            sig = _enrich_signal(row)
            rec = history.get(sig["id"])
            if rec is not None:
                for key in ("explanation", "explanation_model",
                            "explanation_ts"):
                    if rec.get(key) is not None:
                        sig[key] = rec[key]
            signals.append(sig)
            seen_ids.add(sig["id"])
        # Restored history: records whose journal row is gone (redeploy
        # wiped the SQLite file). They already carry outcome fields.
        for rec in history.values():
            if rec["id"] in seen_ids:
                continue
            signals.append(dict(rec))
        signals.sort(key=lambda s: (s["ts"], s["id"]), reverse=True)
        return signals[:limit]

    def _resolve_from_tick(sig: dict, mid: float,
                           tick_ts: int | None) -> dict | None:
        """Outcome of one still-open signal against the latest tick
        mid-price: a tick through TP/SL resolves immediately instead of
        waiting for the next M15 candle to close. Returns None when the
        tick touches neither level."""
        entry, sl, tp = sig["entry"], sig["stop_loss"], sig["take_profit"]
        if entry is None or sl is None or tp is None:
            return None
        direction = (sig.get("direction") or "").lower()
        if direction not in ("buy", "sell"):
            return None
        rr = sig.get("rr")
        if direction == "buy":
            if mid >= float(tp):
                return {"status": "TP", "close_ts": tick_ts,
                        "close_price": float(tp), "r": rr}
            if mid <= float(sl):
                return {"status": "SL", "close_ts": tick_ts,
                        "close_price": float(sl), "r": -1.0}
        else:
            if mid <= float(tp):
                return {"status": "TP", "close_ts": tick_ts,
                        "close_price": float(tp), "r": rr}
            if mid >= float(sl):
                return {"status": "SL", "close_ts": tick_ts,
                        "close_price": float(sl), "r": -1.0}
        return None

    def _refresh_outcomes(signals: list[dict],
                          tick: dict | None = None) -> dict[int, dict]:
        """Resolve outcomes for every signal against fresh M15 candles
        (via _evaluate_outcomes) and then the latest tick for any still
        open — this is the live path: every /api/state and /api/signals
        poll re-checks open signals, so a TP/SL touch resolves within
        one poll instead of waiting for an on-demand call. Newly seen
        or newly resolved signals are persisted to the history
        archive + snapshot."""
        outcomes = _evaluate_outcomes(signals)
        if tick is None:
            tick = _tick_block()
        mid = tick.get("mid")
        if mid is not None:
            for sig in signals:
                outcome = outcomes.get(sig["id"])
                if outcome is None or outcome["status"] != "OPEN":
                    continue
                res = _resolve_from_tick(sig, float(mid), tick.get("ts"))
                if res is not None:
                    outcomes[sig["id"]] = res
                    state["resolved_outcomes"][sig["id"]] = res
        # Sync the durable history: new signals get a record, resolved
        # ones get their outcome. Only changed records are persisted.
        history: dict[int, dict] = state["history"]
        changed: list[dict] = []
        for sig in signals:
            outcome = outcomes.get(sig["id"]) or {
                "status": "OPEN", "close_ts": None,
                "close_price": None, "r": None,
            }
            rec = history.get(sig["id"])
            merged = dict(sig)
            merged.update(outcome)
            if rec is not None:
                for key in ("explanation", "explanation_model",
                            "explanation_ts"):
                    if rec.get(key) is not None:
                        merged[key] = rec[key]
            if rec is None or any(
                merged.get(k) != rec.get(k) for k in merged
            ):
                history[sig["id"]] = merged
                changed.append(merged)
        _persist_history(changed)
        return outcomes

    def _compute_stats(signals: list[dict],
                       outcomes: dict[int, dict]) -> dict:
        """Ledger stats, computed from resolved outcomes only — never
        invented. Hit rate and avg R are null until at least one signal
        has closed; counts and net R are honest zeros when empty."""
        total = len(signals)
        closed = [s for s in signals
                  if outcomes.get(s["id"], {}).get("status") in ("TP", "SL")]
        wins = [s for s in closed
                if outcomes[s["id"]]["status"] == "TP"]
        losses = len(closed) - len(wins)
        net_r = round(sum(
            float(outcomes[s["id"]]["r"] or 0.0) for s in closed
        ), 4)
        return {
            "total_signals": total,
            "open": total - len(closed),
            "closed": len(closed),
            "wins": len(wins),
            "losses": losses,
            "hit_rate": (round(len(wins) / len(closed), 4)
                         if closed else None),
            "net_r": net_r,
            "avg_r": (round(net_r / len(closed), 4) if closed else None),
        }

    # --- routes -------------------------------------------------------

    @app.get("/health")
    def api_health() -> JSONResponse:
        feed = _feed_block()
        return JSONResponse({
            "ok": True,  # the process is alive; feed health is separate
            "ts": int(time.time()),
            "uptime_s": int(time.time() - state["started_at"]),
            "feed": feed.get("name"),
            "feed_ok": feed.get("feed_ok"),
            "last_tick_age_s": feed.get("last_tick_age_s"),
            "kill_switch": _kill_engaged(),
        })

    @app.get("/api/state")
    def api_state(request: Request) -> Response:
        start_ts, _ = journal.today_utc_window()
        today_decisions = journal.list_decisions(since_ts=start_ts, limit=200)
        signals_today = sum(
            1 for d in today_decisions
            if d["action"] == "propose" and d["approved"]
        )
        price = _tick_block()
        # Live outcome refresh: open signals are re-checked against
        # fresh candles + the latest tick on every poll, and the stats
        # below come from that ledger only.
        signals = _merged_signals(limit=1000)
        outcomes = _refresh_outcomes(signals, tick=price)
        for sig in signals:
            sig.update(outcomes.get(sig["id"], {
                "status": "OPEN", "close_ts": None,
                "close_price": None, "r": None,
            }))
        return _gzip_response(
            request,
            {
                "ts": int(time.time()),
                "uptime_s": int(time.time() - state["started_at"]),
                "scan_interval_minutes": int(
                    getattr(getattr(cfg, "schedule", None),
                            "scan_interval_minutes", 15) or 15
                ),
                "kill_switch": _kill_engaged(),
                "feed": _feed_block(),
                "price": price,
                "day": _day_block(),
                "news": _news_block(),
                "telegram": {
                    "enabled": bool(
                        state["components"]
                        and state["components"].get("telegram") is not None
                        and state["components"]["telegram"].enabled
                    )
                },
                "today": {
                    "decision_count": len(today_decisions),
                    "signals": signals_today,
                    "approved": sum(1 for d in today_decisions if d["approved"]),
                    "rejected": sum(1 for d in today_decisions if not d["approved"]),
                },
                "stats": _compute_stats(signals, outcomes),
                "last_signal": _last_signal(),
            }
        )

    @app.get("/api/signals")
    def api_signals(
        request: Request,
        limit: int = Query(_DEFAULT_LIMIT, ge=1, le=1000),
    ) -> Response:
        signals = _merged_signals(limit=limit)
        outcomes = _refresh_outcomes(signals)
        for sig in signals:
            sig.update(outcomes.get(sig["id"], {
                "status": "OPEN", "close_ts": None,
                "close_price": None, "r": None,
            }))
        return _gzip_response(
            request, {"signals": signals, "count": len(signals)})

    @app.get("/api/candles")
    def api_candles(
        request: Request,
        limit: int = Query(200, ge=1, le=1000),
    ) -> Response:
        """Recent M15 candles for the chart, oldest first:
        epoch/open/high/low/close. Reuses the cached _candles_m15 feed
        fetch (60s TTL)."""
        rows = _candles_m15()
        candles = [
            {"epoch": int(r["time"]), "open": float(r["open"]),
             "high": float(r["high"]), "low": float(r["low"]),
             "close": float(r["close"])}
            for r in rows[-limit:]
        ]
        return _gzip_response(
            request,
            {"symbol": "XAUUSD", "timeframe": "M15",
             "candles": candles, "count": len(candles)})

    @app.get("/api/config")
    def api_config(request: Request) -> Response:
        """Read-only connection + strategy params for the dashboard.
        No secrets, keys or tokens are ever included here."""
        news_cfg = getattr(cfg, "news", None)
        risk_cfg = getattr(cfg, "risk", None)
        sig_cfg = getattr(cfg, "signals", None)
        session_cfg = getattr(cfg, "session_hours", None)
        return _gzip_response(
            request,
            {
                "mode": cfg.mode,
                "feed": getattr(sig_cfg, "feed", "auto"),
                "live_feed": _feed_block().get("name"),
                "symbols": list(cfg.symbols),
                "timeframes": list(cfg.timeframes),
                "timeframe": cfg.timeframes[0] if cfg.timeframes else None,
                "scan_interval_minutes": int(
                    getattr(getattr(cfg, "schedule", None),
                            "scan_interval_minutes", 15) or 15
                ),
                "session_hours": {
                    "start": getattr(session_cfg, "start", None),
                    "end": getattr(session_cfg, "end", None),
                    "timezone": getattr(session_cfg, "timezone", "UTC"),
                    "skip_weekends": bool(
                        getattr(session_cfg, "skip_weekends", True)),
                },
                "kill_zones": list(cfg.kill_zones),
                "news": {
                    "enabled": bool(getattr(news_cfg, "enabled", True)),
                    "window_minutes": int(
                        getattr(news_cfg, "window_minutes", 30) or 30),
                    "min_impact": getattr(news_cfg, "min_impact", "High"),
                    "currencies": list(
                        getattr(news_cfg, "currencies", ["USD"])),
                },
                "risk": {
                    "max_risk_pct_per_trade": float(
                        getattr(risk_cfg, "max_risk_pct_per_trade", 1.0)),
                    "max_daily_loss_pct": float(
                        getattr(risk_cfg, "max_daily_loss_pct", 3.0)),
                    "max_trades_per_day": int(
                        getattr(risk_cfg, "max_trades_per_day", 5)),
                    "min_confidence": float(
                        getattr(risk_cfg, "min_confidence", 0.6)),
                    "default_playbook": getattr(
                        risk_cfg, "default_playbook",
                        "sweep_fvg_reclaim"),
                    "breakeven_enabled": bool(
                        getattr(risk_cfg, "breakeven_enabled", True)),
                    "breakeven_at_r": float(
                        getattr(risk_cfg, "breakeven_at_r", 1.0)),
                },
                "signals": {
                    "hypothetical_balance_usd": float(
                        getattr(sig_cfg, "hypothetical_balance_usd",
                                10000.0)),
                    "max_signals_per_day": int(
                        getattr(sig_cfg, "max_signals_per_day", 5)),
                    "telegram_enabled": bool(
                        getattr(sig_cfg, "telegram_enabled", True)),
                },
            })

    @app.post("/explain")
    def api_explain(payload: ExplainRequest) -> JSONResponse:
        """Store an AI explanation on a signal record (journal or
        restored history). The signal itself is untouched."""
        history: dict[int, dict] = state["history"]
        target_id: int | None = None
        if payload.signal_id is not None:
            target_id = int(payload.signal_id)
            row = journal.get_decision(target_id)
            if target_id not in history and (
                row is None or row.get("action") != "propose"
                or not row.get("approved")
            ):
                raise HTTPException(status_code=404,
                                    detail="signal not found")
        else:
            epoch = int(payload.epoch)  # type: ignore[arg-type]
            matches = [r for r in history.values() if r["ts"] == epoch]
            if not matches:
                rows = journal.list_decisions(limit=1000)
                matches = [
                    _enrich_signal(r) for r in rows
                    if r.get("ts") == epoch
                    and r.get("action") == "propose" and r.get("approved")
                ]
            if not matches:
                raise HTTPException(status_code=404,
                                    detail="signal not found")
            target_id = int(max(matches, key=lambda s: s["id"])["id"])
        now = int(time.time())
        rec = history.get(target_id)
        if rec is None:
            row = journal.get_decision(target_id)
            if row is None:
                raise HTTPException(status_code=404,
                                    detail="signal not found")
            rec = _enrich_signal(row)
            rec.update({"status": "OPEN", "close_ts": None,
                        "close_price": None, "r": None})
        rec = dict(rec)
        rec["explanation"] = payload.explanation
        rec["explanation_model"] = payload.model
        rec["explanation_ts"] = now
        history[target_id] = rec
        _persist_history([rec])
        return JSONResponse({
            "ok": True,
            "signal_id": target_id,
            "explanation": payload.explanation,
            "model": payload.model,
            "ts": now,
        })

    @app.get("/api/decisions")
    def api_decisions(
        request: Request,
        limit: int = Query(_DEFAULT_LIMIT, ge=1, le=1000),
        symbol: str | None = None,
    ) -> Response:
        rows = journal.list_decisions(limit=limit, symbol=symbol)
        return _gzip_response(
            request, {"decisions": rows, "count": len(rows)})

    @app.post("/api/kill")
    def api_kill() -> JSONResponse:
        path: Path = state["kill_switch_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return JSONResponse({"kill_switch": True, "path": str(path)})

    @app.post("/api/resume")
    def api_resume() -> JSONResponse:
        path: Path = state["kill_switch_path"]
        if path.exists():
            path.unlink()
        return JSONResponse({"kill_switch": False, "path": str(path)})

    # --- static --------------------------------------------------------

    static_dir = Path(__file__).resolve().parent / "web" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def root(request: Request) -> Response:
        index = static_dir / "index.html"
        if not index.exists():
            raise HTTPException(status_code=404, detail="dashboard not built")
        return _gzip_bytes_response(
            request, index.read_bytes(), "text/html")

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


# Module-level app for `uvicorn app.server:app`.
def _build_default_app() -> FastAPI:
    return create_app("config.yaml")


app = _build_default_app()


__all__ = ["app", "create_app"]
