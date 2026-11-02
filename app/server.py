"""FastAPI dashboard backend (signals-only).

Endpoints:
    GET  /health        process alive + feed status (Render keep-alive)
    GET  /api/state     feed, kill switch, today's decisions/signals, last signal
    GET  /api/signals   recent signaled setups (approved propose decisions)
    GET  /api/decisions?limit=
    POST /api/kill
    POST /api/resume
    GET  /              serves the static dashboard page

The dashboard page lives at app/web/static/index.html and is served as a
single-file vanilla-JS app. It polls /api/state every 10s.

No account, no positions, no equity, no orders — the signals-only bot
has none of those. The feed client is best-effort: when the feed is
down the API returns feed_ok=false and the page says so.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import load_config
from app.journal import Journal

logger = logging.getLogger(__name__)

_DEFAULT_LIMIT = 50


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

    state: dict[str, Any] = {
        "client": components["client"] if components else None,
        "client_factory": client_factory,
        "components": components,
        "kill_switch_path": Path(cfg.paths.kill_switch_path),
        "config_path": str(config_path),
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

    # --- routes -------------------------------------------------------

    @app.get("/health")
    def api_health() -> JSONResponse:
        feed = _feed_block()
        return JSONResponse({
            "ok": True,  # the process is alive; feed health is separate
            "ts": int(time.time()),
            "feed": feed.get("name"),
            "feed_ok": feed.get("feed_ok"),
            "last_tick_age_s": feed.get("last_tick_age_s"),
            "kill_switch": _kill_engaged(),
        })

    @app.get("/api/state")
    def api_state() -> JSONResponse:
        start_ts, _ = journal.today_utc_window()
        today_decisions = journal.list_decisions(since_ts=start_ts, limit=200)
        signals_today = sum(
            1 for d in today_decisions
            if d["action"] == "propose" and d["approved"]
        )
        return JSONResponse(
            {
                "ts": int(time.time()),
                "kill_switch": _kill_engaged(),
                "feed": _feed_block(),
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
                "last_signal": _last_signal(),
            }
        )

    @app.get("/api/signals")
    def api_signals(
        limit: int = Query(_DEFAULT_LIMIT, ge=1, le=1000),
    ) -> JSONResponse:
        rows = journal.list_decisions(limit=1000)
        signals = []
        for row in rows:
            if row.get("action") != "propose" or not row.get("approved"):
                continue
            try:
                prev = json.loads(row.get("raw_llm_json") or "{}")
            except ValueError:
                prev = {}
            signals.append({
                "id": row.get("id"),
                "ts": row.get("ts"),
                "symbol": row.get("symbol"),
                "direction": prev.get("direction"),
                "entry": prev.get("price"),
                "stop_loss": prev.get("stop_loss"),
                "take_profit": prev.get("take_profit"),
                "risk_pct": prev.get("risk_pct"),
                "confidence": row.get("confidence"),
                "playbook": row.get("playbook"),
                "reasoning": row.get("reasoning"),
            })
            if len(signals) >= limit:
                break
        return JSONResponse({"signals": signals, "count": len(signals)})

    @app.get("/api/decisions")
    def api_decisions(
        limit: int = Query(_DEFAULT_LIMIT, ge=1, le=1000),
        symbol: str | None = None,
    ) -> JSONResponse:
        rows = journal.list_decisions(limit=limit, symbol=symbol)
        return JSONResponse({"decisions": rows, "count": len(rows)})

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
    def root() -> FileResponse:
        index = static_dir / "index.html"
        if not index.exists():
            raise HTTPException(status_code=404, detail="dashboard not built")
        return FileResponse(str(index))

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


# Module-level app for `uvicorn app.server:app`.
def _build_default_app() -> FastAPI:
    return create_app("config.yaml")


app = _build_default_app()


__all__ = ["app", "create_app"]
