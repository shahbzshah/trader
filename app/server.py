"""FastAPI dashboard backend.

Endpoints (per KILO_MASTER_PROMPT.md):
    GET  /api/state        account, positions, today's decisions, kill-switch
    GET  /api/decisions?limit=
    GET  /api/equity?days=
    POST /api/kill
    POST /api/resume
    GET  /api/orders?limit=
    GET  /                   serves the static dashboard page

The dashboard page lives at app/web/static/index.html and is served as a
single-file vanilla-JS app. It polls /api/state every 10s and renders
the equity chart on a canvas.

The MT5 client is best-effort: if the terminal isn't running, the API
returns account/positions=null and the page shows "MT5 disconnected".
"""

from __future__ import annotations

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
) -> FastAPI:
    """Build the FastAPI app. ``client_factory`` is optional and used by
    tests; in production the server manages a single MT5Client.
    """
    cfg = load_config(config_path)
    journal = Journal(Path(cfg.paths.data_dir) / "journal.db")

    state: dict[str, Any] = {
        "client": None,
        "client_factory": client_factory,
        "kill_switch_path": Path(cfg.paths.kill_switch_path),
    }

    def _get_client() -> Any:
        """Return a connected MT5 client, or None if MT5 is unavailable."""
        if state["client"] is not None:
            return state["client"]
        factory = state["client_factory"]
        if factory is None:
            from app.mt5_client import MT5Client as _MC
            factory = _MC
        try:
            c = factory()
        except Exception:  # noqa: BLE001
            return None
        try:
            c.connect(0, "", "")
        except Exception:  # noqa: BLE001
            return None
        state["client"] = c
        return c

    def _disconnect_client() -> None:
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

    app = FastAPI(title="Trader Dashboard", version="0.1.0", lifespan=lifespan)

    # --- helpers ------------------------------------------------------

    def _kill_engaged() -> bool:
        try:
            return state["kill_switch_path"].exists()
        except OSError:
            return False

    def _account_block() -> dict | None:
        client = _get_client()
        if client is None:
            return None
        try:
            info = client.account_info()
        except Exception:  # noqa: BLE001
            return None
        return {
            "login": info.login,
            "server": info.server,
            "currency": info.currency,
            "trade_mode": info.trade_mode,
            "balance": float(info.balance),
            "equity": float(info.equity),
            "margin": float(info.margin),
            "free_margin": float(info.free_margin),
            "leverage": int(info.leverage),
        }

    def _positions_block() -> list[dict]:
        client = _get_client()
        if client is None:
            return []
        try:
            positions = client.positions()
        except Exception:  # noqa: BLE001
            return []
        return [
            {
                "ticket": int(p.ticket),
                "symbol": p.symbol,
                "direction": p.direction,
                "volume": float(p.volume),
                "price_open": float(p.price_open),
                "price_current": float(p.price_current),
                "sl": float(p.sl),
                "tp": float(p.tp),
                "profit": float(p.profit),
            }
            for p in positions
        ]

    # --- routes -------------------------------------------------------

    @app.get("/api/state")
    def api_state() -> JSONResponse:
        start_ts, _ = journal.today_utc_window()
        today_decisions = journal.list_decisions(since_ts=start_ts, limit=200)
        return JSONResponse(
            {
                "ts": int(time.time()),
                "kill_switch": _kill_engaged(),
                "account": _account_block(),
                "positions": _positions_block(),
                "today": {
                    "decision_count": len(today_decisions),
                    "approved": sum(1 for d in today_decisions if d["approved"]),
                    "rejected": sum(1 for d in today_decisions if not d["approved"]),
                },
                "mt5_connected": _get_client() is not None,
            }
        )

    @app.get("/api/decisions")
    def api_decisions(
        limit: int = Query(_DEFAULT_LIMIT, ge=1, le=1000),
        symbol: str | None = None,
    ) -> JSONResponse:
        rows = journal.list_decisions(limit=limit, symbol=symbol)
        return JSONResponse({"decisions": rows, "count": len(rows)})

    @app.get("/api/orders")
    def api_orders(
        limit: int = Query(_DEFAULT_LIMIT, ge=1, le=1000),
        symbol: str | None = None,
    ) -> JSONResponse:
        rows = journal.list_orders(limit=limit, symbol=symbol)
        return JSONResponse({"orders": rows, "count": len(rows)})

    @app.get("/api/equity")
    def api_equity(days: int = Query(7, ge=1, le=365)) -> JSONResponse:
        since_ts = int(time.time()) - days * 86400
        rows = journal.list_equity(since_ts=since_ts, limit=5000)
        return JSONResponse({"equity": rows, "count": len(rows), "days": days})

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

    @app.get("/api/health")
    def api_health() -> JSONResponse:
        return JSONResponse({"ok": True, "ts": int(time.time())})

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