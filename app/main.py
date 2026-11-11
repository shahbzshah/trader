"""CLI entrypoint: run | serve | once | status | kill | resume | doctor | review.

Signals-only bot: no MT5, no trade execution. The runner keeps the
market feed warm (OANDA v20 Practice first, Deriv WebSocket as
fallback) and serves the dashboard.

Strategy (live 2026-11-11, Megah's order "combine the 2"): Phil v2
triangles filtered by RUDY's internal trend — see app/strategy.py
and knowledge/playbooks/phil_rudy_v2.md. One open position max;
winners exit at the opposite filtered signal (flip), losses at
Megah's $10 stop; there is no take-profit and no breakeven manager.

Commands:
    run      continuous scan loop (APScheduler) — local use
    serve    scan loop + dashboard web server on $PORT (Render)
    once     single cycle (feed check + candle refresh)
    status   feed connection + latest tick/candles
    doctor   pre-flight self-check (feed, news, Telegram)
    kill     engage the kill switch
    resume   release the kill switch
    review   generate a daily review markdown file
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from app.config import EnvSecrets, load_config
from app.market_data import MarketDataClient, MarketDataError, Rates

logger = logging.getLogger(__name__)


def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]
    try:
        from logging.handlers import RotatingFileHandler

        handlers.append(
            RotatingFileHandler(
                log_dir / "agent.log",
                maxBytes=2_000_000,
                backupCount=3,
                encoding="utf-8",
            )
        )
    except Exception:  # noqa: BLE001 - log file is best-effort
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        handlers=handlers,
    )


def _env_path_for(args: argparse.Namespace) -> Path | None:
    """Locate the ``.env`` file used for credentials.

    Prefers the ``.env`` beside ``config.yaml``; falls back to the repo-root
    ``.env``. ``EnvSecrets.from_env`` tolerates a non-existent path.
    """
    cfg_path = getattr(args, "config", None) or "config.yaml"
    return Path(str(cfg_path)).resolve().parent / ".env"


def _resolve_secrets(args: argparse.Namespace) -> EnvSecrets:
    """Load OANDA + Telegram secrets from .env / process env."""
    return EnvSecrets.from_env(_env_path_for(args))


# --- market-feed construction ---------------------------------------------

def build_market_client(cfg, secrets: EnvSecrets,
                        feed_override: str | None = None
                        ) -> tuple[MarketDataClient, str]:
    """Build and connect the market feed.

    ``feed_override`` (``--feed``) wins over ``cfg.signals.feed``.
    "auto" (default): OANDA when its credentials are present and the
    probe succeeds, otherwise Deriv. "oanda"/"deriv" force that feed
    (OANDA without credentials is an error, not a silent fallback).

    Returns ``(client, source_name)``. Raises :class:`MarketDataError`
    when no feed can be reached.
    """
    from app.deriv_client import DerivClient
    from app.oanda_client import OandaClient

    balance = float(getattr(getattr(cfg, "signals", None),
                            "hypothetical_balance_usd", 10000.0))
    feed = (feed_override or getattr(getattr(cfg, "signals", None),
                                     "feed", "auto") or "auto").lower()

    def _deriv() -> DerivClient:
        client = DerivClient(hypothetical_balance=balance)
        client.connect()
        return client

    def _oanda() -> OandaClient:
        client = OandaClient(
            token=secrets.oanda_api_token,
            account_id=secrets.oanda_account_id,
            hypothetical_balance=balance,
        )
        client.connect()
        return client

    if feed == "deriv":
        return _deriv(), "deriv"
    if feed == "oanda":
        if not secrets.oanda_api_token or not secrets.oanda_account_id:
            raise MarketDataError(
                "feed=oanda but OANDA_API_TOKEN / OANDA_ACCOUNT_ID are not set"
            )
        return _oanda(), "oanda"
    # auto
    if secrets.oanda_api_token and secrets.oanda_account_id:
        try:
            return _oanda(), "oanda"
        except MarketDataError as exc:
            logger.warning("OANDA feed failed (%s); falling back to Deriv", exc)
    else:
        logger.info("no OANDA credentials configured; using Deriv feed")
    return _deriv(), "deriv"


_CHECK_LEVELS = {"ok": "[OK]  ", "warn": "[WARN]", "fail": "[FAIL]"}


def _print_checks(checks: list[tuple[str, str, str]]) -> None:
    for name, status, detail in checks:
        line = f"{_CHECK_LEVELS.get(status, '[?]  ')} {name}"
        if detail:
            line += f": {detail}"
        print(line)


# --- commands ---------------------------------------------------------------

def _cmd_status(args: argparse.Namespace) -> int:
    """Print feed status. Must not crash when no feed is reachable."""
    cfg = load_config(args.config)
    secrets = _resolve_secrets(args)
    try:
        client, source = build_market_client(
            cfg, secrets, feed_override=getattr(args, "feed", None))
    except MarketDataError as exc:
        print(f"no market feed available: {exc}")
        return 0
    try:
        tick = client.symbol_tick("XAUUSD")
        if tick is not None:
            print(
                f"feed={source} XAUUSD bid={tick.bid:.2f} ask={tick.ask:.2f} "
                f"spread={tick.spread if tick.spread is None else round(tick.spread, 1)} pts "
                f"tick_time={_dt.datetime.fromtimestamp(tick.time, tz=_dt.timezone.utc):%H:%M:%S} UTC"
            )
        else:
            print(f"feed={source}: no tick available")
        rates = client.copy_rates("XAUUSD", "M15", 5)
        bars = rates.rates if isinstance(rates, Rates) else list(rates)
        if bars:
            last = bars[-1]
            print(f"M15 candles: {len(bars)} (last close {float(last['close']):.2f})")
        kill = Path(cfg.paths.kill_switch_path).exists()
        print(f"kill switch: {'ENGAGED' if kill else 'disengaged'}")
    finally:
        client.shutdown()
    return 0


def _cmd_kill(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    path = Path(cfg.paths.kill_switch_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    print(f"Kill switch engaged: {path}")
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    path = Path(cfg.paths.kill_switch_path)
    if path.exists():
        path.unlink()
        print(f"Kill switch released: {path}")
    else:
        print("Kill switch was not engaged.")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Continuous scan loop (APScheduler-driven). Local use."""
    from app.scheduler import Scheduler

    cfg = load_config(args.config)
    components = _build_components(args, cfg)
    if components is None:
        return 1

    def cycle() -> None:
        run_once(components)

    sched = Scheduler(cfg, cycle)
    sched.start()
    print(f"running; interval={cfg.schedule.scan_interval_minutes} min. Ctrl-C to stop.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("stopping...")
        sched.stop()
        components["client"].shutdown()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Scan loop + dashboard web server (Render free web service).

    Binds 0.0.0.0:$PORT (Render sets PORT; defaults to the dashboard
    port from config.yaml). The scheduler runs in-process; the
    dashboard reads the same journal + feed.
    """
    from app.scheduler import Scheduler
    from app.server import create_app

    cfg = load_config(args.config)
    components = _build_components(args, cfg)
    if components is None:
        return 1

    client = components["client"]
    if hasattr(client, "start_tick_stream"):
        try:
            client.start_tick_stream("XAUUSD")
        except Exception:  # noqa: BLE001 - stream is monitoring only
            logger.warning("could not start Deriv tick stream", exc_info=True)

    def cycle() -> None:
        run_once(components)

    sched = Scheduler(cfg, cycle)
    sched.start()

    port = int(os.environ.get("PORT") or cfg.dashboard.port)
    host = "0.0.0.0"  # Render routes to the container's $PORT
    print(f"serving dashboard on {host}:{port} (feed={components['feed']}); "
          f"scan interval={cfg.schedule.scan_interval_minutes} min")
    try:
        import uvicorn

        uvicorn.run(
            create_app(args.config, components=components),
            host=host,
            port=port,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("stopping...")
    finally:
        sched.stop()
        client.shutdown()
    return 0




def _cmd_once(args: argparse.Namespace) -> int:
    """Single cycle: feed check + candle refresh + Phil+RUDY strategy."""
    cfg = load_config(args.config)
    components = _build_components(args, cfg)
    if components is None:
        return 1
    res = run_once(components)
    components["client"].shutdown()
    print(
        f"once cycle complete: "
        f"decisions={res['decisions']} signals={res['signals']} "
        f"duplicates={res['duplicates']} rejected={res['rejected']}"
    )
    return 0




def _build_components(args: argparse.Namespace, cfg):
    """Construct feed client, journal and Telegram alerter.

    Returns None and prints a clean error if no market feed can be
    reached.
    """
    from app.journal import Journal
    from app.telegram import TelegramAlerter

    secrets = _resolve_secrets(args)
    try:
        client, source = build_market_client(
            cfg, secrets, feed_override=getattr(args, "feed", None))
    except MarketDataError as exc:
        print(f"no market feed available: {exc}")
        return None

    journal = Journal(Path(cfg.paths.data_dir) / "journal.db")

    telegram = None
    if getattr(getattr(cfg, "signals", None), "telegram_enabled", True):
        telegram = TelegramAlerter(
            token=secrets.telegram_bot_token,
            chat_id=secrets.telegram_chat_id,
        )
        if not telegram.enabled:
            logger.warning(
                "Telegram not configured (TELEGRAM_BOT_TOKEN / "
                "TELEGRAM_CHAT_ID); no alerts will be sent"
            )

    components = {
        "cfg": cfg,
        "client": client,
        "feed": source,
        "journal": journal,
        "telegram": telegram,
    }
    return components





def _open_position_path(cfg) -> Path:
    """Durable open-position state (one position max, per the backtest
    trade engine). Lives in the data dir next to the journal."""
    return Path(cfg.paths.data_dir) / "open_position.json"


def _load_open_position(path: Path) -> dict | None:
    """Read the open position, or None. A corrupt/partial file is
    treated as no position (and logged) — never crash a cycle on it."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return {
            "bar_epoch": int(raw["bar_epoch"]),
            "decision_id": int(raw["decision_id"]),
            "direction": str(raw["direction"]),
            "entry": float(raw["entry"]),
            "stop_loss": float(raw["stop_loss"]),
            "lots": float(raw["lots"]),
        }
    except (KeyError, TypeError, ValueError):
        logger.warning("open position file %s is malformed; ignoring", path)
        return None


def _save_open_position(path: Path, pos: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pos, sort_keys=True), encoding="utf-8")


def _clear_open_position(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _usable_bar(row: Any) -> bool:
    """True when a candle row has finite OHLC + time values."""
    try:
        for key in ("time", "open", "high", "low", "close"):
            value = float(row[key])
            if value != value or value in (float("inf"), float("-inf")):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _close_position(*, journal, pos: dict, exit_price: float,
                    exit_ts: int, reason: str) -> float:
    """Record a paper close in the journal; return realised profit ($).

    profit = (exit - entry) * direction * lots * 100 - $0.50 spread,
    with direction +1 for buy / -1 for sell. The ticket is the journal
    decision id of the signal that opened the position, so the
    dashboard can match a close to its signal.
    """
    direction = pos["direction"]
    sign = 1 if direction == "buy" else -1
    profit = ((float(exit_price) - float(pos["entry"])) * sign
              * float(pos["lots"]) * 100.0 - 0.50)
    journal.record_position_close(
        symbol="XAUUSD",
        direction=direction,
        volume=float(pos["lots"]),
        price_open=float(pos["entry"]),
        price_close=float(exit_price),
        profit=profit,
        ticket=int(pos["decision_id"]),
        sl=float(pos["stop_loss"]),
        tp=None,
        open_ts=int(pos["bar_epoch"]),
        close_ts=int(exit_ts),
    )
    logger.info("position closed (%s): %s %.2f -> %.2f, profit $%.2f",
                reason, direction, pos["entry"], exit_price, profit)
    return profit


def _write_flip_history(cfg, journal, decision_id: int, close_ts: int,
                        close_price: float, r_value: float | None,
                        risk_usd: float, lots: float) -> None:
    """Write a FLIP outcome for one signal into the durable history
    (snapshot + append-only archive, app/history.py) so the dashboard
    ledger shows the flip close. Best-effort: a failure here must
    never break the cycle (the journal close above is the record)."""
    try:
        from app.history import (
            append_archive,
            archive_path,
            load_snapshot,
            snapshot_path,
            write_snapshot,
        )

        row = journal.get_decision(decision_id)
        if row is None:
            return
        try:
            prev = json.loads(row.get("raw_llm_json") or "{}")
        except ValueError:
            prev = {}
        if not isinstance(prev, dict):
            prev = {}
        rec = {
            "id": int(decision_id),
            "ts": int(row["ts"]),
            "symbol": row["symbol"],
            "direction": prev.get("direction"),
            "entry": prev.get("price"),
            "stop_loss": prev.get("stop_loss"),
            "take_profit": None,
            "risk_pct": prev.get("risk_pct"),
            "lots": float(lots),
            "risk_usd": float(risk_usd),
            "per_dollar_usd": float(lots) * 100.0,
            "rr": None,
            "confidence": row.get("confidence"),
            "playbook": row.get("playbook"),
            "reasoning": row.get("reasoning"),
            "status": "FLIP",
            "close_ts": int(close_ts),
            "close_price": float(close_price),
            "r": r_value,
            "explanation": None,
            "explanation_model": None,
            "explanation_ts": None,
        }
        data_dir = Path(cfg.paths.data_dir)
        snap = snapshot_path(data_dir)
        records = [x for x in load_snapshot(snap)
                   if x["id"] != int(decision_id)]
        records.append(rec)
        write_snapshot(snap, records)
        append_archive(archive_path(data_dir), rec)
    except Exception as exc:  # noqa: BLE001 - best-effort history write
        logger.warning("could not write FLIP history for signal %d: %s",
                       decision_id, exc)


def run_once(components: dict) -> dict:
    """One scan cycle: feed warm-up + Phil+RUDY strategy.

    Returns a small result dict for callers/tests (decisions /
    signals / duplicates / rejected this cycle).

    Flow, in order:
    1. scan_watchlist keeps the feed warm and refreshes dashboard
       candles (unchanged).
    2. Fetch 400 M15 bars; drop the forming bar (now < bar_time +
       900s). Only the last CLOSED bar is evaluated for a signal.
    3. If a position is open, check every closed bar after its entry
       bar for the $10 stop (fill at the stop, or at the bar's open
       on a gap through it). A stop close still lets a flip signal
       on the same bar open the next position.
    4. Evaluate the last closed bar (Phil v2 + RUDY itrend). No
       signal -> nothing journaled.
    5. Dedup by signal-bar epoch (journal).
    6. Gates, in order: kill switch, news blackout (High USD ±30
       min, fail-open), max_signals_per_day, daily-loss halt
       (max_daily_loss_pct of the hypothetical balance). A gated
       signal is journaled as a rejected no_trade decision.
    7. Same-direction signal while open is ignored. Opposite signal
       closes the open position at the new signal's entry (flip) and
       opens the new one. Sizing is 1% risk via position_size with
       the $10 SL (0.10 lots on $10,000).
    """
    import json as _json
    import time as _time

    from app import news as news_mod
    from app.market_data import XAUUSD_SPEC
    from app.risk import position_size
    from app.scanner import scan_watchlist
    from app.signals import Signal
    from app.strategy import CONFIDENCE, PLAYBOOK, evaluate_last_bar
    from app.telegram import format_signal_text

    cfg = components["cfg"]
    journal = components["journal"]
    client = components["client"]
    telegram = components.get("telegram")
    feed = components.get("feed", "unknown")
    result = {"decisions": 0, "signals": 0, "duplicates": 0, "rejected": 0}

    snapshots = scan_watchlist(client, cfg)

    blackout = news_mod.current_blackout(cfg)
    if blackout is not None:
        logger.warning(
            "news blackout active: %s (%s) at %s UTC",
            blackout.title,
            blackout.currency,
            blackout.time_utc.strftime("%H:%M"),
        )

    strat = getattr(cfg, "strategy", None)
    phil = getattr(strat, "phil_rudy", None) if strat is not None else None
    warmup = int(getattr(phil, "warmup_bars", 300))
    history_bars = int(getattr(phil, "history_bars", 400))
    max_dist = float(getattr(phil, "max_dist_usd", 20.0))
    min_sep = float(getattr(phil, "min_sep_813_usd", 0.75))
    stop_usd = float(getattr(phil, "stop_usd", 10.0))

    try:
        rates = client.copy_rates("XAUUSD", "M15", history_bars)
    except Exception as exc:  # noqa: BLE001 - feed failure skips strategy
        logger.warning("strategy M15 fetch failed: %s", exc)
        return result
    rows = rates.rates if hasattr(rates, "rates") else list(rates)
    bars = sorted(
        (r for r in rows if _usable_bar(r)),
        key=lambda r: int(r["time"]),
    )
    now = int(_time.time())
    closed = [b for b in bars if now >= int(b["time"]) + 900]
    if len(closed) <= warmup:
        logger.info(
            "scan cycle complete: %d symbol(s), %d closed M15 bars "
            "(warmup %d, no signal possible yet)",
            len(snapshots), len(closed), warmup)
        return result

    pos_path = _open_position_path(cfg)
    pos = _load_open_position(pos_path)

    # --- $10 stop check on closed bars after the entry bar ---
    if pos is not None:
        for bar in closed:
            if int(bar["time"]) <= int(pos["bar_epoch"]):
                continue
            sl = float(pos["stop_loss"])
            direction = pos["direction"]
            hit = (float(bar["low"]) <= sl if direction == "buy"
                   else float(bar["high"]) >= sl)
            if not hit:
                continue
            fill = float(bar["open"])
            if direction == "buy":
                fill = fill if fill < sl else sl
            else:
                fill = fill if fill > sl else sl
            _close_position(journal=journal, pos=pos, exit_price=fill,
                            exit_ts=int(bar["time"]), reason="SL")
            _clear_open_position(pos_path)
            pos = None
            break

    # --- signal on the last closed bar only ---
    sig = evaluate_last_bar(closed, warmup=warmup, max_dist=max_dist,
                            min_sep=min_sep)
    if sig is None:
        logger.info(
            "scan cycle complete: %d symbol(s), no Phil+RUDY signal on "
            "the last closed bar", len(snapshots))
        return result

    direction = "buy" if sig["dir"] == 1 else "sell"
    entry = float(sig["close"])
    sl = entry - stop_usd if direction == "buy" else entry + stop_usd
    cross_word = "above" if direction == "buy" else "below"
    reasoning = (
        f"Phil v2 {direction.upper()}: 21 crossed {cross_word} 55 "
        f"(21 ${sig['ma21']:.2f} vs 55 ${sig['ma55']:.2f}); close "
        f"${entry:.2f} is ${abs(entry - sig['ma55']):.2f} from the 55 "
        f"(max $20.00); 8/13 gap ${abs(sig['ma8'] - sig['ma13']):.2f} "
        f"(min $0.75); RUDY itrend {int(sig['itrend']):+d} agrees. "
        f"Entry = signal-bar close; SL ${sl:.2f} ($10.00); no "
        f"take-profit — exit at the opposite Phil+RUDY signal (flip). "
        f"Confidence 0.48 = backtested win rate, not a prediction."
    )

    # --- dedup by signal-bar epoch ---
    for row in journal.list_decisions(limit=1000, symbol="XAUUSD"):
        if (row.get("action") == "propose" and row.get("approved")
                and int(row.get("ts", -1)) == int(sig["epoch"])):
            result["duplicates"] += 1
            logger.info("duplicate signal bar %s ignored", sig["epoch"])
            return result
    if pos is not None and int(pos["bar_epoch"]) == int(sig["epoch"]):
        result["duplicates"] += 1
        logger.info("duplicate signal bar %s ignored (position open)",
                    sig["epoch"])
        return result

    def _reject(reason: str) -> dict:
        journal.record_decision(
            symbol="XAUUSD", action="no_trade", playbook=PLAYBOOK,
            confidence=CONFIDENCE, risk_pct=1.0, reasoning=reasoning,
            approved=False, reject_reasons=[reason],
            ts=int(sig["epoch"]))
        result["decisions"] += 1
        result["rejected"] += 1
        logger.info("signal rejected: %s", reason)
        return result

    # --- gates, in order ---
    if Path(cfg.paths.kill_switch_path).exists():
        return _reject("kill switch engaged")
    if blackout is not None:
        return _reject(
            f"news blackout: {blackout.title} ({blackout.currency}) at "
            f"{blackout.time_utc:%H:%M} UTC")
    start_ts, _ = journal.today_utc_window()
    approved_today = [
        d for d in journal.list_decisions(limit=1000, since_ts=start_ts)
        if d.get("action") == "propose" and d.get("approved")
    ]
    max_per_day = int(
        getattr(getattr(cfg, "signals", None), "max_signals_per_day", 5))
    if len(approved_today) >= max_per_day:
        return _reject(f"max signals per day ({max_per_day}) reached")
    realized_today = sum(
        (r.get("profit") or 0.0)
        for r in journal.list_positions_history(since_ts=start_ts)
    )
    balance = float(
        getattr(getattr(cfg, "signals", None),
                "hypothetical_balance_usd", 10000.0))
    loss_limit = (-float(getattr(getattr(cfg, "risk", None),
                                 "max_daily_loss_pct", 3.0))
                  / 100.0 * balance)
    if realized_today <= loss_limit:
        return _reject(
            f"daily loss halt: realised ${realized_today:.2f} today "
            f"(limit ${loss_limit:.2f})")

    # --- same-direction signal while open is ignored ---
    if pos is not None and pos["direction"] == direction:
        logger.info("same-direction %s signal while %s open ignored",
                    direction, direction)
        return result

    # --- opposite signal while open: flip close, then open the new one ---
    if pos is not None:
        profit = _close_position(journal=journal, pos=pos,
                                 exit_price=entry,
                                 exit_ts=int(sig["epoch"]), reason="FLIP")
        risk_usd_old = (float(pos["lots"]) * 100.0
                        * abs(float(pos["entry"]) - float(pos["stop_loss"])))
        r_value = profit / risk_usd_old if risk_usd_old else None
        _write_flip_history(cfg, journal, int(pos["decision_id"]),
                            int(sig["epoch"]), entry, r_value,
                            risk_usd_old, float(pos["lots"]))
        _clear_open_position(pos_path)
        pos = None

    lots = position_size(equity=balance, risk_pct=1.0, entry=entry,
                         stop_loss=sl, symbol_info=XAUUSD_SPEC)
    if lots <= 0:
        return _reject("cannot size this trade (lots 0.0)")
    risk_usd = lots * 100.0 * stop_usd
    signal = Signal(
        symbol="XAUUSD", direction=direction, entry=entry, stop_loss=sl,
        take_profit=None, risk_pct=1.0, confidence=CONFIDENCE,
        playbook=PLAYBOOK, reasoning=reasoning, source_feed=feed,
        ts=int(sig["epoch"]), lots=lots, risk_usd=risk_usd,
        per_dollar_usd=lots * 100.0)
    decision_id = journal.record_decision(
        symbol="XAUUSD", action="propose", playbook=PLAYBOOK,
        confidence=CONFIDENCE, risk_pct=1.0, reasoning=reasoning,
        approved=True,
        raw_llm_json=_json.dumps({
            "direction": direction, "price": entry, "stop_loss": sl,
            "take_profit": None, "risk_pct": 1.0,
        }),
        ts=int(sig["epoch"]))
    _save_open_position(pos_path, {
        "bar_epoch": int(sig["epoch"]), "decision_id": decision_id,
        "direction": direction, "entry": entry, "stop_loss": sl,
        "lots": lots,
    })
    if telegram is not None and telegram.enabled:
        try:
            telegram.send_message(format_signal_text(signal))
        except Exception as exc:  # noqa: BLE001 - alert never breaks a cycle
            logger.warning("Telegram alert failed: %s", exc)
    result["decisions"] += 1
    result["signals"] += 1
    logger.info("signal: %s XAUUSD @ %.2f SL %.2f lots %.2f (decision %d)",
                direction, entry, sl, lots, decision_id)
    return result





class _EmptyClient:
    """Fallback when no feed is reachable: empty data, no positions."""

    name = "empty"

    def connect(self):
        return None

    def shutdown(self):
        return None

    def copy_rates(self, symbol, timeframe, count):
        return Rates(rates=[])

    def symbol_tick(self, symbol):
        return None

    def symbol_info(self, symbol):
        from app.market_data import XAUUSD_SPEC
        return XAUUSD_SPEC

    def select_symbol(self, symbol):
        return True

    def positions(self, symbol=None):
        return []

    def account_info(self):
        from app.market_data import PaperAccountInfo
        return PaperAccountInfo(balance=0.0, equity=0.0)

    def feed_status(self):
        return {"name": self.name, "connected": False,
                "last_tick_epoch": None, "last_error": "no feed"}


def _cmd_review(args: argparse.Namespace) -> int:
    """Generate a daily review markdown file (advisory proposals only)."""
    from app.journal import Journal
    from app.review import generate_report

    cfg = load_config(args.config)
    if cfg is None:
        print("config not found")
        return 1

    journal = Journal(Path(cfg.paths.data_dir) / "journal.db")

    day = None
    if args.day:
        try:
            day = _dt.date.fromisoformat(args.day)
        except ValueError:
            print(f"invalid --day value: {args.day!r}; expected YYYY-MM-DD")
            return 2

    path = generate_report(
        journal,
        day=day,
        knowledge_dir=Path(cfg.paths.knowledge_dir),
        reports_dir=Path(cfg.paths.reports_dir),
    )
    print(f"wrote {path}")
    return 0





def _cmd_doctor(args: argparse.Namespace) -> int:
    """Pre-flight self-check against the *real* environment.

    Verifies, in order:
      1. OANDA credentials are configured (warn only — Deriv covers it),
      2. a market feed connects (OANDA, else Deriv),
      3. every configured symbol serves live rates + ticks,
      4. the news calendar feed (fail-open: warn only),
      5. the Telegram token validates via getMe (warn only),
      6. the kill switch is off.

    Returns 0 only when no check FAILs (warnings are non-fatal).
    """
    cfg = load_config(args.config)
    checks: list[tuple[str, str, str]] = []

    def add(name: str, status: str, detail: str = "") -> None:
        checks.append((name, status, detail))

    secrets = _resolve_secrets(args)

    # 1. OANDA credentials (warn only — Deriv is the fallback).
    if secrets.oanda_api_token and secrets.oanda_account_id:
        add("OANDA credentials", "ok", "token + account id configured")
    else:
        add("OANDA credentials", "warn",
            "not set — Deriv fallback will be used (OANDA is the main feed)")

    # 2. Feed connect.
    client = None
    try:
        client, source = build_market_client(
            cfg, secrets, feed_override=getattr(args, "feed", None))
        add("market feed", "ok", f"connected via {source}")
    except MarketDataError as exc:
        add("market feed", "fail", str(exc))
        _print_checks(checks)
        return 1

    # 3. Per-symbol live data.
    primary_tf = cfg.timeframes[0] if cfg.timeframes else "M15"
    for symbol in cfg.symbols:
        try:
            rates = client.copy_rates(symbol, primary_tf, 5)
            bars = rates.rates if isinstance(rates, Rates) else list(rates)
            tick = client.symbol_tick(symbol)
            if not bars:
                raise MarketDataError("no candles returned")
            if tick is None:
                raise MarketDataError("no tick returned")
            add(f"symbol {symbol} live data", "ok",
                f"{len(bars)} {primary_tf} bars, tick {tick.bid:.2f}/{tick.ask:.2f}")
        except Exception as exc:  # noqa: BLE001
            add(f"symbol {symbol} live data", "fail",
                f"{type(exc).__name__}: {exc}")

    # 4. News calendar feed (fail-open: warn only, never fail).
    try:
        from app import news as news_mod

        events = news_mod.get_events(cfg)
        now = _dt.datetime.now(_dt.timezone.utc)
        upcoming = [
            e for e in events
            if e.time_utc >= now - _dt.timedelta(hours=1)
        ][:3]
        detail = f"{len(events)} high-impact event(s) this week"
        if upcoming:
            detail += "; next: " + ", ".join(
                f"{e.title} {e.time_utc.strftime('%a %H:%M')}Z" for e in upcoming
            )
        add("news calendar", "ok", detail)
    except Exception as exc:  # noqa: BLE001
        add("news calendar", "warn", f"feed unreachable, fail-open: {exc}")

    # 5. Telegram token validation (getMe — sends no message).
    try:
        from app.telegram import TelegramAlerter

        alerter = TelegramAlerter(
            token=secrets.telegram_bot_token,
            chat_id=secrets.telegram_chat_id,
        )
        if not alerter.enabled:
            add("Telegram alerts", "warn",
                "not configured — signals will be journaled only")
        else:
            ok, detail = alerter.validate_token()
            add("Telegram alerts", "ok" if ok else "warn",
                f"bot {detail}" if ok else f"token invalid: {detail}")
    except Exception as exc:  # noqa: BLE001
        add("Telegram alerts", "warn", f"check failed: {exc}")

    # 6. Kill switch.
    ks = Path(cfg.paths.kill_switch_path)
    if ks.exists():
        add("kill switch", "warn", f"ENGAGED at {ks}")
    else:
        add("kill switch", "ok", "disengaged")

    client.shutdown()
    _print_checks(checks)
    failed = [c for c in checks if c[1] == "fail"]
    if failed:
        print(f"\n{len(failed)} check(s) FAILED")
        return 1
    print("\nAll checks passed.")
    return 0


def _add_feed_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--feed", default=None, choices=["auto", "oanda", "deriv"],
        help="market feed: auto (default), oanda, or deriv",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trader")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="show feed connection & latest tick")
    _add_feed_arg(p_status)
    p_status.set_defaults(func=_cmd_status)

    sub.add_parser("kill", help="engage kill switch").set_defaults(func=_cmd_kill)
    sub.add_parser("resume", help="release kill switch").set_defaults(func=_cmd_resume)

    p_run = sub.add_parser("run", help="continuous scan loop (local)")
    _add_feed_arg(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_serve = sub.add_parser(
        "serve", help="scan loop + dashboard web server on $PORT (Render)")
    _add_feed_arg(p_serve)
    p_serve.set_defaults(func=_cmd_serve)

    p_once = sub.add_parser("once", help="single cycle")
    _add_feed_arg(p_once)
    p_once.set_defaults(func=_cmd_once)

    p_doctor = sub.add_parser(
        "doctor", help="pre-flight self-check of feed + news + Telegram")
    _add_feed_arg(p_doctor)
    p_doctor.set_defaults(func=_cmd_doctor)

    p_review = sub.add_parser("review", help="generate daily review report")
    p_review.add_argument("--day", default=None,
                           help="ISO date (YYYY-MM-DD); default = today UTC")
    p_review.set_defaults(func=_cmd_review)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = Path(args.config)
    try:
        cfg = load_config(cfg_path)
    except FileNotFoundError:
        cfg = None  # status/kill tolerate missing config
    log_dir = Path("logs") if cfg is None else Path(cfg.paths.logs_dir)
    _configure_logging(log_dir)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
