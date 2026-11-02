"""CLI entrypoint: run | serve | once | analyze | backtest | status | kill | resume.

Signals-only bot: no MT5, no trade execution. The runner scans the
market feed (OANDA v20 Practice first, Deriv WebSocket as fallback),
runs the deterministic rules engine, and turns approved setups into
journaled Telegram signals with paper sizing against a configured
hypothetical balance.

Commands:
    run      continuous scan loop (APScheduler) — local use
    serve    scan loop + dashboard web server on $PORT (Render)
    once     single cycle; --analyze = propose-only (no alerts)
    analyze  decide + print, never journals, never alerts
    status   feed connection + latest tick/candles
    doctor   pre-flight self-check (feed, rules engine, news, Telegram)
    backtest paper backtest on CSV candles (no feed, no network)
    kill     engage the kill switch (stops new signals)
    resume   release the kill switch
    review   generate a daily review markdown file
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import sys
from pathlib import Path

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
    components = _build_components(args, cfg, analyze_only=False)
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
    components = _build_components(args, cfg, analyze_only=False)
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
    """Single full cycle (scan -> decide -> risk -> signal -> journal).

    Ignores session hours; honors kill switch. With ``--analyze`` the
    signal engine is skipped entirely (propose-only, no alerts).
    """
    cfg = load_config(args.config)
    components = _build_components(args, cfg, analyze_only=bool(args.analyze))
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


def _build_components(args: argparse.Namespace, cfg, *, analyze_only: bool):
    """Construct feed client, journal, signal engine and Telegram alerter.

    Returns None and prints a clean error if no market feed can be
    reached.
    """
    from app.journal import Journal
    from app.signals import SignalEngine
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
                "TELEGRAM_CHAT_ID); signals will be journaled only"
            )

    engine = None if analyze_only else SignalEngine(
        client, journal, cfg, telegram=telegram)

    components = {
        "cfg": cfg,
        "client": client,
        "feed": source,
        "journal": journal,
        "engine": engine,
        "telegram": telegram,
        "analyze_only": analyze_only,
    }
    return components


def run_once(components: dict) -> dict:
    """One full scan cycle. Returns a small result dict for callers/tests."""
    from app.brain import decide
    from app.scanner import scan_watchlist

    cfg = components["cfg"]
    journal = components["journal"]
    engine = components["engine"]
    analyze_only = components["analyze_only"]

    snapshots = scan_watchlist(components["client"], cfg)

    decisions = signals = duplicates = rejected = 0
    from app import news as news_mod

    blackout = news_mod.current_blackout(cfg)
    if blackout is not None:
        logger.warning(
            "news blackout active: %s (%s) at %s UTC — standing down this cycle",
            blackout.title,
            blackout.currency,
            blackout.time_utc.strftime("%H:%M"),
        )
    for snap in snapshots:
        if blackout is not None:
            journal.record_decision(
                symbol=snap.get("symbol", ""),
                action="no_trade",
                reasoning=(
                    f"news blackout: {blackout.title} "
                    f"({blackout.currency}, {blackout.impact}) at "
                    f"{blackout.time_utc.strftime('%H:%M')} UTC"
                ),
                approved=True,
                reject_reasons=[],
                raw_llm_json="",
            )
            decisions += 1
            continue
        try:
            decision = decide(snap, cfg)
        except Exception as exc:  # noqa: BLE001
            journal.record_decision(
                symbol=snap.get("symbol", ""),
                action="error",
                reasoning=str(exc),
                approved=False,
                reject_reasons=[f"strategy_error:{exc}"],
                raw_llm_json="",
            )
            rejected += 1
            decisions += 1
            continue

        decisions += 1
        if analyze_only or engine is None:
            # Just journal the decision without signaling.
            journal.record_decision(
                symbol=decision.symbol,
                action=decision.action,
                playbook=decision.playbook or "",
                confidence=float(decision.confidence),
                risk_pct=float(decision.risk_pct),
                reasoning=decision.reasoning,
                approved=True,
                reject_reasons=[],
                raw_llm_json=decision.model_dump_json(),
            )
            continue

        result = engine.process(decision,
                                raw_llm_json=decision.model_dump_json())
        if result.status == "signaled":
            signals += 1
        elif result.status == "duplicate":
            duplicates += 1
        elif result.status in ("rejected", "capped"):
            rejected += 1
        # "skipped" (no_trade) counts as neither.

    # NOTE: no breakeven step — signals-only has no open positions to
    # manage. app.manage.breakeven_moves is retained (pure, tested) for
    # a future execution mode, but the runner never calls it here.
    return {
        "decisions": decisions,
        "signals": signals,
        "duplicates": duplicates,
        "rejected": rejected,
    }


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Decide + print, NEVER journal, NEVER alert.

    Falls back to an empty client when no feed is reachable, so the
    rules engine can still be inspected offline.
    """
    from app.brain import decide
    from app.risk import AccountState, RiskVerdict, check_decision
    from app.scanner import scan_watchlist
    from app.signals import hypothetical_balance, size_signal

    cfg = load_config(args.config)
    if cfg is None:
        print("config not found")
        return 1

    secrets = _resolve_secrets(args)
    try:
        client, source = build_market_client(
            cfg, secrets, feed_override=getattr(args, "feed", None))
        print(f"feed: {source}")
    except MarketDataError as exc:
        print(f"no market feed available ({exc}); using empty data")
        client = _EmptyClient()

    try:
        snapshots = scan_watchlist(client, cfg)
    finally:
        try:
            client.shutdown()
        except Exception:  # noqa: BLE001
            pass

    balance = hypothetical_balance(cfg)
    kill_path = Path(cfg.paths.kill_switch_path)
    account_state = AccountState(
        balance=balance,
        equity=balance,
        starting_balance_today=balance,
        trade_mode=cfg.mode,
        kill_switch_engaged=kill_path.exists(),
    )

    from app import news as news_mod

    blackout_event = news_mod.current_blackout(cfg)
    for snap in snapshots:
        if blackout_event is not None:
            print(
                f"[{snap.get('symbol')}] action=no_trade "
                f"(news blackout: {blackout_event.title})"
            )
            continue
        try:
            decision = decide(snap, cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"[{snap.get('symbol')}] strategy error: {exc}")
            continue

        verdict: RiskVerdict = check_decision(
            decision, account_state, [], [], cfg
        )
        extra = ""
        if decision.action == "propose" and verdict.approved:
            try:
                lots, risk_usd, per_dollar = size_signal(cfg, client, decision)
                extra = (f" sizing: {lots:.2f} lots, risk ${risk_usd:.2f}, "
                         f"${per_dollar:.2f} per $1 move (paper ${balance:,.0f})")
            except ValueError as exc:
                extra = f" sizing failed: {exc}"
        print(
            f"[{snap.get('symbol')}] action={decision.action} "
            f"approved={verdict.approved} "
            f"reasons={verdict.reasons} "
            f"playbook={decision.playbook!r} "
            f"confidence={decision.confidence:.2f} "
            f"reasoning={decision.reasoning[:80]!r}{extra}"
        )
    return 0


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


def _cmd_backtest(args: argparse.Namespace) -> int:
    """Paper backtest of the rules engine on historical CSV candles.

    No feed, no network. Replays closed candles through
    scanner -> strategy -> risk with paper sizing.
    """
    from app.backtest import run_backtest

    cfg = load_config(args.config)
    summary = run_backtest(
        args.csv,
        symbol=args.symbol,
        timeframe=args.timeframe,
        config=cfg,
    )
    print(f"Backtest {summary['symbol']} {summary['timeframe']}: "
          f"{summary['bars']} bars")
    print(f"Trades: {summary['trades']} | "
          f"Wins: {summary['wins']} ({summary['win_rate_pct']:.1f}%) | "
          f"Avg R: {summary['avg_r']:+.2f} | "
          f"Profit factor: {summary['profit_factor']:.2f}")
    print(f"Max drawdown: {summary['max_drawdown_pct']:.2f}% | "
          f"Final equity: ${summary['final_equity']:.2f} "
          f"({summary['return_pct']:+.2f}%)")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Pre-flight self-check against the *real* environment.

    Verifies, in order:
      1. OANDA credentials are configured (warn only — Deriv covers it),
      2. a market feed connects (OANDA, else Deriv),
      3. every configured symbol serves live rates + ticks,
      4. the rules engine passes a self-test,
      5. the news calendar feed (fail-open: warn only),
      6. the Telegram token validates via getMe (warn only),
      7. the kill switch is off.

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

    # 4. Rules engine self-test: a synthetic snapshot must decide cleanly.
    try:
        from app.strategy import decide
        probe = decide({"symbol": "XAUUSD", "timeframes": {}}, cfg)
        assert probe.action == "no_trade"
        add("rules engine", "ok", "strategy.decide self-test passed")
    except Exception as exc:  # noqa: BLE001
        add("rules engine", "fail", f"{type(exc).__name__}: {exc}")

    # 5. News calendar feed (fail-open: warn only, never fail).
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

    # 6. Telegram token validation (getMe — sends no message).
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

    # 7. Kill switch.
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
    p_once.add_argument("--analyze", action="store_true",
                        help="single cycle in propose-only mode (no alerts)")
    _add_feed_arg(p_once)
    p_once.set_defaults(func=_cmd_once)

    p_analyze = sub.add_parser("analyze", help="analyze only, no journal, no alerts")
    p_analyze.add_argument("--dry", action="store_true",
                            help="alias for the default no-journal behavior")
    _add_feed_arg(p_analyze)
    p_analyze.set_defaults(func=_cmd_analyze)

    p_doctor = sub.add_parser(
        "doctor", help="pre-flight self-check of feed + rules engine + Telegram")
    _add_feed_arg(p_doctor)
    p_doctor.set_defaults(func=_cmd_doctor)

    p_review = sub.add_parser("review", help="generate daily review report")
    p_review.add_argument("--day", default=None,
                           help="ISO date (YYYY-MM-DD); default = today UTC")
    p_review.set_defaults(func=_cmd_review)

    p_backtest = sub.add_parser(
        "backtest",
        help="paper backtest of the rules engine on CSV candles (no feed)")
    p_backtest.add_argument("--symbol", default="XAUUSD",
                            help="symbol label for the CSV data")
    p_backtest.add_argument("--timeframe", default="M15",
                            help="timeframe label of the CSV candles")
    p_backtest.add_argument("--csv", required=True,
                            help="CSV with columns time,open,high,low,close,volume")
    p_backtest.set_defaults(func=_cmd_backtest)

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
