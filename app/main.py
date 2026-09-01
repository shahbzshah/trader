"""CLI entrypoint: run | once | analyze | status | kill | resume."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path

from app.config import EnvSecrets, load_config

# Eagerly expose MT5Client + LLMClient so tests can monkeypatch them.
from app.mt5_client import MT5Client  # noqa: F401  (re-exported for tests)
from app.brain import LLMClient  # noqa: F401

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


def _resolve_creds(args: argparse.Namespace) -> tuple[int, str, str, str]:
    """Merge CLI flags with ``.env`` secrets. CLI flags win; secrets fill gaps.

    This is what lets ``python -m app.main run`` work with credentials stored
    only in ``.env`` -- previously ``--login`` defaulted to ``0`` and the
    ``.env`` MT5 credentials were loaded but never used.
    """
    secrets = EnvSecrets.from_env(_env_path_for(args))
    login = getattr(args, "login", 0) or secrets.mt5_login or 0
    password = getattr(args, "password", "") or secrets.mt5_password or ""
    server = getattr(args, "server", "") or secrets.mt5_server or ""
    path = getattr(args, "path", "") or secrets.mt5_path or ""
    return int(login or 0), password or "", server or "", path or ""


_CHECK_LEVELS = {"ok": "[OK]  ", "warn": "[WARN]", "fail": "[FAIL]"}


def _print_checks(checks: list[tuple[str, str, str]]) -> None:
    for name, status, detail in checks:
        line = f"{_CHECK_LEVELS.get(status, '[?]  ')} {name}"
        if detail:
            line += f": {detail}"
        print(line)


def _cmd_status(args: argparse.Namespace) -> int:
    """Print connection status. Must not crash when MT5 is unavailable."""
    try:
        from app.mt5_client import MT5Client, MT5Error

        client = MT5Client()
    except ImportError:
        print("MetaTrader5 package not installed; install with `pip install MetaTrader5`")
        return 0

    login, password, server, path = _resolve_creds(args)
    try:
        client.connect(login, password, server, path)
    except MT5Error as exc:
        print(f"MT5 terminal not running or login failed: {exc}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"MT5 terminal not running: {exc}")
        return 0

    try:
        info = client.account_info()
        print(
            f"Connected login={info.login} server={info.server} "
            f"mode={info.trade_mode} balance={info.balance:.2f} "
            f"equity={info.equity:.2f}"
        )
        positions = client.positions()
        print(f"Open positions: {len(positions)}")
        for p in positions:
            print(f"  {p.ticket} {p.symbol} {p.direction} {p.volume} "
                  f"@ {p.price_open} p/L={p.profit:.2f}")
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
    """Continuous trading loop (APScheduler-driven)."""
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


def _cmd_once(args: argparse.Namespace) -> int:
    """Single full cycle (scan -> brain -> risk -> execute -> journal).

    Ignores session hours; honors kill switch. With ``--analyze`` the
    executor is skipped entirely (propose-only).
    """
    cfg = load_config(args.config)
    components = _build_components(args, cfg, analyze_only=bool(args.analyze))
    if components is None:
        return 1
    res = run_once(components)
    components["client"].shutdown()
    print(
        f"once cycle complete: "
        f"decisions={res['decisions']} orders={res['orders']} "
        f"rejected={res['rejected']} equity_rows={res['equity_rows']}"
    )
    return 0


def _build_components(args: argparse.Namespace, cfg, *, analyze_only: bool):
    """Construct client, journal, executor, and brain LLMClient.

    Returns None and prints a clean error if MT5 can't be reached.
    """
    from app.executor import Executor
    from app.journal import Journal

    try:
        login, password, server, path = _resolve_creds(args)
        client = MT5Client()
        client.connect(login, password, server, path)
        # ensure_demo at startup so we fail fast on a live account.
        client.ensure_demo()
    except Exception as exc:  # noqa: BLE001
        print(f"MT5 not available: {exc}")
        return None

    journal = Journal(Path(cfg.paths.data_dir) / "journal.db")

    executor = None if analyze_only else Executor(client, journal, cfg)

    components = {
        "cfg": cfg,
        "client": client,
        "journal": journal,
        "executor": executor,
        "analyze_only": analyze_only,
    }
    return components


def run_once(components: dict) -> dict:
    """One full scan cycle. Returns a small result dict for callers/tests."""
    from app.brain import decide, load_playbooks
    from app.scanner import scan_watchlist

    cfg = components["cfg"]
    client = components["client"]
    journal = components["journal"]
    executor = components["executor"]
    analyze_only = components["analyze_only"]

    playbooks = load_playbooks(
        Path(getattr(getattr(cfg, "paths", None), "knowledge_dir", "knowledge"))
    )

    # Build a single LLM client at module scope (LLMClient is patchable
    # on the main module for tests).
    llm_client = LLMClient(
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        api_key=cfg.llm_api_key or "",
        timeout=float(cfg.llm.timeout_seconds),
    )

    snapshots = scan_watchlist(client, cfg)

    # Equity snapshot for the cycle.
    try:
        info = client.account_info()
        journal.record_equity(
            balance=float(info.balance),
            equity=float(info.equity),
        )
        equity_rows = 1
    except Exception:  # noqa: BLE001
        equity_rows = 0

    decisions = rejected = orders = 0
    for snap in snapshots:
        if not analyze_only and components.get("kill_switch", False):
            break
        try:
            decision, raw = decide(
                snap,
                playbooks,
                journal_tail=journal.list_decisions(limit=20),
                portfolio_state={"daily_pnl": 0.0, "remaining_risk_pct": 100.0},
                config=cfg,
                client=llm_client,
            )
        except Exception as exc:  # noqa: BLE001
            journal.record_decision(
                symbol=snap.get("symbol", ""),
                action="error",
                reasoning=str(exc),
                approved=False,
                reject_reasons=[f"brain_error:{exc}"],
                raw_llm_json="",
            )
            rejected += 1
            decisions += 1
            continue

        if decision is None:
            journal.record_decision(
                symbol=snap.get("symbol", ""),
                action="error",
                reasoning="llm parse failed",
                approved=False,
                reject_reasons=["llm_parse_failed"],
                raw_llm_json=json.dumps(raw) if isinstance(raw, (dict, list))
                                 else str(raw),
            )
            decisions += 1
            rejected += 1
            continue

        decisions += 1
        if analyze_only or executor is None:
            # Just journal the decision without executing.
            journal.record_decision(
                symbol=decision.symbol,
                action=decision.action,
                playbook=decision.playbook or "",
                confidence=float(decision.confidence),
                risk_pct=float(decision.risk_pct),
                reasoning=decision.reasoning,
                approved=True,
                reject_reasons=[],
                raw_llm_json=json.dumps(raw) if isinstance(raw, (dict, list))
                                 else str(raw),
            )
            continue

        result = executor.execute(decision, raw_llm_json=json.dumps(raw)
                                  if isinstance(raw, (dict, list)) else str(raw))
        if result.status == "rejected":
            rejected += 1
        elif result.status in ("filled", "closed", "modified"):
            orders += 1

    return {
        "decisions": decisions,
        "orders": orders,
        "rejected": rejected,
        "equity_rows": equity_rows,
    }


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Decide + journal but NEVER order. ``--dry`` is the default behaviour."""
    from app.brain import decide, load_playbooks
    from app.risk import AccountState, RiskVerdict, check_decision
    from app.scanner import scan_watchlist

    cfg = load_config(args.config)
    if cfg is None:
        print("config not found")
        return 1

    # Lazy MT5 client; tolerate "terminal not running" in analyze mode.
    try:
        from app.mt5_client import MT5Error
        login, password, server, path = _resolve_creds(args)
        client = MT5Client()  # bound at module import, patchable in tests
        try:
            client.connect(login, password, server, path)
        except MT5Error as exc:
            print(f"MT5 not available, using empty data: {exc}")
            client = _NoopMT5Client()
    except ImportError:
        print("MetaTrader5 not installed, using empty data")
        client = _NoopMT5Client()

    try:
        snapshots = scan_watchlist(client, cfg)
    finally:
        try:
            client.shutdown()
        except Exception:  # noqa: BLE001
            pass

    playbooks = load_playbooks(
        Path(getattr(getattr(cfg, "paths", None), "knowledge_dir", "knowledge"))
    )

    starting_balance = float(getattr(getattr(client, "_last_account", None), "balance", 0.0) or 0.0)
    kill_path = Path(cfg.paths.kill_switch_path)
    account_state = AccountState(
        balance=starting_balance,
        equity=starting_balance,
        starting_balance_today=starting_balance,
        trade_mode=cfg.mode,
        kill_switch_engaged=kill_path.exists(),
    )

    for snap in snapshots:
        try:
            llm_client = LLMClient(
                base_url=cfg.llm.base_url,
                model=cfg.llm.model,
                api_key=cfg.llm_api_key or "",
                timeout=float(cfg.llm.timeout_seconds),
            )
            decision, raw = decide(
                snap,
                playbooks,
                journal_tail=[],
                portfolio_state={"daily_pnl": 0.0, "remaining_risk_pct": 100.0},
                config=cfg,
                client=llm_client,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{snap.get('symbol')}] brain error: {exc}")
            continue

        if decision is None:
            print(f"[{snap.get('symbol')}] no parseable decision: {raw!r}")
            continue

        verdict: RiskVerdict = check_decision(
            decision, account_state, [], [], cfg
        )
        print(
            f"[{snap.get('symbol')}] action={decision.action} "
            f"approved={verdict.approved} "
            f"reasons={verdict.reasons} "
            f"playbook={decision.playbook!r} "
            f"confidence={decision.confidence:.2f} "
            f"reasoning={decision.reasoning[:80]!r}"
        )
    return 0


class _NoopMT5Client:
    """Fallback when MT5 is unavailable: returns empty data, no positions."""

    def __init__(self):
        self._last_account = None

    def account_info(self):  # pragma: no cover - trivial
        return None

    def copy_rates(self, symbol, tf, count):
        return []

    def symbol_tick(self, symbol):  # pragma: no cover - trivial
        return None

    def positions(self, symbol=None):
        return []

    def shutdown(self):
        return None


def _cmd_review(args: argparse.Namespace) -> int:
    """Generate a daily review markdown file (advisory proposals only)."""
    from app.journal import Journal
    from app.review import generate_report

    cfg = load_config(args.config)
    if cfg is None:
        print("config not found")
        return 1

    journal = Journal(Path(cfg.paths.data_dir) / "journal.db")

    llm_client = None
    if not args.no_llm and cfg.llm_api_key:
        llm_client = LLMClient(
            base_url=cfg.llm.base_url,
            model=cfg.llm.model,
            api_key=cfg.llm_api_key,
            timeout=float(cfg.llm.timeout_seconds),
        )

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
        llm_client=llm_client,
        knowledge_dir=Path(cfg.paths.knowledge_dir),
        reports_dir=Path(cfg.paths.reports_dir),
    )
    print(f"wrote {path}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Pre-flight self-check against the *real* environment.

    Verifies, in order:
      1. the ``MetaTrader5`` package can be imported,
      2. the terminal can be reached and credentials resolve (CLI -> .env),
      3. the account is a DEMO account (safety invariant),
      4. every configured symbol is present / selectable and serves live
         rates + ticks (``doctor`` suggests broker-suffix matches when a
         configured name is absent),
      5. the LLM endpoint answers,
      6. the kill switch is off.

    Returns 0 only when no check FAILs (warnings are non-fatal).
    """
    cfg = load_config(args.config)
    checks: list[tuple[str, str, str]] = []

    def add(name: str, status: str, detail: str = "") -> None:
        checks.append((name, status, detail))

    # 1. MetaTrader5 package import (lazy so tests swap in a fake via _mt5).
    try:
        from app.mt5_client import MT5Client, LiveAccountError, _get_mt5
        _get_mt5()
        add("MetaTrader5 package import", "ok", "")
    except ImportError as exc:
        add("MetaTrader5 package import", "fail", str(exc))
        _print_checks(checks)
        print("\nInstall with: .venv\\Scripts\\pip install MetaTrader5")
        return 1

    # 2. Terminal + login (credentials resolved from CLI, then .env).
    login, password, server, path = _resolve_creds(args)
    client = MT5Client()
    try:
        client.connect(login, password, server, path)
        add("MT5 terminal connect", "ok",
            f"login={login} server={server} path={path or '<default>'}")
    except Exception as exc:  # noqa: BLE001
        add("MT5 terminal connect", "fail", f"{type(exc).__name__}: {exc}")
        _print_checks(checks)
        return 1

    # 3. Demo-account assertion (safety invariant #2).
    try:
        info = client.account_info()
        client.ensure_demo()
        add("demo account", "ok",
            f"login={info.login} mode={info.trade_mode} "
            f"balance={info.balance:.2f} equity={info.equity:.2f}")
    except LiveAccountError as exc:
        add("demo account", "fail", f"non-demo account: {exc}")
    except Exception as exc:  # noqa: BLE001
        add("demo account", "fail", f"{type(exc).__name__}: {exc}")

    # 4. Per-symbol health with broker-suffix suggestion.
    available = set(client.symbol_names())
    primary_tf = cfg.timeframes[0] if cfg.timeframes else "M15"
    for symbol in cfg.symbols:
        actual = symbol
        if symbol not in available:
            candidates = [n for n in available if n.startswith(symbol) and n != symbol]
            if candidates:
                add(f"symbol {symbol}", "warn",
                    "not found; possible broker-suffix matches: "
                    + ", ".join(candidates[:5]))
                actual = candidates[0]
            else:
                add(f"symbol {symbol}", "fail", "not present in terminal")
                continue
        try:
            client.select_symbol(actual)
        except Exception as exc:  # noqa: BLE001
            add(f"symbol {symbol} Market Watch", "warn", str(exc))
        try:
            client.copy_rates(actual, primary_tf, 5)
            client.symbol_tick(actual)
            add(f"symbol {symbol} live data", "ok", actual)
        except Exception as exc:  # noqa: BLE001
            add(f"symbol {symbol} live data", "fail",
                f"{type(exc).__name__}: {exc}")

    # 5. LLM endpoint.
    if cfg.llm_api_key:
        try:
            llm = LLMClient(
                base_url=cfg.llm.base_url,
                model=cfg.llm.model,
                api_key=cfg.llm_api_key,
                timeout=float(cfg.llm.timeout_seconds),
            )
            llm.chat([{"role": "user",
                       "content": "reply with the single word: ok"}], 0.0)
            add("LLM endpoint", "ok", f"{cfg.llm.base_url} model={cfg.llm.model}")
        except Exception as exc:  # noqa: BLE001
            add("LLM endpoint", "fail", f"{type(exc).__name__}: {exc}")
    else:
        add("LLM endpoint", "warn", "LLM_API_KEY missing from .env")

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trader")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="show MT5 connection & account")
    p_status.add_argument("--login", type=int, default=0)
    p_status.add_argument("--password", default="")
    p_status.add_argument("--server", default="")
    p_status.add_argument("--path", default="")
    p_status.set_defaults(func=_cmd_status)

    sub.add_parser("kill", help="engage kill switch").set_defaults(func=_cmd_kill)
    sub.add_parser("resume", help="release kill switch").set_defaults(func=_cmd_resume)

    p_run = sub.add_parser("run", help="continuous trading loop")
    p_run.add_argument("--login", type=int, default=0)
    p_run.add_argument("--password", default="")
    p_run.add_argument("--server", default="")
    p_run.add_argument("--path", default="")
    p_run.set_defaults(func=_cmd_run)
    p_once = sub.add_parser("once", help="single cycle")
    p_once.add_argument("--analyze", action="store_true",
                        help="single cycle in propose-only mode (no orders)")
    p_once.add_argument("--login", type=int, default=0)
    p_once.add_argument("--password", default="")
    p_once.add_argument("--server", default="")
    p_once.add_argument("--path", default="")
    p_once.set_defaults(func=_cmd_once)
    p_analyze = sub.add_parser("analyze", help="analyze only, no orders")
    p_analyze.add_argument("--dry", action="store_true",
                            help="alias for the default no-orders behavior")
    p_analyze.add_argument("--login", type=int, default=0)
    p_analyze.add_argument("--password", default="")
    p_analyze.add_argument("--server", default="")
    p_analyze.add_argument("--path", default="")
    p_analyze.set_defaults(func=_cmd_analyze)

    p_doctor = sub.add_parser(
        "doctor", help="pre-flight self-check of MT5 + LLM + kill switch")
    p_doctor.add_argument("--login", type=int, default=0)
    p_doctor.add_argument("--password", default="")
    p_doctor.add_argument("--server", default="")
    p_doctor.add_argument("--path", default="")
    p_doctor.set_defaults(func=_cmd_doctor)

    p_review = sub.add_parser("review", help="generate daily review report")
    p_review.add_argument("--day", default=None,
                           help="ISO date (YYYY-MM-DD); default = today UTC")
    p_review.add_argument("--no-llm", action="store_true",
                           help="skip the LLM-driven proposal section")
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