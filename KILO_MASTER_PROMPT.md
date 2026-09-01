# Master Build Brief — Personal Trading Agent (MT5 · Paper-Trading v1)

You are building a standalone personal trading agent for a coder-trader on Windows.
It connects to MetaTrader 5, watches a watchlist on a schedule, analyzes each symbol
against **his written trading playbooks** using an LLM, and executes decisions on a
**DEMO account**. Every decision — taken or skipped — is journaled to SQLite.
Build it milestone by milestone. Each milestone has acceptance criteria; do not
start the next milestone until the current one passes.

## Non-negotiable safety invariants

1. `app/risk.py` is the ONLY authority on position sizing and limits. The LLM
   proposes; code disposes. Any decision failing a risk check is rejected and
   journaled with the rejection reasons — no matter what the LLM output says.
2. **Demo account only in v1.** `config.yaml` has `mode: demo`. The MT5 connector
   must assert `account_info().trade_mode` is a demo account before ANY order and
   refuse otherwise. Do not build a live-trading code path.
3. **Kill switch**: a flag file `data/KILL_SWITCH`, a dashboard button, and a CLI
   flag all halt trading instantly. The flag is checked before every order and
   before every scan cycle.
4. Secrets live only in `.env` (add to `.gitignore`). Config in `config.yaml`.
   Never hardcode credentials or keys.
5. ONLY `app/mt5_client.py` imports the `MetaTrader5` package. No other module
   may import it.
6. The LLM must never receive or produce account passwords; it only sees market
   data, playbooks, and portfolio state.

## Stack

- Python 3.11+, Windows, run from the repo root
- Packages: `MetaTrader5`, `pandas`, `pydantic>=2`, `PyYAML`, `python-dotenv`,
  `httpx` (LLM over plain HTTP — OpenAI-compatible `/chat/completions`; no vendor
  SDK), `fastapi`, `uvicorn`, `apscheduler`, `pytest`
- LLM is configured via `config.yaml` (`llm.base_url`, `llm.model`,
  `llm.temperature`) and `.env` (`LLM_API_KEY`) so any OpenAI-compatible provider
  works (Z.ai GLM, OpenAI, local Ollama, ...)
- SQLite via stdlib `sqlite3`, WAL mode. No ORM.
- All logging through the `logging` module: console + rotating file `logs/agent.log`.
  No bare `print()` outside the CLI entrypoints.

## Repository layout

```
trader/
  KILO_MASTER_PROMPT.md      # this file
  config.yaml                # mode, symbols, schedule, risk limits, llm settings
  .env.example               # MT5_LOGIN / MT5_PASSWORD / MT5_SERVER / LLM_API_KEY
  requirements.txt
  knowledge/
    playbooks/               # one .md file per setup (see M3)
    risk_rules.md
    market_notes.md
  app/
    __init__.py
    config.py                # typed config loader (pydantic Settings)
    mt5_client.py            # the ONLY MetaTrader5 import
    scanner.py               # candles + indicators -> snapshot dicts
    brain.py                 # LLM decision engine, strict pydantic Decision schema
    risk.py                  # PURE functions: sizing + veto checks
    journal.py               # sqlite: decisions, orders, equity, reviews
    executor.py              # journal-verified order placement via mt5_client
    scheduler.py             # APScheduler wiring + session-hours guard
    main.py                  # CLI entrypoint: run | once | analyze | status | kill
    server.py                # FastAPI dashboard backend
    web/static/index.html    # single-file dashboard page
  tests/                     # pytest, no MT5/LLM/network needed (mocks + fixtures)
  data/                      # journal.db, KILL_SWITCH flag (gitignored)
  reports/                   # generated daily/weekly review markdown
```

## Module contracts

### app/config.py
Pydantic settings loaded from `config.yaml` + `.env`. Sections: `mode` (always
"demo" for v1), `symbols: list[str]`, `timeframes: list[str]` (e.g. ["M15","H1"]),
`schedule.scan_interval_minutes`, `session_hours` (start/end, skip weekends),
`risk` (`max_risk_pct_per_trade` default 1.0, `max_daily_loss_pct` default 3.0,
`max_open_trades` default 3, `max_trades_per_day` default 5,
`default_playbook` name), `llm` (`base_url`, `model`, `temperature`, timeout),
`dashboard` (`host`, `port` default 8765).

### app/mt5_client.py
Thin, typed wrapper class `MT5Client`: `connect(login, password, server, path)`,
`ensure_demo()` (raises `LiveAccountError` if the account is not demo),
`account_info()`, `positions()`, `copy_rates(symbol, timeframe, count)`,
`symbol_tick(symbol)`, `market_order(symbol, direction, lots, sl, tp, comment)`,
`close_position(ticket)`, `modify_position(ticket, sl, tp)`. Convert MT5 enums to
plain strings at the boundary. Every call wrapped with clear exceptions and
logging. Retry connect once on transient failure.

### app/scanner.py
Pure function `build_snapshot(symbol, rates_by_tf, account, positions) -> dict`
plus `scan_watchlist(client, config)`. Snapshot per symbol: last close per TF,
EMA20/50/200, RSI14, ATR14, session high/low, current spread, recent swing
high/lows (last 20 bars), a tail of the last 50 closed candles per TF (OHLC
rounded), open positions for that symbol. Indicators computed with pandas
(hand-rolled EMA/RSI/ATR to avoid extra deps — keep functions pure and tested).

### app/brain.py
`decide(snapshot, playbooks, journal_tail, portfolio_state, config) -> Decision`
- System prompt = risk_rules.md summary + playbook contents (each playbook has a
  `## Name` heading) + explicit rules: "default action is no_trade; never exceed
  stated risk; output ONLY the JSON object".
- User message = snapshot JSON + open positions + last ~20 journal entries +
  today's realized P/L and remaining risk budget.
- Response parsed into pydantic `Decision`; on parse failure retry once with a
  stricter instruction, then record an `error` decision row.
- `Decision` schema (pydantic): `action` = "no_trade" | "propose" | "close" |
  "modify"; `symbol: str`; `direction: "buy"|"sell"` (propose only); `entry_type:
  "market"|"limit"`; `price: float|None`; `stop_loss: float`; `take_profit: float`;
  `risk_pct: float`; `confidence: float 0..1`; `playbook: str` (must name the
  playbook used); `reasoning: str` (must reference concrete snapshot numbers).

### app/risk.py  (pure functions, exhaustively tested)
- `position_size(equity, risk_pct, entry, stop_loss, symbol_info) -> lots`
  (respect min/max lot and lot step; round DOWN to step)
- `check_decision(decision, account_state, open_positions, journal, config)
  -> RiskVerdict(approved: bool, reasons: list[str])` enforcing:
  kill-switch off; demo mode; symbol in watchlist; SL present and on the correct
  side of entry; risk_pct <= max_risk_pct_per_trade (use the config value, not
  the LLM's, if lower); projected daily loss + this trade risk <= max_daily_loss_pct;
  open trades < max_open_trades; trades today < max_trades_per_day; confidence >=
  config minimum (default 0.6).
- `daily_realized_pl(journal_rows) -> float`
- No I/O inside this module. All inputs passed in. 100% unit-testable.

### app/journal.py
Tables: `decisions(id, ts, symbol, action, playbook, confidence, risk_pct,
reasoning, approved, reject_reasons_json, raw_llm_json)`,
`orders(id, ts, decision_id, ticket, symbol, direction, lots, entry, sl, tp,
status, error)`, `equity(id, ts, balance, equity, floating_pl)`,
`reviews(id, ts, path)` + `positions_history` if needed. Write a row for EVERY
LLM decision, approved or not. Equity snapshot on every scan. Thread-safe via a
single connection + lock or a connection per call.

### app/executor.py
Takes an approved Decision, re-validates against risk.py immediately before
placement (state may have changed), places the order via mt5_client, records the
order row, and links it to the decision id. Handles partial failures with clear
status strings. Also handles close/modify actions for open positions.

### app/scheduler.py + app/main.py
APScheduler job: every `scan_interval_minutes`, inside session hours only:
scan watchlist → brain decide → risk check → execute → journal. CLI
(`argparse`): `run` (loop), `once` (single cycle, ignores session hours),
`analyze` (decide + journal but NEVER order — propose only), `status`
(connection + account + open positions), `kill` (create flag file),
`resume` (remove flag file).

### app/server.py + app/web/static/index.html
FastAPI endpoints: `GET /api/state` (account, positions, today's decisions,
kill-switch status), `GET /api/decisions?limit=`, `GET /api/equity?days=`,
`POST /api/kill`, `POST /api/resume`. The dashboard is ONE static HTML page
(vanilla JS + canvas equity chart, dark theme, clear kill/resume buttons, red
banner when kill switch is active). Refresh every 10 s.

### app/review.py (M6)
Generates `reports/YYYY-MM-DD.md`: stats (trades, win/loss, P/L, risk used),
every decision with reasoning, and an LLM-generated section "Playbook update
proposals" comparing outcomes to playbooks. Proposals are advisory only — never
edit knowledge/ automatically.

## knowledge/ content model (M3)

`risk_rules.md`: structured markdown with `## Rule` headings, each with a
one-line machine-readable summary line starting with `RULE:` (key: value) the
brain can parse (e.g. `RULE: max_risk_pct: 1.0`).
`playbooks/<name>.md`: template per setup with sections: Name, Market/Session,
Trigger (entry conditions), Confirmation, Stop loss logic, Take profit logic,
Invalidation (stand down when...), Notes. Ship TWO example templates
(`breakout_retest.md`, `trend_pullback.md`) clearly marked as placeholders for
the user's real rules.

## Milestones & acceptance criteria

- **M1 — Skeleton, config, MT5 client**: venv instructions, requirements.txt,
  config.yaml, .env.example, all module stubs with real signatures, config loader
  tested, mt5_client fully implemented + unit tests with a fake MT5 module.
  Accept: `pytest` green (no MT5 needed); `python -m app.main status` prints a
  clean "MT5 terminal not running" message instead of crashing.
- **M2 — Scanner**: snapshot builder + unit tests on synthetic candle data
  (deterministic fixtures). Accept: tests green; snapshot JSON is stable and
  rounded (no float noise).
- **M3 — Knowledge pack, brain, risk**: playbook templates + risk_rules.md,
  brain with mocked httpx tests (valid JSON, invalid JSON retry, refusal path),
  risk.py exhaustively tested (sizing math incl. lot step rounding, every veto
  branch, daily loss math, kill switch).
  Accept: tests green; `python -m app.main analyze --dry` runs end-to-end with a
  mocked LLM and prints a Decision without touching orders.
- **M4 — Journal, executor, scheduler**: sqlite schema + migrations, full loop,
  `once` mode. Accept: tests green with mocked MT5; a full `--once` cycle on a
  mocked environment writes decision+order rows.
- **M5 — Dashboard**: endpoints + page. Accept: `uvicorn app.server:app` serves;
  manual check of state/decisions/equity/kill buttons; API smoke tests via
  httpx TestClient.
- **M6 — Review**: report generator + tests. Accept: generates a report from
  seeded journal fixture rows.

## Working rules for you (Kilo)

- Work milestone by milestone. After each: run the full test suite, report which
  tests cover which acceptance criteria, and list any deviations from this brief.
- Match the brief exactly where specified; where unspecified, choose the simplest
  robust option and note the choice.
- No TODO placeholders in delivered code; every function either works or raises
  NotImplementedError clearly marked as pending-milestone work.
- Ask before adding any dependency not listed above.
