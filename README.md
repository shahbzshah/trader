# trader

Rules-based trading agent for **MetaTrader 5** — demo account only.

It watches a watchlist on a schedule, builds technical snapshots (EMAs, RSI, ATR,
swing levels, fair value gaps, liquidity sweeps, market structure) from live
candle data, runs a **deterministic rules engine** (no LLM, no network) for the
SMC "sweep + FVG reclaim" setup, enforces hard risk limits in pure code, executes
on a **MT5 demo account**, and journals every decision — taken or skipped — to
SQLite. Ships with a pre-flight `doctor` command, a kill switch, a paper
backtester, and a dark-theme web dashboard.

```
scan watchlist → rules engine → risk veto/sizing → execute (demo) → journal → review
```

## Safety model (non-negotiable)

| Invariant | Enforcement |
|---|---|
| **Demo account only** | `mode: demo` in `config.yaml` is validated by the loader; `MT5Client` asserts `account_info().trade_mode` is demo at connect **and before every order**. There is no live-trading code path. |
| **The strategy proposes, code disposes** | `app/risk.py` is the *only* authority on position sizing and vetoes. Any decision failing a check is rejected and journaled with the reasons — regardless of what the rules engine suggested. |
| **Kill switch** | Flag file `data/KILL_SWITCH` + `kill`/`resume` CLI commands + dashboard buttons. Checked before every order and every scan cycle; the dashboard shows a red banner while engaged. |
| **Secrets never leak** | MT5 credentials live only in `.env` (gitignored). There are no API keys and no network calls in the decision path. |
| **Single MT5 boundary** | Only `app/mt5_client.py` imports the `MetaTrader5` package. All other modules are mockable and testable without a terminal. |

## Setup

### Prerequisites

- **Windows** + **Python 3.11+**
- **MetaTrader 5 desktop terminal** installed, logged into a **demo** account once
  (the Python package cannot work without the terminal — `doctor` will tell you
  exactly this if it's missing)

### Install

```powershell
git clone https://github.com/shahbzshah/trader.git
cd trader

python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

copy .env.example .env
# then edit .env:
#   MT5_LOGIN / MT5_PASSWORD / MT5_SERVER   (from your broker's demo email)
#   MT5_PATH                                (optional path to terminal64.exe)

python -m app.main doctor     # pre-flight — get every check to [OK]
```

### Verification order

```powershell
python -m app.main status                # real account, balance, positions
python -m app.main analyze --dry         # snapshots + rules-engine decisions, zero orders
python -m app.main once --analyze        # one full propose-only cycle
python -m app.main run                   # scheduled loop
uvicorn app.server:app                   # dashboard on http://127.0.0.1:8765
```

## CLI commands

| Command | What it does |
|---|---|
| `python -m app.main doctor` | Pre-flight self-check: MT5 package, terminal connect, demo assert, per-symbol live data (suggests broker-suffix matches like `EURUSD.a`), rules-engine self-test, kill switch. Exit code 0 only when everything passes. |
| `python -m app.main status` | Connection status, account balance/equity, open positions. Never crashes when MT5 is unavailable. |
| `python -m app.main analyze --dry` | Build snapshots, run the rules engine, journal decisions. **Never orders.** |
| `python -m app.main once` | One full trading cycle (ignores session hours). |
| `python -m app.main once --analyze` | One propose-only cycle. |
| `python -m app.main run` | Continuous loop on the configured schedule, session-hours aware. |
| `python -m app.main kill` / `resume` | Engage / release the kill switch. |
| `python -m app.main review [--day YYYY-MM-DD]` | Daily report to `reports/`: stats, every decision + reasoning, deterministic advisory proposals. |
| `python -m app.main backtest --symbol XAUUSD --timeframe M15 --csv data.csv` | Paper backtest of the rules engine on CSV candles (`time,open,high,low,close,volume`). No MT5, no network. Prints trades, win rate, avg R, profit factor, max drawdown. |

Credentials are resolved from CLI flags first, then `.env` — so `run` works
without repeating them on the command line.

## Strategy: sweep + FVG reclaim

One setup, traded mechanically (`app/strategy.py`, spec in
`knowledge/playbooks/sweep_fvg_reclaim.md`):

- **Bias (H1):** market-structure trend up/down (BOS/CHoCH) aligned with EMA50.
- **Trigger (M15):** liquidity sweep of the recent swing high/low within the
  last 5 bars — wick beyond the level, close back inside.
- **Entry:** fresh, unfilled M15 fair value gap formed at/after the sweep;
  limit order at the FVG midpoint.
- **Stop:** beyond the sweep extreme, never closer than 1x ATR14.
- **Target:** single take-profit at 2R.
- **Filters:** configured session hours only, spread cap, weekends skipped,
  and a **news blackout** — no trades within ±30 min of a high-impact USD
  event (free ForexFactory calendar feed, cached to `data/news_cache.json`,
  fail-open with a logged warning; see `news:` in `config.yaml`).

All thresholds are module-level constants at the top of `app/strategy.py`.
The engine is deterministic: the same snapshot always yields the same
decision. `no_trade` is the default action.

Execution: the strategy's `entry_type="limit"` decisions are placed as
broker **pending orders** (`app/executor.py` → `MT5Client.pending_order`),
resting at the FVG midpoint until price touches them and expiring after 75
minutes (matching the backtest's 5-bar pending window). Limit placement is
validated against the live tick (buy limit below ask / sell limit above
bid); mispriced or `entry_type`-unknown decisions are rejected before they
reach the broker. `entry_type="market"` decisions still fill immediately.

## Configuration highlights (`config.yaml`)

- `symbols`: XAUUSD (gold only) — `timeframes`: M15, H1
- `schedule.scan_interval_minutes`: 15 — `session_hours`: 08:00–20:00 UTC, weekends skipped

## Knowledge pack

- `knowledge/risk_rules.md` — risk invariants (caps, demo-only, kill switch).
- `knowledge/playbooks/sweep_fvg_reclaim.md` — the human-readable spec of the
  encoded setup (Trigger / Confirmation / Stop / Take-profit / Invalidation).
  Keep it in sync with the constants in `app/strategy.py`.

Review-report proposals are advisory only; nothing in `knowledge/` is ever
edited automatically.

## Project structure

```
trader/
├── app/
│   ├── main.py          # CLI: run | once | analyze | backtest | status | doctor | kill | resume | review
│   ├── mt5_client.py    # the ONLY MetaTrader5 import; connection, rates, orders
│   ├── scanner.py       # live candles + indicators + SMC (FVG/sweep/structure) → snapshots
│   ├── strategy.py      # deterministic rules engine + Decision schema (no LLM, no network)
│   ├── brain.py         # decision interface: re-exports Decision/decide, playbook loader
│   ├── backtest.py      # paper backtester: scanner → strategy → risk on CSV candles
│   ├── risk.py          # PURE sizing + veto checks (the safety authority)
│   ├── executor.py      # journal-verified order placement
│   ├── journal.py       # SQLite (WAL): decisions, orders, equity
│   ├── scheduler.py     # APScheduler wiring + session-hours guard
│   ├── server.py        # FastAPI dashboard API (state/decisions/equity/kill)
│   ├── review.py        # daily markdown reports (deterministic proposals)
│   ├── config.py        # typed pydantic config + .env secrets loader
│   └── web/static/index.html   # single-file dark-theme dashboard
├── knowledge/           # risk rules + the sweep+FVG playbook
├── tests/               # no MT5/network needed (fakes + fixtures)
├── data/                # runtime: journal.db, KILL_SWITCH (gitignored)
├── reports/             # generated daily reports (gitignored)
├── config.yaml          # mode, symbols, schedule, risk limits
└── .env.example         # template for secrets
```

## Testing

```powershell
.venv\Scripts\pytest -q        # full suite, no MT5/network needed
```

The suite runs fully offline: a fake MetaTrader5 module emulates the terminal
API, and hand-crafted candle fixtures exercise the FVG/sweep/structure
detectors and the rules engine — including edge cases like filled gaps,
missing bias, wide spreads, and order rejections.

## Runtime artifacts (gitignored)

`data/journal.db` (SQLite, WAL mode) · `data/KILL_SWITCH` · `logs/agent.log` ·
`reports/*.md`

## Disclaimer

Personal, educational project running on **demo accounts only**. Not financial
advice. Trading involves substantial risk; the authors take no responsibility
for any losses.

