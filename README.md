# trader — gold signals bot (XAUUSD, signals-only)

Rules-based **signals-only** bot for gold (XAUUSD). No MT5, no trade
execution, no broker account.

It watches gold on a schedule, builds technical snapshots (EMAs, RSI, ATR,
swing levels, fair value gaps, liquidity sweeps, market structure) from live
candle data, runs a **deterministic rules engine** (no LLM, no network) for
the SMC "sweep + FVG reclaim" setup, enforces hard risk limits in pure code,
sizes each setup against a **configured hypothetical balance** (default
$10,000 — no real money anywhere), journals every signal, and posts it to
Telegram. Ships with a pre-flight `doctor` command, a kill switch, a paper
backtester, and a dark-theme web dashboard.

```
scan watchlist → rules engine → risk veto/paper sizing → Telegram signal → journal
```

## Safety model (non-negotiable)

| Invariant | Enforcement |
|---|---|
| **Signals only — never trades** | There is no order-placement code anywhere. `app/executor.py` and `app/mt5_client.py` were deleted; no module imports MetaTrader5. A signal is a message, not an order. |
| **The strategy proposes, code disposes** | `app/risk.py` is the *only* authority on sizing and vetoes. Any decision failing a check is rejected and journaled with the reasons — regardless of what the rules engine suggested. |
| **Kill switch** | Flag file `data/KILL_SWITCH` + `kill`/`resume` CLI commands + dashboard buttons. Checked by the risk gate before every signal; the dashboard shows a red banner while engaged. |
| **Secrets never leak** | OANDA + Telegram credentials live only in `.env` (gitignored) or the Render dashboard. They are never printed, logged, or committed. The Deriv fallback needs no credentials at all. |
| **Paper sizing** | Every signal prints risk$ and lots computed from `signals.hypothetical_balance_usd` (default $10,000) at `risk.max_risk_pct_per_trade` (default 1%): lots = risk$ / (SL$ x 100) for gold (1 lot = 100 oz). |

## Market data

| Feed | Role | Auth | Notes |
|---|---|---|---|
| **OANDA v20 Practice** (`api-fxpractice.oanda.com`) | MAIN | `OANDA_API_TOKEN` + `OANDA_ACCOUNT_ID` from env | Free practice account; M15/H1 candles + live pricing for `XAU_USD`. |
| **Deriv WebSocket v3** | FALLBACK / cross-check | None (public `app_id` 1089) | `frxXAUUSD` ticks + native M15/H1 candles; ~1 tick/s; 5-endpoint failover; live tick stream for feed health. |

`feed: auto` (default) uses OANDA when its credentials are present and the
probe succeeds, otherwise Deriv. `--feed oanda|deriv` forces one feed.
OANDA practice signup from Kenya is **unconfirmed** — try it and see; if
OANDA won't onboard you, the bot runs fine on Deriv alone.

## Setup

### Prerequisites

- **Python 3.11+** (any OS — no Windows, no MT5 terminal needed)

### Install

```bash
git clone https://github.com/shahbzshah/trader.git
cd trader

python -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
# then edit .env:
#   OANDA_API_TOKEN / OANDA_ACCOUNT_ID   (free Practice account, see .env.example)
#   TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (your own bot via @BotFather)

python -m app.main doctor     # pre-flight — get every check to [OK]
```

### Verification order

```bash
python -m app.main status --feed deriv   # feed + latest tick (works with no credentials)
python -m app.main analyze --dry         # snapshots + rules-engine decisions + sizing preview, zero alerts
python -m app.main once --feed deriv     # one full cycle (journals signals, sends Telegram if configured)
python -m app.main run                   # scheduled loop (local)
python -m app.main serve                 # scan loop + dashboard on $PORT (Render)
```

## CLI commands

| Command | What it does |
|---|---|
| `python -m app.main doctor` | Pre-flight self-check: OANDA credentials (warn if missing), feed connect, per-symbol live data, rules-engine self-test, news calendar, Telegram token validation (`getMe` — sends no message), kill switch. Exit code 0 only when everything passes. |
| `python -m app.main status` | Feed connection, latest tick, recent M15 candles, kill-switch state. Never crashes when no feed is reachable. |
| `python -m app.main analyze --dry` | Build snapshots, run the rules engine, print decisions + paper sizing. **Never journals, never alerts.** |
| `python -m app.main once` | One full cycle (ignores session hours). Signals are journaled + sent to Telegram. |
| `python -m app.main once --analyze` | One propose-only cycle (no alerts). |
| `python -m app.main run` | Continuous loop on the configured schedule, session-hours aware (local use). |
| `python -m app.main serve` | Scan loop + dashboard web server on `0.0.0.0:$PORT` (Render free web service). |
| `python -m app.main kill` / `resume` | Engage / release the kill switch. |
| `python -m app.main review [--day YYYY-MM-DD]` | Daily report to `reports/`: stats, every decision + reasoning, deterministic advisory proposals. |
| `python -m app.main backtest --symbol XAUUSD --timeframe M15 --csv data.csv` | Paper backtest of the rules engine on CSV candles (`time,open,high,low,close,volume`). No feed, no network. Prints trades, win rate, avg R, profit factor, max drawdown. |

## Strategy: sweep + FVG reclaim

One setup, run mechanically (`app/strategy.py`, spec in
`knowledge/playbooks/sweep_fvg_reclaim.md`):

- **Bias (H1):** market-structure trend up/down (BOS/CHoCH) aligned with EMA50.
- **Trigger (M15):** liquidity sweep of the recent swing high/low within the
  last 5 bars — wick beyond the level, close back inside.
- **Entry:** fresh, unfilled M15 fair value gap formed at/after the sweep;
  limit entry at the FVG midpoint.
- **Stop:** beyond the sweep extreme, never closer than 1x ATR14.
- **Target:** single take-profit at 2R.
- **Filters:** configured session hours only, spread cap, weekends skipped,
  and a **news blackout** — no signals within ±30 min of a high-impact USD
  event (free ForexFactory calendar feed, cached to `data/news_cache.json`,
  fail-open with a logged warning; see `news:` in `config.yaml`).

All thresholds are module-level constants at the top of `app/strategy.py`.
The engine is deterministic: the same snapshot always yields the same
decision. `no_trade` is the default action.

Signals: the strategy's `entry_type="limit"` decisions become Telegram
alerts with entry/SL/TP + paper sizing. Limit placement is validated
against the live tick (buy limit below ask / sell limit above bid);
mispriced decisions are rejected before they are alerted. Identical
signals are de-duplicated (same symbol/direction/entry/SL/TP already
alerted today is not re-sent), and `signals.max_signals_per_day`
(default 5) caps alert spam.

## Configuration highlights (`config.yaml`)

- `symbols`: XAUUSD (gold only) — `timeframes`: M15, H1
- `schedule.scan_interval_minutes`: 15 — `session_hours`: 00:00–23:59 UTC (all sessions), weekends skipped — `kill_zones`: none (removed 2026-11-03)
- `signals.hypothetical_balance_usd`: 10000.0 — `signals.feed`: auto — `signals.max_signals_per_day`: 5

## Deploying to Render (free)

The bot runs as a **free web service** on a second Render workspace
(separate Gmail account — the Crash 1000 bot already uses the first, and
one workspace's 750 free instance-hours/month is not enough for two
24/7 services).

```bash
# In the Render dashboard (second account), create a Web Service from this repo:
#   runtime: Docker, plan: Free, region: Frankfurt
#   health check path: /health
# Then set these env vars in the dashboard (never commit them):
#   OANDA_API_TOKEN, OANDA_ACCOUNT_ID, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
# Or use the blueprint: render.yaml (env vars are sync:false — set them in the dashboard)
```

Keep the service awake with a 5-minute pinger (e.g. UptimeRobot) on
`/health` — it returns `{"ok": true, "feed_ok": ..., "last_tick_age_s": ...}`.

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
│   ├── main.py          # CLI: run | serve | once | analyze | backtest | status | doctor | kill | resume | review
│   ├── market_data.py   # shared feed types + MarketDataClient interface (no MT5)
│   ├── oanda_client.py  # OANDA v20 Practice feed (MAIN)
│   ├── deriv_client.py  # Deriv WebSocket v3 feed (FALLBACK) + live tick stream
│   ├── signals.py       # signals pipeline: risk gate → paper sizing → dedupe → Telegram
│   ├── telegram.py      # Telegram Bot API sender (send-only)
│   ├── http_client.py   # httpx client construction (sandbox proxy-safe)
│   ├── scanner.py       # live candles + indicators + SMC (FVG/sweep/structure) → snapshots
│   ├── strategy.py      # deterministic rules engine + Decision schema (no LLM, no network)
│   ├── brain.py         # decision interface: re-exports Decision/decide, playbook loader
│   ├── backtest.py      # paper backtester: scanner → strategy → risk on CSV candles
│   ├── risk.py          # PURE sizing + veto checks (the safety authority)
│   ├── manage.py        # breakeven logic (retained, parked — no positions in signals-only)
│   ├── journal.py       # SQLite (WAL): decisions (orders table stays empty by design)
│   ├── scheduler.py     # APScheduler wiring + session-hours guard
│   ├── server.py        # FastAPI dashboard API (health/state/signals/decisions/kill)
│   ├── review.py        # daily markdown reports (deterministic proposals)
│   ├── config.py        # typed pydantic config + .env secrets loader
│   └── web/static/index.html   # single-file dark-theme dashboard
├── knowledge/           # risk rules + the sweep+FVG playbook
├── tests/               # mocked feeds + live Deriv test (marked `live`)
├── data/                # runtime: journal.db, KILL_SWITCH (gitignored)
├── reports/             # generated daily reports (gitignored)
├── config.yaml          # mode, symbols, schedule, risk limits, signals
├── Dockerfile           # Render free web service image
├── render.yaml          # Render blueprint (env vars set in dashboard)
└── .env.example         # template for secrets
```

## Testing

```bash
.venv/bin/pytest -q        # full suite: mocked feeds + live Deriv test
.venv/bin/pytest -q -m "not live"   # offline only
```

The suite runs mostly offline: mocked OANDA/Telegram HTTP, a mocked Deriv
transport, and hand-crafted candle fixtures exercise the FVG/sweep/structure
detectors, the rules engine, sizing, dedupe, and the kill switch. One test
(`test_live_tick_and_candles`, marked `live`) hits the real Deriv public
gateway and is skipped when the network is unavailable.

## Runtime artifacts (gitignored)

`data/journal.db` (SQLite, WAL mode) · `data/KILL_SWITCH` · `logs/agent.log` ·
`reports/*.md`

## Disclaimer

Personal, educational project. Signals only — this bot never places a trade.
Not financial advice. Trading involves substantial risk; the authors take no
responsibility for any losses.
