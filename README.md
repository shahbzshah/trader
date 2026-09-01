# trader

LLM-driven trading agent for **MetaTrader 5** — demo account only.

It watches a watchlist on a schedule, builds technical snapshots (EMAs, RSI, ATR,
swing levels) from live candle data, asks an LLM to decide **against your written
playbooks**, enforces hard risk limits in pure code, executes on a **MT5 demo
account**, and journals every decision — taken or skipped — to SQLite. Ships with
a pre-flight `doctor` command, a kill switch, and a dark-theme web dashboard.

```
scan watchlist → LLM decide → risk veto/sizing → execute (demo) → journal → review
```

## Safety model (non-negotiable)

| Invariant | Enforcement |
|---|---|
| **Demo account only** | `mode: demo` in `config.yaml` is validated by the loader; `MT5Client` asserts `account_info().trade_mode` is demo at connect **and before every order**. There is no live-trading code path. |
| **The LLM proposes, code disposes** | `app/risk.py` is the *only* authority on position sizing and vetoes. Any decision failing a check is rejected and journaled with the reasons — regardless of what the model output says. |
| **Kill switch** | Flag file `data/KILL_SWITCH` + `kill`/`resume` CLI commands + dashboard buttons. Checked before every order and every scan cycle; the dashboard shows a red banner while engaged. |
| **Secrets never leak** | Credentials and API keys live only in `.env` (gitignored). The LLM never sees account credentials — only market data, playbooks, and portfolio state. |
| **Single MT5 boundary** | Only `app/mt5_client.py` imports the `MetaTrader5` package. All other modules are mockable and testable without a terminal. |

## Setup

### Prerequisites

- **Windows** + **Python 3.11+**
- **MetaTrader 5 desktop terminal** installed, logged into a **demo** account once
  (the Python package cannot work without the terminal — `doctor` will tell you
  exactly this if it's missing)
- An API key from any OpenAI-compatible LLM provider

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
#   LLM_API_KEY                             (your provider key)

python -m app.main doctor     # pre-flight — get every check to [OK]
```

### Verification order

```powershell
python -m app.main status                # real account, balance, positions
python -m app.main analyze --dry         # snapshots + LLM decisions, zero orders
python -m app.main once --analyze        # one full propose-only cycle
python -m app.main run                   # scheduled loop
uvicorn app.server:app                   # dashboard on http://127.0.0.1:8765
```

## CLI commands

| Command | What it does |
|---|---|
| `python -m app.main doctor` | Pre-flight self-check: MT5 package, terminal connect, demo assert, per-symbol live data (suggests broker-suffix matches like `EURUSD.a`), LLM ping, kill switch. Exit code 0 only when everything passes. |
| `python -m app.main status` | Connection status, account balance/equity, open positions. Never crashes when MT5 is unavailable. |
| `python -m app.main analyze --dry` | Build snapshots, ask the LLM, journal decisions. **Never orders.** |
| `python -m app.main once` | One full trading cycle (ignores session hours). |
| `python -m app.main once --analyze` | One propose-only cycle. |
| `python -m app.main run` | Continuous loop on the configured schedule, session-hours aware. |
| `python -m app.main kill` / `resume` | Engage / release the kill switch. |
| `python -m app.main review [--day YYYY-MM-DD] [--no-llm]` | Daily report to `reports/`: stats, every decision + reasoning, advisory playbook-update proposals. |

Credentials are resolved from CLI flags first, then `.env` — so `run` works
without repeating them on the command line.

## LLM provider

Any **OpenAI-compatible** `/chat/completions` endpoint works. The default is
[Poolside](https://poolside.ai) (`https://inference.poolside.ai/v1`, model
`poolside/laguna-s-2.1`). To switch providers, edit two lines in `config.yaml`
and keep the key in `.env`:

```yaml
llm:
  base_url: "https://inference.poolside.ai/v1"   # or Z.ai GLM, OpenAI, local Ollama, ...
  model: "poolside/laguna-s-2.1"
```

The brain expects strict JSON decisions (`no_trade` is the default action) and
retries / stands down on malformed or refused responses.

## Configuration highlights (`config.yaml`)

- `symbols`: EURUSD, GBPUSD, XAUUSD — `timeframes`: M15, H1
- `schedule.scan_interval_minutes`: 15 — `session_hours`: 08:00–20:00 UTC, weekends skipped

## Knowledge pack

- `knowledge/risk_rules.md` — structured markdown with machine-readable
  `RULE: key: value` lines the brain parses into the system prompt.
- `knowledge/playbooks/*.md` — per-setup templates (`breakout_retest.md`,
  `trend_pullback.md`) with Trigger / Confirmation / Stop / Take-profit /
  Invalidation sections. **Placeholders** — replace with your real rules.

Review-report proposals are advisory only; nothing in `knowledge/` is ever
edited automatically.

## Project structure

```
trader/
├── app/
│   ├── main.py          # CLI: run | once | analyze | status | doctor | kill | resume | review
│   ├── mt5_client.py    # the ONLY MetaTrader5 import; connection, rates, orders
│   ├── scanner.py       # live candles + indicators → snapshot dicts
│   ├── brain.py         # LLM client + prompt composition + strict Decision schema
│   ├── risk.py          # PURE sizing + veto checks (the safety authority)
│   ├── executor.py      # journal-verified order placement
│   ├── journal.py       # SQLite (WAL): decisions, orders, equity
│   ├── scheduler.py     # APScheduler wiring + session-hours guard
│   ├── server.py        # FastAPI dashboard API (state/decisions/equity/kill)
│   ├── review.py        # daily markdown reports
│   ├── config.py        # typed pydantic config + .env secrets loader
│   └── web/static/index.html   # single-file dark-theme dashboard
├── knowledge/           # risk rules + playbook templates
├── tests/               # 192 tests — no MT5/LLM/network needed (fakes + fixtures)
├── data/                # runtime: journal.db, KILL_SWITCH (gitignored)
├── reports/             # generated daily reports (gitignored)
├── config.yaml          # mode, symbols, schedule, risk limits, LLM settings
└── .env.example         # template for secrets
```

## Testing

```powershell
.venv\Scripts\pytest -q        # 192 passed
```

The suite runs fully offline: a fake MetaTrader5 module emulates the terminal
API, and a fake HTTP transport mocks the LLM endpoint — including edge cases
like malformed JSON, refusals, and order rejections.

## Runtime artifacts (gitignored)

`data/journal.db` (SQLite, WAL mode) · `data/KILL_SWITCH` · `logs/agent.log` ·
`reports/*.md`

## Disclaimer

Personal, educational project running on **demo accounts only**. Not financial
advice. Trading involves substantial risk; the authors take no responsibility
for any losses.

