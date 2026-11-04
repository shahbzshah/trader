"""Typed config loader.

Reads `config.yaml` plus `.env` (via python-dotenv) and validates against
pydantic models. The single public function is :func:`load_config`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field, field_validator


class ScheduleConfig(BaseModel):
    scan_interval_minutes: int = Field(ge=1, le=1440)


def _validate_hhmm(value: str) -> str:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"must be HH:MM, got {value!r}")
    h, m = int(parts[0]), int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"HH:MM out of range: {value!r}")
    return f"{h:02d}:{m:02d}"


class SessionHoursConfig(BaseModel):
    start: str
    end: str
    timezone: str = "UTC"
    skip_weekends: bool = True

    @field_validator("start", "end")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        return _validate_hhmm(v)


class RiskConfig(BaseModel):
    max_risk_pct_per_trade: float = Field(default=1.0, ge=0.0, le=100.0)
    max_daily_loss_pct: float = Field(default=3.0, ge=0.0, le=100.0)
    max_open_trades: int = Field(default=3, ge=0)
    max_trades_per_day: int = Field(default=5, ge=0)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    breakeven_enabled: bool = True
    breakeven_at_r: float = Field(default=1.0, ge=0.1, le=10.0)


class SignalsConfig(BaseModel):
    """Signals-only operation (no MT5, no trade execution)."""

    # Hypothetical account balance every signal is sized against
    # (lots = risk$ / (SL$ x 100) for gold). No real money anywhere.
    hypothetical_balance_usd: float = Field(default=10000.0, ge=0.0)
    # Which feed to use: "oanda" (main), "deriv" (fallback), "auto"
    # (OANDA when its credentials are present and the probe succeeds,
    # otherwise Deriv).
    feed: Literal["auto", "oanda", "deriv"] = "auto"
    telegram_enabled: bool = True
    max_signals_per_day: int = Field(default=5, ge=0)


class DashboardConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)


class NewsConfig(BaseModel):
    """Economic-calendar blackout. Fail-open: feed trouble never blocks trading."""

    enabled: bool = True
    feed_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    cache_path: str = "data/news_cache.json"
    cache_ttl_hours: int = Field(default=12, ge=1, le=168)
    currencies: list[str] = Field(default_factory=lambda: ["USD"])
    min_impact: str = "High"
    window_minutes: int = Field(default=30, ge=0, le=240)

    @field_validator("currencies")
    @classmethod
    def _upper_currencies(cls, v: list[str]) -> list[str]:
        return [c.upper() for c in v]


class PathsConfig(BaseModel):
    data_dir: str = "data"
    logs_dir: str = "logs"
    reports_dir: str = "reports"
    knowledge_dir: str = "knowledge"
    kill_switch_path: str = "data/KILL_SWITCH"


class AppConfig(BaseModel):
    mode: Literal["demo"] = "demo"
    symbols: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=list)
    schedule: ScheduleConfig
    session_hours: SessionHoursConfig
    kill_zones: list[str] = Field(default_factory=list)
    risk: RiskConfig = RiskConfig()
    news: NewsConfig = NewsConfig()
    signals: SignalsConfig = SignalsConfig()
    dashboard: DashboardConfig = DashboardConfig()
    paths: PathsConfig = PathsConfig()

    @field_validator("kill_zones")
    @classmethod
    def _zone_format(cls, v: list[str]) -> list[str]:
        out = []
        for z in v:
            parts = str(z).split("-")
            if len(parts) != 2:
                raise ValueError(f"kill zone must be 'HH:MM-HH:MM', got {z!r}")
            out.append(f"{_validate_hhmm(parts[0])}-{_validate_hhmm(parts[1])}")
        return out

    @field_validator("symbols")
    @classmethod
    def _nonempty_symbols(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one symbol must be configured")
        return [s.upper() for s in v]

    @field_validator("timeframes")
    @classmethod
    def _nonempty_tfs(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one timeframe must be configured")
        return v


class EnvSecrets(BaseModel):
    """Secrets that must come from .env, never config.yaml.

    OANDA v20 Practice (main feed) + Telegram (alerts). No MT5
    credentials exist in the signals-only bot.
    """

    oanda_api_token: str = ""
    oanda_account_id: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @classmethod
    def from_env(cls, env_path: Path | None) -> "EnvSecrets":
        env: dict[str, str | None] = {}
        if env_path is not None and env_path.exists():
            env = {k: v for k, v in dotenv_values(env_path).items() if v is not None}
        for k in ("OANDA_API_TOKEN", "OANDA_ACCOUNT_ID",
                  "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
            if k not in env and os.environ.get(k) is not None:
                env[k] = os.environ.get(k)
        return cls(
            oanda_api_token=env.get("OANDA_API_TOKEN", "") or "",
            oanda_account_id=env.get("OANDA_ACCOUNT_ID", "") or "",
            telegram_bot_token=env.get("TELEGRAM_BOT_TOKEN", "") or "",
            telegram_chat_id=env.get("TELEGRAM_CHAT_ID", "") or "",
        )


def load_config(
    config_path: Path | str = "config.yaml",
    env_path: Path | str | None = ".env",
    *,
    base_dir: Path | str | None = None,
) -> AppConfig:
    """Load and validate configuration from YAML and .env.

    Secrets (OANDA + Telegram) are read from .env / process env. The
    ``mode`` is asserted to be ``demo`` per the safety invariant.
    Unknown config keys (e.g. a legacy ``llm:`` section) are ignored.
    """
    config_path = Path(config_path)
    if base_dir is not None:
        config_path = Path(base_dir) / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if raw.get("mode") != "demo":
        raise ValueError(
            "config.yaml mode must be 'demo' (safety invariant: demo-only v1)"
        )

    cfg = AppConfig(**raw)
    return cfg


__all__ = [
    "AppConfig",
    "DashboardConfig",
    "EnvSecrets",
    "NewsConfig",
    "PathsConfig",
    "RiskConfig",
    "ScheduleConfig",
    "SessionHoursConfig",
    "SignalsConfig",
    "load_config",
]