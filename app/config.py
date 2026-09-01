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


class SessionHoursConfig(BaseModel):
    start: str
    end: str
    timezone: str = "UTC"
    skip_weekends: bool = True

    @field_validator("start", "end")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError("session hours must be HH:MM")
        h, m = int(parts[0]), int(parts[1])
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError("session hours out of range")
        return f"{h:02d}:{m:02d}"


class RiskConfig(BaseModel):
    max_risk_pct_per_trade: float = Field(default=1.0, ge=0.0, le=100.0)
    max_daily_loss_pct: float = Field(default=3.0, ge=0.0, le=100.0)
    max_open_trades: int = Field(default=3, ge=0)
    max_trades_per_day: int = Field(default=5, ge=0)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    default_playbook: str = "breakout_retest"


class LLMConfig(BaseModel):
    base_url: str
    model: str
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=60.0, gt=0.0)


class DashboardConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)


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
    risk: RiskConfig = RiskConfig()
    llm: LLMConfig
    dashboard: DashboardConfig = DashboardConfig()
    paths: PathsConfig = PathsConfig()
    llm_api_key: str = ""

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
    """Secrets that must come from .env, never config.yaml."""

    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    mt5_path: str = ""
    llm_api_key: str = ""

    @classmethod
    def from_env(cls, env_path: Path | None) -> "EnvSecrets":
        env: dict[str, str | None] = {}
        if env_path is not None and env_path.exists():
            env = {k: v for k, v in dotenv_values(env_path).items() if v is not None}
        for k in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_PATH", "LLM_API_KEY"):
            if k not in env and os.environ.get(k) is not None:
                env[k] = os.environ.get(k)
        try:
            login = int(env.get("MT5_LOGIN") or 0)
        except ValueError:
            login = 0
        return cls(
            mt5_login=login,
            mt5_password=env.get("MT5_PASSWORD", "") or "",
            mt5_server=env.get("MT5_SERVER", "") or "",
            mt5_path=env.get("MT5_PATH", "") or "",
            llm_api_key=env.get("LLM_API_KEY", "") or "",
        )


def load_config(
    config_path: Path | str = "config.yaml",
    env_path: Path | str | None = ".env",
    *,
    base_dir: Path | str | None = None,
) -> AppConfig:
    """Load and validate configuration from YAML and .env.

    Secrets (MT5 creds, LLM_API_KEY) are read from .env / process env and
    injected into the returned AppConfig. The ``mode`` is asserted to be
    ``demo`` per the safety invariant.
    """
    config_path = Path(config_path)
    if base_dir is not None:
        config_path = Path(base_dir) / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    secrets = EnvSecrets.from_env(
        Path(env_path) if env_path is not None else None
    )

    if raw.get("mode") != "demo":
        raise ValueError(
            "config.yaml mode must be 'demo' (safety invariant: demo-only v1)"
        )

    cfg = AppConfig(**raw, llm_api_key=secrets.llm_api_key)
    return cfg


__all__ = [
    "AppConfig",
    "DashboardConfig",
    "EnvSecrets",
    "LLMConfig",
    "PathsConfig",
    "RiskConfig",
    "ScheduleConfig",
    "SessionHoursConfig",
    "load_config",
]