"""Economic-calendar news blackout (fail-open).

Uses the free ForexFactory weekly JSON feed (no API key):

    https://nfs.faireconomy.media/ff_calendar_thisweek.json

The feed is fetched at most once per ``cache_ttl_hours`` and cached to
``data/news_cache.json``. If the fetch fails, the stale cache is used;
if there is no cache either, there is no blackout (fail-open) and a
warning is logged. For a demo bot this is the honest tradeoff — a live
system would want fail-closed.

The MT5 Python package exposes no calendar function, so an MQL5 bridge
was rejected (it would break the single-MT5-boundary design).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import httpx

logger = logging.getLogger(__name__)

FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Ordered impact levels; min_impact keeps everything at or above it.
_IMPACT_RANK = {"Holiday": 0, "Low": 1, "Medium": 2, "High": 3}


@dataclass(frozen=True)
class CalEvent:
    title: str
    currency: str
    time_utc: _dt.datetime
    impact: str


def _now_utc() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _parse_time(raw: str) -> _dt.datetime | None:
    try:
        dt = _dt.datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt.astimezone(_dt.timezone.utc)


def parse_events(
    raw: Iterable[Mapping[str, Any]],
    currencies: Iterable[str] = ("USD",),
    min_impact: str = "High",
) -> list[CalEvent]:
    """Filter a raw feed list down to trade-relevant events.

    Keeps events whose ``country`` is in ``currencies`` and whose impact
    rank is at or above ``min_impact``. Returns UTC-normalised events
    sorted by time.
    """
    want = {c.upper() for c in currencies}
    min_rank = _IMPACT_RANK.get(min_impact, 3)
    events: list[CalEvent] = []
    for row in raw:
        try:
            currency = str(row.get("country", "")).upper()
            impact = str(row.get("impact", ""))
            when = _parse_time(str(row.get("date", "")))
        except Exception:  # noqa: BLE001 - malformed row, skip it
            continue
        if currency not in want:
            continue
        if _IMPACT_RANK.get(impact, -1) < min_rank:
            continue
        if when is None:
            continue
        events.append(
            CalEvent(
                title=str(row.get("title", "")),
                currency=currency,
                time_utc=when,
                impact=impact,
            )
        )
    events.sort(key=lambda e: e.time_utc)
    return events


def _read_cache(cache_path: Path) -> tuple[list[dict], _dt.datetime | None]:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        fetched = _parse_time(str(payload.get("fetched_at", "")))
        events = payload.get("events", [])
        if isinstance(events, list):
            return events, fetched
    except Exception:  # noqa: BLE001 - corrupt cache behaves as no cache
        pass
    return [], None


def _write_cache(cache_path: Path, events: list[dict]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"fetched_at": _now_utc().isoformat(), "events": events}
            ),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001 - caching is best-effort
        logger.warning("could not write news cache to %s", cache_path)


def fetch_calendar(
    url: str = FF_CALENDAR_URL,
    cache_path: str | Path = "data/news_cache.json",
    ttl_hours: float = 12,
) -> list[dict]:
    """Return the raw weekly calendar list, using the disk cache.

    Fail-open: on any fetch problem the stale cache is used; with no
    cache at all an empty list is returned and a warning is logged.
    """
    path = Path(cache_path)
    cached, fetched_at = _read_cache(path)
    if fetched_at is not None and (_now_utc() - fetched_at) < _dt.timedelta(
        hours=ttl_hours
    ):
        return cached
    try:
        resp = httpx.get(url, timeout=15)
        resp.raise_for_status()
        events = resp.json()
        if not isinstance(events, list):
            raise ValueError("unexpected calendar payload")
        _write_cache(path, events)
        return events
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        logger.warning("news calendar fetch failed (%s); fail-open", exc)
        return cached


def in_blackout(
    now: _dt.datetime,
    events: Iterable[CalEvent],
    window_min: float = 30,
) -> CalEvent | None:
    """Return the event whose ±window covers ``now``, else None."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=_dt.timezone.utc)
    window = _dt.timedelta(minutes=window_min)
    for event in events:
        if abs(event.time_utc - now) <= window:
            return event
    return None


def get_events(
    cfg: Any,
    cache_path: str | Path | None = None,
) -> list[CalEvent]:
    """Fetch + filter events per config. Never raises for feed issues."""
    news_cfg = getattr(cfg, "news", None)
    url = getattr(news_cfg, "feed_url", FF_CALENDAR_URL)
    path = cache_path or getattr(news_cfg, "cache_path", "data/news_cache.json")
    ttl = float(getattr(news_cfg, "cache_ttl_hours", 12))
    currencies = list(getattr(news_cfg, "currencies", ["USD"]) or ["USD"])
    min_impact = str(getattr(news_cfg, "min_impact", "High"))
    raw = fetch_calendar(url, path, ttl)
    return parse_events(raw, currencies, min_impact)


def current_blackout(cfg: Any, now: _dt.datetime | None = None) -> CalEvent | None:
    """Return the active blackout event right now, or None.

    Respects ``news.enabled``. Never raises for feed issues (fail-open).
    """
    news_cfg = getattr(cfg, "news", None)
    if news_cfg is not None and not getattr(news_cfg, "enabled", True):
        return None
    try:
        events = get_events(cfg)
        window = float(getattr(news_cfg, "window_minutes", 30))
        return in_blackout(now or _now_utc(), events, window)
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        logger.warning("news blackout check failed (%s); fail-open", exc)
        return None


__all__ = [
    "CalEvent",
    "FF_CALENDAR_URL",
    "current_blackout",
    "fetch_calendar",
    "get_events",
    "in_blackout",
    "parse_events",
]
