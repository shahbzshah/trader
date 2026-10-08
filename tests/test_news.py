"""Tests for the news blackout module (app/news.py)."""

import datetime as dt
import json

import pytest

from app import news
from app.config import load_config


def _raw(title, country, iso, impact):
    return {
        "title": title,
        "country": country,
        "date": iso,
        "impact": impact,
        "forecast": "",
        "previous": "",
    }


@pytest.fixture
def fixture_rows():
    return [
        _raw("FOMC Minutes", "USD", "2026-10-07T14:00:00-04:00", "High"),
        _raw("CPI m/m", "USD", "2026-10-08T08:30:00-04:00", "High"),
        _raw("Retail Sales", "USD", "2026-10-09T08:30:00-04:00", "Medium"),
        _raw("ECB Presser", "EUR", "2026-10-08T14:15:00-04:00", "High"),
        _raw("Bank Holiday", "USD", "2026-10-07T00:00:00-04:00", "Holiday"),
        {"title": "broken", "country": "USD"},  # missing date/impact -> skipped
    ]


def test_parse_events_filters_currency_impact_and_time(fixture_rows):
    events = news.parse_events(fixture_rows, currencies=["USD"], min_impact="High")
    titles = [e.title for e in events]
    assert titles == ["FOMC Minutes", "CPI m/m"]
    # 14:00 EDT -> 18:00 UTC
    assert events[0].time_utc == dt.datetime(
        2026, 10, 7, 18, 0, tzinfo=dt.timezone.utc
    )
    assert events[0].currency == "USD"


def test_parse_events_medium_kept_when_threshold_lower(fixture_rows):
    events = news.parse_events(fixture_rows, currencies=["USD"], min_impact="Medium")
    assert "Retail Sales" in [e.title for e in events]


def test_in_blackout_window_edges():
    now = dt.datetime(2026, 10, 7, 18, 0, tzinfo=dt.timezone.utc)
    event = news.CalEvent("FOMC Minutes", "USD", now, "High")
    assert news.in_blackout(now, [event], window_min=30) is event
    before = now - dt.timedelta(minutes=29)
    assert news.in_blackout(before, [event], window_min=30) is event
    outside = now - dt.timedelta(minutes=31)
    assert news.in_blackout(outside, [event], window_min=30) is None
    assert news.in_blackout(now, [], window_min=30) is None


def test_in_blackout_accepts_naive_now():
    naive = dt.datetime(2026, 10, 7, 18, 0)  # treated as UTC
    event = news.CalEvent(
        "X", "USD", dt.datetime(2026, 10, 7, 18, 0, tzinfo=dt.timezone.utc), "High"
    )
    assert news.in_blackout(naive, [event], window_min=30) is event


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_calendar_caches_and_reuses(tmp_path, monkeypatch, fixture_rows):
    cache = tmp_path / "news_cache.json"
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return _FakeResp(fixture_rows)

    monkeypatch.setattr(news.httpx, "get", fake_get)
    first = news.fetch_calendar(cache_path=cache, ttl_hours=12)
    assert len(first) == 6
    assert cache.exists()
    second = news.fetch_calendar(cache_path=cache, ttl_hours=12)
    assert second == first
    assert len(calls) == 1  # second call served from cache


def test_fetch_calendar_fail_open_uses_stale_cache(tmp_path, monkeypatch, fixture_rows):
    cache = tmp_path / "news_cache.json"
    cache.write_text(
        json.dumps({"fetched_at": "2000-01-01T00:00:00+00:00", "events": fixture_rows}),
        encoding="utf-8",
    )

    def boom(url, timeout):
        raise RuntimeError("network down")

    monkeypatch.setattr(news.httpx, "get", boom)
    assert news.fetch_calendar(cache_path=cache, ttl_hours=12) == fixture_rows


def test_fetch_calendar_fail_open_no_cache(tmp_path, monkeypatch):
    def boom(url, timeout):
        raise RuntimeError("network down")

    monkeypatch.setattr(news.httpx, "get", boom)
    assert news.fetch_calendar(cache_path=tmp_path / "missing.json") == []


def _cfg_with_news(tmp_path, **overrides):
    """Load the real config.yaml with news-section overrides applied."""
    import yaml

    from app.config import AppConfig

    with open("config.yaml", encoding="utf-8") as fh:
        merged = yaml.safe_load(fh) or {}
    news_section = dict(merged.get("news") or {})
    for key, value in overrides.items():
        news_section[key] = value
    merged["news"] = news_section
    return AppConfig(**merged)


def test_current_blackout_disabled_returns_none(tmp_path):
    cfg = _cfg_with_news(tmp_path, enabled=False)
    assert news.current_blackout(cfg) is None


def test_current_blackout_active_event(tmp_path, monkeypatch, fixture_rows):
    cfg = _cfg_with_news(tmp_path, enabled=True, window_minutes=120)
    monkeypatch.setattr(
        news, "fetch_calendar", lambda *a, **k: fixture_rows
    )
    event = news.current_blackout(
        cfg, now=dt.datetime(2026, 10, 7, 18, 0, tzinfo=dt.timezone.utc)
    )
    assert event is not None
    assert event.title == "FOMC Minutes"


def test_current_blackout_no_event_now(tmp_path, monkeypatch, fixture_rows):
    cfg = _cfg_with_news(tmp_path, enabled=True, window_minutes=30)
    monkeypatch.setattr(
        news, "fetch_calendar", lambda *a, **k: fixture_rows
    )
    event = news.current_blackout(
        cfg, now=dt.datetime(2026, 10, 1, 12, 0, tzinfo=dt.timezone.utc)
    )
    assert event is None


def test_news_config_defaults_from_real_yaml():
    cfg = load_config("config.yaml", env_path="/nonexistent/.env")
    assert cfg.news.enabled is True
    assert cfg.news.currencies == ["USD"]
    assert cfg.news.min_impact == "High"
    assert cfg.news.window_minutes == 30
