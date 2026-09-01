"""Tests for app/scheduler.py: in_session guard."""

from __future__ import annotations

import datetime as _dt

import pytest

from app.scheduler import in_session


class SH:
    def __init__(self, start="08:00", end="20:00", skip_weekends=True,
                 timezone="UTC"):
        self.start = start
        self.end = end
        self.skip_weekends = skip_weekends
        self.timezone = timezone


class Cfg:
    def __init__(self, **kw):
        self.session_hours = SH(**kw)


def test_in_session_inside_window():
    cfg = Cfg()
    now = _dt.datetime(2026, 8, 31, 12, 0, tzinfo=_dt.timezone.utc)  # Monday
    assert in_session(cfg, now) is True


def test_in_session_before_window():
    cfg = Cfg()
    now = _dt.datetime(2026, 8, 31, 5, 0, tzinfo=_dt.timezone.utc)
    assert in_session(cfg, now) is False


def test_in_session_after_window():
    cfg = Cfg()
    now = _dt.datetime(2026, 8, 31, 21, 0, tzinfo=_dt.timezone.utc)
    assert in_session(cfg, now) is False


def test_in_session_skips_weekend():
    cfg = Cfg(skip_weekends=True)
    saturday = _dt.datetime(2026, 8, 29, 12, 0, tzinfo=_dt.timezone.utc)
    sunday = _dt.datetime(2026, 8, 30, 12, 0, tzinfo=_dt.timezone.utc)
    assert in_session(cfg, saturday) is False
    assert in_session(cfg, sunday) is False


def test_in_session_allows_weekend_when_not_skipping():
    cfg = Cfg(skip_weekends=False)
    saturday = _dt.datetime(2026, 8, 29, 12, 0, tzinfo=_dt.timezone.utc)
    assert in_session(cfg, saturday) is True


def test_in_session_overnight_window_always_inside():
    """If end <= start we treat it as 24/7 enabled."""
    cfg = Cfg(start="22:00", end="06:00", skip_weekends=False)
    friday_night = _dt.datetime(2026, 8, 28, 23, 0, tzinfo=_dt.timezone.utc)
    assert in_session(cfg, friday_night) is True