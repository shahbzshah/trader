"""APScheduler wiring + session-hours guard.

``in_session`` checks whether the current UTC time falls within the
configured session window and (optionally) skips weekends.

``Scheduler`` wraps APScheduler to fire ``cycle`` on
``config.schedule.scan_interval_minutes`` minutes, but only inside the
session window. The cycle is a callable that takes no arguments (it
will fetch the live config + client itself).
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


def _parse_hhmm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def in_session(config: Any, now: _dt.datetime | None = None) -> bool:
    """Return True if ``now`` (default = current UTC) is within the
    configured trading session.

    The session is defined by ``config.session_hours.start`` and
    ``config.session_hours.end`` (HH:MM strings). The window is assumed
    to be UTC unless the config has a ``timezone`` field (currently
    unused; we treat the times as UTC).

    If ``session_hours.skip_weekends`` is true, Saturday and Sunday
    (UTC weekday 5/6) are excluded.
    """
    now = now or _dt.datetime.now(_dt.timezone.utc)
    sh = config.session_hours
    if getattr(sh, "skip_weekends", False) and now.weekday() >= 5:
        return False
    start_h, start_m = _parse_hhmm(sh.start)
    end_h, end_m = _parse_hhmm(sh.end)
    start = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    if end <= start:
        # Session wraps midnight (e.g. 22:00 -> 06:00). Treat as 24h enabled.
        return True
    return start <= now <= end


class Scheduler:
    """Run a cycle on a fixed interval, gated by session hours."""

    def __init__(self, config: Any, cycle: Callable[[], None]) -> None:
        self.config = config
        self.cycle = cycle
        self._scheduler: Optional[BackgroundScheduler] = None

    def start(self) -> None:
        if self._scheduler is not None:
            return
        interval = int(self.config.schedule.scan_interval_minutes)
        self._scheduler = BackgroundScheduler(daemon=True, timezone="UTC")
        self._scheduler.add_job(
            self._safe_cycle,
            trigger=IntervalTrigger(minutes=interval),
            id="trader_cycle",
            max_instances=1,
            coalesce=True,
            next_run_time=_dt.datetime.now(_dt.timezone.utc),
        )
        self._scheduler.start()
        logger.info("scheduler started, interval=%d min", interval)

    def stop(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None

    def _safe_cycle(self) -> None:
        if not in_session(self.config):
            logger.info("outside session hours, skipping cycle")
            return
        if self.config.paths.kill_switch_path and \
                self._kill_switch_engaged():
            logger.warning("kill switch engaged, skipping cycle")
            return
        try:
            self.cycle()
        except Exception:  # noqa: BLE001
            logger.exception("cycle failed")

    def _kill_switch_engaged(self) -> bool:
        import os
        try:
            return os.path.exists(self.config.paths.kill_switch_path)
        except OSError:
            return False


__all__ = ["Scheduler", "in_session"]