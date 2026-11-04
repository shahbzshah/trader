"""Daily/weekly review report generator.

Generates ``reports/YYYY-MM-DD.md`` (and ``reports/YYYY-MM-DD-week.md``)
with:
  - a stats block (trades, win/loss, realised P/L, risk used)
  - every decision with reasoning
  - a short deterministic "Playbook update proposals" section derived
    from journal statistics (heuristic flags, no LLM).

The proposals are **advisory only**; this module never edits
``knowledge/``.

The function is intentionally synchronous and pure with respect to
its inputs (a Journal). It can be invoked manually via ``app review``
or wired into the scheduler.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from app.journal import Journal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ReviewStats:
    decision_count: int
    approved_count: int
    rejected_count: int
    orders_filled: int
    orders_rejected: int
    realized_pl: float
    win_count: int
    loss_count: int
    starting_balance: float
    ending_balance: float
    risk_used_pct: float

    def to_dict(self) -> dict:
        return {
            "decision_count": self.decision_count,
            "approved_count": self.approved_count,
            "rejected_count": self.rejected_count,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "realized_pl": round(self.realized_pl, 5),
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "starting_balance": round(self.starting_balance, 5),
            "ending_balance": round(self.ending_balance, 5),
            "risk_used_pct": round(self.risk_used_pct, 5),
        }


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------


def _start_of_day_utc(day: _dt.date) -> int:
    return int(_dt.datetime(day.year, day.month, day.day, tzinfo=_dt.timezone.utc).timestamp())


def _end_of_day_utc(day: _dt.date) -> int:
    return _start_of_day_utc(day) + 86400


def compute_stats(
    journal: Journal,
    *,
    day: _dt.date | None = None,
) -> ReviewStats:
    """Compute stats for the given UTC day (default: today UTC)."""
    target = day or _dt.datetime.now(_dt.timezone.utc).date()
    start_ts = _start_of_day_utc(target)
    end_ts = _end_of_day_utc(target)

    decisions = journal.list_decisions(since_ts=start_ts, limit=10000)
    decisions = [d for d in decisions if d["ts"] < end_ts]
    orders = journal.list_orders(since_ts=start_ts, limit=10000)
    orders = [o for o in orders if o["ts"] < end_ts]
    equity_rows = journal.list_equity(since_ts=start_ts, limit=10000)
    equity_rows = [e for e in equity_rows if e["ts"] < end_ts]
    closes = journal.list_equity  # placeholder, real P/L via positions_history below

    # Realised P/L from positions_history (if any closed trades today).
    realised = 0.0
    win = loss = 0
    try:
        from app.journal import Journal as _J  # for type checker
        with journal._lock:  # noqa: SLF001 - direct access ok in same package
            import sqlite3
            conn = sqlite3.connect(str(journal.path))
            try:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT profit FROM positions_history "
                    "WHERE close_ts >= ? AND close_ts < ?",
                    (start_ts, end_ts),
                ).fetchall()
            finally:
                conn.close()
    except Exception:  # noqa: BLE001
        rows = []
    for r in rows:
        try:
            p = float(r["profit"])
        except (TypeError, ValueError, KeyError):
            continue
        realised += p
        if p > 0:
            win += 1
        elif p < 0:
            loss += 1

    starting = float(equity_rows[0]["balance"]) if equity_rows else 0.0
    ending = float(equity_rows[-1]["balance"]) if equity_rows else starting
    if starting > 0:
        risk_used = abs(realised) / starting * 100.0
    else:
        risk_used = 0.0

    return ReviewStats(
        decision_count=len(decisions),
        approved_count=sum(1 for d in decisions if d["approved"]),
        rejected_count=sum(1 for d in decisions if not d["approved"]),
        orders_filled=sum(1 for o in orders if o["status"] == "filled"),
        orders_rejected=sum(1 for o in orders if o["status"] == "rejected"),
        realized_pl=realised,
        win_count=win,
        loss_count=loss,
        starting_balance=starting,
        ending_balance=ending,
        risk_used_pct=risk_used,
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_money(v: float) -> str:
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"


def _render_decisions_md(decisions: Sequence[Mapping]) -> str:
    if not decisions:
        return "_No decisions recorded for this period._\n"
    lines = [
        "| Time (UTC) | Symbol | Action | Playbook | Conf. | Status | Reasoning |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for d in decisions[:200]:
        ts = _dt.datetime.fromtimestamp(int(d["ts"]), tz=_dt.timezone.utc)
        ts_str = ts.strftime("%Y-%m-%d %H:%M")
        status = "approved" if d["approved"] else "rejected"
        reasoning = (d.get("reasoning") or "").replace("|", "\\|")[:120]
        pb = d.get("playbook") or "—"
        lines.append(
            f"| {ts_str} | {d['symbol']} | {d['action']} | {pb} | "
            f"{float(d.get('confidence') or 0):.2f} | {status} | {reasoning} |"
        )
    return "\n".join(lines) + "\n"


def _render_proposals_md(proposals: Sequence[Mapping]) -> str:
    if not proposals:
        return "_No proposals -- not enough data or nothing flagged._\n"
    lines: list[str] = []
    for p in proposals:
        title = str(p.get("title") or p.get("playbook") or "Proposal")
        body = str(p.get("body") or p.get("rationale") or "")
        lines.append(f"### {title}\n")
        lines.append(body.strip() + "\n")
    return "\n".join(lines)


def render_report(
    stats: ReviewStats,
    decisions: Sequence[Mapping],
    proposals: Sequence[Mapping],
    *,
    day: _dt.date | None = None,
) -> str:
    """Render the full markdown report. Pure: no I/O, no LLM calls."""
    target = day or _dt.datetime.now(_dt.timezone.utc).date()
    date_str = target.isoformat()
    parts: list[str] = [
        f"# Daily Review — {date_str}",
        "",
        f"_Generated at {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}_",
        "",
        "## Stats",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Decisions | {stats.decision_count} |",
        f"| Approved / Rejected | {stats.approved_count} / {stats.rejected_count} |",
        f"| Orders filled | {stats.orders_filled} |",
        f"| Orders rejected | {stats.orders_rejected} |",
        f"| Realised P/L | {_fmt_money(stats.realized_pl)} |",
        f"| Wins / Losses | {stats.win_count} / {stats.loss_count} |",
        f"| Starting balance | {_fmt_money(stats.starting_balance)} |",
        f"| Ending balance | {_fmt_money(stats.ending_balance)} |",
        f"| Risk used (vs. starting) | {stats.risk_used_pct:.2f}% |",
        "",
        "## Decisions",
        "",
        _render_decisions_md(decisions),
        "## Playbook update proposals (advisory)",
        "",
        _render_proposals_md(proposals),
        "",
        "> Proposals are advisory only. They are never written to `knowledge/` automatically. "
        "Review the suggestions, edit the relevant playbook file by hand, then re-run the agent.",
        "",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Deterministic proposal generation (no LLM)
# ---------------------------------------------------------------------------


def generate_proposals(decisions: Sequence[Mapping]) -> list[dict]:
    """Advisory playbook-update proposals from journal statistics.

    Pure and deterministic. Flags structural issues a human should look
    at: setups that never trigger, proposals the risk gate keeps vetoing,
    and dominant rejection reasons. Never edits ``knowledge/``.
    """
    proposals: list[dict] = []
    if not decisions:
        return proposals

    by_pb: dict[str, list] = {}
    for d in decisions:
        by_pb.setdefault(str(d.get("playbook") or "unknown"), []).append(d)

    for pb, rows in sorted(by_pb.items()):
        proposes = [r for r in rows if r.get("action") == "propose"]
        if len(rows) >= 20 and len(proposes) / len(rows) < 0.05:
            proposals.append({
                "playbook": pb,
                "title": f"{pb}: setup rarely triggers",
                "body": (f"Only {len(proposes)}/{len(rows)} decisions proposed a trade. "
                         "If this persists across weeks, consider widening the trigger "
                         "or checking the session filter."),
            })
        vetoed = [r for r in proposes if not r.get("approved")]
        if len(proposes) >= 10 and len(vetoed) / len(proposes) > 0.5:
            proposals.append({
                "playbook": pb,
                "title": f"{pb}: risk gate vetoes most proposals",
                "body": (f"{len(vetoed)}/{len(proposes)} proposals were rejected by risk.py. "
                         "Review the dominant reject reasons below -- the setup may be "
                         "over-trading into the daily-loss or open-trade caps."),
            })

    # Most common rejection reasons across all decisions.
    reason_counts: dict[str, int] = {}
    for d in decisions:
        for r in (d.get("reject_reasons") or []):
            key = str(r).split(":")[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1
    if reason_counts:
        top = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[0]
        if top[1] >= 5:
            proposals.append({
                "playbook": "",
                "title": f"Dominant reject reason: {top[0]} ({top[1]}x)",
                "body": ("This veto fired most often. If it is 'max_trades_per_day_reached' "
                         "or 'max_open_trades_reached', the caps are doing their job. If it is "
                         "'confidence_below_min' or 'spread', the setup may be firing in "
                         "conditions it should skip -- tighten the entry filters."),
            })
    return proposals[:5]


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def generate_report(
    journal: Journal,
    *,
    day: _dt.date | None = None,
    knowledge_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
) -> Path:
    """Generate a markdown review report for ``day`` (default today UTC).

    Returns the path of the written file. The report is also recorded
    in the ``reviews`` table.
    """
    target = day or _dt.datetime.now(_dt.timezone.utc).date()
    start_ts = _start_of_day_utc(target)
    end_ts = _end_of_day_utc(target)

    stats = compute_stats(journal, day=target)
    decisions = [
        d for d in journal.list_decisions(since_ts=start_ts, limit=10000)
        if d["ts"] < end_ts
    ]

    # Deterministic advisory proposals from journal statistics.
    proposals = generate_proposals(decisions)

    md = render_report(stats, decisions, proposals, day=target)

    rdir = Path(reports_dir) if reports_dir else Path("reports")
    rdir.mkdir(parents=True, exist_ok=True)
    out = rdir / f"{target.isoformat()}.md"
    out.write_text(md, encoding="utf-8")
    journal.record_review(path=str(out))
    return out


__all__ = [
    "ReviewStats",
    "compute_stats",
    "generate_proposals",
    "generate_report",
    "render_report",
]