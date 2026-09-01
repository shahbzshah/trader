"""Daily/weekly review report generator.

Generates ``reports/YYYY-MM-DD.md`` (and ``reports/YYYY-MM-DD-week.md``)
with:
  - a stats block (trades, win/loss, realised P/L, risk used)
  - every decision with reasoning
  - a short LLM-generated "Playbook update proposals" section that
    compares recent outcomes to the existing playbook text.

The LLM proposals are **advisory only**; this module never edits
``knowledge/``.

The function is intentionally synchronous and pure with respect to
its inputs (a Journal and an LLMClient). It can be invoked manually
via ``app review`` or wired into the scheduler.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.brain import LLMClient, extract_json_object, load_playbooks
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
        return "_No proposals generated (insufficient data or LLM declined)._\n"
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
# LLM-driven proposal generation
# ---------------------------------------------------------------------------


_PROPOSAL_SYSTEM = """You are a trading coach reviewing yesterday's agent decisions
and outcomes. Compare them to the playbooks in the system prompt and
identify concrete, minimal changes the human trader should consider.

Rules:
- Output ONLY a JSON object. No prose, no markdown fences.
- The JSON schema:
  {{"proposals": [
    {{"playbook": "<filename>", "title": "<short>",
      "rationale": "<one paragraph citing concrete decision IDs/numbers>"}}
  ]}}
- If everything looks fine, return {"proposals": []}.
- Do NOT propose changes you cannot justify with the decisions provided.
- Do NOT exceed 5 proposals.
"""


def _summarise_decisions_for_llm(decisions: Sequence[Mapping]) -> str:
    rows: list[dict] = []
    for d in decisions[:200]:
        rows.append({
            "id": d["id"],
            "ts": d["ts"],
            "symbol": d["symbol"],
            "action": d["action"],
            "playbook": d.get("playbook"),
            "confidence": d.get("confidence"),
            "approved": d["approved"],
            "reject_reasons": d.get("reject_reasons", []),
            "reasoning": (d.get("reasoning") or "")[:200],
        })
    return json.dumps({"decisions": rows, "stats_count": len(decisions)}, indent=2)


def generate_proposals(
    decisions: Sequence[Mapping],
    playbooks: Sequence[Mapping],
    llm_client: LLMClient,
    *,
    temperature: float = 0.2,
) -> list[dict]:
    """Ask the LLM for advisory playbook-update proposals.

    Returns an empty list on network / parse failure (advisory, never
    raises into the report writer).
    """
    if not decisions or not playbooks:
        return []
    playbook_text = "\n\n".join(
        f"## {p['name']}\n{p['body']}" for p in playbooks
    )
    system_msg = {
        "role": "system",
        "content": _PROPOSAL_SYSTEM + "\n\n# Playbooks (reference)\n" + playbook_text,
    }
    user_msg = {
        "role": "user",
        "content": "Decisions from the most recent review window:\n\n"
                    + _summarise_decisions_for_llm(decisions),
    }
    try:
        resp = llm_client.chat([system_msg, user_msg], temperature=temperature)
    except Exception as exc:  # noqa: BLE001
        logger.warning("review LLM call failed: %s", exc)
        return []
    parsed = extract_json_object(resp.content) or {}
    proposals = parsed.get("proposals", [])
    if not isinstance(proposals, list):
        return []
    out: list[dict] = []
    for p in proposals[:5]:
        if not isinstance(p, dict):
            continue
        out.append({
            "playbook": str(p.get("playbook", "")),
            "title": str(p.get("title", "Proposal")),
            "body": str(p.get("rationale", p.get("body", ""))),
        })
    return out


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def generate_report(
    journal: Journal,
    *,
    day: _dt.date | None = None,
    llm_client: LLMClient | None = None,
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

    # Optional LLM-driven proposals.
    proposals: list[dict] = []
    if llm_client is not None:
        kdir = Path(knowledge_dir) if knowledge_dir else Path("knowledge")
        playbooks = load_playbooks(kdir)
        proposals = generate_proposals(decisions, playbooks, llm_client)

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