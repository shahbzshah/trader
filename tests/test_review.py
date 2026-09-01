"""Tests for app/review.py: stats, markdown shape, advisory-only.

Acceptance: generates a report from seeded journal fixture rows.
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.brain import LLMClient
from app.journal import Journal
from app.review import (
    ReviewStats,
    compute_stats,
    generate_proposals,
    generate_report,
    render_report,
)
from tests.fake_http import make_chat_response, recording_transport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_journal(j: Journal, *, day: _dt.date) -> dict:
    """Seed decisions, orders, equity, and a position close for ``day``.

    Returns a dict of the seeded counts so tests can assert on them.
    """
    start = int(_dt.datetime(day.year, day.month, day.day,
                              tzinfo=_dt.timezone.utc).timestamp())
    ts = start + 3600

    # Decisions.
    d1 = j.record_decision(symbol="EURUSD", action="propose",
                            playbook="breakout_retest", confidence=0.8,
                            risk_pct=0.5, reasoning="trend confirmed",
                            approved=True, ts=ts)
    d2 = j.record_decision(symbol="EURUSD", action="no_trade",
                            confidence=0.6, approved=True, ts=ts + 60)
    d3 = j.record_decision(symbol="GBPUSD", action="propose",
                            confidence=0.4, approved=False,
                            reject_reasons=["confidence_below_min"],
                            reasoning="low conf", ts=ts + 120)
    # An old decision outside the window.
    j.record_decision(symbol="EURUSD", action="no_trade", ts=start - 3600)

    # Orders: one filled, one rejected, linked to d1.
    o1 = j.record_order(decision_id=d1, symbol="EURUSD", direction="buy",
                        lots=0.20, status="filled", ticket=1234,
                        entry=1.1, sl=1.095, tp=1.12, ts=ts + 5)
    j.record_order(decision_id=d3, symbol="GBPUSD", direction="buy",
                   lots=0.10, status="rejected", error="low confidence",
                   ts=ts + 125)

    # Equity snapshots.
    j.record_equity(balance=10000, equity=10050, ts=ts - 60)
    j.record_equity(balance=10000, equity=10080, ts=ts + 1800)

    # Position close: one win, one loss.
    j.record_position_close(symbol="EURUSD", direction="buy", volume=0.20,
                            price_open=1.1, price_close=1.12, profit=200.0,
                            ticket=999, close_ts=ts + 1500)
    j.record_position_close(symbol="EURUSD", direction="sell", volume=0.10,
                            price_open=1.12, price_close=1.115, profit=-50.0,
                            ticket=998, close_ts=ts + 1600)

    return {"decisions": 4, "orders": 2, "equity": 2, "closes": 2,
            "d1": d1, "d2": d2, "d3": d3, "o1": o1}


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------


def test_compute_stats_aggregates_journal_rows(tmp_path: Path):
    j = Journal(tmp_path / "j.db")
    day = _dt.datetime(2026, 8, 31, tzinfo=_dt.timezone.utc).date()
    _seed_journal(j, day=day)

    stats = compute_stats(j, day=day)
    assert stats.decision_count == 3       # 3 in the window, 1 outside
    assert stats.approved_count == 2
    assert stats.rejected_count == 1
    assert stats.orders_filled == 1
    assert stats.orders_rejected == 1
    assert stats.win_count == 1
    assert stats.loss_count == 1
    assert stats.realized_pl == pytest.approx(150.0, abs=1e-6)
    assert stats.starting_balance == 10000.0
    assert stats.ending_balance == 10000.0
    assert stats.risk_used_pct == pytest.approx(1.5, abs=1e-6)


def test_compute_stats_empty_journal(tmp_path: Path):
    j = Journal(tmp_path / "j.db")
    day = _dt.datetime(2026, 8, 31, tzinfo=_dt.timezone.utc).date()
    stats = compute_stats(j, day=day)
    assert stats.decision_count == 0
    assert stats.realized_pl == 0.0
    assert stats.starting_balance == 0.0
    assert stats.risk_used_pct == 0.0


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------


def test_render_report_includes_stats_and_decisions(tmp_path: Path):
    j = Journal(tmp_path / "j.db")
    day = _dt.datetime(2026, 8, 31, tzinfo=_dt.timezone.utc).date()
    _seed_journal(j, day=day)
    stats = compute_stats(j, day=day)
    decisions = j.list_decisions(since_ts=int(
        _dt.datetime(day.year, day.month, day.day, tzinfo=_dt.timezone.utc).timestamp()
    ), limit=1000)

    md = render_report(stats, decisions, [], day=day)
    # Header.
    assert "# Daily Review — 2026-08-31" in md
    # Stats table.
    assert "## Stats" in md
    assert "Decisions" in md
    assert "Realised P/L" in md
    # Decisions table.
    assert "## Decisions" in md
    assert "EURUSD" in md
    # Advisory section always present.
    assert "## Playbook update proposals" in md
    assert "advisory only" in md.lower()
    # Without LLM proposals, the empty-state copy is shown.
    assert "No proposals generated" in md


def test_render_report_includes_proposals(tmp_path: Path):
    j = Journal(tmp_path / "j.db")
    day = _dt.datetime(2026, 8, 31, tzinfo=_dt.timezone.utc).date()
    _seed_journal(j, day=day)
    stats = compute_stats(j, day=day)
    decisions = j.list_decisions(limit=1000)
    proposals = [
        {"playbook": "breakout_retest.md",
         "title": "Tighten stop on ranging sessions",
         "body": "Two of three losing trades came from ranging markets."},
    ]
    md = render_report(stats, decisions, proposals, day=day)
    assert "Tighten stop on ranging sessions" in md
    assert "breakout_retest.md" in md or "breakout_retest" in md
    assert "advisory only" in md.lower()


def test_render_report_handles_no_decisions():
    stats = ReviewStats(
        decision_count=0, approved_count=0, rejected_count=0,
        orders_filled=0, orders_rejected=0, realized_pl=0.0,
        win_count=0, loss_count=0, starting_balance=0.0, ending_balance=0.0,
        risk_used_pct=0.0,
    )
    md = render_report(stats, [], [], day=_dt.date(2026, 8, 31))
    assert "_No decisions recorded for this period._" in md


# ---------------------------------------------------------------------------
# generate_proposals (with mocked LLM)
# ---------------------------------------------------------------------------


def test_generate_proposals_parses_llm_response(tmp_path: Path):
    playbooks = [
        {"name": "breakout_retest", "body": "Body A", "path": "a.md"},
    ]
    transport, _ = recording_transport([make_chat_response(json.dumps({
        "proposals": [
            {"playbook": "a.md", "title": "Tighten SL",
             "rationale": "Two losses came from wide stops."},
            {"playbook": "a.md", "title": "Skip news",
             "rationale": "Losses cluster near 16:00 UTC."},
        ]
    }))])
    client = LLMClient(base_url="http://x", model="m", api_key="", timeout=5.0,
                         transport=transport)
    out = generate_proposals(
        decisions=[{"id": 1, "ts": 1, "symbol": "EURUSD", "action": "propose",
                    "playbook": "a.md", "confidence": 0.8, "approved": True,
                    "reject_reasons": [], "reasoning": "x"}],
        playbooks=playbooks,
        llm_client=client,
    )
    assert len(out) == 2
    assert out[0]["title"] == "Tighten SL"
    assert "Two losses" in out[0]["body"]


def test_generate_proposals_empty_when_no_decisions():
    transport, _ = recording_transport([])
    client = LLMClient(base_url="http://x", model="m", api_key="", timeout=5.0,
                         transport=transport)
    out = generate_proposals(
        decisions=[], playbooks=[{"name": "x", "body": "y", "path": "x.md"}],
        llm_client=client,
    )
    assert out == []


def test_generate_proposals_empty_proposals_response():
    transport, _ = recording_transport([make_chat_response(json.dumps({
        "proposals": []
    }))])
    client = LLMClient(base_url="http://x", model="m", api_key="", timeout=5.0,
                         transport=transport)
    out = generate_proposals(
        decisions=[{"id": 1, "ts": 1, "symbol": "EURUSD", "action": "no_trade",
                    "playbook": "a", "confidence": 0.5, "approved": True,
                    "reject_reasons": [], "reasoning": "x"}],
        playbooks=[{"name": "x", "body": "y", "path": "x.md"}],
        llm_client=client,
    )
    assert out == []


def test_generate_proposals_swallows_network_error():
    from tests.fake_http import make_raising_transport
    import httpx
    transport = make_raising_transport(httpx.ConnectError("boom"))
    client = LLMClient(base_url="http://x", model="m", api_key="", timeout=1.0,
                         transport=transport)
    out = generate_proposals(
        decisions=[{"id": 1, "ts": 1, "symbol": "EURUSD", "action": "no_trade",
                    "playbook": "a", "confidence": 0.5, "approved": True,
                    "reject_reasons": [], "reasoning": "x"}],
        playbooks=[{"name": "x", "body": "y", "path": "x.md"}],
        llm_client=client,
    )
    assert out == []


# ---------------------------------------------------------------------------
# generate_report (top-level: writes file + records review)
# ---------------------------------------------------------------------------


def test_generate_report_writes_markdown_and_journals(tmp_path: Path):
    j = Journal(tmp_path / "j.db")
    day = _dt.datetime(2026, 8, 31, tzinfo=_dt.timezone.utc).date()
    _seed_journal(j, day=day)

    out = generate_report(
        j,
        day=day,
        llm_client=None,
        knowledge_dir=Path("knowledge"),
        reports_dir=tmp_path / "reports",
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Daily Review — 2026-08-31")
    # Journal row recorded.
    revs = j.list_reviews(limit=5)
    assert any(r["path"] == str(out) for r in revs)


def test_generate_report_default_day_is_today(tmp_path: Path):
    j = Journal(tmp_path / "j.db")
    # No seed: file should still be written.
    out = generate_report(
        j,
        knowledge_dir=Path("knowledge"),
        reports_dir=tmp_path / "reports",
    )
    assert out.exists()
    today = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    assert today in out.read_text(encoding="utf-8")


def test_generate_report_with_mocked_llm(tmp_path: Path):
    j = Journal(tmp_path / "j.db")
    day = _dt.datetime(2026, 8, 31, tzinfo=_dt.timezone.utc).date()
    _seed_journal(j, day=day)
    transport, _ = recording_transport([make_chat_response(json.dumps({
        "proposals": [{"playbook": "breakout_retest.md",
                        "title": "Refine confirmation",
                        "rationale": "Confirmations fired too late."}]
    }))])
    client = LLMClient(base_url="http://x", model="m", api_key="", timeout=5.0,
                         transport=transport)
    out = generate_report(
        j, day=day, llm_client=client,
        knowledge_dir=Path("knowledge"),
        reports_dir=tmp_path / "reports",
    )
    text = out.read_text(encoding="utf-8")
    assert "Refine confirmation" in text


def test_review_does_not_edit_knowledge(tmp_path: Path):
    """The review must NEVER write to knowledge/."""
    # Make a fresh knowledge dir we can scan for changes.
    kdir = tmp_path / "knowledge"
    pdir = kdir / "playbooks"
    pdir.mkdir(parents=True)
    pdir.joinpath("custom.md").write_text("Original content\n", encoding="utf-8")

    j = Journal(tmp_path / "j.db")
    day = _dt.datetime(2026, 8, 31, tzinfo=_dt.timezone.utc).date()
    _seed_journal(j, day=day)
    generate_report(
        j, day=day, llm_client=None,
        knowledge_dir=kdir, reports_dir=tmp_path / "reports",
    )
    # Knowledge file untouched.
    assert pdir.joinpath("custom.md").read_text(encoding="utf-8") == "Original content\n"


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_main_review_command_writes_file(tmp_path: Path, monkeypatch, capsys):
    from app import main as main_module

    cfg_text = f"""
mode: demo
symbols: [EURUSD]
timeframes: [M15]
schedule: {{scan_interval_minutes: 15}}
session_hours: {{start: '00:00', end: '23:59'}}
risk: {{max_risk_pct_per_trade: 1.0, max_daily_loss_pct: 3.0,
       max_open_trades: 3, max_trades_per_day: 5, min_confidence: 0.6,
       default_playbook: breakout_retest}}
llm: {{base_url: 'http://x', model: 'm'}}
paths: {{data_dir: '{tmp_path.as_posix().replace(chr(92), '/')}',
         logs_dir: '{tmp_path.as_posix().replace(chr(92), '/')}',
         reports_dir: '{(tmp_path / 'reports').as_posix()}',
         knowledge_dir: 'knowledge',
         kill_switch_path: '{(tmp_path / 'KILL_SWITCH').as_posix()}'}}
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    # Seed journal.
    j = Journal(tmp_path / "journal.db")
    _seed_journal(j, day=_dt.datetime.now(_dt.timezone.utc).date())

    rc = main_module.main(["--config", str(cfg_path), "review", "--no-llm"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "wrote" in out
    assert (tmp_path / "reports").exists()
    md_files = list((tmp_path / "reports").glob("*.md"))
    assert len(md_files) == 1