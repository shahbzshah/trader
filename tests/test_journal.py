"""Tests for app/journal.py: schema + roundtrip."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.journal import Journal, SCHEMA_VERSION


@pytest.fixture
def j(tmp_path: Path) -> Journal:
    return Journal(tmp_path / "journal.db")


def test_journal_creates_file_and_schema(j: Journal, tmp_path: Path):
    assert (tmp_path / "journal.db").exists()
    # Open raw connection to assert schema_meta + tables.
    conn = sqlite3.connect(str(tmp_path / "journal.db"))
    try:
        ver = conn.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone()
        assert ver[0] == str(SCHEMA_VERSION)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        for t in ("decisions", "orders", "equity", "reviews", "positions_history",
                  "schema_meta"):
            assert t in tables
    finally:
        conn.close()


def test_journal_idempotent_init(tmp_path: Path):
    Journal(tmp_path / "x.db")
    # Second init must not raise.
    Journal(tmp_path / "x.db")


def test_record_decision_roundtrip(j: Journal):
    did = j.record_decision(
        symbol="EURUSD",
        action="propose",
        playbook="breakout_retest",
        confidence=0.8,
        risk_pct=0.5,
        reasoning="test",
        approved=True,
        reject_reasons=[],
        raw_llm_json='{"a":1}',
    )
    assert did > 0
    rows = j.list_decisions(limit=10)
    assert len(rows) == 1
    r = rows[0]
    assert r["symbol"] == "EURUSD"
    assert r["action"] == "propose"
    assert r["approved"] is True
    assert r["reject_reasons"] == []
    assert r["raw_llm_json"] == '{"a":1}'


def test_record_decision_rejected(j: Journal):
    j.record_decision(
        symbol="EURUSD", action="propose", approved=False,
        reject_reasons=["kill_switch_engaged", "max_open_trades_reached"],
    )
    r = j.list_decisions(limit=1)[0]
    assert r["approved"] is False
    assert "kill_switch_engaged" in r["reject_reasons"]
    assert "max_open_trades_reached" in r["reject_reasons"]


def test_list_decisions_filtered(j: Journal):
    j.record_decision(symbol="EURUSD", action="no_trade")
    j.record_decision(symbol="GBPUSD", action="no_trade")
    j.record_decision(symbol="EURUSD", action="propose", approved=True)
    eur = j.list_decisions(symbol="EURUSD", limit=10)
    assert all(d["symbol"] == "EURUSD" for d in eur)
    assert len(eur) == 2


def test_record_order_links_to_decision(j: Journal):
    did = j.record_decision(symbol="EURUSD", action="propose", approved=True)
    oid = j.record_order(
        decision_id=did, symbol="EURUSD", direction="buy", lots=0.10,
        status="submitted", entry=1.1, sl=1.09, tp=1.12,
    )
    assert oid > 0
    o = j.list_orders(limit=1)[0]
    assert o["decision_id"] == did
    assert o["status"] == "submitted"
    # Update to filled.
    j.update_order_status(oid, status="filled", ticket=12345)
    o2 = j.list_orders(limit=1)[0]
    assert o2["status"] == "filled"
    assert o2["ticket"] == 12345


def test_record_equity_and_listing(j: Journal):
    j.record_equity(balance=10000, equity=10050, floating_pl=50)
    j.record_equity(balance=10000, equity=9990, floating_pl=-10)
    rows = j.list_equity()
    assert len(rows) == 2
    assert rows[0]["equity"] == 10050
    assert rows[1]["equity"] == 9990


def test_record_review(j: Journal):
    rid = j.record_review(path="reports/2026-08-31.md")
    assert rid > 0
    revs = j.list_reviews()
    assert revs[0]["path"] == "reports/2026-08-31.md"


def test_record_position_close(j: Journal):
    pid = j.record_position_close(
        symbol="EURUSD", direction="buy", volume=0.1,
        price_open=1.1, price_close=1.11, profit=100.0, ticket=42,
    )
    assert pid > 0


def test_today_utc_window(j: Journal):
    start, end = j.today_utc_window()
    assert end - start == 86400


def test_is_kill_switch_engaged(j: Journal, tmp_path: Path):
    flag = tmp_path / "KILL_SWITCH"
    assert j.is_kill_switch_engaged(flag) is False
    flag.touch()
    assert j.is_kill_switch_engaged(flag) is True


def test_journal_thread_safe_concurrent_writes(tmp_path: Path):
    """Hammer the journal with concurrent writes to confirm the lock works."""
    import threading

    j = Journal(tmp_path / "concurrent.db")
    errors: list[Exception] = []

    def writer(n: int) -> None:
        try:
            for i in range(n):
                j.record_decision(symbol="EURUSD", action="no_trade",
                                   confidence=0.5, reasoning=f"writer-{n}-{i}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(20,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(j.list_decisions(limit=1000)) == 80


def test_wal_mode_active(j: Journal, tmp_path: Path):
    """The journal must enable WAL mode (not delete-mode)."""
    conn = sqlite3.connect(str(tmp_path / "journal.db"))
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
        assert mode == "wal"
    finally:
        conn.close()