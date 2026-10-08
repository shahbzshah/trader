"""Tests for app/brain.py (rules-based decision interface)."""

from __future__ import annotations

import pytest

from app.brain import (
    VALID_ACTIONS,
    VALID_DIRECTIONS,
    VALID_ENTRY_TYPES,
    Decision,
    decide,
    load_playbooks,
)
from app.strategy import Decision as StrategyDecision


def test_decision_schema_reexported_from_strategy():
    assert Decision is StrategyDecision


def test_decision_constants_present():
    assert set(VALID_ACTIONS) == {"no_trade", "propose", "close", "modify"}
    assert set(VALID_DIRECTIONS) == {"buy", "sell"}
    assert set(VALID_ENTRY_TYPES) == {"market", "limit"}


def test_decision_defaults():
    d = Decision(action="no_trade", symbol="EURUSD")
    assert d.direction is None
    assert d.entry_type == "market"
    assert d.price is None and d.stop_loss is None and d.take_profit is None
    assert d.risk_pct == 0.0
    assert d.confidence == 0.0
    assert d.playbook == "" and d.reasoning == ""


def test_decision_confidence_clamped():
    with pytest.raises(Exception):
        Decision(action="no_trade", symbol="EURUSD", confidence=1.5)
    with pytest.raises(Exception):
        Decision(action="no_trade", symbol="EURUSD", confidence=-0.1)
    d = Decision(action="no_trade", symbol="EURUSD", confidence=0.75)
    assert d.confidence == 0.75


def test_decide_delegates_to_strategy_and_needs_no_network():
    # Empty snapshot -> deterministic no_trade, no client, no I/O.
    d = decide({"symbol": "EURUSD", "timeframes": {}})
    assert isinstance(d, Decision)
    assert d.action == "no_trade"
    assert d.symbol == "EURUSD"


def test_load_playbooks_parses_repo_file(tmp_path):
    pb_dir = tmp_path / "playbooks"
    pb_dir.mkdir()
    (pb_dir / "sweep_fvg_reclaim.md").write_text(
        "## Name\nSweep + FVG Reclaim\n\n## Trigger\nsweep\n",
        encoding="utf-8",
    )
    books = load_playbooks(tmp_path)
    assert len(books) == 1
    assert books[0]["name"] == "Sweep + FVG Reclaim"


def test_load_playbooks_missing_dir_returns_empty(tmp_path):
    assert load_playbooks(tmp_path / "nope") == []
