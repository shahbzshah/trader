"""SQLite journal for the trading agent.

Tables (per KILO_MASTER_PROMPT.md):
    decisions(id, ts, symbol, action, playbook, confidence, risk_pct,
              reasoning, approved, reject_reasons_json, raw_llm_json)
    orders(id, ts, decision_id, ticket, symbol, direction, lots, entry,
           sl, tp, status, error)
    equity(id, ts, balance, equity, floating_pl)
    reviews(id, ts, path)
    positions_history (id, ts, ticket, symbol, direction, volume,
                       price_open, price_close, sl, tp, profit, close_ts)

Every LLM decision is journaled, approved or not. Equity snapshots
are recorded on every scan cycle. The connection uses WAL mode and is
thread-safe via a per-instance lock.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    playbook TEXT,
    confidence REAL,
    risk_pct REAL,
    reasoning TEXT,
    approved INTEGER NOT NULL,
    reject_reasons_json TEXT,
    raw_llm_json TEXT
);

CREATE INDEX IF NOT EXISTS decisions_symbol_ts ON decisions(symbol, ts);
CREATE INDEX IF NOT EXISTS decisions_ts ON decisions(ts);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    decision_id INTEGER,
    ticket INTEGER,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    lots REAL NOT NULL,
    entry REAL,
    sl REAL,
    tp REAL,
    status TEXT NOT NULL,
    error TEXT,
    FOREIGN KEY (decision_id) REFERENCES decisions(id)
);

CREATE INDEX IF NOT EXISTS orders_decision_id ON orders(decision_id);
CREATE INDEX IF NOT EXISTS orders_symbol_ts ON orders(symbol, ts);

CREATE TABLE IF NOT EXISTS equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    balance REAL NOT NULL,
    equity REAL NOT NULL,
    floating_pl REAL
);

CREATE INDEX IF NOT EXISTS equity_ts ON equity(ts);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    open_ts INTEGER,
    close_ts INTEGER,
    ticket INTEGER,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    volume REAL,
    price_open REAL,
    price_close REAL,
    sl REAL,
    tp REAL,
    profit REAL
);

CREATE INDEX IF NOT EXISTS positions_history_symbol ON positions_history(symbol);
"""


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


@dataclass
class DecisionRow:
    id: int | None
    ts: int
    symbol: str
    action: str
    playbook: str
    confidence: float
    risk_pct: float
    reasoning: str
    approved: bool
    reject_reasons: list[str]
    raw_llm_json: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ts": self.ts,
            "symbol": self.symbol,
            "action": self.action,
            "playbook": self.playbook,
            "confidence": self.confidence,
            "risk_pct": self.risk_pct,
            "reasoning": self.reasoning,
            "approved": self.approved,
            "reject_reasons": list(self.reject_reasons),
            "raw_llm_json": self.raw_llm_json,
        }


@dataclass
class OrderRow:
    id: int | None
    ts: int
    decision_id: int | None
    ticket: int | None
    symbol: str
    direction: str
    lots: float
    entry: float | None
    sl: float | None
    tp: float | None
    status: str
    error: str | None


class Journal:
    """Thread-safe SQLite journal."""

    def __init__(self, path: Path | str, *, auto_init: bool = True) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        if auto_init:
            self._init_schema()

    # --- low-level connection ------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.path),
            isolation_level=None,  # autocommit; we manage transactions
            check_same_thread=False,
            timeout=10.0,
        )
        conn.row_factory = sqlite3.Row
        # WAL + foreign keys
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA_SQL)
                conn.execute(
                    "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)",
                    ("version", str(SCHEMA_VERSION)),
                )
            finally:
                conn.close()

    def close(self) -> None:
        # The connection is opened per-operation; nothing to keep alive.
        pass

    # --- decisions ------------------------------------------------------

    def record_decision(
        self,
        *,
        symbol: str,
        action: str,
        playbook: str = "",
        confidence: float = 0.0,
        risk_pct: float = 0.0,
        reasoning: str = "",
        approved: bool = False,
        reject_reasons: Sequence[str] | None = None,
        raw_llm_json: str = "",
        ts: int | None = None,
    ) -> int:
        ts = int(ts) if ts is not None else _now_ts()
        reasons_json = json.dumps(list(reject_reasons or []))
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO decisions (
                        ts, symbol, action, playbook, confidence, risk_pct,
                        reasoning, approved, reject_reasons_json, raw_llm_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        symbol,
                        action,
                        playbook,
                        float(confidence),
                        float(risk_pct),
                        reasoning,
                        1 if approved else 0,
                        reasons_json,
                        raw_llm_json,
                    ),
                )
                return int(cur.lastrowid)
            finally:
                conn.close()

    def list_decisions(
        self,
        *,
        limit: int = 50,
        symbol: str | None = None,
        since_ts: int | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM decisions"
        args: list[Any] = []
        clauses: list[str] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            args.append(symbol)
        if since_ts is not None:
            clauses.append("ts >= ?")
            args.append(int(since_ts))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, args).fetchall()
            finally:
                conn.close()
        return [_decision_to_dict(r) for r in rows]

    def get_decision(self, decision_id: int) -> dict | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM decisions WHERE id = ?", (int(decision_id),)
                ).fetchone()
            finally:
                conn.close()
        return _decision_to_dict(row) if row else None

    # --- orders ---------------------------------------------------------

    def record_order(
        self,
        *,
        decision_id: int | None,
        symbol: str,
        direction: str,
        lots: float,
        status: str,
        ticket: int | None = None,
        entry: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        error: str | None = None,
        ts: int | None = None,
    ) -> int:
        ts = int(ts) if ts is not None else _now_ts()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO orders (
                        ts, decision_id, ticket, symbol, direction, lots,
                        entry, sl, tp, status, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        decision_id,
                        ticket,
                        symbol,
                        direction,
                        float(lots),
                        entry,
                        sl,
                        tp,
                        status,
                        error,
                    ),
                )
                return int(cur.lastrowid)
            finally:
                conn.close()

    def list_orders(
        self,
        *,
        limit: int = 50,
        symbol: str | None = None,
        since_ts: int | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM orders"
        args: list[Any] = []
        clauses: list[str] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            args.append(symbol)
        if since_ts is not None:
            clauses.append("ts >= ?")
            args.append(int(since_ts))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, args).fetchall()
            finally:
                conn.close()
        return [_order_to_dict(r) for r in rows]

    def update_order_status(
        self,
        order_id: int,
        *,
        status: str,
        ticket: int | None = None,
        error: str | None = None,
    ) -> None:
        sets = ["status = ?"]
        args: list[Any] = [status]
        if ticket is not None:
            sets.append("ticket = ?")
            args.append(int(ticket))
        if error is not None:
            sets.append("error = ?")
            args.append(error)
        args.append(int(order_id))
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    f"UPDATE orders SET {', '.join(sets)} WHERE id = ?",
                    args,
                )
            finally:
                conn.close()

    # --- equity ---------------------------------------------------------

    def record_equity(
        self,
        *,
        balance: float,
        equity: float,
        floating_pl: float | None = None,
        ts: int | None = None,
    ) -> int:
        ts = int(ts) if ts is not None else _now_ts()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO equity (ts, balance, equity, floating_pl)
                    VALUES (?, ?, ?, ?)
                    """,
                    (ts, float(balance), float(equity),
                     None if floating_pl is None else float(floating_pl)),
                )
                return int(cur.lastrowid)
            finally:
                conn.close()

    def list_equity(
        self,
        *,
        since_ts: int | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        sql = "SELECT * FROM equity"
        args: list[Any] = []
        if since_ts is not None:
            sql += " WHERE ts >= ?"
            args.append(int(since_ts))
        sql += " ORDER BY ts ASC, id ASC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, args).fetchall()
            finally:
                conn.close()
        return [
            {
                "id": int(r["id"]),
                "ts": int(r["ts"]),
                "balance": float(r["balance"]),
                "equity": float(r["equity"]),
                "floating_pl": (None if r["floating_pl"] is None
                                 else float(r["floating_pl"])),
            }
            for r in rows
        ]

    # --- reviews --------------------------------------------------------

    def record_review(self, *, path: str, ts: int | None = None) -> int:
        ts = int(ts) if ts is not None else _now_ts()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT INTO reviews (ts, path) VALUES (?, ?)",
                    (ts, path),
                )
                return int(cur.lastrowid)
            finally:
                conn.close()

    def list_reviews(self, *, limit: int = 20) -> list[dict]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM reviews ORDER BY ts DESC, id DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
            finally:
                conn.close()
        return [
            {"id": int(r["id"]), "ts": int(r["ts"]), "path": str(r["path"])}
            for r in rows
        ]

    # --- positions history ---------------------------------------------

    def record_position_close(
        self,
        *,
        symbol: str,
        direction: str,
        volume: float,
        price_open: float,
        price_close: float,
        profit: float,
        ticket: int | None = None,
        sl: float | None = None,
        tp: float | None = None,
        open_ts: int | None = None,
        close_ts: int | None = None,
    ) -> int:
        close_ts = int(close_ts) if close_ts is not None else _now_ts()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO positions_history (
                        open_ts, close_ts, ticket, symbol, direction, volume,
                        price_open, price_close, sl, tp, profit
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (open_ts, close_ts, ticket, symbol, direction, volume,
                     price_open, price_close, sl, tp, profit),
                )
                return int(cur.lastrowid)
            finally:
                conn.close()

    # --- introspection --------------------------------------------------

    def is_kill_switch_engaged(self, kill_switch_path: Path | str) -> bool:
        return Path(kill_switch_path).exists()

    def today_utc_window(self) -> tuple[int, int]:
        """Return (start_ts, end_ts) for the current UTC calendar day."""
        now = datetime.now(timezone.utc)
        start = int(datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp())
        return start, start + 86400


def _decision_to_dict(row: sqlite3.Row) -> dict:
    reasons_raw = row["reject_reasons_json"] or "[]"
    try:
        reasons = json.loads(reasons_raw)
    except (TypeError, ValueError):
        reasons = []
    return {
        "id": int(row["id"]),
        "ts": int(row["ts"]),
        "symbol": str(row["symbol"]),
        "action": str(row["action"]),
        "playbook": (row["playbook"] or ""),
        "confidence": (None if row["confidence"] is None else float(row["confidence"])),
        "risk_pct": (None if row["risk_pct"] is None else float(row["risk_pct"])),
        "reasoning": (row["reasoning"] or ""),
        "approved": bool(row["approved"]),
        "reject_reasons": reasons,
        "raw_llm_json": (row["raw_llm_json"] or ""),
    }


def _order_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": int(row["id"]),
        "ts": int(row["ts"]),
        "decision_id": (None if row["decision_id"] is None else int(row["decision_id"])),
        "ticket": (None if row["ticket"] is None else int(row["ticket"])),
        "symbol": str(row["symbol"]),
        "direction": str(row["direction"]),
        "lots": float(row["lots"]),
        "entry": (None if row["entry"] is None else float(row["entry"])),
        "sl": (None if row["sl"] is None else float(row["sl"])),
        "tp": (None if row["tp"] is None else float(row["tp"])),
        "status": str(row["status"]),
        "error": row["error"],
    }


__all__ = [
    "DecisionRow",
    "Journal",
    "OrderRow",
    "SCHEMA_VERSION",
]