"""Durable signal history: snapshot + append-only archive.

Mirrors the Crash 1000 bot's ``history_snapshot.py`` pattern: the
journal (SQLite) lives on Render's ephemeral disk and resets on every
deploy, so the signal ledger's durable record is a pair of repo files:

* ``signal_history.jsonl`` — append-only archive, one JSON line per
  signal-state change (signal first seen, outcome resolved,
  explanation stored). A sidecar commits this file; it is the
  permanent record.
* ``history_snapshot.json`` — the latest state of every signal, latest
  100 records, oldest first, rebuilt from the archive. The server
  loads it at startup so history (and TP/SL tracking of still-open
  signals) survives deploys.

Startup must never fail on a missing or corrupt snapshot: bad files
yield an empty history. No secrets of any kind are stored here — a
signal record is entry/SL/TP, outcome and an optional AI explanation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SNAPSHOT_FILENAME = "history_snapshot.json"
ARCHIVE_FILENAME = "signal_history.jsonl"
HISTORY_CAP = 100

_OUTCOMES = ("OPEN", "TP", "SL")
_DIRECTIONS = ("buy", "sell")


def snapshot_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / SNAPSHOT_FILENAME


def archive_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / ARCHIVE_FILENAME


def valid_record(rec: Any) -> bool:
    """A dict the server can hold in its history as-is."""
    if not isinstance(rec, dict):
        return False
    if not isinstance(rec.get("id"), int):
        return False
    if not isinstance(rec.get("ts"), int):
        return False
    if not isinstance(rec.get("symbol"), str) or not rec.get("symbol"):
        return False
    if rec.get("direction") not in _DIRECTIONS:
        return False
    for key in ("entry", "stop_loss", "take_profit"):
        if not isinstance(rec.get(key), (int, float)):
            return False
    if rec.get("status") not in _OUTCOMES:
        return False
    if rec.get("r") is not None and not isinstance(rec.get("r"), (int, float)):
        return False
    if rec.get("close_ts") is not None and not isinstance(rec.get("close_ts"), int):
        return False
    if rec.get("close_price") is not None and not isinstance(
        rec.get("close_price"), (int, float)
    ):
        return False
    explanation = rec.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        return False
    return True


def load_snapshot(path: Path | str) -> list[dict]:
    """Read a snapshot file. Missing, corrupt, or wrong-shaped files
    yield an empty list — startup must never fail on a bad snapshot."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    recs = [r for r in data if valid_record(r)]
    recs.sort(key=lambda r: (r["ts"], r["id"]))
    return recs[-HISTORY_CAP:]


def append_archive(path: Path | str, record: dict) -> None:
    """Append one record line to the archive. Best-effort: a write
    failure must never break the request that triggered it."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        pass


def write_snapshot(path: Path | str, records: list[dict]) -> None:
    """Atomic write: temp file + replace, so a crash mid-write never
    leaves a half-written snapshot for the next startup."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        recs = sorted(
            [r for r in records if valid_record(r)],
            key=lambda r: (r["ts"], r["id"]),
        )[-HISTORY_CAP:]
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(recs, fh, sort_keys=True)
            fh.write("\n")
        tmp.replace(path)
    except OSError:
        pass


def build_snapshot(archive: Path | str, cap: int = HISTORY_CAP) -> list[dict]:
    """Collapse an append-only archive to the latest state of every
    record: the last line per signal id wins, so a record first
    archived open and later closed comes back closed."""
    latest: dict[int, dict] = {}
    try:
        with open(archive, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not valid_record(rec):
                    continue
                latest[int(rec["id"])] = rec
    except OSError:
        return []
    recs = sorted(latest.values(), key=lambda r: (r["ts"], r["id"]))
    return recs[-cap:]


__all__ = [
    "ARCHIVE_FILENAME",
    "HISTORY_CAP",
    "SNAPSHOT_FILENAME",
    "append_archive",
    "archive_path",
    "build_snapshot",
    "load_snapshot",
    "snapshot_path",
    "valid_record",
    "write_snapshot",
]
