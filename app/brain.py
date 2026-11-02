"""Decision interface (rules-based).

This module previously hosted the LLM client. The decision engine is now a
deterministic rules engine in :mod:`app.strategy` -- no LLM, no network, no
prompt composition, no retries.

This module keeps its name and re-exports the ``Decision`` schema so that
``app.signals``, ``app.journal`` and ``app.main`` imports keep working, and
delegates decisions to the strategy. :func:`load_playbooks` is retained as a
pure markdown loader (used by the review report to display playbook text).
"""

from __future__ import annotations

from pathlib import Path

from app.strategy import (
    VALID_ACTIONS,
    VALID_DIRECTIONS,
    VALID_ENTRY_TYPES,
    Decision,
    decide,
)


def _split_md_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs by '## ' headings."""
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line[3:].strip()
            current_body = []
        else:
            current_body.append(line)
    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_body).strip()))
    return sections


def load_playbooks(knowledge_dir: Path | str) -> list[dict]:
    """Load every .md under knowledge/playbooks/ into ``[{name, body, raw}]``."""
    base = Path(knowledge_dir) / "playbooks"
    playbooks: list[dict] = []
    if not base.exists():
        return playbooks
    for path in sorted(base.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        name = path.stem
        # If the file has a "## Name" heading, prefer that.
        for heading, body in _split_md_sections(text):
            if heading.lower() == "name":
                first_line = body.splitlines()[0].strip() if body else ""
                if first_line:
                    name = first_line
                break
        playbooks.append({"name": name, "path": str(path), "body": text, "raw": text})
    return playbooks


__all__ = [
    "VALID_ACTIONS",
    "VALID_DIRECTIONS",
    "VALID_ENTRY_TYPES",
    "Decision",
    "decide",
    "load_playbooks",
]
