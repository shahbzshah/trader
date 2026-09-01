"""LLM decision engine.

The brain is the only module that talks to the LLM. It composes:
  - a system prompt from the risk rules and playbooks, plus explicit
    rules ("default action is no_trade; never exceed stated risk;
    output ONLY the JSON object")
  - a user message with the per-symbol snapshot, open positions, the
    last ~20 journal entries, and today's realised P/L + remaining risk
    budget
  - one call to the OpenAI-compatible /chat/completions endpoint

The LLM response is parsed into a ``Decision``. If parsing fails, the
brain retries once with a stricter instruction; if it still fails, an
``error`` action decision is recorded.

The HTTP client is injected so tests can swap in a transport that
returns canned JSON.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision schema
# ---------------------------------------------------------------------------


VALID_ACTIONS = ("no_trade", "propose", "close", "modify")
VALID_DIRECTIONS = ("buy", "sell")
VALID_ENTRY_TYPES = ("market", "limit")


class Decision(BaseModel):
    action: str
    symbol: str
    direction: str | None = None
    entry_type: str = "market"
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_pct: float = 0.0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    playbook: str = ""
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Knowledge loader
# ---------------------------------------------------------------------------


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


def load_risk_rules(path: Path | str) -> list[dict]:
    """Parse knowledge/risk_rules.md into ``[{name, rule_line, body}, ...]``."""
    text = Path(path).read_text(encoding="utf-8")
    sections = _split_md_sections(text)
    rules: list[dict] = []
    for heading, body in sections:
        if not heading.lower().startswith("rule"):
            continue
        rule_line = ""
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("RULE:"):
                rule_line = stripped[len("RULE:"):].strip()
                break
        rules.append({"name": heading, "rule": rule_line, "body": body})
    # Also include the top-level RULE: lines that have no heading.
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("RULE:") and not any(r["rule"] == s[len("RULE:"):].strip()
                                              for r in rules):
            rules.append({"name": "global", "rule": s[len("RULE:"):].strip(), "body": ""})
    return rules


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


def _format_rules_for_prompt(rules: Sequence[dict]) -> str:
    lines = ["# Risk Rules (authoritative)"]
    for r in rules:
        lines.append(f"- {r['name']}: {r['rule']}" if r["rule"] else f"- {r['name']}")
    return "\n".join(lines)


def _format_playbooks_for_prompt(playbooks: Sequence[dict]) -> str:
    parts = ["# Playbooks (templates)"]
    for p in playbooks:
        parts.append(f"\n## {p['name']}\n{p['body']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    content: str
    raw: dict


class LLMClient:
    """Thin wrapper over an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._transport = transport

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
        }
        client_kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        with httpx.Client(**client_kwargs) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected LLM response shape: {data!r}") from exc
        return LLMResponse(content=content, raw=data)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict | None:
    """Pull the first JSON object out of a (possibly chatty) LLM response."""
    if not text:
        return None
    # 1. Try fenced ```json ... ```
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 2. Try first { ... } block
    m = _JSON_OBJECT_RE.search(text)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try trimming trailing junk by counting braces.
            pass
    # 3. Whole string parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Prompt composition
# ---------------------------------------------------------------------------


SYSTEM_PROMPT_TEMPLATE = """You are a disciplined, conservative trading analyst operating on a \
MetaTrader 5 DEMO account only.

{rule_block}

{playbook_block}

# Hard rules
1. The default action is `no_trade`. Only propose a trade if the snapshot \
   clearly satisfies the chosen playbook's trigger and confirmation rules.
2. Never exceed the per-trade risk cap or the daily loss cap from the risk \
   rules. If unsure, output `no_trade`.
3. Output ONLY a single JSON object. No prose before or after. No markdown \
   fences. No explanations outside the JSON. The JSON schema:
{{
  "action": "no_trade" | "propose" | "close" | "modify",
  "symbol": "EURUSD",
  "direction": "buy" | "sell" | null,
  "entry_type": "market" | "limit",
  "price": <float or null>,
  "stop_loss": <float or null>,
  "take_profit": <float or null>,
  "risk_pct": <float>,
  "confidence": <float between 0 and 1>,
  "playbook": "<playbook name or empty>",
  "reasoning": "<one short paragraph citing concrete snapshot numbers>"
}}
"""


def _system_prompt(rules: list[dict], playbooks: list[dict]) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        rule_block=_format_rules_for_prompt(rules),
        playbook_block=_format_playbooks_for_prompt(playbooks),
    )


def _user_prompt(
    snapshot: Mapping,
    open_positions: Sequence[Mapping],
    journal_tail: Sequence[Mapping],
    daily_pnl: float,
    remaining_risk_pct: float,
) -> str:
    return json.dumps(
        {
            "snapshot": snapshot,
            "open_positions": list(open_positions),
            "journal_tail": list(journal_tail),
            "today": {
                "realized_pl": round(float(daily_pnl), 5),
                "remaining_risk_pct": round(float(remaining_risk_pct), 5),
            },
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# The decide() entry point
# ---------------------------------------------------------------------------


def decide(
    snapshot: Mapping,
    playbooks: Sequence[dict],
    journal_tail: Sequence[Mapping],
    portfolio_state: Mapping,
    config: Any,
    *,
    client: LLMClient | None = None,
) -> tuple[Decision | None, dict]:
    """Return (Decision_or_None, raw_llm_json).

    If the LLM output cannot be parsed into a valid Decision after one
    retry, returns (None, raw_dict). The caller is responsible for
    journaling the failure.

    Args:
        snapshot: a dict from app.scanner.build_snapshot.
        playbooks: list of playbook dicts (from load_playbooks).
        journal_tail: last ~20 journal rows.
        portfolio_state: mapping with keys "daily_pnl" and
            "remaining_risk_pct".
        config: the loaded AppConfig (for LLM settings and risk limits).
        client: optional pre-constructed LLMClient. If None, one is
            built from config.
    """
    if client is None:
        llm_cfg = getattr(config, "llm", None)
        client = LLMClient(
            base_url=getattr(llm_cfg, "base_url", "http://localhost"),
            model=getattr(llm_cfg, "model", ""),
            api_key=getattr(config, "llm_api_key", "") or "",
            timeout=float(getattr(llm_cfg, "timeout_seconds", 60.0)),
        )

    llm_cfg = getattr(config, "llm", None)
    temperature = float(getattr(llm_cfg, "temperature", 0.2))

    # Pre-load the rules and the system prompt once; the rules are
    # carried by the playbooks loader's parent directory.
    knowledge_dir = Path(getattr(getattr(config, "paths", None), "knowledge_dir", "knowledge"))
    risk_rules_path = knowledge_dir / "risk_rules.md"
    try:
        rules = load_risk_rules(risk_rules_path)
    except FileNotFoundError:
        rules = []

    system_msg = {
        "role": "system",
        "content": _system_prompt(rules, playbooks),
    }
    user_msg = {
        "role": "user",
        "content": _user_prompt(
            snapshot,
            snapshot.get("open_positions", []),
            journal_tail,
            float(portfolio_state.get("daily_pnl", 0.0)),
            float(portfolio_state.get("remaining_risk_pct", 100.0)),
        ),
    }

    # First attempt
    last_raw: dict = {}
    try:
        resp = client.chat([system_msg, user_msg], temperature=temperature)
        parsed = extract_json_object(resp.content)
        last_raw = {"content": resp.content, "raw": resp.raw}
        decision = _coerce_decision(parsed) if parsed is not None else None
    except Exception as exc:  # noqa: BLE001 - network errors
        logger.warning("LLM call failed: %s", exc)
        return None, {"error": str(exc), "phase": "request"}

    if decision is not None:
        return decision, last_raw

    # Retry once with a stricter instruction.
    retry_system = {
        "role": "system",
        "content": (
            system_msg["content"]
            + "\n\nIMPORTANT: Your previous reply could not be parsed. "
            "Reply with ONLY the raw JSON object, no fences, no prose."
        ),
    }
    try:
        resp = client.chat([retry_system, user_msg], temperature=temperature)
        parsed = extract_json_object(resp.content)
        last_raw = {"content": resp.content, "raw": resp.raw, "retried": True}
        decision = _coerce_decision(parsed) if parsed is not None else None
    except Exception as exc:  # noqa: BLE001
        return None, {"error": str(exc), "phase": "retry"}

    return decision, last_raw


def _coerce_decision(parsed: Any) -> Decision | None:
    """Validate parsed JSON against the Decision schema. Returns None on failure."""
    if not isinstance(parsed, dict):
        return None
    # Provide defaults so partial responses still validate.
    defaults = {
        "action": "no_trade",
        "symbol": "",
        "direction": None,
        "entry_type": "market",
        "price": None,
        "stop_loss": None,
        "take_profit": None,
        "risk_pct": 0.0,
        "confidence": 0.0,
        "playbook": "",
        "reasoning": "",
    }
    merged = {**defaults, **parsed}
    # Type coerce risky fields.
    for k in ("price", "stop_loss", "take_profit", "risk_pct", "confidence"):
        v = merged.get(k)
        if v is not None:
            try:
                merged[k] = float(v)
            except (TypeError, ValueError):
                merged[k] = None if k != "risk_pct" else 0.0
                if k in ("risk_pct", "confidence"):
                    merged[k] = 0.0
    # Action whitelist
    if merged["action"] not in VALID_ACTIONS:
        return None
    if merged.get("direction") not in (None,) + VALID_DIRECTIONS:
        return None
    if merged.get("entry_type") not in VALID_ENTRY_TYPES:
        merged["entry_type"] = "market"
    try:
        return Decision(**merged)
    except ValidationError:
        return None


__all__ = [
    "Decision",
    "LLMClient",
    "LLMResponse",
    "VALID_ACTIONS",
    "VALID_DIRECTIONS",
    "VALID_ENTRY_TYPES",
    "decide",
    "extract_json_object",
    "load_playbooks",
    "load_risk_rules",
]