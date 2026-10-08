"""Open-position management: breakeven stop moves.

Pure functions over position snapshots. No I/O. The caller (main.run_only /
run_once) executes the returned ``modify`` Decisions through the normal
executor pipeline, so risk.py still vets them (demo-mode, kill switch).

A position qualifies for a breakeven move when it is up at least
``risk.breakeven_at_r`` R and its stop has not been moved to entry yet.
Moving the stop to entry turns a would-be loser into a scratch trade --
it never locks in profit, it only removes the loss.
"""

from __future__ import annotations

from typing import Any, Iterable

from app.strategy import Decision

_EPS = 1e-9


def breakeven_moves(positions: Iterable[Any], config: Any) -> list[Decision]:
    """Propose SL->entry modifies for qualifying open positions.

    Args:
        positions: iterable of position-like objects with ``ticket``,
            ``symbol``, ``direction`` ("buy"/"sell"), ``price_open``,
            ``price_current``, ``sl`` and ``tp`` attributes.
        config: app config; reads ``config.risk.breakeven_enabled``,
            ``config.risk.breakeven_at_r`` and ``config.symbols``.

    Returns:
        A list of ``action="modify"`` Decisions (ticket carried in
        ``.price``, per the executor's modify convention). Empty when
        breakeven is disabled or nothing qualifies.
    """
    risk_cfg = getattr(config, "risk", None)
    if not bool(getattr(risk_cfg, "breakeven_enabled", True)):
        return []
    try:
        at_r = float(getattr(risk_cfg, "breakeven_at_r", 1.0))
    except (TypeError, ValueError):
        return []
    if at_r <= 0:
        return []

    watchlist = {s.upper() for s in (list(getattr(config, "symbols", []) or []))}
    out: list[Decision] = []
    for p in positions:
        symbol = str(getattr(p, "symbol", "") or "").upper()
        if watchlist and symbol not in watchlist:
            continue
        direction = getattr(p, "direction", None)
        if direction not in ("buy", "sell"):
            continue
        try:
            ticket = int(getattr(p, "ticket", 0) or 0)
            entry = float(getattr(p, "price_open", 0.0) or 0.0)
            cur = float(getattr(p, "price_current", 0.0) or 0.0)
            sl = float(getattr(p, "sl", 0.0) or 0.0)
            tp = float(getattr(p, "tp", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if ticket <= 0 or entry <= 0 or cur <= 0 or sl <= 0:
            continue
        risk_dist = abs(entry - sl)
        if risk_dist <= 0:
            continue
        if direction == "buy":
            r_now = (cur - entry) / risk_dist
            already = sl >= entry - _EPS
        else:
            r_now = (entry - cur) / risk_dist
            already = sl <= entry + _EPS
        if r_now < at_r or already:
            continue
        out.append(
            Decision(
                action="modify",
                symbol=symbol,
                direction=direction,
                price=float(ticket),
                stop_loss=entry,
                take_profit=tp,
                risk_pct=0.0,
                confidence=1.0,
                playbook="breakeven",
                reasoning=(
                    f"breakeven: {symbol} ticket {ticket} at {r_now:.2f}R, "
                    f"SL {sl} -> entry {entry}"
                ),
            )
        )
    return out


__all__ = ["breakeven_moves"]
