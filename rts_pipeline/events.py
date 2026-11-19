"""Event parsing, validation, canonical ordering, and digesting."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .canonical import canonical_line, hash_value
from .constants import ACTIONS, SCHEMA_VERSION
from .errors import PipelineError

EVENT_FIELDS = (
    "tick",
    "actor",
    "action",
    "target",
    "amount",
    "rng_seed_delta",
    "local_seq",
)


def event_key(event: dict[str, Any]) -> tuple[int, str, int]:
    return (event["tick"], event["actor"], event["local_seq"])


def validate_event(raw: Any, rules: dict[str, Any], line_number: int | None = None) -> dict[str, Any]:
    where = {"line": line_number} if line_number is not None else {}
    if not isinstance(raw, dict):
        raise PipelineError("E_MALFORMED_EVENT", "event must be a JSON object", **where)
    missing = [field for field in EVENT_FIELDS if field not in raw]
    if missing:
        raise PipelineError("E_MALFORMED_EVENT", "event is missing required fields", missing=missing, **where)
    tick = raw["tick"]
    actor = raw["actor"]
    action = raw["action"]
    target = raw["target"]
    amount = raw["amount"]
    delta = raw["rng_seed_delta"]
    local_seq = raw["local_seq"]
    if not isinstance(tick, int) or isinstance(tick, bool) or tick < 0:
        raise PipelineError("E_MALFORMED_EVENT", "tick must be a non-negative integer", **where)
    if not isinstance(actor, str) or not actor:
        raise PipelineError("E_MALFORMED_EVENT", "actor must be a non-empty string", **where)
    if actor not in rules["factions"]:
        raise PipelineError("E_MALFORMED_EVENT", "actor has no faction rule", actor=actor, **where)
    if action not in ACTIONS:
        raise PipelineError("E_MALFORMED_EVENT", "unknown action", action=action, **where)
    if not isinstance(target, str) or not target:
        raise PipelineError("E_MALFORMED_EVENT", "target must be a non-empty string", **where)
    if action in ("produce", "dispatch") and target not in rules["factions"][actor]["unit_costs"]:
        raise PipelineError("E_MALFORMED_EVENT", "target unit is not defined by faction rules", actor=actor, target=target, **where)
    if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
        raise PipelineError("E_MALFORMED_EVENT", "amount must be a positive integer", **where)
    if not isinstance(delta, int) or isinstance(delta, bool) or delta < 0:
        raise PipelineError("E_MALFORMED_EVENT", "rng_seed_delta must be a non-negative integer", **where)
    if not isinstance(local_seq, int) or isinstance(local_seq, bool) or local_seq < 0:
        raise PipelineError("E_MALFORMED_EVENT", "local_seq must be a non-negative integer", **where)
    return {
        "tick": tick,
        "actor": actor,
        "action": action,
        "target": target,
        "amount": amount,
        "rng_seed_delta": delta,
        "local_seq": local_seq,
    }


def normalize_events(events: Iterable[Any], rules: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = [validate_event(event, rules) for event in events]
    normalized.sort(key=event_key)
    seen: set[tuple[int, str, int]] = set()
    for event in normalized:
        key = event_key(event)
        if key in seen:
            raise PipelineError(
                "E_MALFORMED_EVENT",
                "duplicate event ordering key",
                tick=event["tick"],
                actor=event["actor"],
                local_seq=event["local_seq"],
            )
        seen.add(key)
    return normalized


def events_digest(events: Iterable[dict[str, Any]]) -> str:
    return hash_value(list(events))


def read_events(path: Path, rules: dict[str, Any]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise PipelineError("E_MALFORMED_EVENT", "blank event line", line=line_number)
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PipelineError(
                        "E_MALFORMED_EVENT",
                        "event line is not valid JSON",
                        line=line_number,
                        error=str(exc),
                    ) from exc
                parsed.append(validate_event(raw, rules, line_number))
    except OSError as exc:
        raise PipelineError("E_IO", "event stream is unreadable", path=str(path), error=str(exc)) from exc
    return normalize_events(parsed, rules)


def write_events(path: Path, events: list[dict[str, Any]]) -> None:
    from .canonical import atomic_write_bytes

    payload = b"".join(canonical_line(event) for event in events)
    atomic_write_bytes(Path(path), payload)


def game_document(seed: int, tick_start: int, tick_end: int, chunk_ticks: int, scenario: str, events: list[dict[str, Any]], rules: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {faction: 0 for faction in sorted(rules["factions"])}
    for event in events:
        counts[event["actor"]] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "game_id": f"game-{seed}",
        "scenario": scenario,
        "seed": seed,
        "tick_start": tick_start,
        "tick_end": tick_end,
        "tick_span": tick_end - tick_start + 1,
        "chunk_ticks": chunk_ticks,
        "event_count": len(events),
        "faction_event_counts": counts,
        "events_hash": events_digest(events),
        "rules_hash": hash_value(rules),
    }

