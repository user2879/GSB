"""Event stream model and strict input validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import ErrorCode, RtsError

ACTION_COLLECT = "collect"
ACTION_PRODUCE = "produce"
ACTION_MOVE = "move"
VALID_ACTIONS = frozenset({ACTION_COLLECT, ACTION_PRODUCE, ACTION_MOVE})

_REQUIRED_FIELDS = ("tick", "actor", "action", "target", "amount", "rng_seed_delta")


@dataclass(frozen=True)
class Event:
    """A single deterministic match command.

    ``seq`` is the actor-local monotonic submission ordinal. It is required so
    that sequence gaps (``E_SEQ_GAP``) can be detected even after the stream has
    been grouped and re-sorted by tick.
    """

    tick: int
    actor: str
    action: str
    target: str
    amount: int
    rng_seed_delta: int
    seq: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "actor": self.actor,
            "action": self.action,
            "target": self.target,
            "amount": self.amount,
            "rng_seed_delta": self.rng_seed_delta,
            "seq": self.seq,
        }


def event_from_dict(raw: Any, *, line_no: int | None = None) -> Event:
    """Validate one raw JSON object and convert it to an :class:`Event`."""
    location = f"line {line_no}" if line_no is not None else "input object"
    if not isinstance(raw, dict):
        raise RtsError(
            ErrorCode.E_INPUT_MALFORMED,
            f"event at {location} must be a JSON object",
            context={"line_no": line_no},
        )
    missing = [name for name in _REQUIRED_FIELDS if name not in raw]
    if missing:
        raise RtsError(
            ErrorCode.E_INPUT_MALFORMED,
            f"event at {location} missing fields: {','.join(missing)}",
            context={"line_no": line_no, "missing": missing},
        )
    if "seq" not in raw:
        raise RtsError(
            ErrorCode.E_INPUT_MALFORMED,
            f"event at {location} missing required field 'seq'",
            context={"line_no": line_no},
        )

    values: dict[str, Any] = {}
    for name in ("tick", "amount", "rng_seed_delta", "seq"):
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise RtsError(
                ErrorCode.E_INPUT_MALFORMED,
                f"event at {location} field '{name}' must be an integer",
                context={"line_no": line_no, "field": name, "value": value},
            )
        values[name] = value
    for name in ("actor", "action", "target"):
        value = raw[name]
        if not isinstance(value, str) or not value:
            raise RtsError(
                ErrorCode.E_INPUT_MALFORMED,
                f"event at {location} field '{name}' must be a non-empty string",
                context={"line_no": line_no, "field": name},
            )
        values[name] = value

    if values["tick"] < 0 or values["seq"] < 0 or values["amount"] < 0:
        raise RtsError(
            ErrorCode.E_INPUT_MALFORMED,
            f"event at {location} has negative tick/seq/amount",
            context={"line_no": line_no, "raw": raw},
        )
    if values["action"] not in VALID_ACTIONS:
        raise RtsError(
            ErrorCode.E_INPUT_MALFORMED,
            f"event at {location} has unknown action '{values['action']}'",
            context={"line_no": line_no, "action": values["action"]},
        )
    return Event(**values)


def load_events(path: str) -> list[Event]:
    """Load a JSON-lines event stream, reporting malformed lines with a code."""
    events: list[Event] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = __import__("json").loads(line)
            except ValueError as exc:
                raise RtsError(
                    ErrorCode.E_INPUT_MALFORMED,
                    f"invalid JSON at line {line_no}: {exc}",
                    context={"line_no": line_no},
                ) from exc
            events.append(event_from_dict(raw, line_no=line_no))
    if not events:
        raise RtsError(
            ErrorCode.E_INPUT_MALFORMED, "event stream is empty", context={"path": path}
        )
    return events


def sort_key(event: Event) -> tuple[int, str, int]:
    """Deterministic submission order: tick, then actor, then local seq."""
    return event.tick, event.actor, event.seq
