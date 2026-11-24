from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import content_hash, read_json
from .errors import PlanError


@dataclass(frozen=True)
class Arrival:
    arrival_seq: int
    order_index: int
    event_id: str
    delay_ms: int
    declared_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "arrival_seq": self.arrival_seq,
            "event_id": self.event_id,
            "delay_ms": self.delay_ms,
        }


@dataclass(frozen=True)
class InjectionPlan:
    arrivals: tuple[Arrival, ...]
    plan_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "arrivals": [arrival.as_dict() for arrival in self.arrivals],
            "plan_hash": self.plan_hash,
        }


def _require_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PlanError(f"{name} must be an integer >= {minimum}")
    return value


def _seq(entry: Mapping[str, Any]) -> Any:
    return entry.get("arrival_seq", entry.get("arrival_sequence", entry.get("sequence")))


def _event_id(entry: Mapping[str, Any]) -> Any:
    return entry.get("event_id", entry.get("id"))


def _delay(entry: Mapping[str, Any]) -> Any:
    return entry.get("delay_ms", entry.get("arrival_delay_ms", entry.get("delay", 0)))


def _normalise_entries(raw: Any) -> tuple[list[dict[str, Any]], int]:
    variable_sized = 0
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        entries = []
        for event_id, value in raw.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("event_id", event_id)
            else:
                item = {"event_id": event_id, "arrival_seq": value}
            entries.append(item)
    else:
        raise PlanError("injection plan must be an array or object")
    normalised: list[dict[str, Any]] = []
    for declared, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PlanError(f"arrival entry {declared} must be an object")
        if set(entry) <= {"event_id", "arrival_seq", "delay_ms"}:
            variable_sized += 0
        normalised.append(dict(entry))
    return normalised, variable_sized


def parse_plan(raw: Any) -> InjectionPlan:
    if isinstance(raw, dict):
        if "arrivals" in raw:
            raw_entries = raw["arrivals"]
        elif "events" in raw:
            raw_entries = raw["events"]
        elif "entries" in raw:
            raw_entries = raw["entries"]
        elif "plan" in raw:
            raw_entries = raw["plan"]
        else:
            # A direct event_id -> sequence mapping is also accepted.
            raw_entries = raw
    else:
        raw_entries = raw
    entries, _ = _normalise_entries(raw_entries)
    if not entries:
        raise PlanError("injection plan must contain at least one arrival")

    arrivals: list[dict[str, Any]] = []
    seen_seq: set[int] = set()
    seen_ids: set[str] = set()
    for declared, entry in enumerate(entries):
        seq = _require_int(_seq(entry), f"arrivals[{declared}].arrival_seq")
        event_id = _event_id(entry)
        if not isinstance(event_id, str) or not event_id:
            raise PlanError(f"arrivals[{declared}].event_id must be a non-empty string")
        delay = _require_int(_delay(entry), f"arrivals[{declared}].delay_ms")
        if seq in seen_seq:
            raise PlanError(f"duplicate arrival_seq {seq}")
        if event_id in seen_ids:
            # Keep duplicate event IDs representable so the replay layer can
            # emit E_EVENT_INVALID exactly at the duplicated delivery.
            pass
        seen_seq.add(seq)
        seen_ids.add(event_id)
        arrivals.append(
            {
                "arrival_seq": seq,
                "event_id": event_id,
                "delay_ms": delay,
                "declared_index": declared,
            }
        )

    arrivals.sort(key=lambda item: item["arrival_seq"])
    stable: list[Arrival] = []
    for order_index, item in enumerate(arrivals):
        stable.append(
            Arrival(
                arrival_seq=item["arrival_seq"],
                order_index=order_index,
                event_id=item["event_id"],
                delay_ms=item["delay_ms"],
                declared_index=item["declared_index"],
            )
        )
    plan_hash = content_hash([arrival.as_dict() for arrival in stable])
    return InjectionPlan(arrivals=tuple(stable), plan_hash=plan_hash)


def load_plan(path: str | Path) -> InjectionPlan:
    try:
        raw = read_json(Path(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read injection plan: {exc}") from exc
    return parse_plan(raw)


def validate_plan_event_ids(plan: InjectionPlan, event_ids: Iterable[str]) -> None:
    known = set(event_ids)
    missing = sorted({arrival.event_id for arrival in plan.arrivals if arrival.event_id not in known})
    if missing:
        preview = ", ".join(missing[:8])
        raise PlanError(f"injection plan references {len(missing)} unknown event id(s): {preview}")
