"""Seed-driven deterministic event generation and ingress validation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .spec import Spec, SourceSpec
from .util import digest_of


@dataclass(frozen=True)
class Event:
    source_id: str
    source_seq: int
    timestamp_ms: int
    group_key: str
    value: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_seq": self.source_seq,
            "timestamp_ms": self.timestamp_ms,
            "group_key": self.group_key,
            "value": self.value,
        }


@dataclass(frozen=True)
class RejectedEvent:
    source_id: str
    source_seq: int
    field: str
    reason: str
    watermark_ms: int
    planned: dict[str, Any]


def _rng_for(source: SourceSpec) -> random.Random:
    return random.Random(f"swa:v1:{source.source_id}:{source.seed}")


def generate_source_events(
    source: SourceSpec,
    spec: Spec,
    overrides: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> list[Event]:
    """Generate the full ordered event stream for one source.

    The random stream is consumed in a fixed per-event order (timestamp,
    group, value), so identical (seed, count) always yields identical bytes.
    """
    overrides = overrides or {}
    rng = _rng_for(source)
    events: list[Event] = []
    current_ms = 0
    for seq in range(1, source.event_count + 1):
        gap = rng.randint(1, 500)
        current_ms += gap
        group_key = spec.group_keys[rng.randrange(len(spec.group_keys))]
        value = rng.randint(spec.value_min, spec.value_max)
        planned: dict[str, Any] = {
            "source_id": source.source_id,
            "source_seq": seq,
            "timestamp_ms": current_ms,
            "group_key": group_key,
            "value": value,
        }
        planned.update(
            {key: item for key, item in overrides.get((source.source_id, seq), {}).items() if key not in ("source_id", "source_seq")}
        )
        events.append(
            Event(
                source_id=planned["source_id"],
                source_seq=planned["source_seq"],
                timestamp_ms=planned["timestamp_ms"],
                group_key=planned["group_key"],
                value=planned["value"],
            )
        )
    return events


def generate_all_events(
    spec: Spec, overrides: dict[tuple[str, int], dict[str, Any]] | None = None
) -> dict[str, list[Event]]:
    return {source.source_id: generate_source_events(source, spec, overrides) for source in spec.sources}


def event_stream_digest(streams: dict[str, list[Event]]) -> str:
    material = [
        [event.to_dict() for event in streams[source_id]]
        for source_id in sorted(streams)
    ]
    return digest_of(material)


def validate_event(event: Event, spec: Spec, source_watermark_ms: int) -> RejectedEvent | None:
    """Ingress validation; a rejection must never advance the source watermark."""
    if not isinstance(event.group_key, str) or not event.group_key.strip():
        return RejectedEvent(
            event.source_id,
            event.source_seq,
            "group_key",
            "group_key must be a non-empty string",
            source_watermark_ms,
            event.to_dict(),
        )
    if not isinstance(event.value, int) or isinstance(event.value, bool) or not (spec.value_min <= event.value <= spec.value_max):
        return RejectedEvent(
            event.source_id,
            event.source_seq,
            "value",
            f"value outside [{spec.value_min}, {spec.value_max}]",
            source_watermark_ms,
            event.to_dict(),
        )
    if not isinstance(event.timestamp_ms, int) or isinstance(event.timestamp_ms, bool) or event.timestamp_ms < 0:
        return RejectedEvent(
            event.source_id,
            event.source_seq,
            "timestamp_ms",
            "timestamp_ms must be a non-negative integer",
            source_watermark_ms,
            event.to_dict(),
        )
    if event.timestamp_ms <= source_watermark_ms:
        return RejectedEvent(
            event.source_id,
            event.source_seq,
            "timestamp_ms",
            "source timestamps must be strictly monotonic",
            source_watermark_ms,
            event.to_dict(),
        )
    return None


def build_round_robin_plan(streams: dict[str, list[Event]]) -> list[Event]:
    """Interleave sources round-robin while preserving each source's order."""
    queues = {source_id: list(events) for source_id, events in streams.items()}
    order = sorted(queues)
    plan: list[Event] = []
    while any(queues.values()):
        for source_id in order:
            if queues[source_id]:
                plan.append(queues[source_id].pop(0))
    return plan


def rejection_to_dict(rejection: RejectedEvent) -> dict[str, Any]:
    return {
        "source_id": rejection.source_id,
        "source_seq": rejection.source_seq,
        "field": rejection.field,
        "reason": rejection.reason,
        "watermark_ms": rejection.watermark_ms,
        "planned": rejection.planned,
    }
