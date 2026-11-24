"""Deterministic local event-stream simulation from integer seeds."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from .errors import OrderError
from .spec import SessionBundle, SourceSpec


@dataclass
class Event:
    source_id: str
    source_event_index: int
    seq: int
    event_time_ms: int
    event_id: str
    key_values: dict[str, Any]
    measure_raw: int
    measure_decimals: int
    weight_raw: int
    weight_decimals: int
    injection: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_event_index": self.source_event_index,
            "seq": self.seq,
            "event_time_ms": self.event_time_ms,
            "event_id": self.event_id,
            "key_values": self.key_values,
            "measure_raw": self.measure_raw,
            "measure_decimals": self.measure_decimals,
            "weight_raw": self.weight_raw,
            "weight_decimals": self.weight_decimals,
        }


def _generate_source_events(source: SourceSpec) -> list[Event]:
    rng = np.random.Generator(np.random.PCG64(int(source.seed)))
    count = source.event_count
    if count == 0:
        return []

    timestamps = np.array(
        [source.start_ms + index * source.pace_ms for index in range(count)],
        dtype=np.int64,
    )
    disorder_count = int((count * source.out_of_order_ppm) // 10000)
    if disorder_count >= 2:
        if disorder_count >= count:
            positions = np.arange(count, dtype=np.int64)
        else:
            positions = rng.choice(count, size=disorder_count, replace=False)
        selected = timestamps[positions].copy()
        permutation = rng.permutation(len(positions))
        timestamps[positions] = selected[permutation]

    key_names = sorted(source.partition_values)
    per_event_keys: list[dict[str, Any]] = []
    for _ in range(count):
        keys: dict[str, Any] = {}
        for name in key_names:
            values = source.partition_values[name]
            pick = int(rng.integers(0, len(values)))
            keys[name] = values[pick]
        per_event_keys.append(keys)

    measure_span = source.measure_range[1] - source.measure_range[0] + 1
    weight_span = source.weight_range[1] - source.weight_range[0] + 1
    events: list[Event] = []
    for index in range(count):
        measure_raw = source.measure_range[0] + int(rng.integers(0, measure_span))
        weight_raw = source.weight_range[0] + int(rng.integers(0, weight_span))
        events.append(
            Event(
                source_id=source.source_id,
                source_event_index=index,
                seq=-1,
                event_time_ms=int(timestamps[index]),
                event_id=f"{source.source_id}:{index:08d}",
                key_values=per_event_keys[index],
                measure_raw=measure_raw,
                measure_decimals=source.measure_decimals,
                weight_raw=weight_raw,
                weight_decimals=source.weight_decimals,
            )
        )
    return events


def generate_events(bundle: SessionBundle) -> list[Event]:
    """Generate one canonical stream independent of source argument order."""

    per_source = {
        source.source_id: _generate_source_events(source)
        for source in sorted(bundle.sources, key=lambda item: item.source_id)
    }
    source_ids = sorted(per_source)
    max_count = max((len(rows) for rows in per_source.values()), default=0)
    stream: list[Event] = []
    seq = 0
    for local_index in range(max_count):
        for source_id in source_ids:
            rows = per_source[source_id]
            if local_index >= len(rows):
                continue
            event = rows[local_index]
            event.seq = seq
            seq += 1
            stream.append(event)
    validate_out_of_order_budget(bundle, stream)
    return stream


def validate_out_of_order_budget(bundle: SessionBundle, events: Iterable[Event]) -> None:
    """Reject a stream whose observed source-local regression rate exceeds ppm."""

    rows_by_source: dict[str, list[Event]] = {}
    for event in events:
        rows_by_source.setdefault(event.source_id, []).append(event)
    source_by_id = {source.source_id: source for source in bundle.sources}
    for source_id in sorted(rows_by_source):
        rows = sorted(
            rows_by_source[source_id], key=lambda item: item.source_event_index
        )
        source = source_by_id[source_id]
        regressions = 0
        first: dict[str, Any] | None = None
        for previous, current in zip(rows, rows[1:]):
            if current.event_time_ms < previous.event_time_ms:
                regressions += 1
                if first is None:
                    first = {
                        "source_id": source_id,
                        "event_id": current.event_id,
                        "previous_event_id": previous.event_id,
                        "previous_event_time_ms": previous.event_time_ms,
                        "event_time_ms": current.event_time_ms,
                    }
        denominator = max(1, len(rows) - 1)
        if regressions * 10000 > source.out_of_order_ppm * denominator:
            raise OrderError(
                "event stream exceeds declared source-local out-of-order ppm",
                details={
                    "source_id": source_id,
                    "regressions": regressions,
                    "pairs": denominator,
                    "observed_ppm": (regressions * 10000) // denominator,
                    "declared_ppm": source.out_of_order_ppm,
                    "first_regression": first,
                },
            )


def apply_injection(events: list[Event], mode: str | None) -> list[Event]:
    """Annotate a deterministic stream with one runtime anomaly trigger."""

    if mode is None:
        return events
    if mode not in {"order", "budget", "watermark"}:
        raise ValueError(f"unsupported injection mode: {mode}")
    if not events:
        return events
    if mode == "order":
        index = max(1, len(events) * 55 // 100)
        if index >= len(events):
            index = len(events) - 1
        event = events[index]
        same_source = [
            item
            for item in events[:index]
            if item.source_id == event.source_id
        ]
        if same_source:
            previous_time = same_source[-1].event_time_ms
            events[index] = replace(
                event,
                event_time_ms=max(0, previous_time - 1),
                injection="order",
            )
        else:
            events[index] = replace(event, injection="order")
    elif mode == "budget":
        index = max(1, len(events) * 65 // 100)
        events[min(index, len(events) - 1)] = replace(
            events[min(index, len(events) - 1)], injection="budget"
        )
    else:
        index = max(1, len(events) * 45 // 100)
        events[min(index, len(events) - 1)] = replace(
            events[min(index, len(events) - 1)], injection="watermark"
        )
    return events
