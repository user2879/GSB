from __future__ import annotations

from dataclasses import dataclass

from .canonical import digest_obj
from .model import Event, SessionSpec, SourceSpec
from .prng import SplitMix64


@dataclass(frozen=True)
class SourceGeneration:
    source: SourceSpec
    events: tuple[Event, ...]
    digest: str
    displaced_events: int
    actual_out_of_order_ratio_bp: int


def _resolved_partition_keys(source: SourceSpec, session_partition_keys: tuple[str, ...]) -> tuple[str, ...]:
    return source.partition_keys or session_partition_keys


def generate_source(
    source: SourceSpec,
    available_partition_keys: tuple[str, ...],
    *,
    arrival_offset: int = 0,
) -> SourceGeneration:
    partition_keys = _resolved_partition_keys(source, available_partition_keys)
    if not partition_keys:
        raise ValueError("no partition keys available for source generation")
    rng = SplitMix64(source.seed)
    local: list[tuple[int, int, str, int]] = []
    step = source.time_step_ms
    jitter_span = max(1, min(step, max(1, step // 2)))
    for source_seq in range(source.event_count):
        timestamp = source.base_time_ms + source_seq * step + rng.randbelow(jitter_span)
        partition_key = partition_keys[rng.randbelow(len(partition_keys))]
        measure = rng.randint(source.measure_min, source.measure_max)
        local.append((source_seq, timestamp, partition_key, measure))

    displaced = (source.event_count * source.out_of_order_ratio_bp) // 10_000
    if displaced >= source.event_count:
        displaced = source.event_count - 1
    selected: set[int] = set()
    if displaced > 0 and source.event_count > 1:
        selected = set(rng.sample_indices(source.event_count - 1, displaced))

    max_delay = source.reorder_delay_events
    if selected and max_delay <= 0:
        max_delay = 1
    schedule: list[tuple[int, int]] = []
    for source_seq in range(source.event_count):
        if source_seq in selected:
            delay = rng.randint(1, max_delay)
            release = min(source.event_count - 1, source_seq + delay)
        else:
            release = source_seq
        schedule.append((release, source_seq))
    schedule.sort()

    events: list[Event] = []
    digest_rows: list[dict[str, object]] = []
    for local_arrival, (_, source_seq) in enumerate(schedule):
        seq, timestamp, partition_key, measure = local[source_seq]
        event = Event(
            event_id=f"{source.source_id}:{seq:08d}",
            source_id=source.source_id,
            source_seq=seq,
            arrival_no=arrival_offset + local_arrival,
            partition_key=partition_key,
            event_time_ms=timestamp,
            measure=measure,
        )
        events.append(event)
        digest_rows.append(
            {
                "event_id": event.event_id,
                "source_id": event.source_id,
                "source_seq": event.source_seq,
                "local_arrival": local_arrival,
                "partition_key": event.partition_key,
                "event_time_ms": event.event_time_ms,
                "measure": event.measure,
            }
        )
    actual_ratio = (displaced * 10_000) // source.event_count if source.event_count else 0
    return SourceGeneration(
        source=source,
        events=tuple(events),
        digest=digest_obj(digest_rows),
        displaced_events=displaced,
        actual_out_of_order_ratio_bp=actual_ratio,
    )


def generate_session(spec: SessionSpec) -> tuple[list[Event], dict[str, SourceGeneration]]:
    partition_keys = spec.partition_keys
    generations = [generate_source(source, partition_keys) for source in spec.sources]
    max_length = max(len(generation.events) for generation in generations)
    merged: list[Event] = []
    for slot in range(max_length):
        for generation in generations:
            if slot < len(generation.events):
                event = generation.events[slot]
                merged.append(event)
    final_events: list[Event] = []
    for arrival_no, event in enumerate(merged):
        final_events.append(
            Event(
                event_id=event.event_id,
                source_id=event.source_id,
                source_seq=event.source_seq,
                arrival_no=arrival_no,
                partition_key=event.partition_key,
                event_time_ms=event.event_time_ms,
                measure=event.measure,
            )
        )
    by_source = {generation.source.source_id: generation for generation in generations}
    return final_events, by_source


def source_event_digest(spec: SessionSpec, source_id: str) -> str:
    _, by_source = generate_session(spec)
    return by_source[source_id].digest
