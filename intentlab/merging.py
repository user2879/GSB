"""Adjacent-query merge and deterministic cross-query slot alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .fingerprints import query_content_fingerprint, segment_cache_key
from .models import AlignedSlot, Query, RuntimeConfig, ScoreResult, Segment


@dataclass(frozen=True)
class QueryScorePair:
    query: Query
    score: ScoreResult


def _can_merge(previous: QueryScorePair, current: QueryScorePair, config: RuntimeConfig) -> bool:
    if previous.query.session_id != current.query.session_id:
        return False
    delta = current.query.turn - previous.query.turn
    if delta < 1 or delta > config.max_turn_gap:
        return False
    return (
        previous.score.predicted_intent == current.score.predicted_intent
        and previous.score.predicted_intent != config.fallback_intent
    )


def partition_query_pairs(
    pairs: Iterable[QueryScorePair],
    config: RuntimeConfig,
) -> list[tuple[QueryScorePair, ...]]:
    ordered = sorted(
        pairs,
        key=lambda pair: (pair.query.session_id, pair.query.turn, pair.query.query_id),
    )
    groups: list[list[QueryScorePair]] = []
    for pair in ordered:
        if not groups:
            groups.append([pair])
            continue
        if _can_merge(groups[-1][-1], pair, config):
            groups[-1].append(pair)
        else:
            groups.append([pair])
    return [tuple(group) for group in groups]


def segment_key_for(
    group: Sequence[QueryScorePair],
    config: RuntimeConfig,
) -> tuple[str, str]:
    intent = group[0].score.predicted_intent
    key = segment_cache_key(
        session_id=group[0].query.session_id,
        intent=intent,
        score_keys=[pair.score.cache_key for pair in group],
        query_fingerprints=[query_content_fingerprint(pair.query) for pair in group],
        merge_config={
            "max_turn_gap": config.max_turn_gap,
            "term_gap": config.term_gap,
            "slot_priority": ["earlier_query", "longer_span", "query_id", "slot_name"],
        },
    )
    segment_id = "seg-" + key[:16]
    return segment_id, key


def build_segment(
    group: Sequence[QueryScorePair],
    config: RuntimeConfig,
    *,
    segment_id: str | None = None,
    cache_key: str | None = None,
) -> Segment:
    computed_id, computed_key = segment_key_for(group, config)
    segment_id = segment_id or computed_id
    cache_key = cache_key or computed_key
    intent = group[0].score.predicted_intent

    base_by_query: dict[str, int] = {}
    combined_terms: list[str] = []
    for index, pair in enumerate(group):
        if index > 0:
            combined_terms.extend("<gap>" for _ in range(config.term_gap))
        base_by_query[pair.query.query_id] = len(combined_terms)
        combined_terms.extend(pair.query.terms)

    mapped: list[dict[str, object]] = []
    for pair in group:
        base = base_by_query[pair.query.query_id]
        for slot in pair.query.slots:
            mapped.append(
                {
                    "priority": (
                        pair.query.turn,
                        -(slot.end - slot.start),
                        pair.query.query_id,
                        slot.name,
                        slot.start,
                    ),
                    "name": slot.name,
                    "query_id": pair.query.query_id,
                    "local_start": slot.start,
                    "local_end": slot.end,
                    "start": base + slot.start,
                    "end": base + slot.end,
                }
            )
    mapped.sort(key=lambda item: item["priority"])

    accepted: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []
    aligned: list[AlignedSlot] = []
    for item in mapped:
        start = int(item["start"])
        end = int(item["end"])
        conflict_target: dict[str, object] | None = None
        for existing in accepted:
            existing_start = int(existing["start"])
            existing_end = int(existing["end"])
            if start < existing_end and end > existing_start:
                conflict_target = existing
                break
        if conflict_target is None:
            accepted.append(item)
            aligned.append(
                AlignedSlot(
                    name=str(item["name"]),
                    query_id=str(item["query_id"]),
                    local_start=int(item["local_start"]),
                    local_end=int(item["local_end"]),
                    start=start,
                    end=end,
                    status="accepted",
                )
            )
            continue
        conflict_name = f"{conflict_target['query_id']}:{conflict_target['name']}"
        aligned.append(
            AlignedSlot(
                name=str(item["name"]),
                query_id=str(item["query_id"]),
                local_start=int(item["local_start"]),
                local_end=int(item["local_end"]),
                start=start,
                end=end,
                status="conflicted",
                conflict_with=conflict_name,
            )
        )
        conflicts.append(
            {
                "rejected": {
                    "name": item["name"],
                    "query_id": item["query_id"],
                    "local_start": item["local_start"],
                    "local_end": item["local_end"],
                    "start": start,
                    "end": end,
                },
                "accepted": {
                    "name": conflict_target["name"],
                    "query_id": conflict_target["query_id"],
                    "start": conflict_target["start"],
                    "end": conflict_target["end"],
                },
                "decision": "overlap_offset_mapping",
                "priority": ["earlier_query", "longer_span", "query_id", "slot_name"],
            }
        )

    aligned.sort(key=lambda slot: (slot.start, slot.end, slot.name, slot.query_id))
    conflicts.sort(
        key=lambda item: (
            item["rejected"]["start"],
            item["rejected"]["end"],
            item["rejected"]["name"],
            item["rejected"]["query_id"],
        )
    )
    return Segment(
        segment_id=segment_id,
        session_id=group[0].query.session_id,
        intent=intent,
        query_ids=tuple(pair.query.query_id for pair in group),
        start_turn=group[0].query.turn,
        end_turn=group[-1].query.turn,
        terms=tuple(combined_terms),
        aligned_slots=tuple(aligned),
        conflicts=tuple(conflicts),
        cache_key=cache_key,
    )


def build_segments(
    pairs: Iterable[QueryScorePair],
    config: RuntimeConfig,
) -> tuple[Segment, ...]:
    groups = partition_query_pairs(pairs, config)
    return tuple(build_segment(group, config) for group in groups)


def segment_from_dict(data: Mapping[str, object]) -> Segment:
    return Segment(
        segment_id=str(data["segment_id"]),
        session_id=str(data["session_id"]),
        intent=str(data["intent"]),
        query_ids=tuple(str(item) for item in data.get("query_ids", [])),
        start_turn=int(data["start_turn"]),
        end_turn=int(data["end_turn"]),
        terms=tuple(str(item) for item in data.get("terms", [])),
        aligned_slots=tuple(AlignedSlot(**item) for item in data.get("aligned_slots", [])),
        conflicts=tuple(dict(item) for item in data.get("conflicts", [])),
        cache_key=str(data.get("cache_key", "")),
    )

