"""Deterministic lexical, collaborative-filtering, and popularity recall."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .models import AppConfig, CHANNEL_ORDER, Interaction, Item


@dataclass(frozen=True)
class RecallContext:
    catalog: dict[str, Item]
    histories: dict[str, list[Interaction]]
    global_max_ms: int
    item_heat: dict[str, float]
    item_heat_max: float
    latest_event_ms: dict[str, int]
    item_user_counts: dict[str, int]
    item_similarities: dict[str, dict[str, float]]

    @classmethod
    def build(cls, catalog: dict[str, Item], histories: dict[str, list[Interaction]], config: AppConfig) -> "RecallContext":
        all_records = [record for records in histories.values() for record in records]
        global_max_ms = max((record.event_ms for record in all_records), default=0)
        item_heat = {item_id: 0.0 for item_id in catalog}
        latest_event_ms: dict[str, int] = {}
        item_user_sets: dict[str, set[str]] = {item_id: set() for item_id in catalog}

        for user_id, records in histories.items():
            for record in records:
                delta = max(0, global_max_ms - record.event_ms)
                weight = config.event_weights[record.event_type]
                item_heat[record.item_id] += weight * math.pow(2.0, -delta / config.half_life_ms)
                latest_event_ms[record.item_id] = max(latest_event_ms.get(record.item_id, 0), record.event_ms)
                item_user_sets[record.item_id].add(user_id)

        item_heat_max = max(item_heat.values(), default=0.0)
        pair_counts: dict[tuple[str, str], int] = {}
        for user_id in sorted(histories):
            unique_items = sorted({record.item_id for record in histories[user_id]})
            for left_index, left in enumerate(unique_items):
                for right in unique_items[left_index + 1 :]:
                    pair_counts[(left, right)] = pair_counts.get((left, right), 0) + 1

        item_user_counts = {item_id: len(users) for item_id, users in item_user_sets.items()}
        similarities: dict[str, dict[str, float]] = {item_id: {} for item_id in catalog}
        for (left, right), count in sorted(pair_counts.items()):
            denominator = math.sqrt(item_user_counts[left] * item_user_counts[right])
            if denominator <= 0:
                continue
            similarity = count / denominator
            similarities[left][right] = similarity
            similarities[right][left] = similarity

        return cls(
            catalog=catalog,
            histories=histories,
            global_max_ms=global_max_ms,
            item_heat=item_heat,
            item_heat_max=item_heat_max,
            latest_event_ms=latest_event_ms,
            item_user_counts=item_user_counts,
            item_similarities=similarities,
        )


def _rank_scores(scores: dict[str, float], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(scores.items(), key=lambda entry: (-entry[1], entry[0]))[:limit]
    return [
        {"item_id": item_id, "raw_score": float(score), "rank": rank}
        for rank, (item_id, score) in enumerate(ordered, start=1)
    ]


def _user_terms(records: list[Interaction], catalog: dict[str, Item]) -> set[str]:
    terms: set[str] = set()
    for record in records:
        terms.update(catalog[record.item_id].terms)
    return terms


def _lexical_scores(user_terms: set[str], seen: set[str], context: RecallContext) -> dict[str, float]:
    if not user_terms:
        return {}
    scores: dict[str, float] = {}
    user_term_count = len(user_terms)
    for item_id in sorted(context.catalog):
        if item_id in seen:
            continue
        item_terms = set(context.catalog[item_id].terms)
        overlap = len(user_terms.intersection(item_terms))
        if overlap == 0:
            continue
        scores[item_id] = overlap / math.sqrt(user_term_count * len(item_terms))
    return scores


def _collaborative_scores(records: list[Interaction], seen: set[str], context: RecallContext) -> dict[str, float]:
    history_items = sorted({record.item_id for record in records})
    if not history_items:
        return {}
    scores: dict[str, float] = {}
    for history_item in history_items:
        for candidate, similarity in context.item_similarities.get(history_item, {}).items():
            if candidate in seen:
                continue
            scores[candidate] = scores.get(candidate, 0.0) + similarity
    return scores


def _popularity_scores(seen: set[str], context: RecallContext, config: AppConfig) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item_id in sorted(context.catalog):
        if item_id in seen:
            continue
        scores[item_id] = max(float(config.popularity_floor), context.item_heat.get(item_id, 0.0))
    return scores


def run_recall_for_user(user_id: str, context: RecallContext, config: AppConfig) -> dict[str, Any]:
    """Compute all three recall channels for one user.

    The returned record is precision-independent and is the input to both the
    primary and reference fixed-point fusion calculations.
    """

    records = context.histories.get(user_id, [])
    seen = {record.item_id for record in records}
    user_terms = _user_terms(records, context.catalog)

    scores_by_channel: dict[str, dict[str, float]] = {
        "lexical": {},
        "collaborative": {},
        "popularity": {},
    }
    if config.channels["lexical"].enabled:
        scores_by_channel["lexical"] = _lexical_scores(user_terms, seen, context)
    if config.channels["collaborative"].enabled:
        scores_by_channel["collaborative"] = _collaborative_scores(records, seen, context)
    if config.channels["popularity"].enabled:
        scores_by_channel["popularity"] = _popularity_scores(seen, context, config)

    channel_records: dict[str, Any] = {}
    candidate_ids: set[str] = set()
    for name in CHANNEL_ORDER:
        channel_config = config.channels[name]
        candidates = _rank_scores(scores_by_channel[name], channel_config.candidate_limit)
        channel_records[name] = {
            "name": name,
            "enabled": channel_config.enabled,
            "candidate_limit": channel_config.candidate_limit,
            "weight": channel_config.weight,
            "candidate_count": len(candidates),
            "score_max": max((candidate["raw_score"] for candidate in candidates), default=0.0),
            "score_min": min((candidate["raw_score"] for candidate in candidates), default=0.0),
            "candidates": candidates,
            "stable_task_key": f"{user_id}|{name}",
        }
        candidate_ids.update(candidate["item_id"] for candidate in candidates)

    feature_inputs: dict[str, dict[str, float]] = {}
    for item_id in sorted(candidate_ids):
        item_terms = set(context.catalog[item_id].terms)
        overlap_ratio = len(user_terms.intersection(item_terms)) / max(1, len(user_terms))
        if item_id in context.latest_event_ms:
            delta = max(0, context.global_max_ms - context.latest_event_ms[item_id])
            freshness = math.pow(2.0, -delta / config.half_life_ms)
        else:
            freshness = 0.0
        prior = context.item_heat.get(item_id, 0.0) / context.item_heat_max if context.item_heat_max > 0 else 0.0
        feature_inputs[item_id] = {
            "history_overlap": min(1.0, max(0.0, overlap_ratio)),
            "freshness": min(1.0, max(0.0, freshness)),
            "popularity_prior": min(1.0, max(0.0, prior)),
        }

    return {
        "user_id": user_id,
        "task_key": f"{user_id}|channels={','.join(CHANNEL_ORDER)}",
        "channel_task_keys": [f"{user_id}|{name}" for name in CHANNEL_ORDER],
        "history_item_count": len(seen),
        "history_term_count": len(user_terms),
        "candidate_count": len(candidate_ids),
        "channels": channel_records,
        "feature_inputs": feature_inputs,
    }
