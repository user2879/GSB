"""Fixed-point intent scoring and float64/high-width references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .fingerprints import score_cache_key
from .models import (
    Contribution,
    IntentScore,
    MatchResult,
    Query,
    RuntimeConfig,
    ScoreResult,
    TriggerEntry,
)


def _quantize(value: int, quantum: int) -> int:
    if quantum <= 1:
        return value
    return ((value + quantum // 2) // quantum) * quantum


def _decision(score: int, threshold: int, tie_rule: str) -> bool:
    if score == threshold:
        return tie_rule == "accept_on_tie"
    return score > threshold


def _winner(scores: Iterable[IntentScore]) -> IntentScore | None:
    ordered = sorted(scores, key=lambda item: (-item.score, item.intent_id))
    return ordered[0] if ordered else None


def score_query_fixed(
    query: Query,
    match: MatchResult,
    entries: Iterable[TriggerEntry],
    config: RuntimeConfig,
    *,
    fraction_bits: int | None = None,
    quantum: int | None = None,
    cache_key: str = "",
) -> ScoreResult:
    bits = config.fraction_bits if fraction_bits is None else fraction_bits
    quant = config.quantum if quantum is None else quantum
    scale = 1 << bits
    entry_list = sorted(entries, key=lambda entry: (entry.entry_order, entry.trigger_id))
    entry_by_id = {entry.trigger_id: entry for entry in entry_list}
    candidates = tuple(dict.fromkeys(query.candidates))

    hit_counts: dict[str, int] = {}
    for span in match.selected:
        entry = entry_by_id.get(span.trigger_id)
        if entry is None or span.terms != entry.terms:
            continue
        if entry.intent_id not in candidates:
            continue
        hit_counts[span.trigger_id] = hit_counts.get(span.trigger_id, 0) + 1

    contributions: list[Contribution] = []
    raw_by_intent: dict[str, int] = {intent_id: 0 for intent_id in candidates}
    for intent_id in sorted(candidates):
        for entry in entry_list:
            if entry.intent_id != intent_id:
                continue
            count = hit_counts.get(entry.trigger_id, 0)
            if count <= 0:
                continue
            numerator = entry.weight * count * scale
            contribution = numerator // max(1, len(query.terms))
            raw_by_intent[intent_id] += contribution
            contributions.append(
                Contribution(
                    trigger_id=entry.trigger_id,
                    intent_id=intent_id,
                    weight=entry.weight,
                    hit_count=count,
                    numerator=numerator,
                    contribution=contribution,
                    entry_order=entry.entry_order,
                )
            )

    scores: list[IntentScore] = []
    for intent_id in sorted(candidates):
        raw = raw_by_intent[intent_id]
        quantized = _quantize(raw, quant)
        threshold = config.threshold_for(intent_id) * scale
        margin = quantized - threshold
        boundary = abs(margin) <= quant // 2
        scores.append(
            IntentScore(
                intent_id=intent_id,
                raw_score=raw,
                score=quantized,
                threshold=threshold,
                decision=_decision(quantized, threshold, config.tie_rule),
                boundary=boundary,
                margin=margin,
            )
        )

    winning = _winner(scores)
    predicted = config.fallback_intent
    boundary = False
    margin = 0
    if winning is not None:
        margin = winning.margin
        boundary = winning.boundary
        if winning.decision and winning.score > 0:
            predicted = winning.intent_id
    referenced = tuple(sorted(hit_counts))
    return ScoreResult(
        query_id=query.query_id,
        predicted_intent=predicted,
        scores=tuple(scores),
        contributions=tuple(contributions),
        referenced_trigger_ids=referenced,
        boundary=boundary,
        margin=margin,
        fraction_bits=bits,
        quantum=quant,
        cache_key=cache_key,
    )


def score_query(
    query: Query,
    match: MatchResult,
    view_entries: Iterable[TriggerEntry],
    config: RuntimeConfig,
) -> ScoreResult:
    entries = tuple(view_entries)
    by_id: Mapping[str, TriggerEntry] = {entry.trigger_id: entry for entry in entries}
    selected_ids = sorted({span.trigger_id for span in match.selected})
    referenced = [by_id[trigger_id] for trigger_id in selected_ids if trigger_id in by_id]
    key = score_cache_key(query, match.cache_key, referenced, config)
    return score_query_fixed(query, match, entries, config, cache_key=key)


@dataclass(frozen=True)
class FloatIntentScore:
    intent_id: str
    score: float
    threshold: float
    decision: bool
    margin: float


@dataclass(frozen=True)
class FloatScoreResult:
    predicted_intent: str
    scores: tuple[FloatIntentScore, ...]
    margin: float


def score_query_float64(
    query: Query,
    match: MatchResult,
    entries: Iterable[TriggerEntry],
    config: RuntimeConfig,
) -> FloatScoreResult:
    entry_list = sorted(entries, key=lambda entry: (entry.entry_order, entry.trigger_id))
    entry_by_id = {entry.trigger_id: entry for entry in entry_list}
    hit_counts: dict[str, int] = {}
    for span in match.selected:
        entry = entry_by_id.get(span.trigger_id)
        if entry is not None and span.terms == entry.terms and entry.intent_id in query.candidates:
            hit_counts[entry.trigger_id] = hit_counts.get(entry.trigger_id, 0) + 1

    results: list[FloatIntentScore] = []
    for intent_id in sorted(query.candidates):
        total = np.float64(0.0)
        for entry in entry_list:
            if entry.intent_id != intent_id:
                continue
            count = hit_counts.get(entry.trigger_id, 0)
            if count <= 0:
                continue
            total = np.float64(total + np.float64(entry.weight * count) / np.float64(max(1, len(query.terms))))
        threshold = np.float64(config.threshold_for(intent_id))
        margin = np.float64(total - threshold)
        if total == threshold:
            decision = config.tie_rule == "accept_on_tie"
        else:
            decision = bool(total > threshold)
        results.append(
            FloatIntentScore(
                intent_id=intent_id,
                score=float(total),
                threshold=float(threshold),
                decision=decision,
                margin=float(margin),
            )
        )
    ordered = sorted(results, key=lambda item: (-item.score, item.intent_id))
    winning = ordered[0] if ordered else None
    predicted = config.fallback_intent
    margin = 0.0
    if winning is not None:
        margin = winning.margin
        if winning.decision and winning.score > 0:
            predicted = winning.intent_id
    return FloatScoreResult(
        predicted_intent=predicted,
        scores=tuple(results),
        margin=margin,
    )


def high_width_reference(
    query: Query,
    match: MatchResult,
    entries: Iterable[TriggerEntry],
    config: RuntimeConfig,
) -> ScoreResult:
    bits = config.fraction_bits + config.reference_extra_bits
    return score_query_fixed(
        query,
        match,
        entries,
        config,
        fraction_bits=bits,
        quantum=config.quantum,
    )


def score_result_from_dict(data: Mapping[str, object]) -> ScoreResult:
    scores = tuple(IntentScore(**item) for item in data.get("scores", []))
    contributions = tuple(Contribution(**item) for item in data.get("contributions", []))
    return ScoreResult(
        query_id=str(data["query_id"]),
        predicted_intent=str(data["predicted_intent"]),
        scores=scores,
        contributions=contributions,
        referenced_trigger_ids=tuple(str(item) for item in data.get("referenced_trigger_ids", [])),
        boundary=bool(data["boundary"]),
        margin=int(data["margin"]),
        fraction_bits=int(data["fraction_bits"]),
        quantum=int(data["quantum"]),
        cache_key=str(data.get("cache_key", "")),
    )
