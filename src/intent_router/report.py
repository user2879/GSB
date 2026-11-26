"""Diagnostic report assembly and intent distribution statistics."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .engine import EngineResult
from .errors import FailureCollector
from .fixedpoint import QuantGear
from .statistics import fixed_point_distribution, permille_share


def intent_distribution(result: EngineResult) -> dict[str, int]:
    """Distribution of authoritative conclusions (dedup-aware)."""
    counter: Counter[str] = Counter()
    for text_id in sorted(result.scores):
        counter[result.final_intent(text_id)] += 1
    return dict(sorted(counter.items()))


def distribution_with_share(
    result: EngineResult, intents: tuple[str, ...]
) -> dict[str, dict[str, int]]:
    decisions = [
        result.final_intent(text_id) for text_id in sorted(result.scores)
    ]
    ordered = list(intents)
    extra = sorted(set(decisions) - set(ordered))
    ordered.extend(extra)
    counts = fixed_point_distribution(ordered, decisions)
    shares = permille_share(counts)
    return {
        intent: {"count": counts[intent], "permille": shares[intent]}
        for intent in ordered
    }


def build_report(
    result: EngineResult,
    failures: FailureCollector,
    version: int,
    config_hash: str,
    gear: QuantGear,
    config_intents: tuple[str, ...] = (),
) -> dict[str, Any]:
    per_text: list[dict[str, Any]] = []
    for text_id in sorted(result.scores):
        score = result.scores[text_id]
        member = result.removed.get(text_id)
        per_text.append(
            {
                "text_id": text_id,
                "channel": result.texts[text_id].channel,
                "gold_label": result.texts[text_id].gold_label,
                "raw_intent": score.decided_intent,
                "final_intent": result.final_intent(text_id),
                "boundary": score.boundary,
                "boundary_reasons": list(score.boundary_reasons),
                "is_duplicate": member is not None,
                "representative_id": (
                    member.representative_id if member else None
                ),
                "similarity_permille": (
                    member.similarity_permille if member else None
                ),
                "suppressed": result.matches[text_id].to_dict()["suppressed"],
                "paths": [
                    score.paths[intent].to_dict(gear)
                    for intent in sorted(score.paths)
                ],
            }
        )

    flips = [flip.to_dict() for flip in result.flips if flip.flipped]
    flip_source_counts = Counter(flip["source"] for flip in flips)

    report = {
        "version": version,
        "config_hash": config_hash,
        "summary": {
            "texts_total": len(result.scores),
            "clusters": len(result.clusters),
            "duplicates_removed": len(result.removed),
            "pending": len(result.pending),
            "invalidations": len(result.ledger.invalidations),
            "reuses": len(result.ledger.reuses),
            "flips": len(flips),
        },
        "intent_distribution": intent_distribution(result),
        "intent_distribution_detail": distribution_with_share(
            result, config_intents
        ),
        "flip_attribution": dict(sorted(flip_source_counts.items())),
        "flip_details": flips,
        "pending": [item.to_dict() for item in result.pending],
        "consumed_budget": dict(result.consumed),
        "failure_counts": failures.count_by_code(),
        "failures": failures.to_list(),
        "per_text": per_text,
        "invalidations": [r.to_dict() for r in result.ledger.invalidations],
        "reuses": [r.to_dict() for r in result.ledger.reuses],
    }
    return report
