"""Two-precision error propagation diagnostics and boundary-sensitive triplets."""

from __future__ import annotations

from typing import Any

from .fusion import triple_sort_key
from .loader import FixedPointConfig, Sentence


STAGE_LABELS = (
    "entity_quantization",
    "relation_quantization",
    "fusion_weighting",
    "threshold_decision",
)


def _scale_units(units: int | None, factor: int) -> int | None:
    return None if units is None else units * factor


def _delta(configured: int | None, reference: int | None) -> int | None:
    if configured is None or reference is None:
        return None
    return configured - reference


def _stage_summary(items: list[dict[str, Any]], tolerance_units: int) -> dict[str, Any]:
    deltas = [abs(item["delta_units"]) for item in items if item["delta_units"] is not None]
    missing = any(item.get("missing") is not None for item in items)
    explicitly_over = any(item.get("over_tolerance", False) for item in items)
    max_delta = max(deltas) if deltas else 0
    return {
        "over_tolerance": missing or explicitly_over or max_delta > tolerance_units,
        "max_abs_delta_units": max_delta,
        "has_missing_items": missing,
        "items": items,
    }


def _entity_items(
    low: dict[str, Any], high: dict[str, Any], factor: int
) -> list[dict[str, Any]]:
    low_map = {
        f"{item['start']}:{item['end']}:{item['type']}": item
        for item in low["entities"]
    }
    high_map = {
        f"{item['start']}:{item['end']}:{item['type']}": item
        for item in high["entities"]
    }
    items: list[dict[str, Any]] = []
    for key in sorted(set(low_map).union(high_map)):
        low_item = low_map.get(key)
        high_item = high_map.get(key)
        configured = _scale_units(
            None if low_item is None else low_item["confidence_units"], factor
        )
        reference = None if high_item is None else high_item["confidence_units"]
        items.append(
            {
                "key": key,
                "configured_units": configured,
                "reference_units": reference,
                "delta_units": _delta(configured, reference),
                "missing": "both" if low_item is None and high_item is None else (
                    "configured" if low_item is None else (
                        "reference" if high_item is None else None
                    )
                ),
            }
        )
    return items


def _relation_items(
    low: dict[str, Any], high: dict[str, Any], factor: int
) -> tuple[list[dict[str, Any]], bool]:
    low_map = {item["pair_key"]: item for item in low["relations"]}
    high_map = {item["pair_key"]: item for item in high["relations"]}
    type_shift = False
    items: list[dict[str, Any]] = []
    for key in sorted(set(low_map).union(high_map)):
        low_item = low_map.get(key)
        high_item = high_map.get(key)
        low_score = _scale_units(
            None if low_item is None else low_item["score_units"], factor
        )
        high_score = None if high_item is None else high_item["score_units"]
        low_confidence = _scale_units(
            None if low_item is None else low_item["confidence_units"], factor
        )
        high_confidence = (
            None if high_item is None else high_item["confidence_units"]
        )
        low_type = None if low_item is None else low_item["type"]
        high_type = None if high_item is None else high_item["type"]
        if low_type is not None and high_type is not None and low_type != high_type:
            type_shift = True
        score_delta = _delta(low_score, high_score)
        confidence_delta = _delta(low_confidence, high_confidence)
        # The selected score is the primary relation-stage quantity. Confidence
        # is included separately because fusion consumes it downstream.
        delta_for_tolerance = score_delta
        if delta_for_tolerance is None:
            delta_for_tolerance = confidence_delta
        items.append(
            {
                "key": key,
                "configured_type": low_type,
                "reference_type": high_type,
                "configured_score_units": low_score,
                "reference_score_units": high_score,
                "score_delta_units": score_delta,
                "configured_confidence_units": low_confidence,
                "reference_confidence_units": high_confidence,
                "confidence_delta_units": confidence_delta,
                "delta_units": delta_for_tolerance,
                "over_tolerance": bool(
                    low_type is not None
                    and high_type is not None
                    and low_type != high_type
                ),
                "missing": "both" if low_item is None and high_item is None else (
                    "configured" if low_item is None else (
                        "reference" if high_item is None else None
                    )
                ),
            }
        )
    return items, type_shift


def _fusion_key(candidate: dict[str, Any]) -> str:
    return "|".join(
        (
            candidate["pair_key"],
            candidate["predicate"],
            str(candidate["subject_span"]),
            str(candidate["object_span"]),
        )
    )


def _fusion_items(
    low: dict[str, Any], high: dict[str, Any], factor: int
) -> list[dict[str, Any]]:
    low_map = {_fusion_key(item): item for item in low["candidates"]}
    high_map = {_fusion_key(item): item for item in high["candidates"]}
    items: list[dict[str, Any]] = []
    for key in sorted(set(low_map).union(high_map)):
        low_item = low_map.get(key)
        high_item = high_map.get(key)
        configured = _scale_units(
            None if low_item is None else low_item["fusion_score_units"], factor
        )
        reference = None if high_item is None else high_item["fusion_score_units"]
        items.append(
            {
                "key": key,
                "configured_units": configured,
                "reference_units": reference,
                "delta_units": _delta(configured, reference),
                "configured_permille": (
                    None if low_item is None else low_item["confidence_permille"]
                ),
                "reference_permille": (
                    None if high_item is None else high_item["confidence_permille"]
                ),
                "missing": "both" if low_item is None and high_item is None else (
                    "configured" if low_item is None else (
                        "reference" if high_item is None else None
                    )
                ),
            }
        )
    return items


def _threshold_items(
    low: dict[str, Any], high: dict[str, Any], factor: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    low_map = {_fusion_key(item): item for item in low["candidates"]}
    high_map = {_fusion_key(item): item for item in high["candidates"]}
    items: list[dict[str, Any]] = []
    boundary: list[dict[str, Any]] = []
    for key in sorted(set(low_map).union(high_map)):
        low_item = low_map.get(key)
        high_item = high_map.get(key)
        low_accepted = None if low_item is None else low_item["accepted"]
        high_accepted = None if high_item is None else high_item["accepted"]
        differs = low_accepted != high_accepted
        item = {
            "key": key,
            "configured_accepted": low_accepted,
            "reference_accepted": high_accepted,
            "decision_differs": differs,
            "delta_units": 1 if differs else 0,
            "over_tolerance": differs,
            "configured_units": (
                None
                if low_item is None
                else _scale_units(low_item["fusion_score_units"], factor)
            ),
            "reference_units": (
                None if high_item is None else high_item["fusion_score_units"]
            ),
        }
        items.append(item)
        if differs:
            source_item = low_item if low_item is not None else high_item
            boundary.append(
                {
                    "sentence_id": source_item["sentence_id"],
                    "subject": source_item["subject"],
                    "predicate": source_item["predicate"],
                    "object": source_item["object"],
                    "subject_span": source_item["subject_span"],
                    "object_span": source_item["object_span"],
                    "configured": (
                        None
                        if low_item is None
                        else {
                            "accepted": low_item["accepted"],
                            "fusion_score_units": low_item["fusion_score_units"],
                            "fusion_score": low_item["fusion_score"],
                            "confidence_permille": low_item["confidence_permille"],
                            "threshold_units": low_item["threshold_units"],
                        }
                    ),
                    "reference": (
                        None
                        if high_item is None
                        else {
                            "accepted": high_item["accepted"],
                            "fusion_score_units": high_item["fusion_score_units"],
                            "fusion_score": high_item["fusion_score"],
                            "confidence_permille": high_item["confidence_permille"],
                            "threshold_units": high_item["threshold_units"],
                        }
                    ),
                    "score_source": source_item.get("score_source", "unknown"),
                }
            )
    return items, boundary


def diagnose_sentence(
    sentence: Sentence,
    entity_low: dict[str, Any],
    entity_high: dict[str, Any],
    relation_low: dict[str, Any],
    relation_high: dict[str, Any],
    fusion_low: dict[str, Any],
    fusion_high: dict[str, Any],
    fixed_low: FixedPointConfig,
    fixed_high: FixedPointConfig,
) -> dict[str, Any]:
    factor = 10 ** (fixed_high.decimals - fixed_low.decimals)
    tolerance_units = fixed_low.tolerance_units

    entity_items = _entity_items(entity_low, entity_high, factor)
    entity_summary = _stage_summary(entity_items, tolerance_units)
    relation_items, relation_type_shift = _relation_items(
        relation_low, relation_high, factor
    )
    relation_summary = _stage_summary(relation_items, tolerance_units)
    fusion_items = _fusion_items(fusion_low, fusion_high, factor)
    fusion_summary = _stage_summary(fusion_items, tolerance_units)
    threshold_items, boundary_triples = _threshold_items(
        fusion_low, fusion_high, factor
    )
    threshold_summary = _stage_summary(threshold_items, tolerance_units)

    low_spans = {(item["start"], item["end"]) for item in entity_low["entities"]}
    high_spans = {(item["start"], item["end"]) for item in entity_high["entities"]}
    entity_boundary_shift = low_spans != high_spans
    low_types = {
        (item["start"], item["end"]): item["type"] for item in entity_low["entities"]
    }
    high_types = {
        (item["start"], item["end"]): item["type"] for item in entity_high["entities"]
    }
    entity_type_shift = any(
        span in low_types and span in high_types and low_types[span] != high_types[span]
        for span in low_types.keys() & high_types.keys()
    )

    stages = [
        {
            "stage": "entity_quantization",
            **entity_summary,
        },
        {
            "stage": "relation_quantization",
            **relation_summary,
            "classification_shift": relation_type_shift,
        },
        {
            "stage": "fusion_weighting",
            **fusion_summary,
        },
        {
            "stage": "threshold_decision",
            **threshold_summary,
        },
    ]
    first_over = next(
        (stage["stage"] for stage in stages if stage["over_tolerance"]), None
    )

    if entity_boundary_shift or entity_type_shift:
        primary_source = "entity_boundary_shift"
    elif relation_type_shift:
        primary_source = "relation_classification"
    elif entity_summary["over_tolerance"]:
        primary_source = "entity_quantization"
    elif relation_summary["over_tolerance"]:
        primary_source = "relation_quantization"
    elif fusion_summary["over_tolerance"]:
        primary_source = "fusion_quantization"
    elif threshold_summary["over_tolerance"]:
        primary_source = "threshold_decision"
    else:
        primary_source = "none"

    for item in boundary_triples:
        item["primary_diagnostic_source"] = primary_source
    boundary_triples.sort(
        key=lambda item: (
            item["subject"],
            item["predicate"],
            item["object"],
            item["sentence_id"],
        )
    )

    return {
        "sentence_id": sentence.sentence_id,
        "configured_precision": {
            "decimals": fixed_low.decimals,
            "quantization_decimals": fixed_low.quantization_decimals,
        },
        "reference_precision": {
            "decimals": fixed_high.decimals,
            "quantization_decimals": fixed_high.quantization_decimals,
        },
        "unit_scale_factor": factor,
        "tolerance_units": tolerance_units,
        "stages": stages,
        "first_stage_over_tolerance": first_over,
        "entity_boundary_shift": entity_boundary_shift,
        "entity_type_shift": entity_type_shift,
        "relation_classification_shift": relation_type_shift,
        "primary_source": primary_source,
        "boundary_sensitive_triples": boundary_triples,
    }


def score_histogram(
    items: list[dict[str, Any]], field: str = "confidence_permille"
) -> dict[str, Any]:
    bins = [
        {
            "lower": lower,
            "upper": 1000 if lower == 900 else lower + 99,
            "count": 0,
        }
        for lower in range(0, 1000, 100)
    ]
    for item in items:
        value = item.get(field)
        if value is None:
            continue
        value = max(0, min(1000, int(value)))
        index = min(value // 100, 9)
        bins[index]["count"] += 1
    return {
        "field": field,
        "bins": bins,
        "total": sum(item["count"] for item in bins),
    }


def boundary_triplets_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item["subject"],
        item["predicate"],
        item["object"],
        item["sentence_id"],
    )
