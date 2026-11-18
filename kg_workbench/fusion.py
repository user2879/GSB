"""Fixed-point temperature fusion and threshold decisions."""

from __future__ import annotations

from typing import Any

from .loader import FixedPointConfig, Sentence, WorkbenchConfig
from .util import (
    format_fixed,
    permille,
    quantize_probability,
    quantize_units,
    round_div_half_up,
    sigmoid_from_units,
)


def triple_identity(
    sentence: Sentence, relation: dict[str, Any]
) -> tuple[str, str, str, str, int, int, int, int]:
    subject_start = relation["subject"]["start"]
    subject_end = relation["subject"]["end"]
    object_start = relation["object"]["start"]
    object_end = relation["object"]["end"]
    subject_text = "".join(sentence.words[subject_start : subject_end + 1])
    object_text = "".join(sentence.words[object_start : object_end + 1])
    return (
        subject_text,
        relation["type"],
        object_text,
        sentence.sentence_id,
        subject_start,
        subject_end,
        object_start,
        object_end,
    )


def triple_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item["subject"],
        item["predicate"],
        item["object"],
        item["sentence_id"],
        item["subject_span"][0],
        item["subject_span"][1],
        item["object_span"][0],
        item["object_span"][1],
    )


def fuse_sentence(
    sentence: Sentence,
    relation_result: dict[str, Any],
    config: WorkbenchConfig,
    fixed: FixedPointConfig,
    *,
    threshold_equal_injection_id: str | None = None,
) -> dict[str, Any]:
    scale = fixed.scale
    entity_weight_units = quantize_units(
        config.fixed.entity_weight, fixed.decimals, fixed.quantization_decimals
    )
    relation_weight_units = quantize_units(
        config.fixed.relation_weight, fixed.decimals, fixed.quantization_decimals
    )
    bias_units = quantize_units(
        config.fixed.bias, fixed.decimals, fixed.quantization_decimals
    )
    temperature_units = quantize_units(
        config.fixed.temperature, fixed.decimals, fixed.quantization_decimals
    )
    if temperature_units <= 0:
        temperature_units = fixed.quantum
    threshold_units = quantize_units(
        config.fixed.confidence_threshold,
        fixed.decimals,
        fixed.quantization_decimals,
    )

    candidates: list[dict[str, Any]] = []
    injection_events: list[dict[str, Any]] = []
    for relation in relation_result["relations"]:
        entity_units = relation["entity_score_units"]
        relation_units = relation["confidence_units"]
        weighted_numerator = (
            entity_weight_units * entity_units
            + relation_weight_units * relation_units
            + bias_units * scale
        )
        weighted_margin_units = round_div_half_up(weighted_numerator, scale)
        scaled_margin_units = round_div_half_up(
            weighted_margin_units * scale, temperature_units
        )
        fusion_raw = sigmoid_from_units(scaled_margin_units, fixed.decimals)
        fusion_units = quantize_probability(
            fusion_raw, fixed.decimals, fixed.quantization_decimals
        )
        injected = False
        if threshold_equal_injection_id is not None:
            fusion_units = threshold_units
            injected = True
            injection_events.append(
                {
                    "injection_id": threshold_equal_injection_id,
                    "sentence_id": sentence.sentence_id,
                    "stage": "fusion",
                    "kind": "threshold_equal",
                    "applied": True,
                    "business_code": None,
                    "pair_key": relation["pair_key"],
                    "fusion_units": fusion_units,
                    "threshold_units": threshold_units,
                }
            )
        boundary = fusion_units == threshold_units
        accepted = fusion_units > threshold_units or (
            boundary and config.fixed.threshold_equal_policy == "accept"
        )

        entity_contribution = round_div_half_up(entity_weight_units * entity_units, scale)
        relation_contribution = round_div_half_up(
            relation_weight_units * relation_units, scale
        )
        if (
            threshold_equal_injection_id is not None
            and abs(entity_contribution) < abs(relation_contribution)
        ):
            score_source = "relation_score"
        elif (
            threshold_equal_injection_id is not None
            and abs(entity_contribution) >= abs(relation_contribution)
        ):
            score_source = "entity_score"
        else:
            score_source = (
                "relation_score"
                if abs(relation_contribution) > abs(entity_contribution)
                else "entity_score"
            )

        identity = triple_identity(sentence, relation)
        candidates.append(
            {
                "sentence_id": sentence.sentence_id,
                "subject": identity[0],
                "predicate": identity[1],
                "object": identity[2],
                "subject_span": [relation["subject"]["start"], relation["subject"]["end"]],
                "object_span": [relation["object"]["start"], relation["object"]["end"]],
                "subject_type": relation["subject"]["type"],
                "object_type": relation["object"]["type"],
                "pair_key": relation["pair_key"],
                "entity_score_units": entity_units,
                "entity_score": format_fixed(entity_units, fixed.decimals),
                "relation_score_units": relation_units,
                "relation_score": format_fixed(relation_units, fixed.decimals),
                "entity_weight_units": entity_weight_units,
                "relation_weight_units": relation_weight_units,
                "bias_units": bias_units,
                "temperature_units": temperature_units,
                "weighted_margin_units": weighted_margin_units,
                "scaled_margin_units": scaled_margin_units,
                "fusion_score_units": fusion_units,
                "fusion_score": format_fixed(fusion_units, fixed.decimals),
                "confidence_permille": permille(fusion_units, fixed.decimals),
                "threshold_units": threshold_units,
                "threshold": format_fixed(threshold_units, fixed.decimals),
                "boundary_sample": boundary,
                "accepted": accepted,
                "score_source": score_source,
                "entity_contribution_units": entity_contribution,
                "relation_contribution_units": relation_contribution,
                "suppressed": list(relation["suppressed"]),
                "trigger": relation["trigger"],
                "injected_threshold_equal": injected,
            }
        )

    candidates.sort(key=triple_sort_key)
    triples = [item for item in candidates if item["accepted"]]
    return {
        "sentence_id": sentence.sentence_id,
        "precision": {
            "decimals": fixed.decimals,
            "quantization_decimals": fixed.quantization_decimals,
            "scale": fixed.scale,
        },
        "threshold_units": threshold_units,
        "threshold_equal_policy": config.fixed.threshold_equal_policy,
        "candidates": candidates,
        "triples": triples,
        "injection_events": injection_events,
    }
