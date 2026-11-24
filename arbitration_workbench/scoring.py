"""Integer-domain scoring, decay, relationship application and aggregation."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable

from .canonical import (
    fixed_str,
    quantize_floor,
    round_half_up_div,
    seal_payload,
    to_fixed,
)
from .codes import BusinessFault, E_HALF_LIFE_ZERO, E_RELATION_UNKNOWN_TYPE, E_THRESHOLD_INVALID
from .models import ArbitrationConfig, Evidence, Relation, RulePackage


DECAY_FRACTION_BITS = 32
DECAY_ALGORITHM = "BINARY_FRACTIONAL_SQUARING_FLOOR_V1"
RELATION_ALGORITHM = "EXCLUSIONS_THEN_SUPPORTS_V1"
AGGREGATION_ALGORITHM = "ORDERED_FUSION_QUANTIZE_FLOOR_V1"


def _fixed_decimal(value: Decimal, config: ArbitrationConfig) -> int:
    return to_fixed(value, config.fixed_decimal_width)


def _fixed_quantized(value: Decimal, config: ArbitrationConfig) -> int:
    return quantize_floor(_fixed_decimal(value, config), config.quantization_step)


def _binary_decay(elapsed_ms: int, half_life_ms: int) -> tuple[int, bool]:
    """Return coefficient in 2**DECAY_FRACTION_BITS and exact-boundary flag."""

    if half_life_ms <= 0:
        raise BusinessFault(E_HALF_LIFE_ZERO, "half_life_ms must be greater than zero", "decay")
    whole, remainder = divmod(elapsed_ms, half_life_ms)
    exact_boundary = remainder == 0
    if remainder == 0:
        coefficient = 1 << DECAY_FRACTION_BITS
    else:
        from .canonical import binary_pow2_fraction

        coefficient = binary_pow2_fraction(remainder, half_life_ms, DECAY_FRACTION_BITS)
    if whole >= DECAY_FRACTION_BITS:
        coefficient = 0
    elif whole:
        coefficient >>= whole
    return coefficient, exact_boundary


def compute_decay_stage(
    workorder_id: str,
    evidence: list[Evidence],
    rules: RulePackage,
    config: ArbitrationConfig,
    *,
    effective_half_life_ms: int | None = None,
    injection_id: str | None = None,
) -> dict[str, Any]:
    if not evidence:
        raise BusinessFault("E_EVIDENCE_EMPTY", "work order has no evidence", "decay", workorder_id=workorder_id)
    half_life = rules.half_life_ms if effective_half_life_ms is None else effective_half_life_ms
    if half_life <= 0:
        raise BusinessFault(
            E_HALF_LIFE_ZERO,
            "half_life_ms must be greater than zero",
            "decay",
            workorder_id=workorder_id,
            injection_id=injection_id,
        )

    event_times = [record.event_ms for record in evidence]
    base_event_ms = min(event_times)
    scale = config.scale
    denominator = scale**3
    traces: list[dict[str, Any]] = []
    boundary_sequences: list[int] = []

    for record in sorted(evidence, key=lambda item: item.evidence_seq):
        elapsed = int(record.event_ms) - int(base_event_ms)
        coefficient_bits, exact_boundary = _binary_decay(elapsed, half_life)
        numerator = coefficient_bits * scale
        coefficient_q = round_half_up_div(numerator, 1 << DECAY_FRACTION_BITS)
        decay_residual_num = numerator - coefficient_q * (1 << DECAY_FRACTION_BITS)
        if exact_boundary:
            boundary_sequences.append(record.evidence_seq)

        weight_q = _fixed_decimal(rules.weights[record.evidence_type], config)
        role_q = _fixed_decimal(rules.role_credibility[record.source_role], config)
        confidence_q = _fixed_decimal(record.confidence, config)
        product = weight_q * role_q * confidence_q * coefficient_q
        raw_score_q = round_half_up_div(product, denominator)
        score_q = quantize_floor(raw_score_q, config.quantization_step)
        score_residual_q = raw_score_q - score_q

        traces.append(
            {
                "evidence_seq": record.evidence_seq,
                "evidence_type": record.evidence_type,
                "source_role": record.source_role,
                "event_ms": record.event_ms,
                "elapsed_ms": elapsed,
                "terms": list(record.terms),
                "confidence": record.confidence_text,
                "weight_q": weight_q,
                "role_credibility_q": role_q,
                "confidence_q": confidence_q,
                "decay_coefficient_q": coefficient_q,
                "decay_coefficient": fixed_str(coefficient_q, config.fixed_decimal_width),
                "decay_quantization_residual_num": decay_residual_num,
                "decay_quantization_denominator": 1 << DECAY_FRACTION_BITS,
                "half_life_boundary": exact_boundary,
                "raw_score_q": raw_score_q,
                "score_q": score_q,
                "score": fixed_str(score_q, config.fixed_decimal_width),
                "quantization_residual_q": score_residual_q,
                "evidence_fingerprint": record.fingerprint(),
            }
        )

    payload: dict[str, Any] = {
        "stage": "decay",
        "workorder_id": workorder_id,
        "input_fingerprint": _evidence_list_fingerprint(evidence),
        "metadata": {
            "base_event_ms": base_event_ms,
            "half_life_ms": half_life,
            "decay_algorithm": DECAY_ALGORITHM,
            "decay_fraction_bits": DECAY_FRACTION_BITS,
            "fixed_decimal_width": config.fixed_decimal_width,
            "quantization_step": config.quantization_step,
            "boundary_decay_policy": "EXACT_POWER_OF_TWO_SHIFT",
            "rounding_policy": "HALF_UP_TO_FIXED_SCALE_THEN_FLOOR_TO_QUANTIZATION_STEP",
            "exact_half_life_boundaries": boundary_sequences,
            "exact_half_life_boundary_count": len(boundary_sequences),
        },
        "scores": traces,
    }
    return seal_payload(payload)


def _evidence_list_fingerprint(evidence: Iterable[Evidence]) -> str:
    from .canonical import sha256_obj

    return sha256_obj([record.canonical() for record in sorted(evidence, key=lambda item: item.evidence_seq)])


def relation_is_applicable(relation: Relation, evidence: list[Evidence]) -> bool:
    types = [record.evidence_type for record in evidence]
    if relation.left_type == relation.right_type:
        return types.count(relation.left_type) >= 2
    return relation.left_type in types and relation.right_type in types


def applicable_relations(
    evidence: list[Evidence],
    rules: RulePackage,
    extra_relations: Iterable[Relation] = (),
) -> list[Relation]:
    result = [relation for relation in rules.relations if relation_is_applicable(relation, evidence)]
    for relation in extra_relations:
        # Injected relation faults deliberately reference a missing type.  Keep
        # such an entry visible so the relation stage can emit the business code
        # instead of silently ignoring the injected defect.
        if (
            relation_is_applicable(relation, evidence)
            or relation.left_type not in rules.weights
            or relation.right_type not in rules.weights
        ):
            result.append(relation)
    return sorted(result, key=lambda relation: relation.relation_id)


def relation_strength_q(relation: Relation, config: ArbitrationConfig) -> int:
    if relation.strength is not None:
        return _fixed_quantized(relation.strength, config)
    value = config.exclusion_strength if relation.kind == "mutual_exclusion" else config.support_strength
    return _fixed_quantized(value, config)


def relation_pair_sequences(relation: Relation, evidence: list[Evidence]) -> list[tuple[int, int]]:
    left = sorted(record.evidence_seq for record in evidence if record.evidence_type == relation.left_type)
    right = sorted(record.evidence_seq for record in evidence if record.evidence_type == relation.right_type)
    pairs: list[tuple[int, int]] = []
    if relation.left_type == relation.right_type:
        for index, first in enumerate(left):
            for second in left[index + 1 :]:
                pairs.append((first, second))
    else:
        for first in left:
            for second in right:
                pairs.append((first, second))
    return sorted(pairs)


def compute_relation_stage(
    workorder_id: str,
    evidence: list[Evidence],
    decay_payload: dict[str, Any],
    relations: list[Relation],
    rules: RulePackage,
    config: ArbitrationConfig,
    *,
    injection_ids_by_relation: dict[str, str] | None = None,
) -> dict[str, Any]:
    injection_ids_by_relation = injection_ids_by_relation or {}
    scores = {item["evidence_seq"]: int(item["score_q"]) for item in decay_payload["scores"]}
    initial_scores = dict(scores)
    zero_floor = _fixed_quantized(config.zero_lower_bound, config)
    adjustments: list[dict[str, Any]] = []
    applicable_ids: list[str] = []

    ordered_relations: list[tuple[Relation, list[tuple[int, int]]]] = []
    for relation in sorted(relations, key=lambda item: item.relation_id):
        for endpoint in (relation.left_type, relation.right_type):
            if endpoint not in rules.weights:
                raise BusinessFault(
                    E_RELATION_UNKNOWN_TYPE,
                    f"relation {relation.relation_id} references unknown evidence type {endpoint}",
                    "relations",
                    workorder_id=workorder_id,
                    relation_id=relation.relation_id,
                    injection_id=injection_ids_by_relation.get(relation.relation_id),
                )
        pairs = relation_pair_sequences(relation, evidence)
        if pairs:
            applicable_ids.append(relation.relation_id)
            ordered_relations.append((relation, pairs))

    application_order = sorted(
        ordered_relations,
        key=lambda item: (0 if item[0].kind == "mutual_exclusion" else 1, item[0].relation_id),
    )
    adjustment_index = 0
    for relation, pairs in application_order:
        strength_q = relation_strength_q(relation, config)
        for left_seq, right_seq in pairs:
            for target_seq in (left_seq, right_seq):
                before = scores[target_seq]
                if relation.kind == "mutual_exclusion":
                    requested = -strength_q
                    after = max(zero_floor, before - strength_q)
                else:
                    requested = strength_q
                    after = max(zero_floor, before + strength_q)
                applied = after - before
                scores[target_seq] = after
                adjustment_index += 1
                adjustments.append(
                    {
                        "adjustment_index": adjustment_index,
                        "relation_id": relation.relation_id,
                        "kind": relation.kind,
                        "left_seq": left_seq,
                        "right_seq": right_seq,
                        "target_seq": target_seq,
                        "requested_delta_q": requested,
                        "applied_delta_q": applied,
                        "before_q": before,
                        "after_q": after,
                        "clamped": applied != requested,
                        "injection_id": injection_ids_by_relation.get(relation.relation_id),
                    }
                )

    evidence_scores = []
    for seq in sorted(scores):
        evidence_scores.append(
            {
                "evidence_seq": seq,
                "initial_score_q": initial_scores[seq],
                "final_score_q": scores[seq],
                "score": fixed_str(scores[seq], config.fixed_decimal_width),
            }
        )
    payload: dict[str, Any] = {
        "stage": "relations",
        "workorder_id": workorder_id,
        "input_fingerprint": decay_payload["content_fingerprint"],
        "metadata": {
            "relation_algorithm": RELATION_ALGORITHM,
            "application_order": "ALL_EXCLUSIONS_THEN_ALL_SUPPORTS",
            "zero_lower_bound_q": zero_floor,
            "zero_lower_bound": fixed_str(zero_floor, config.fixed_decimal_width),
            "fixed_decimal_width": config.fixed_decimal_width,
            "quantization_step": config.quantization_step,
        },
        "referenced_relation_ids": sorted(applicable_ids),
        "evidence_scores": evidence_scores,
        "adjustments": adjustments,
    }
    return seal_payload(payload)


def compute_aggregation_stage(
    workorder_id: str,
    relation_payload: dict[str, Any],
    rules: RulePackage,
    config: ArbitrationConfig,
) -> dict[str, Any]:
    ordered = sorted(relation_payload["evidence_scores"], key=lambda item: item["evidence_seq"])
    running = 0
    steps: list[dict[str, Any]] = []
    for item in ordered:
        value = int(item["final_score_q"])
        running += value
        steps.append(
            {
                "evidence_seq": item["evidence_seq"],
                "input_score_q": value,
                "running_sum_q": running,
            }
        )
    if not steps:
        raise BusinessFault("E_EVIDENCE_EMPTY", "cannot aggregate an empty work order", "aggregation", workorder_id=workorder_id)

    if config.fusion_mode == "MEAN":
        raw_decision_q = round_half_up_div(running, len(steps))
    elif config.fusion_mode == "SUM":
        raw_decision_q = running
    else:
        raise BusinessFault("E_CONFIG_SCHEMA", "unsupported fusion mode", "aggregation", workorder_id=workorder_id)

    decision_q = quantize_floor(raw_decision_q, config.quantization_step)
    quantization_residual_q = raw_decision_q - decision_q
    threshold_qs = [
        {
            "priority": threshold.priority,
            "min_score_q": _fixed_decimal(threshold.min_score, config),
            "min_score": fixed_str(_fixed_decimal(threshold.min_score, config), config.fixed_decimal_width),
        }
        for threshold in sorted(rules.thresholds, key=lambda item: (item.min_score, item.priority))
    ]
    selected: dict[str, Any] | None = None
    matched_boundaries: list[str] = []
    for threshold in threshold_qs:
        if decision_q == threshold["min_score_q"]:
            matched_boundaries.append(threshold["priority"])
        if decision_q >= threshold["min_score_q"]:
            selected = threshold
    if selected is None:
        raise BusinessFault(
            E_THRESHOLD_INVALID,
            "decision score is below every configured priority threshold",
            "aggregation",
            workorder_id=workorder_id,
        )

    payload: dict[str, Any] = {
        "stage": "aggregation",
        "workorder_id": workorder_id,
        "input_fingerprint": relation_payload["content_fingerprint"],
        "metadata": {
            "aggregation_algorithm": AGGREGATION_ALGORITHM,
            "fusion_mode": config.fusion_mode,
            "ordered_by": "evidence_seq_ascending",
            "fixed_decimal_width": config.fixed_decimal_width,
            "quantization_step": config.quantization_step,
            "threshold_boundary_policy": "INCLUSIVE_UPPER",
        },
        "steps": steps,
        "raw_decision_q": raw_decision_q,
        "decision_score_q": decision_q,
        "decision_score": fixed_str(decision_q, config.fixed_decimal_width),
        "quantization_residual_q": quantization_residual_q,
        "thresholds": threshold_qs,
        "boundary": bool(matched_boundaries),
        "boundary_threshold_priorities": matched_boundaries,
        "matched_threshold_priority": selected["priority"] if matched_boundaries else None,
        "upgrade_priority": selected["priority"],
    }
    return seal_payload(payload)
