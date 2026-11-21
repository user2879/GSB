"""Fixed-point relation adjustments, fusion and upgrade decisions."""

from __future__ import annotations

from decimal import Decimal

from .constants import REASON_ORDER
from .model import ArbitrationConfig, Evidence, Rules
from .numeric import decimal_text, fingerprint, quantize_fixed, quantize_units
from .rules import matched_pairs


def fuse_scores(scores: list[int], fusion: str, quantum: int = 1) -> int:
    if not scores:
        return 0
    if fusion == "sum":
        return sum(scores)
    if fusion == "max":
        return max(scores)
    if fusion == "min":
        return min(scores)
    if fusion == "top2_sum":
        return sum(sorted(scores, reverse=True)[:2])
    if fusion == "mean":
        return quantize_units(Decimal(sum(scores)) / Decimal(len(scores)), quantum)
    raise ValueError(f"unknown fusion method {fusion!r}")


def _reason_sort_key(reason: str) -> tuple[int, str]:
    try:
        return (REASON_ORDER.index(reason), reason)
    except ValueError:
        if reason.startswith("R_PRIORITY_"):
            return (len(REASON_ORDER), reason)
        if reason.startswith("R_REL_"):
            return (len(REASON_ORDER) + 1, reason)
        return (len(REASON_ORDER) + 2, reason)


def resolve_priority(score_fixed: int, rules: Rules, config: ArbitrationConfig) -> tuple[int, str]:
    candidates = []
    for band in rules.priority_bands:
        minimum = quantize_fixed(band.min_score, config.scale, config.quantization_units)
        if score_fixed >= minimum:
            candidates.append((minimum, band.level, band.name))
    if not candidates:
        band = rules.priority_bands[0]
        return band.level, band.name
    _, level, name = max(candidates, key=lambda item: (item[0], -item[1], item[2]))
    return level, name


def apply_relations(
    ticket_id: str,
    evidences: tuple[Evidence, ...],
    score_result: dict,
    rules: Rules,
    config: ArbitrationConfig,
) -> dict:
    raw_by_seq = {item["evidence_seq"]: item["score_fixed"] for item in score_result["evidence_scores"]}
    adjusted = dict(raw_by_seq)
    contributions: list[dict] = []
    matched_relation_ids: set[str] = set()

    exclusion_fixed = quantize_fixed(config.exclusion_strength, config.scale, config.quantization_units)
    for relation in rules.relations:
        pairs = matched_pairs(relation, evidences)
        for left, right in pairs:
            matched_relation_ids.add(relation.relation_id)
            if relation.kind == "exclusive":
                nominal_total = -exclusion_fixed
                left_nominal = -((exclusion_fixed + 1) // 2)
                right_nominal = -(exclusion_fixed // 2)
            else:
                bonus_source = relation.bonus if relation.bonus is not None else config.support_bonus
                bonus_fixed = quantize_fixed(bonus_source, config.scale, config.quantization_units)
                nominal_total = bonus_fixed
                left_nominal = (bonus_fixed + 1) // 2
                right_nominal = bonus_fixed // 2

            left_before = adjusted[left.evidence_seq]
            right_before = adjusted[right.evidence_seq]
            left_after = max(0, left_before + left_nominal)
            right_after = max(0, right_before + right_nominal)
            adjusted[left.evidence_seq] = left_after
            adjusted[right.evidence_seq] = right_after
            left_applied = left_after - left_before
            right_applied = right_after - right_before
            contributions.append(
                {
                    "relation_id": relation.relation_id,
                    "kind": relation.kind,
                    "left_seq": left.evidence_seq,
                    "right_seq": right.evidence_seq,
                    "left_type": left.evidence_type,
                    "right_type": right.evidence_type,
                    "nominal_delta_fixed": nominal_total,
                    "applied_delta_fixed": left_applied + right_applied,
                    "left_delta_fixed": left_applied,
                    "right_delta_fixed": right_applied,
                    "left_score_before": left_before,
                    "left_score_after": left_after,
                    "right_score_before": right_before,
                    "right_score_after": right_after,
                }
            )

    contributions.sort(key=lambda item: (item["relation_id"], item["left_seq"], item["right_seq"], item["kind"]))
    adjusted_scores = [adjusted[seq] for seq in sorted(adjusted)]
    score_fixed = fuse_scores(adjusted_scores, config.fusion, config.quantization_units)

    result = {
        "ticket_id": ticket_id,
        "score_fixed": score_fixed,
        "score": decimal_text(Decimal(score_fixed) / Decimal(config.scale)),
        "precision": f"fixed_{config.fixed_decimal_places}",
        "fusion": config.fusion,
        "raw_total_fixed": score_result["raw_total_fixed"],
        "relation_delta_fixed": score_fixed - score_result["raw_total_fixed"],
        "adjusted_evidence_scores": [
            {"evidence_seq": seq, "score_fixed": adjusted[seq]} for seq in sorted(adjusted)
        ],
        "relation_contributions": contributions,
        "matched_relation_ids": sorted(matched_relation_ids),
        "evidence_scores": score_result["evidence_scores"],
        "adjusted_scores": adjusted_scores,
    }
    result["relation_fingerprint"] = fingerprint(result)
    return result


def decide_ticket(
    ticket_id: str,
    relation_result: dict,
    rules: Rules,
    config: ArbitrationConfig,
) -> dict:
    score_fixed = relation_result["score_fixed"]
    threshold_fixed = quantize_fixed(config.decision_threshold, config.scale, config.quantization_units)
    boundary = score_fixed == threshold_fixed
    if score_fixed > threshold_fixed:
        upgraded = True
    elif boundary:
        upgraded = config.threshold_equality_upgrades
    else:
        upgraded = False
    priority_level, priority_name = resolve_priority(score_fixed, rules, config)

    contributions = relation_result["relation_contributions"]
    reasons = {"R_UPGRADE" if upgraded else "R_NO_UPGRADE"}
    if boundary:
        reasons.add("R_THRESHOLD_BOUNDARY")
    if any(item["kind"] == "exclusive" and item["applied_delta_fixed"] != 0 for item in contributions):
        reasons.add("R_EXCLUSIVE_APPLIED")
    if any(item["kind"] == "support" and item["applied_delta_fixed"] != 0 for item in contributions):
        reasons.add("R_SUPPORT_APPLIED")
    for relation_id in relation_result["matched_relation_ids"]:
        reasons.add(f"R_REL_{relation_id}")
    reasons.add(f"R_PRIORITY_{priority_name}")
    reason_codes = sorted(reasons, key=_reason_sort_key)

    result = dict(relation_result)
    result.update(
        {
            "upgrade_code": 1 if upgraded else 0,
            "priority_level": priority_level,
            "priority_name": priority_name,
            "boundary": boundary,
            "reason_codes": reason_codes,
            "missing": 0,
            "threshold_fixed": threshold_fixed,
            "threshold": decimal_text(config.decision_threshold),
        }
    )
    # The relation-stage fingerprint is intentionally not replaced: conversion and
    # downstream equality can still trace the exact upstream relation product.
    result["decision_fingerprint"] = fingerprint(
        {key: value for key, value in sorted(result.items()) if key != "decision_fingerprint"}
    )
    return result
