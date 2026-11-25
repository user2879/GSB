"""Boundary diagnostics for reservoir and global truncation decisions."""

from __future__ import annotations

from typing import Any

from .reservoir import RANDOM_SPACE, quantization_step, quantized_weight


def _fixed_digits(quantization: dict[str, Any]) -> int:
    decimals = int(quantization.get("decimals", 3))
    if str(quantization.get("tier", "fine")) == "coarse":
        return max(1, decimals - 1)
    return decimals


def _round(value: float, digits: int) -> float:
    return float(f"{value:.{digits}f}")


def _classify_flip(
    *,
    exact_random: float,
    fixed_random: float,
    exact_threshold: float,
    fixed_threshold: float,
) -> str:
    random_side_flips = (fixed_random < exact_threshold) != (
        exact_random < exact_threshold
    )
    ordinal_side_flips = (exact_random < fixed_threshold) != (
        exact_random < exact_threshold
    )
    if random_side_flips and ordinal_side_flips:
        return "MIXED"
    if random_side_flips:
        return "RANDOM_SOURCE_DIGEST"
    if ordinal_side_flips:
        return "ORDINAL_CALCULATION"
    return "ROUNDING_EDGE"


def analyze_reservoir_admission(event: dict[str, Any], quantization: dict[str, Any]) -> dict[str, Any]:
    ordinal = int(event["arrival_ordinal"])
    threshold_num = int(event["threshold_numerator"])
    threshold_den = int(event["threshold_denominator"])
    random_numerator = int(event["random_numerator_hex"], 16)
    exact_admitted = random_numerator * threshold_den < threshold_num * RANDOM_SPACE
    unit = float(event["random_unit"])
    threshold = threshold_num / threshold_den
    digits = _fixed_digits(quantization)
    fixed_unit = _round(unit, digits)
    fixed_threshold = _round(threshold, digits)
    fixed_admitted = fixed_unit < fixed_threshold
    coarse = {
        "decimals": int(quantization.get("decimals", 3)),
        "tier": "coarse",
    }
    fine = {
        "decimals": int(quantization.get("decimals", 3)),
        "tier": "fine",
    }
    coarse_digits = _fixed_digits(coarse)
    fine_digits = _fixed_digits(fine)
    coarse_admitted = _round(unit, coarse_digits) < _round(threshold, coarse_digits)
    fine_admitted = _round(unit, fine_digits) < _round(threshold, fine_digits)
    half_step = 0.5 * (10 ** -digits)
    distance = abs(unit - threshold)
    if distance > half_step:
        return {}

    flip = exact_admitted != fixed_admitted
    flip_source = "NONE"
    if flip:
        flip_source = _classify_flip(
            exact_random=unit,
            fixed_random=fixed_unit,
            exact_threshold=threshold,
            fixed_threshold=fixed_threshold,
        )
    if flip_source == "ROUNDING_EDGE":
        flip_source = "MIXED"
    if fine_admitted != coarse_admitted:
        tier_source = _classify_flip(
            exact_random=unit,
            fixed_random=_round(unit, coarse_digits),
            exact_threshold=threshold,
            fixed_threshold=_round(threshold, fine_digits),
        )
        if flip_source == "NONE":
            flip_source = tier_source
    if flip_source == "ORDINAL_CALCULATION":
        code = "E_D_FLIP_ORDINAL"
    elif flip_source == "RANDOM_SOURCE_DIGEST":
        code = "E_D_FLIP_RANDOM_SOURCE"
    elif flip_source == "MIXED":
        code = "E_D_FLIP_MIXED"
    else:
        code = "D_RANDOM_BOUNDARY_STABLE"
    if exact_admitted == fixed_admitted:
        direction = "stable"
    elif fixed_admitted:
        direction = "fixed_admits_reference_rejects"
    else:
        direction = "fixed_rejects_reference_admits"
    return {
        "type": "reservoir_boundary",
        "log_id": event["log_id"],
        "bucket_id": event.get("bucket_id"),
        "arrival_ordinal": ordinal,
        "threshold": threshold,
        "random_unit": unit,
        "distance": distance,
        "half_step": half_step,
        "reference_decision": "admit" if exact_admitted else "reject",
        "fixed_decision": "admit" if fixed_admitted else "reject",
        "fine_decision": "admit" if fine_admitted else "reject",
        "coarse_decision": "admit" if coarse_admitted else "reject",
        "tier_flip": fine_admitted != coarse_admitted,
        "difference_direction": direction,
        "flip_source": flip_source,
        "business_code": code,
        "random_digest": event["random_digest"],
    }


def _nullable_record(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "log_id": value["log_id"],
        "bucket_id": value["bucket_id"],
        "rank": value["global_rank"],
        "priority": value["priority"],
        "arrival_ordinal": value["arrival_ordinal"],
        "raw_weight": float(value["metric"]) * 1000.0,
    }


def analyze_truncation_boundary(
    left: dict[str, Any],
    right: dict[str, Any],
    quantization: dict[str, Any],
) -> dict[str, Any]:
    step = quantization_step(quantization)
    half_step = step / 2
    left_raw = float(left["metric"]) * 1000.0
    right_raw = float(right["metric"]) * 1000.0
    fine = {"decimals": int(quantization.get("decimals", 3)), "tier": "fine"}
    coarse = {"decimals": int(quantization.get("decimals", 3)), "tier": "coarse"}
    left_fine = quantized_weight(left["metric"], fine)
    right_fine = quantized_weight(right["metric"], fine)
    left_coarse = quantized_weight(left["metric"], coarse)
    right_coarse = quantized_weight(right["metric"], coarse)
    raw_delta = abs(left_raw - right_raw)
    if raw_delta > half_step:
        return {}
    reference_retains_left = left_raw >= right_raw
    fine_retains_left = left_fine >= right_fine
    coarse_retains_left = left_coarse >= right_coarse
    tier_flip = fine_retains_left != coarse_retains_left
    actual_tie = (
        int(left["priority"]) == int(right["priority"])
        and int(left["arrival_ordinal"]) == int(right["arrival_ordinal"])
    )
    if tier_flip:
        code = "E_D_FLIP_TRUNCATION_TIE"
        flip_source = "TRUNCATION_TIE"
    else:
        code = "D_TRUNCATION_BOUNDARY_STABLE"
        flip_source = "NONE"
    if reference_retains_left == fine_retains_left:
        direction = "stable"
    elif fine_retains_left:
        direction = "fixed_retains_right_reference_retains_left"
    else:
        direction = "fixed_retains_left_reference_retains_right"
    return {
        "type": "truncation_boundary",
        "cut_rank": int(left["global_rank"]),
        "left": _nullable_record(left),
        "right": _nullable_record(right),
        "left_coarse_weight": left_coarse,
        "right_coarse_weight": right_coarse,
        "right_fine_weight": right_fine,
        "left_fine_weight": left_fine,
        "raw_weight_delta": raw_delta,
        "half_quantum": half_step,
        "reference_decision": "retain_left" if reference_retains_left else "retain_right",
        "fine_decision": "retain_left" if fine_retains_left else "retain_right",
        "coarse_decision": "retain_left" if coarse_retains_left else "retain_right",
        "actual_rank_decision": "retain_left_by_priority_ordinal_log_id",
        "priority_ordinal_tie": actual_tie,
        "tier_flip": tier_flip,
        "difference_direction": direction,
        "flip_source": flip_source,
        "business_code": code,
    }


def compute_diagnostics(
    buckets: dict[str, dict[str, Any]],
    truncation: dict[str, Any],
    quantization: dict[str, Any],
) -> dict[str, Any]:
    reservoir_boundaries: list[dict[str, Any]] = []
    for bucket_id, bucket in sorted(buckets.items()):
        for event in bucket["history"]:
            if int(event["arrival_ordinal"]) <= int(bucket["capacity"]):
                continue
            item = analyze_reservoir_admission(
                {**event, "bucket_id": bucket_id}, quantization
            )
            if item:
                reservoir_boundaries.append(item)

    truncation_boundaries: list[dict[str, Any]] = []
    budget = int(truncation["global_budget"])
    ranked = truncation["ranked_candidates"]
    if 0 < budget < len(ranked):
        left = ranked[budget - 1]
        right = ranked[budget]
        item = analyze_truncation_boundary(left, right, quantization)
        if item:
            truncation_boundaries.append(item)
    tier_flips = [
        item
        for item in [*reservoir_boundaries, *truncation_boundaries]
        if item.get("tier_flip")
    ]
    return {
        "schema": "lds-diagnostics-v1",
        "quantization": quantization,
        "reservoir_boundaries": reservoir_boundaries,
        "truncation_boundaries": truncation_boundaries,
        "tier_flips": tier_flips,
        "summary": {
            "reservoir_boundary_count": len(reservoir_boundaries),
            "truncation_boundary_count": len(truncation_boundaries),
            "tier_flip_count": len(tier_flips),
            "business_codes": sorted(
                {
                    item["business_code"]
                    for item in [*reservoir_boundaries, *truncation_boundaries]
                }
            ),
        },
    }
