"""Deterministic hotspot ranking and fixed-point error annotations."""

from __future__ import annotations

import math
from typing import Any

from .aggregate import AggregationResult
from .canonical import canonical_json_bytes
from .models import ProfilePolicy

RANKING_RULE = "self_time_ms desc, cumulative_time_ms desc, frame lexicographic asc"
QUANTIZATION_RULE = "round-half-up using integer arithmetic"


def _reduce(numerator: int, denominator: int) -> tuple[int, int]:
    if numerator == 0:
        return 0, 1
    divisor = math.gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor


def _ratio_error(q: int, value: int, total: int, scale: int) -> tuple[int, int]:
    if total == 0:
        return 0, 1
    return _reduce(abs(q * total - value * scale), scale * total)


def _round_half_up_ratio(value: int, total: int, scale: int) -> int:
    if total == 0:
        return 0
    return (2 * value * scale + total) // (2 * total)


def build_ranking(result: AggregationResult, policy: ProfilePolicy) -> dict[str, Any]:
    total = result.total_time_ms
    contexts = result.context_map()
    ranked = sorted(
        result.frame_stats,
        key=lambda stat: (-stat.self_time_ms, -stat.cumulative_time_ms, stat.frame),
    )
    top_count = min(policy.top_k, len(ranked))
    top = ranked[:top_count]
    all_quantized = [
        _round_half_up_ratio(stat.self_time_ms, total, policy.scale) for stat in ranked
    ]
    top_quantized = [
        _round_half_up_ratio(stat.self_time_ms, total, policy.scale) for stat in top
    ]

    tied_by_index: dict[int, list[int]] = {index: [] for index in range(top_count)}
    for left in range(top_count):
        for right in range(left + 1, top_count):
            if abs(top_quantized[left] - top_quantized[right]) <= 1:
                tied_by_index[left].append(right)
                tied_by_index[right].append(left)

    boundary_tie = (
        top_count < len(ranked) and abs(all_quantized[top_count - 1] - all_quantized[top_count]) <= 1
    )

    entries: list[dict[str, Any]] = []
    for index, stat in enumerate(top):
        q = top_quantized[index]
        err_num, err_den = _ratio_error(q, stat.self_time_ms, total, policy.scale)
        relation_context = contexts.get(stat.frame, {"callers": (), "callees": ()})
        ties = sorted(
            (
                {
                    "frame": top[other].frame,
                    "fixed_ratio_delta": abs(q - top_quantized[other]),
                    "combined_error_bound_units_numerator": 1,
                    "combined_error_bound_units_denominator": 1,
                    "order_decision": "stable exact tie-break used; fixed-point intervals overlap",
                }
                for other in tied_by_index[index]
            ),
            key=lambda item: item["frame"],
        )
        entries.append(
            {
                "rank": index + 1,
                "frame": stat.frame,
                "self_time_ms": stat.self_time_ms,
                "cumulative_time_ms": stat.cumulative_time_ms,
                "participant_weight": stat.participant_weight,
                "occurrence_count": stat.occurrence_count,
                "self_ratio_fixed": q,
                "self_ratio_exact_numerator": stat.self_time_ms,
                "self_ratio_exact_denominator": total if total else 1,
                "quantization_error": {"numerator": err_num, "denominator": err_den},
                "quantization_order_error_bound_dominated": bool(ties)
                or (boundary_tie and index == top_count - 1),
                "error_band_ties": ties,
                "callers": [item.to_dict() for item in relation_context["callers"]],
                "callees": [item.to_dict() for item in relation_context["callees"]],
            }
        )

    boundary: dict[str, Any]
    if top_count and top_count < len(ranked):
        boundary = {
            "last_selected_frame": top[-1].frame,
            "first_excluded_frame": ranked[top_count].frame,
            "last_selected_self_time_ms": top[-1].self_time_ms,
            "first_excluded_self_time_ms": ranked[top_count].self_time_ms,
            "last_selected_self_ratio_fixed": all_quantized[top_count - 1],
            "first_excluded_self_ratio_fixed": all_quantized[top_count],
            "quantization_order_error_bound_dominated_with_first_excluded": boundary_tie,
            "excluded_is_not_better": (
                -ranked[top_count].self_time_ms,
                -ranked[top_count].cumulative_time_ms,
                ranked[top_count].frame,
            )
            > (-top[-1].self_time_ms, -top[-1].cumulative_time_ms, top[-1].frame),
        }
    else:
        boundary = {
            "last_selected_frame": top[-1].frame if top else None,
            "first_excluded_frame": None,
            "excluded_is_not_better": True,
        }

    top_q_sum = sum(top_quantized)
    top_self_sum = sum(stat.self_time_ms for stat in top)
    aggregate_error_num, aggregate_error_den = _ratio_error(
        top_q_sum, top_self_sum, total, policy.scale
    )
    bound_num, bound_den = _reduce(top_count, 2 * policy.scale)
    ambiguous_pair_keys = {
        tuple(sorted((entries[left]["frame"], entries[right]["frame"])))
        for left in range(top_count)
        for right in tied_by_index[left]
        if left < right
    }
    if boundary_tie:
        ambiguous_pair_keys.add(tuple(sorted((top[-1].frame, ranked[top_count].frame))))

    return {
        "metric_version": policy.metric_version,
        "ranking_rule": RANKING_RULE,
        "requested_k": policy.top_k,
        "reported_k": top_count,
        "ranking": entries,
        "error_annotation": {
            "fixed_point_scale": policy.scale,
            "quantization_rule": QUANTIZATION_RULE,
            "one_fixed_point_unit": {"numerator": 1, "denominator": policy.scale},
            "single_frame_max_quantization_error": {
                "numerator": 1,
                "denominator": 2 * policy.scale,
                "explanation": "absolute error is at most half a fixed-point unit",
            },
            "combined_pair_error_bound": {
                "numerator": 1,
                "denominator": policy.scale,
                "fixed_point_units": 1,
                "explanation": "two half-unit errors; fixed values within one unit have overlapping intervals",
            },
            "stable_tie_break": RANKING_RULE,
            "error_bound_dominated_pair_count": len(ambiguous_pair_keys),
            "error_bound_dominated_pairs": [
                {"frames": list(pair)} for pair in sorted(ambiguous_pair_keys)
            ],
            "top_k_contains_true_top_k": True,
            "top_k_membership_basis": (
                "ranking uses exact unbounded integer self_time_ms, cumulative_time_ms and "
                "frame name; fixed-point ratios are presentation-only, so the reported set "
                "is exactly the true comparator top-k set"
            ),
            "top_k_boundary_check": boundary,
            "top_k_quantized_ratio_sum": top_q_sum,
            "top_k_quantization_error": {
                "numerator": aggregate_error_num,
                "denominator": aggregate_error_den,
            },
            "top_k_total_ratio_error_upper_bound": {
                "numerator": bound_num,
                "denominator": bound_den,
                "source_numerator": top_count,
                "source_denominator": 2 * policy.scale,
                "explanation": "sum of k half-unit per-frame bounds",
            },
        },
    }


def canonical_core_bytes(result: AggregationResult, ranking_report: dict[str, Any]) -> bytes:
    """Canonical bytes containing only path-independent conclusions."""

    payload = {
        "metric_version": ranking_report["metric_version"],
        "frame_stats": [stat.to_dict() for stat in result.frame_stats],
        "total_weight": result.total_weight,
        "total_time_ms": result.total_time_ms,
        "ranking": ranking_report["ranking"],
        "error_annotation": ranking_report["error_annotation"],
    }
    return canonical_json_bytes(payload)
