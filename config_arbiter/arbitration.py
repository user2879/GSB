from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable

from .chain import build_chains_for_view
from .errors import BindingError
from .fixedpoint import (
    add_i64,
    comparison_error_bound,
    midpoint_trunc,
    quantize_weight,
    saturate_i64,
    scale_factor,
    sub_i64,
    tolerance_fixed,
)
from .spec import KeyPolicy, Specs, View
from .util import decimal_text

STABLE_TIE_BREAK = (
    "environment_priority_asc,environment_id_lex,source_id_lex,fixed_value_asc; "
    "weighted groups additionally order by weighted_support_desc before environment priority"
)


def _source_ref(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "environment_id": candidate["environment_id"],
        "source_id": candidate["source_id"],
        "source_type": candidate["source_type"],
    }


def _candidate_public(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "environment_id": candidate["environment_id"],
        "source_id": candidate["source_id"],
        "source_type": candidate["source_type"],
        "value_text": candidate["value_text"],
        "value_fixed": candidate["value_fixed"],
        "observed_seq": candidate["observed_seq"],
        "captured_at_ms": candidate["captured_at_ms"],
        "priority_bias": candidate["priority_bias"],
        "environment_priority_index": candidate["environment_priority_index"],
        "weight_raw": candidate["weight_raw"],
        "weight_fixed": candidate["weight_fixed"],
        "quantization_error_fixed_units": candidate["quantization_error_fixed_units"],
        "quantization_error_upper_fixed_units": candidate["quantization_error_upper_fixed_units"],
        "saturated": candidate["saturated"],
    }


def _stable_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["environment_priority_index"],
        candidate["environment_id"],
        candidate["source_id"],
        candidate["value_fixed"],
    )


def _sorted_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=_stable_key)


def _sum_i64(values: Iterable[int]) -> int:
    total = 0
    for value in values:
        total = add_i64(total, value)
    return total


def _candidate_map(
    specs: Specs,
    view: View,
    key: str,
    chains: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    priority_index = {environment_id: index for index, environment_id in enumerate(view.environment_priority)}
    chain_by_environment = {chain["environment_id"]: chain for chain in chains}
    candidates: list[dict[str, Any]] = []
    for environment_id in view.environment_priority:
        chain = chain_by_environment[environment_id]
        effective = chain["effective"]
        if effective is None:
            continue
        source = specs.environments[environment_id].sources[effective["source_id"]]
        candidates.append(
            {
                "environment_id": environment_id,
                "environment_priority_index": priority_index[environment_id],
                "environment_depth": specs.environments[environment_id].depth,
                "source_id": effective["source_id"],
                "source_type": effective["source_type"],
                "value_text": effective["value_text"],
                "value_fixed": effective["value_fixed"],
                "observed_seq": effective["observed_seq"],
                "captured_at_ms": effective["captured_at_ms"],
                "priority_bias": source.priority_bias,
                "quantization_error_fixed_units": effective["quantization_error_fixed_units"],
                "quantization_error_upper_fixed_units": effective["quantization_error_upper_fixed_units"],
                "saturated": effective["saturated"],
                "weight_raw": view.environment_weights[environment_id],
                "weight_fixed": quantize_weight(view.environment_weights[environment_id], specs.session.scale_power),
            }
        )
    return _sorted_candidates(candidates)


def _error_pairs(
    candidates: list[dict[str, Any]],
    nominal: dict[str, Any],
    bound: int,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for other in _sorted_candidates(candidates):
        if other is nominal:
            continue
        difference = abs(other["value_fixed"] - nominal["value_fixed"])
        if difference <= bound:
            pairs.append(
                {
                    "left": _source_ref(nominal),
                    "right": _source_ref(other),
                    "left_fixed": nominal["value_fixed"],
                    "right_fixed": other["value_fixed"],
                    "fixed_difference": difference,
                    "bound": bound,
                }
            )
    return pairs


def _base_result(
    *,
    specs: Specs,
    view: View,
    key: str,
    policy: KeyPolicy,
    candidates: list[dict[str, Any]],
    chains: list[dict[str, Any]],
    comparison_bound: int,
    tolerance: int,
    max_error: int,
) -> dict[str, Any]:
    return {
        "key": key,
        "strategy": policy.strategy,
        "policy": {"strategy": policy.strategy, "tolerance": policy.tolerance},
        "scale_policy_version": specs.session.scale_policy_version,
        "arbitration_policy_version": specs.session.arbitration_policy_version,
        "error_model": {
            "scale_power": specs.session.scale_power,
            "scale_factor": scale_factor(specs.session.scale_power),
            "scale_policy_version": specs.session.scale_policy_version,
            "truncation": specs.session.truncation,
            "overflow": specs.session.overflow,
            "tolerance_fixed_units": tolerance,
            "single_key_quantization_errors": [
                {
                    "environment_id": candidate["environment_id"],
                    "source_id": candidate["source_id"],
                    **candidate["quantization_error_fixed_units"],
                    "upper_fixed_units": candidate["quantization_error_upper_fixed_units"],
                    "saturated": candidate["saturated"],
                }
                for candidate in candidates
            ],
            "single_key_quantization_error_upper_fixed_units": max_error,
            "comparison_error_bound_fixed_units": comparison_bound,
            "comparison_error_formula": "tolerance_fixed + 2 * max_single_candidate_quantization_error_upper",
        },
        "chains": chains,
        "candidate_count": len(candidates),
        "stable_tie_break_rule": STABLE_TIE_BREAK,
    }


def _latest_result(
    base: dict[str, Any],
    candidates: list[dict[str, Any]],
    comparison_bound: int,
) -> dict[str, Any]:
    nominal = sorted(
        candidates,
        key=lambda candidate: (
            -candidate["captured_at_ms"],
            -candidate["observed_seq"],
            candidate["environment_priority_index"],
            candidate["environment_id"],
            candidate["source_id"],
        ),
    )[0]
    near = [candidate for candidate in candidates if abs(candidate["value_fixed"] - nominal["value_fixed"]) <= comparison_bound]
    final = min(near, key=_stable_key) if len(near) > 1 else nominal
    error_influenced = len(near) > 1
    rejected: list[dict[str, Any]] = []
    for candidate in _sorted_candidates(candidates):
        if candidate is final:
            continue
        near_candidate = candidate in near
        if candidate is nominal and final is not nominal:
            reason = "error_band_overridden_by_stable_tie_break"
        elif near_candidate:
            reason = "error_band_lost_by_stable_tie_break"
        else:
            reason = "older_observation"
        rejected.append(
            {
                **_candidate_public(candidate),
                "reason": reason,
                "detail": {
                    "nominal_environment_id": nominal["environment_id"],
                    "nominal_source_id": nominal["source_id"],
                    "difference_to_nominal_fixed": abs(candidate["value_fixed"] - nominal["value_fixed"]),
                    "error_band": comparison_bound,
                    "inside_error_band": near_candidate,
                },
            }
        )
    result = dict(base)
    result.update(
        {
            "authoritative": {
                **{
                    field: final[field]
                    for field in (
                        "environment_id",
                        "source_id",
                        "source_type",
                        "value_text",
                        "value_fixed",
                        "observed_seq",
                        "captured_at_ms",
                    )
                },
                "origin": "latest_observation",
            },
            "accepted": [
                {
                    **_candidate_public(final),
                    "decision": "accepted_latest_observation",
                }
            ],
            "rejected": rejected,
            "error_influenced": error_influenced,
            "error_attribution": {
                "nominal_choice": _source_ref(nominal),
                "final_choice": _source_ref(final),
                "nominal_value_fixed": nominal["value_fixed"],
                "final_value_fixed": final["value_fixed"],
                "candidates_in_error_band": [_source_ref(candidate) for candidate in near],
                "error_band_pairs": _error_pairs(candidates, nominal, comparison_bound),
                "stable_tie_break_applied": error_influenced,
            },
            "key_aggregation_error_upper_fixed_units": max(
                candidate["quantization_error_upper_fixed_units"] for candidate in candidates
            ),
            "aggregation_error_formula": "selected_candidate_quantization_error_upper",
        }
    )
    return result


def _weighted_result(
    base: dict[str, Any],
    candidates: list[dict[str, Any]],
    comparison_bound: int,
) -> dict[str, Any]:
    groups_by_value: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        groups_by_value.setdefault(candidate["value_fixed"], []).append(candidate)
    groups: list[dict[str, Any]] = []
    for value_fixed, members in groups_by_value.items():
        members = _sorted_candidates(members)
        total_weight = 0
        for member in members:
            total_weight = add_i64(total_weight, member["weight_fixed"])
        groups.append(
            {
                "value_fixed": value_fixed,
                "members": members,
                "representative": members[0],
                "total_weight_fixed": total_weight,
            }
        )

    def group_rank(group: dict[str, Any]) -> tuple[Any, ...]:
        return (-group["total_weight_fixed"], _stable_key(group["representative"]))

    groups.sort(key=group_rank)
    nominal = groups[0]
    near_groups = [group for group in groups if abs(group["value_fixed"] - nominal["value_fixed"]) <= comparison_bound]
    final = min(near_groups, key=group_rank) if len(near_groups) > 1 else nominal
    error_influenced = len(near_groups) > 1
    accepted_members = _sorted_candidates(final["members"])
    accepted_ids = {(member["environment_id"], member["source_id"]) for member in accepted_members}
    rejected: list[dict[str, Any]] = []
    for candidate in _sorted_candidates(candidates):
        if (candidate["environment_id"], candidate["source_id"]) in accepted_ids:
            continue
        candidate_group = next(group for group in groups if group["value_fixed"] == candidate["value_fixed"])
        inside_band = candidate_group in near_groups
        rejected.append(
            {
                **_candidate_public(candidate),
                "reason": "error_band_lost_by_stable_weighted_tie_break"
                if inside_band
                else "weighted_support_lower",
                "detail": {
                    "group_weight_fixed": candidate_group["total_weight_fixed"],
                    "winning_group_weight_fixed": final["total_weight_fixed"],
                    "difference_to_nominal_fixed": abs(candidate["value_fixed"] - nominal["value_fixed"]),
                    "error_band": comparison_bound,
                    "inside_error_band": inside_band,
                },
            }
        )
    representative = final["representative"]
    result = dict(base)
    result.update(
        {
            "authoritative": {
                **{
                    field: representative[field]
                    for field in (
                        "environment_id",
                        "source_id",
                        "source_type",
                        "value_text",
                        "value_fixed",
                        "observed_seq",
                        "captured_at_ms",
                    )
                },
                "origin": "weighted_group_representative",
            },
            "accepted": [
                {
                    **_candidate_public(member),
                    "decision": "accepted_majority_value",
                }
                for member in accepted_members
            ],
            "rejected": rejected,
            "error_influenced": error_influenced,
            "weighted_support": {
                "accepted_weight_fixed": final["total_weight_fixed"],
                "all_candidate_weight_fixed": _sum_i64(group["total_weight_fixed"] for group in groups),
                "nominal_group_value_fixed": nominal["value_fixed"],
                "nominal_group_weight_fixed": nominal["total_weight_fixed"],
                "final_group_value_fixed": final["value_fixed"],
                "final_group_weight_fixed": final["total_weight_fixed"],
            },
            "error_attribution": {
                "nominal_choice": {
                    "value_fixed": nominal["value_fixed"],
                    "sources": [_source_ref(member) for member in nominal["members"]],
                },
                "final_choice": {
                    "value_fixed": final["value_fixed"],
                    "sources": [_source_ref(member) for member in final["members"]],
                },
                "groups_in_error_band": [
                    {
                        "value_fixed": group["value_fixed"],
                        "weight_fixed": group["total_weight_fixed"],
                        "sources": [_source_ref(member) for member in group["members"]],
                    }
                    for group in near_groups
                ],
                "stable_tie_break_applied": error_influenced,
            },
            "key_aggregation_error_upper_fixed_units": max(
                candidate["quantization_error_upper_fixed_units"] for candidate in candidates
            ),
            "aggregation_error_formula": "selected_candidate_quantization_error_upper",
        }
    )
    return result


def _interval_result(
    base: dict[str, Any],
    candidates: list[dict[str, Any]],
    comparison_bound: int,
    tolerance: int,
    max_error: int,
) -> dict[str, Any]:
    intervals: list[dict[str, Any]] = []
    for candidate in candidates:
        radius = add_i64(tolerance, candidate["quantization_error_upper_fixed_units"])
        intervals.append(
            {
                "candidate": candidate,
                "lo": sub_i64(candidate["value_fixed"], radius),
                "hi": add_i64(candidate["value_fixed"], radius),
            }
        )
    intervals.sort(key=lambda interval: (_stable_key(interval["candidate"]), interval["lo"], interval["hi"]))
    full_lo = max(interval["lo"] for interval in intervals)
    full_hi = min(interval["hi"] for interval in intervals)
    conflict = full_lo > full_hi
    if not conflict:
        accepted_intervals = intervals
        intersection_lo = full_lo
        intersection_hi = full_hi
    else:
        endpoints = sorted({interval["lo"] for interval in intervals} | {interval["hi"] for interval in intervals})
        clusters: list[dict[str, Any]] = []
        for point in endpoints:
            members = [interval for interval in intervals if interval["lo"] <= point <= interval["hi"]]
            if not members:
                continue
            lo = max(interval["lo"] for interval in members)
            hi = min(interval["hi"] for interval in members)
            clusters.append(
                {
                    "members": members,
                    "lo": lo,
                    "hi": hi,
                    "span": hi - lo,
                    "stable": min(_stable_key(interval["candidate"]) for interval in members),
                }
            )
        clusters.sort(key=lambda cluster: (-len(cluster["members"]), cluster["span"], cluster["stable"]))
        best = clusters[0]
        accepted_intervals = best["members"]
        intersection_lo = best["lo"]
        intersection_hi = best["hi"]

    midpoint = midpoint_trunc(intersection_lo, intersection_hi)
    accepted_candidates = _sorted_candidates(interval["candidate"] for interval in accepted_intervals)
    accepted_ids = {(candidate["environment_id"], candidate["source_id"]) for candidate in accepted_candidates}
    rejected: list[dict[str, Any]] = []
    for candidate in _sorted_candidates(candidates):
        if (candidate["environment_id"], candidate["source_id"]) in accepted_ids:
            continue
        rejected.append(
            {
                **_candidate_public(candidate),
                "reason": "interval_disjoint_from_authoritative_cluster",
                "detail": {
                    "authoritative_interval_lo": intersection_lo,
                    "authoritative_interval_hi": intersection_hi,
                },
            }
        )
    near_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            difference = abs(left["value_fixed"] - right["value_fixed"])
            if difference <= comparison_bound:
                near_pairs.append(
                    {
                        "left": _source_ref(left),
                        "right": _source_ref(right),
                        "left_fixed": left["value_fixed"],
                        "right_fixed": right["value_fixed"],
                        "fixed_difference": difference,
                        "bound": comparison_bound,
                    }
                )
    error_influenced = bool(near_pairs)
    representative = min(accepted_candidates, key=_stable_key)
    result = dict(base)
    result.update(
        {
            "authoritative": {
                "environment_id": None,
                "source_id": None,
                "source_type": None,
                "value_text": decimal_text(Decimal(midpoint) / Decimal(base["error_model"]["scale_factor"])),
                "value_fixed": midpoint,
                "observed_seq": None,
                "captured_at_ms": None,
                "origin": "interval_intersection_midpoint",
            },
            "authoritative_interval": {
                "lo_fixed": intersection_lo,
                "hi_fixed": intersection_hi,
                "midpoint_truncation": "toward_zero",
                "all_intervals_intersect": not conflict,
            },
            "accepted": [
                {
                    **_candidate_public(candidate),
                    "decision": "accepted_interval_overlap",
                }
                for candidate in accepted_candidates
            ],
            "rejected": rejected,
            "error_influenced": error_influenced,
            "error_attribution": {
                "nominal_choice": _source_ref(representative),
                "final_choice": {
                    "origin": "interval_intersection_midpoint",
                    "value_fixed": midpoint,
                },
                "interval_conflict": conflict,
                "accepted_sources": [_source_ref(candidate) for candidate in accepted_candidates],
                "error_band_pairs": near_pairs,
                "stable_tie_break_applied": error_influenced or conflict,
            },
            "key_aggregation_error_upper_fixed_units": saturate_i64(tolerance + max_error + 1),
            "aggregation_error_formula": "tolerance_fixed + max_quantization_error_upper + midpoint_truncation_allowance(1)",
        }
    )
    return result


def arbitrate_key(specs: Specs, view: View, key: str) -> dict[str, Any]:
    policy = view.key_policies[key]
    chains = build_chains_for_view(specs, view.view_id, key)
    candidates = _candidate_map(specs, view, key, chains)
    if not candidates:
        raise BindingError(
            "no environment has an effective value satisfying the key binding",
            details={"view_id": view.view_id, "key": key, "chains": chains},
        )
    max_error = max(candidate["quantization_error_upper_fixed_units"] for candidate in candidates)
    tolerance = tolerance_fixed(policy.tolerance, specs.session.scale_power)
    comparison_bound = comparison_error_bound(policy.tolerance, specs.session.scale_power, max_error)
    base = _base_result(
        specs=specs,
        view=view,
        key=key,
        policy=policy,
        candidates=candidates,
        chains=chains,
        comparison_bound=comparison_bound,
        tolerance=tolerance,
        max_error=max_error,
    )
    if policy.strategy == "latest":
        return _latest_result(base, candidates, comparison_bound)
    if policy.strategy == "weighted_majority":
        return _weighted_result(base, candidates, comparison_bound)
    return _interval_result(base, candidates, comparison_bound, tolerance, max_error)
