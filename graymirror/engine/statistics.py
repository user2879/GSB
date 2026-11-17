"""Count aggregation and level by level fixed point error propagation.

All math uses :class:`fractions.Fraction`; no floating point touches the bounds.
The only ordering used is explicit deterministic sorting, so permuting the
request stream never changes the reported bounds or contribution ranking.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from .fixedpoint import GRID_BASIS
from .router import GRAY, STABLE, MirrorRecord, QuantizedRule, RouteDecision

# A contribution is "primary" only when it owns at least this share of the total
# bound; lower shares are reported as minor. The threshold is explicit and ties
# break on (rule_id, level), making the verdict independent of request order.
CONTRIBUTION_THRESHOLD = Fraction(1, 100)  # 1%


def _frac(numerator: int, denominator: int) -> Fraction:
    return Fraction(numerator, denominator) if denominator else Fraction(0)


def _per_10000(frac: Fraction) -> Fraction:
    return frac * GRID_BASIS


def compute_statistics(
    rules: list,
    quantized: dict[str, QuantizedRule],
    decisions: list[RouteDecision],
    mirrors: list[MirrorRecord],
) -> dict[str, Any]:
    per_rule: dict[str, dict[str, Any]] = {
        rule.rule_id: {
            "rule_id": rule.rule_id, "scale": quantized[rule.rule_id].scale,
            "total": 0, STABLE: 0, GRAY: 0,
            "main_success": 0, "main_failed": 0,
            "mirror_candidates": 0, "mirrored": 0,
            "mirror_success": 0, "mirror_timeout": 0,
        }
        for rule in rules
    }
    for decision in decisions:
        bucket = per_rule[decision.rule_id]
        bucket["total"] += 1
        bucket[decision.channel] += 1
        if decision.status == "SUCCESS":
            bucket["main_success"] += 1
        else:
            bucket["main_failed"] += 1
    for mirror in mirrors:
        bucket = per_rule[mirror.rule_id]
        bucket["mirrored"] += 1
        if mirror.status == "SUCCESS":
            bucket["mirror_success"] += 1
        else:
            bucket["mirror_timeout"] += 1

    for rule in rules:
        qrule = quantized[rule.rule_id]
        bucket = per_rule[rule.rule_id]
        total = bucket["total"]
        bucket["mirror_candidates"] = total
        bucket["gray_share"] = _frac(bucket[GRAY], total)
        bucket["stable_share"] = _frac(bucket[STABLE], total)
        bucket["main_success_rate"] = _frac(bucket["main_success"], total)
        bucket["mirror_share"] = _frac(bucket["mirrored"], total)
        bucket["mirror_success_rate"] = _frac(
            bucket["mirror_success"], bucket["mirrored"])
        bucket["bounds"] = _rule_error_bounds(qrule, total)
    return {"per_rule": per_rule, "global": _global_bounds(rules, per_rule)}


def _rule_error_bounds(qrule: QuantizedRule, total: int) -> dict[str, Any]:
    # Level 1, routing: floor truncation moves the boundary by at most one
    # fixed point unit 1/S, so either channel share shifts by at most 1/S.
    route_fraction = Fraction(1, qrule.scale)
    # Level 2, mirror sampling: same absolute grid step.
    mirror_abs_fraction = Fraction(1, qrule.scale)
    mirror_rate = qrule.mirror
    if mirror_rate.raw in (0, GRID_BASIS):
        # 0 stays exactly 0 and the full scale quantizes exactly under floor.
        mirror_relative = Fraction(0)
    else:
        mirror_relative = Fraction(GRID_BASIS,
                                   mirror_rate.raw * qrule.scale)
    # Level 3, success rate aggregation: only the boundary request can flip,
    # hence at most 1/n of the rule's requests.
    sr_fraction = Fraction(1, total) if total else Fraction(0)
    e2e_fraction = route_fraction + mirror_abs_fraction
    return {
        "route_share_bound_fraction": str(route_fraction),
        "route_share_bound_per_10000": str(_per_10000(route_fraction)),
        "mirror_abs_bound_fraction": str(mirror_abs_fraction),
        "mirror_abs_bound_per_10000": str(_per_10000(mirror_abs_fraction)),
        "mirror_relative_deviation_bound": str(mirror_relative),
        "success_rate_bound_fraction": str(sr_fraction),
        "success_rate_bound_per_10000": str(_per_10000(sr_fraction)),
        "end_to_end_bound_fraction": str(e2e_fraction),
        "end_to_end_bound_per_10000": str(_per_10000(e2e_fraction)),
    }


def _global_bounds(rules: list, per_rule: dict[str, dict]) -> dict[str, Any]:
    total = sum(bucket["total"] for bucket in per_rule.values())
    total_main_success = sum(bucket["main_success"] for bucket in per_rule.values())
    total_mirrored = sum(bucket["mirrored"] for bucket in per_rule.values())
    total_mirror_success = sum(
        bucket["mirror_success"] for bucket in per_rule.values())
    observed = {
        "total_requests": total,
        "total_stable": sum(bucket[STABLE] for bucket in per_rule.values()),
        "total_gray": sum(bucket[GRAY] for bucket in per_rule.values()),
        "main_success_rate": str(_frac(total_main_success, total)),
        "mirrored": total_mirrored,
        "mirror_share": str(_frac(total_mirrored, total)),
        "mirror_success_rate": str(
            _frac(total_mirror_success, total_mirrored)),
    }
    if total:
        global_e2e = sum(
            Fraction(bucket["bounds"]["end_to_end_bound_fraction"])
            * bucket["total"] for bucket in per_rule.values()
        ) / total
        global_sr = Fraction(1, total)
    else:
        global_e2e = Fraction(0)
        global_sr = Fraction(0)
    observed["global_end_to_end_bound_fraction"] = str(global_e2e)
    observed["global_end_to_end_bound_per_10000"] = str(_per_10000(global_e2e))
    observed["global_success_rate_bound_fraction"] = str(global_sr)
    observed["global_success_rate_bound_per_10000"] = str(_per_10000(global_sr))
    observed["contributions"] = _contribution_ranking(rules, per_rule)
    return observed


def _contribution_ranking(rules: list, per_rule: dict[str, dict]) -> list[dict]:
    total = sum(bucket["total"] for bucket in per_rule.values()) or 1
    entries: list[dict[str, Any]] = []
    for rule in sorted(rules, key=lambda item: item.rule_id):
        bucket = per_rule[rule.rule_id]
        volume_weight = Fraction(bucket["total"], total)
        for level in ("route", "mirror"):
            value = volume_weight * Fraction(1, bucket["scale"])
            entries.append({
                "rule_id": rule.rule_id, "level": level,
                "bound_contribution_fraction": str(value),
            })
    summed = sum(
        Fraction(item["bound_contribution_fraction"]) for item in entries
    ) or Fraction(1)
    for item in entries:
        share = Fraction(item["bound_contribution_fraction"]) / summed
        item["share_of_total_bound"] = str(share)
        item["class"] = (
            "primary" if share >= CONTRIBUTION_THRESHOLD else "minor")
    entries.sort(key=lambda item: (
        -Fraction(item["share_of_total_bound"]), item["rule_id"], item["level"]))
    return entries
