from __future__ import annotations

import copy
from typing import Any

from .fingerprint import stable_float
from .woe import compute_woe


TOLERANCE = 1e-12


def _violations(bins: list[dict[str, Any]], direction: str) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    if direction == "none":
        return violations
    ordered = [(index, item) for index, item in enumerate(bins) if not item["is_missing"]]
    for position in range(len(ordered) - 1):
        left_index, left = ordered[position]
        right_index, right = ordered[position + 1]
        if direction == "ascending" and left["woe"] > right["woe"] + TOLERANCE:
            gap = left["woe"] - right["woe"]
        elif direction == "descending" and left["woe"] < right["woe"] - TOLERANCE:
            gap = right["woe"] - left["woe"]
        else:
            continue
        violations.append(
            {
                "left_index": left_index,
                "right_index": right_index,
                "position": position,
                "left_boundary": left.get("lower", left.get("label", "")),
                "right_boundary": right.get("upper", right.get("label", "")),
                "gap": stable_float(gap),
                "combined_count": int(left["count"] + right["count"]),
                "left_woe": left["woe"],
                "right_woe": right["woe"],
            }
        )
    return violations


def _selection_key(violation: dict[str, Any], priority: list[str]) -> tuple[Any, ...]:
    key: list[Any] = []
    for rule in priority:
        if rule == "left":
            key.append(int(violation["position"]))
        elif rule == "right":
            key.append(-int(violation["position"]))
        elif rule == "max_gap":
            key.append(-float(violation["gap"]))
        elif rule == "min_count":
            key.append(int(violation["combined_count"]))
        elif rule == "max_count":
            key.append(-int(violation["combined_count"]))
    key.extend([int(violation["left_index"]), int(violation["right_index"])])
    return tuple(key)


def _select_violation(violations: list[dict[str, Any]], priority: list[str]) -> dict[str, Any]:
    return min(violations, key=lambda item: _selection_key(item, priority))


def _rule_value(violation: dict[str, Any], rule: str) -> Any:
    if rule == "left":
        return int(violation["position"])
    if rule == "right":
        return -int(violation["position"])
    if rule == "max_gap":
        return -float(violation["gap"])
    if rule == "min_count":
        return int(violation["combined_count"])
    if rule == "max_count":
        return -int(violation["combined_count"])
    return 0


def _hit_rule(
    violations: list[dict[str, Any]],
    selected: dict[str, Any],
    priority: list[str],
) -> str:
    for rule in priority:
        selected_value = _rule_value(selected, rule)
        values = [_rule_value(item, rule) for item in violations]
        minimum = min(values)
        if selected_value == minimum and any(value > minimum for value in values):
            return rule
    return "stable_order"


def _merge_pair(bins: list[dict[str, Any]], left_index: int, right_index: int) -> None:
    left = bins[left_index]
    right = bins[right_index]
    left["upper"] = right["upper"]
    left["count"] = int(left["count"] + right["count"])
    left["good"] = int(left["good"] + right["good"])
    left["bad"] = int(left["bad"] + right["bad"])
    del bins[right_index]
    for index, item in enumerate(bins):
        item["index"] = index


def solve_monotonic(
    column: str,
    column_type: str,
    base_bins: list[dict[str, Any]],
    direction: str,
    conflict_priority: list[str],
    max_rounds: int,
    smoothing: float,
    category_to_group: dict[str, int] | None = None,
) -> dict[str, Any]:
    bins = copy.deepcopy(base_bins)
    category_mapping = dict(category_to_group or {})
    initial_woe = compute_woe(column, column_type, bins, smoothing)
    bins = initial_woe["bins"]

    if column_type != "numeric" or direction == "none":
        return {
            **initial_woe,
            "direction": direction,
            "status": "not_applicable" if column_type != "numeric" or direction == "none" else "ok",
            "corrections": [],
            "rounds": 0,
            "max_rounds": max_rounds,
            "conflict_priority": list(conflict_priority),
            "unresolved_violations": [],
            "category_to_group": category_mapping,
            "missing_index": next((item["index"] for item in bins if item["is_missing"]), None),
        }

    corrections: list[dict[str, Any]] = []
    for round_number in range(1, max_rounds + 1):
        current_violations = _violations(bins, direction)
        if not current_violations:
            break
        selected = _select_violation(current_violations, conflict_priority)
        hit_rule = _hit_rule(current_violations, selected, conflict_priority)
        left_index = int(selected["left_index"])
        right_index = int(selected["right_index"])
        before_bounds = [
            [bins[left_index].get("lower"), bins[left_index].get("upper")],
            [bins[right_index].get("lower"), bins[right_index].get("upper")],
        ]
        before_woe = [bins[left_index]["woe"], bins[right_index]["woe"]]
        before_counts = [bins[left_index]["count"], bins[right_index]["count"]]
        _merge_pair(bins, left_index, right_index)
        recomputed = compute_woe(column, column_type, bins, smoothing)
        after_bins = recomputed["bins"]
        after_woe = after_bins[left_index]["woe"]
        corrections.append(
            {
                "round": round_number,
                "direction": direction,
                "merged_bin_indices": [left_index, right_index],
                "merged_bounds": before_bounds,
                "new_bounds": [after_bins[left_index].get("lower"), after_bins[left_index].get("upper")],
                "before_woe": before_woe,
                "after_woe": after_woe,
                "before_counts": before_counts,
                "after_count": after_bins[left_index]["count"],
                "violation_gap": selected["gap"],
                "hit_rule": hit_rule,
                "priority": list(conflict_priority),
            }
        )
        bins = after_bins

    remaining = _violations(bins, direction)
    final_woe = compute_woe(column, column_type, bins, smoothing)
    return {
        **final_woe,
        "direction": direction,
        "status": "ok" if not remaining else "unresolved",
        "corrections": corrections,
        "rounds": len(corrections),
        "max_rounds": max_rounds,
        "conflict_priority": list(conflict_priority),
        "unresolved_violations": [
            {
                "left_index": item["left_index"],
                "right_index": item["right_index"],
                "left_woe": item["left_woe"],
                "right_woe": item["right_woe"],
                "gap": item["gap"],
            }
            for item in remaining
        ],
        "category_to_group": category_mapping,
        "missing_index": next((item["index"] for item in final_woe["bins"] if item["is_missing"]), None),
    }
