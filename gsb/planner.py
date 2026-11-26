"""Predicate evaluation and deterministic block-group skip planning."""

from __future__ import annotations

from typing import Any, Iterable

from .hashing import canonical_hash
from .spec import normalize_predicate, predicate_sort_key


def canonical_predicates(predicates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [normalize_predicate(p) for p in predicates]
    return sorted(normalized, key=predicate_sort_key)


def compare_values(op: str, actual: Any, literal: Any) -> bool | None:
    if op in {"is_null"}:
        return actual is None
    if op in {"is_not_null"}:
        return actual is not None
    if actual is None:
        return None

    if op in {"=", "=="}:
        return actual == literal
    if op in {"!=", "<>"}:
        return actual != literal
    if op == "<":
        return actual < literal
    if op == "<=":
        return actual <= literal
    if op == ">":
        return actual > literal
    if op == ">=":
        return actual >= literal
    if op == "in":
        values = list(literal or [])
        non_null = [value for value in values if value is not None]
        if actual in non_null:
            return True
        if any(value is None for value in values):
            return None
        return False
    if op == "not_in":
        values = list(literal or [])
        non_null = [value for value in values if value is not None]
        if actual in non_null:
            return False
        if any(value is None for value in values):
            return None
        return True
    if op == "between":
        low, high = literal
        return low <= actual <= high
    raise ValueError(f"unsupported operator {op}")


def row_matches(predicates: Iterable[dict[str, Any]], row: dict[str, Any]) -> bool:
    """SQL three-valued AND; only logical TRUE is a match."""

    result: bool | None = True
    for predicate in predicates:
        current = compare_values(predicate["op"], row.get(predicate["column"]), predicate.get("value"))
        if current is False:
            return False
        if current is None:
            result = None
    return result is True


def conjunction_values(predicates: Iterable[dict[str, Any]], row: dict[str, Any]) -> bool | None:
    result: bool | None = True
    for predicate in predicates:
        current = compare_values(predicate["op"], row.get(predicate["column"]), predicate.get("value"))
        if current is False:
            return False
        if current is None:
            result = None
    return result


def _snapshot(index: dict[str, Any], column: str, row_count: int) -> dict[str, Any]:
    stats = index.get(column, {}) if isinstance(index, dict) else {}
    return {
        "min": stats.get("min"),
        "max": stats.get("max"),
        "null_count": int(stats.get("null_count", 0) or 0),
    }


def evaluate_index_predicate(
    predicate: dict[str, Any],
    skip_index: dict[str, Any],
    row_count: int,
) -> dict[str, Any]:
    predicate = normalize_predicate(predicate)
    column = predicate["column"]
    op = predicate["op"]
    literal = predicate.get("value")
    snapshot = _snapshot(skip_index, column, row_count)
    minimum = snapshot["min"]
    maximum = snapshot["max"]
    null_count = snapshot["null_count"]

    evaluation: dict[str, Any] = {
        "predicate": predicate,
        "index_column": column,
        "index_snapshot": snapshot,
        "can_skip": False,
        "basis": "index_inconclusive",
        "index_fields": [] if op in {"is_null", "is_not_null"} else ["min", "max"],
    }

    if op == "is_null":
        evaluation["index_fields"] = ["null_count"]
        if null_count == 0:
            evaluation.update(can_skip=True, basis="null_count_zero")
        return evaluation

    if op == "is_not_null":
        evaluation["index_fields"] = ["null_count"]
        if null_count == row_count:
            evaluation.update(can_skip=True, basis="all_rows_are_null")
        return evaluation

    if null_count == row_count:
        evaluation.update(can_skip=True, basis="all_rows_are_null", index_fields=["null_count"])
        return evaluation

    if minimum is None or maximum is None:
        evaluation.update(basis="index_range_missing", index_fields=["min", "max", "null_count"])
        return evaluation

    if op == "=":
        evaluation["index_fields"] = ["min", "max"]
        if literal < minimum:
            evaluation.update(can_skip=True, basis="literal_below_min", index_fields=["min"])
        elif literal > maximum:
            evaluation.update(can_skip=True, basis="literal_above_max", index_fields=["max"])
    elif op in {"!=", "<>"}:
        evaluation["index_fields"] = ["min", "max"]
        if minimum == maximum == literal:
            evaluation.update(can_skip=True, basis="single_value_equals_literal", index_fields=["min", "max"])
    elif op == ">":
        evaluation["index_fields"] = ["max"]
        if maximum <= literal:
            evaluation.update(can_skip=True, basis="max_at_or_below_literal", index_fields=["max"])
    elif op == ">=":
        evaluation["index_fields"] = ["max"]
        if maximum < literal:
            evaluation.update(can_skip=True, basis="max_less_than_literal", index_fields=["max"])
    elif op == "<":
        evaluation["index_fields"] = ["min"]
        if minimum >= literal:
            evaluation.update(can_skip=True, basis="min_at_or_above_literal", index_fields=["min"])
    elif op == "<=":
        evaluation["index_fields"] = ["min"]
        if minimum > literal:
            evaluation.update(can_skip=True, basis="min_greater_than_literal", index_fields=["min"])
    elif op == "in":
        evaluation["index_fields"] = ["min", "max"]
        literals = [value for value in (literal or []) if value is not None]
        if literals and all(value < minimum for value in literals):
            evaluation.update(can_skip=True, basis="in_literals_below_min", index_fields=["min"])
        elif literals and all(value > maximum for value in literals):
            evaluation.update(can_skip=True, basis="in_literals_above_max", index_fields=["max"])
        elif literals and all(value < minimum or value > maximum for value in literals):
            evaluation.update(can_skip=True, basis="in_literals_outside_min_max", index_fields=["min", "max"])
    elif op == "not_in":
        evaluation["index_fields"] = ["min", "max"]
        literals = list(literal or [])
        if minimum == maximum and minimum in literals:
            evaluation.update(can_skip=True, basis="single_value_in_not_in_list", index_fields=["min", "max"])
    elif op == "between":
        low, high = literal
        evaluation["index_fields"] = ["min", "max"]
        if high < minimum:
            evaluation.update(can_skip=True, basis="between_below_min", index_fields=["min"])
        elif low > maximum:
            evaluation.update(can_skip=True, basis="between_above_max", index_fields=["max"])

    return evaluation


def compile_block_group_plan(query: dict[str, Any], block_group: dict[str, Any]) -> dict[str, Any]:
    predicates = canonical_predicates(query.get("predicates", []))
    evaluations = [
        evaluate_index_predicate(predicate, block_group.get("skip_index", {}), block_group["row_count"])
        for predicate in predicates
    ]
    witness = next((item for item in evaluations if item["can_skip"]), None)
    return {
        "block_group_id": block_group["id"],
        "decision": "skip" if witness else "verify",
        "evaluations": evaluations,
        "skip_basis": witness["basis"] if witness else None,
        "witness_predicate": witness["predicate"] if witness else None,
        "witness_index_fields": witness["index_fields"] if witness else [],
    }


def compile_query_plan(query: dict[str, Any], block_groups: list[dict[str, Any]]) -> dict[str, Any]:
    predicates = canonical_predicates(query.get("predicates", []))
    ordered_groups = sorted(block_groups, key=lambda group: group["id"])
    group_plans = [compile_block_group_plan(query, group) for group in ordered_groups]
    skip_groups = [item["block_group_id"] for item in group_plans if item["decision"] == "skip"]
    verify_groups = [item["block_group_id"] for item in group_plans if item["decision"] == "verify"]
    plan = {
        "query_id": query["id"],
        "predicate_order": predicates,
        "block_groups": group_plans,
        "skip_block_groups": skip_groups,
        "verify_block_groups": verify_groups,
    }
    plan["digest"] = canonical_hash({key: value for key, value in plan.items() if key != "digest"})
    return plan


def plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "digest": plan["digest"],
        "skip_block_groups": list(plan["skip_block_groups"]),
        "verify_block_groups": list(plan["verify_block_groups"]),
    }
