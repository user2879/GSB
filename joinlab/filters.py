from __future__ import annotations

from typing import Any, Iterable

from .spec import FilterSpec


def matches_filter(actual: Any, predicate: FilterSpec) -> bool:
    op = predicate.op
    expected = predicate.value
    if op == "is_null":
        return actual is None
    if op == "not_null":
        return actual is not None
    if actual is None:
        if op == "eq" and expected is None:
            return True
        if op == "ne" and expected is not None:
            return False
        return False
    if expected is None:
        return op == "ne"
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "lt":
        return actual < expected
    if op == "le":
        return actual <= expected
    if op == "gt":
        return actual > expected
    if op == "ge":
        return actual >= expected
    return False


def apply_filters(rows: Iterable[dict[str, Any]], predicates: tuple[FilterSpec, ...]) -> list[dict[str, Any]]:
    if not predicates:
        return [dict(row) for row in rows]
    result: list[dict[str, Any]] = []
    for row in rows:
        if all(matches_filter(row.get(predicate.column), predicate) for predicate in predicates):
            result.append(dict(row))
    return result

