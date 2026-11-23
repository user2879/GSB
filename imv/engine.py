from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .errors import BusinessError, fail
from .fixed import AggregateAccumulator, FixedPointContext
from .model import BaseTable, Change, ViewDef
from .util import canonical_json_bytes, canonical_json_text, digest_bytes, digest_obj


GroupKey = tuple[Any, ...]


@dataclass(frozen=True)
class Transition:
    seq: int
    op: str
    row_id: Any
    before: dict[str, Any] | None
    after: dict[str, Any] | None


@dataclass(frozen=True)
class MaintenanceOutcome:
    payload: dict[str, Any]
    result_json: str
    result_bytes: bytes
    digest: str
    impacted_groups: tuple[list[Any], ...]
    recomputed_groups: tuple[list[Any], ...]
    group_count: int

    def trace_payload(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "group_count": self.group_count,
            "impacted_groups": [list(key) for key in self.impacted_groups],
            "recomputed_groups": [list(key) for key in self.recomputed_groups],
        }


@dataclass(frozen=True)
class Difference:
    view_id: str
    group_key: list[Any]
    field: str
    cached: Any = None
    authoritative: Any = None

    def location(self) -> dict[str, Any]:
        return {"view_id": self.view_id, "group_key": self.group_key, "field": self.field}

    def details(self) -> dict[str, Any]:
        return {
            "cached": self.cached,
            "authoritative": self.authoritative,
            "group_key": self.group_key,
            "field": self.field,
        }


def _compare_scalar(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return type(left) is type(right) and left == right
    return left == right


def _compare_filter(node: dict[str, Any] | None, row: dict[str, Any]) -> bool:
    if node is None:
        return True
    op = node["op"]
    if op == "and":
        return all(_compare_filter(arg, row) for arg in node["args"])
    if op == "or":
        return any(_compare_filter(arg, row) for arg in node["args"])
    if op == "not":
        return not _compare_filter(node["arg"], row)
    value = row.get(node["column"])
    if op == "is_null":
        return value is None
    if op == "is_not_null":
        return value is not None
    if value is None:
        return False
    if op == "in":
        return any(_compare_scalar(value, candidate) for candidate in node["value"] if candidate is not None)
    if op == "not_in":
        return not any(_compare_scalar(value, candidate) for candidate in node["value"] if candidate is not None)
    expected = node.get("value")
    if expected is None:
        return False
    if op == "eq":
        return _compare_scalar(value, expected)
    if op == "ne":
        return not _compare_scalar(value, expected)
    if op == "gt":
        return value > expected
    if op == "gte":
        return value >= expected
    if op == "lt":
        return value < expected
    if op == "lte":
        return value <= expected
    fail("E_SCHEMA", f"unsupported filter operator at evaluation time: {op}")


def group_key_for_row(view: ViewDef, row: dict[str, Any]) -> GroupKey | None:
    if not _compare_filter(view.filter, row):
        return None
    key = tuple(row.get(column) for column in view.group_by)
    if any(value is None for value in key):
        return None
    return key


def _sort_group_keys(keys: Iterable[GroupKey]) -> list[GroupKey]:
    return sorted(keys, key=lambda key: canonical_json_bytes(list(key)))


def _aggregate_group(
    view: ViewDef,
    fixed: FixedPointContext,
    rows: list[dict[str, Any]],
) -> dict[str, int | None]:
    accumulators = [AggregateAccumulator(spec.payload(), fixed) for spec in view.aggregates]
    for row in rows:
        for accumulator in accumulators:
            accumulator.add(row)
    return {accumulator.alias: accumulator.value() for accumulator in accumulators}


def _base_payload(
    view: ViewDef,
    fixed: FixedPointContext,
    *,
    cursor: int,
    param_version: int,
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "view_id": view.view_id,
        "param_version": param_version,
        "cursor": cursor,
        "group_by": list(view.group_by),
        "aggregates": [spec.payload() for spec in view.aggregates],
        "fixed_point": {
            "scale_factor": fixed.scale_factor,
            "truncation": fixed.truncation,
            "overflow": fixed.overflow,
        },
        "groups": groups,
    }


def _finalize_outcome(
    payload: dict[str, Any],
    impacted: Iterable[GroupKey],
    recomputed: Iterable[GroupKey],
) -> MaintenanceOutcome:
    result_json = canonical_json_text(payload)
    result_bytes = result_json.encode("utf-8")
    return MaintenanceOutcome(
        payload=payload,
        result_json=result_json,
        result_bytes=result_bytes,
        digest=digest_bytes(result_bytes),
        impacted_groups=tuple(list(key) for key in _sort_group_keys(impacted)),
        recomputed_groups=tuple(list(key) for key in _sort_group_keys(recomputed)),
        group_count=len(payload["groups"]),
    )


def build_view_result(
    view: ViewDef,
    fixed: FixedPointContext,
    rows: dict[Any, dict[str, Any]],
    *,
    cursor: int,
    param_version: int,
) -> MaintenanceOutcome:
    buckets: dict[GroupKey, list[dict[str, Any]]] = {}
    for row in rows.values():
        key = group_key_for_row(view, row)
        if key is None:
            continue
        buckets.setdefault(key, []).append(row)
    groups = [
        {"key": list(key), "values": _aggregate_group(view, fixed, buckets[key])}
        for key in _sort_group_keys(buckets)
    ]
    payload = _base_payload(view, fixed, cursor=cursor, param_version=param_version, groups=groups)
    return _finalize_outcome(payload, buckets.keys(), buckets.keys())


def maintain_view_incrementally(
    view: ViewDef,
    fixed: FixedPointContext,
    previous: MaintenanceOutcome | dict[str, Any],
    rows_after: dict[Any, dict[str, Any]],
    transitions: list[Transition],
    *,
    cursor_before: int,
    cursor_after: int,
    param_version: int,
) -> MaintenanceOutcome:
    previous_payload = previous.payload if isinstance(previous, MaintenanceOutcome) else previous
    if previous_payload.get("view_id") != view.view_id:
        fail("E_STATE", "previous result belongs to a different view", location={"view_id": view.view_id})
    if previous_payload.get("cursor") != cursor_before:
        fail(
            "E_STATE",
            "previous result cursor does not match the maintenance cursor",
            location={"view_id": view.view_id},
            details={"expected": cursor_before, "actual": previous_payload.get("cursor")},
        )

    previous_groups: dict[GroupKey, dict[str, Any]] = {
        tuple(group["key"]): group["values"] for group in previous_payload.get("groups", [])
    }
    impacted: set[GroupKey] = set()
    for transition in transitions:
        for row in (transition.before, transition.after):
            if row is None:
                continue
            key = group_key_for_row(view, row)
            if key is not None:
                impacted.add(key)

    buckets: dict[GroupKey, list[dict[str, Any]]] = {key: [] for key in impacted}
    if impacted:
        for row in rows_after.values():
            key = group_key_for_row(view, row)
            if key in buckets:
                buckets[key].append(row)

    final_groups: dict[GroupKey, dict[str, Any]] = dict(previous_groups)
    recomputed: set[GroupKey] = set(impacted)
    for key in impacted:
        if buckets[key]:
            final_groups[key] = _aggregate_group(view, fixed, buckets[key])
        else:
            final_groups.pop(key, None)

    groups = [
        {"key": list(key), "values": final_groups[key]}
        for key in _sort_group_keys(final_groups)
    ]
    payload = _base_payload(view, fixed, cursor=cursor_after, param_version=param_version, groups=groups)
    return _finalize_outcome(payload, impacted, recomputed)


def replay_base_rows(
    initial_rows: dict[Any, dict[str, Any]],
    changes: Iterable[Change],
    *,
    cursor: int | None = None,
) -> dict[Any, dict[str, Any]]:
    rows = {row_id: dict(row) for row_id, row in initial_rows.items()}
    limit = len(changes) if cursor is None else cursor
    for change in list(changes)[:limit]:
        if change.op == "delete":
            rows.pop(change.row_id, None)
        else:
            assert change.row is not None
            rows[change.row_id] = dict(change.row)
    return rows


def authoritative_results(
    base: BaseTable,
    definitions: Any,
    changes: tuple[Change, ...],
    *,
    cursor: int,
    param_version: int,
) -> dict[str, MaintenanceOutcome]:
    rows = replay_base_rows(base.rows, changes, cursor=cursor)
    results: dict[str, MaintenanceOutcome] = {}
    for view_id in sorted(definitions.views):
        view = definitions.views[view_id]
        results[view_id] = build_view_result(
            view,
            definitions.fixed,
            rows,
            cursor=cursor,
            param_version=param_version,
        )
    return results


def _parse_result(result_json: str, *, view_id: str) -> dict[str, Any]:
    import json

    try:
        payload = json.loads(result_json)
    except json.JSONDecodeError as exc:
        fail("E_PREVI", "cached result is not valid JSON", location={"view_id": view_id, "group_key": [], "field": "__json__"}, details={"error": str(exc)})
    if not isinstance(payload, dict):
        fail("E_PREVI", "cached result is not an object", location={"view_id": view_id, "group_key": [], "field": "__json__"})
    return payload


def find_first_difference(
    view_id: str,
    cached_json: str,
    authoritative_payload: dict[str, Any],
) -> Difference | None:
    authoritative_json = canonical_json_text(authoritative_payload)
    if cached_json == authoritative_json:
        return None
    cached = _parse_result(cached_json, view_id=view_id)

    for field in ("view_id", "param_version", "cursor", "group_by", "aggregates", "fixed_point"):
        if cached.get(field) != authoritative_payload.get(field):
            return Difference(view_id, [], f"__meta__.{field}", cached.get(field), authoritative_payload.get(field))

    cached_groups = {canonical_json_text(group.get("key", [])): group for group in cached.get("groups", [])}
    auth_groups = {canonical_json_text(group.get("key", [])): group for group in authoritative_payload.get("groups", [])}
    all_keys = sorted(set(cached_groups) | set(auth_groups), key=lambda value: value.encode("utf-8"))
    for encoded_key in all_keys:
        key = cached_groups.get(encoded_key, auth_groups.get(encoded_key, {})).get("key", [])
        if encoded_key not in cached_groups:
            return Difference(view_id, key, "__group__", None, "present")
        if encoded_key not in auth_groups:
            return Difference(view_id, key, "__group__", "present", None)
        cached_values = cached_groups[encoded_key].get("values", {})
        auth_values = auth_groups[encoded_key].get("values", {})
        for alias in sorted(set(cached_values) | set(auth_values)):
            if alias not in cached_values or alias not in auth_values or not _compare_scalar(cached_values[alias], auth_values[alias]):
                return Difference(view_id, key, alias, cached_values.get(alias), auth_values.get(alias))
    return Difference(view_id, [], "__bytes__", digest_obj(cached), digest_obj(authoritative_payload))


def assert_byte_identical(
    view_id: str,
    cached_json: str,
    authoritative_payload: dict[str, Any],
) -> None:
    difference = find_first_difference(view_id, cached_json, authoritative_payload)
    if difference is not None:
        raise BusinessError(
            "E_PREVI",
            f"incremental result differs from authoritative recompute at field {difference.field}",
            difference.location(),
            difference.details(),
        )


def view_result_summary(view_id: str, result_json: str, *, stale: bool, inconsistent: bool) -> dict[str, Any]:
    payload = _parse_result(result_json, view_id=view_id)
    return {
        "view_id": view_id,
        "digest": digest_bytes(result_json.encode("utf-8")),
        "group_count": len(payload.get("groups", [])),
        "cursor": payload.get("cursor"),
        "param_version": payload.get("param_version"),
        "stale": stale,
        "inconsistent": inconsistent,
    }
