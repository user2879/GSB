from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from .canonical import digest_json
from .data import batch_to_numpy


def _filter_mask(values: np.ndarray[Any, Any], operator: str, target: Any) -> np.ndarray[Any, Any]:
    if operator == "=":
        return values == target
    if operator == "!=":
        return values != target
    if operator == "<":
        return values < target
    if operator == "<=":
        return values <= target
    if operator == ">":
        return values > target
    if operator == ">=":
        return values >= target
    if operator == "in":
        return np.isin(values, np.asarray(target, dtype=values.dtype))
    raise ValueError(f"unsupported predicate operator {operator}")


def apply_scan_filters(
    columns: dict[str, np.ndarray[Any, Any]],
    filters: list[dict[str, Any]],
) -> dict[str, np.ndarray[Any, Any]]:
    mask: np.ndarray[Any, Any] | None = None
    for predicate in filters:
        column_name = predicate["column"]
        if column_name not in columns:
            column_name = column_name.split(".", 1)[1]
        current = _filter_mask(columns[column_name], predicate["operator"], predicate["value"])
        mask = current if mask is None else np.logical_and(mask, current)
    if mask is None:
        return columns
    return {name: values[mask] for name, values in columns.items()}


def _predicate_mask(
    left: Mapping[str, np.ndarray[Any, Any]],
    right: Mapping[str, np.ndarray[Any, Any]],
    predicate: dict[str, Any],
) -> np.ndarray[Any, Any]:
    left_name = predicate["left"]
    right_name = predicate["right"]
    if left_name in left and right_name in right:
        left_values, right_values = left[left_name], right[right_name]
    elif right_name in left and left_name in right:
        left_values, right_values = left[right_name], right[left_name]
    else:
        raise ValueError(f"predicate does not cross the join boundary: {predicate}")
    return _filter_mask(left_values, predicate["operator"], right_values)


def _materialize(
    left: Mapping[str, np.ndarray[Any, Any]],
    right: Mapping[str, np.ndarray[Any, Any]],
    left_indices: np.ndarray[Any, Any],
    right_indices: np.ndarray[Any, Any],
    predicates: list[dict[str, Any]],
) -> dict[str, np.ndarray[Any, Any]]:
    left_frame = {name: values[left_indices] for name, values in left.items()}
    right_frame = {name: values[right_indices] for name, values in right.items()}
    if left_indices.size:
        mask: np.ndarray[Any, Any] | None = None
        for predicate in predicates:
            current = _predicate_mask(left_frame, right_frame, predicate)
            mask = current if mask is None else np.logical_and(mask, current)
        if mask is not None:
            left_frame = {name: values[mask] for name, values in left_frame.items()}
            right_frame = {name: values[mask] for name, values in right_frame.items()}
    left_frame.update(right_frame)
    return left_frame


def _key_tuples(
    left: Mapping[str, np.ndarray[Any, Any]],
    right: Mapping[str, np.ndarray[Any, Any]],
    predicates: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    left_names: list[str] = []
    right_names: list[str] = []
    for predicate in predicates:
        if predicate["operator"] != "=":
            continue
        if predicate["left"] in left and predicate["right"] in right:
            left_names.append(predicate["left"])
            right_names.append(predicate["right"])
        elif predicate["right"] in left and predicate["left"] in right:
            left_names.append(predicate["right"])
            right_names.append(predicate["left"])
        else:
            raise ValueError(f"equality predicate does not cross join boundary: {predicate}")
    return left_names, right_names


def _hash_join(
    left: Mapping[str, np.ndarray[Any, Any]],
    right: Mapping[str, np.ndarray[Any, Any]],
    predicates: list[dict[str, Any]],
) -> dict[str, np.ndarray[Any, Any]]:
    left_keys, right_keys = _key_tuples(left, right, predicates)
    if not left_keys:
        return _nested_loop_join(left, right, predicates)
    right_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    right_arrays = [right[name].tolist() for name in right_keys]
    for index, key in enumerate(zip(*right_arrays, strict=True)):
        right_groups[tuple(int(item) for item in key)].append(index)
    left_arrays = [left[name].tolist() for name in left_keys]
    left_indices: list[int] = []
    right_indices: list[int] = []
    for index, key in enumerate(zip(*left_arrays, strict=True)):
        matches = right_groups.get(tuple(int(item) for item in key), ())
        if matches:
            left_indices.extend([index] * len(matches))
            right_indices.extend(matches)
    return _materialize(
        left,
        right,
        np.asarray(left_indices, dtype=np.int64),
        np.asarray(right_indices, dtype=np.int64),
        predicates,
    )


def _sort_merge_join(
    left: Mapping[str, np.ndarray[Any, Any]],
    right: Mapping[str, np.ndarray[Any, Any]],
    predicates: list[dict[str, Any]],
) -> dict[str, np.ndarray[Any, Any]]:
    left_keys, right_keys = _key_tuples(left, right, predicates)
    if not left_keys:
        return _nested_loop_join(left, right, predicates)
    left_arrays = [left[name] for name in left_keys]
    right_arrays = [right[name] for name in right_keys]
    left_order = np.lexsort(tuple(reversed(left_arrays)))
    right_order = np.lexsort(tuple(reversed(right_arrays)))
    left_tuples = [tuple(int(left[name][index]) for name in left_keys) for index in left_order]
    right_tuples = [tuple(int(right[name][index]) for name in right_keys) for index in right_order]
    left_indices: list[int] = []
    right_indices: list[int] = []
    left_pointer = 0
    right_pointer = 0
    while left_pointer < len(left_tuples) and right_pointer < len(right_tuples):
        left_key = left_tuples[left_pointer]
        right_key = right_tuples[right_pointer]
        if left_key < right_key:
            left_pointer += 1
        elif left_key > right_key:
            right_pointer += 1
        else:
            left_end = left_pointer + 1
            while left_end < len(left_tuples) and left_tuples[left_end] == left_key:
                left_end += 1
            right_end = right_pointer + 1
            while right_end < len(right_tuples) and right_tuples[right_end] == right_key:
                right_end += 1
            for left_index in left_order[left_pointer:left_end]:
                for right_index in right_order[right_pointer:right_end]:
                    left_indices.append(int(left_index))
                    right_indices.append(int(right_index))
            left_pointer = left_end
            right_pointer = right_end
    return _materialize(
        left,
        right,
        np.asarray(left_indices, dtype=np.int64),
        np.asarray(right_indices, dtype=np.int64),
        predicates,
    )


def _nested_loop_join(
    left: Mapping[str, np.ndarray[Any, Any]],
    right: Mapping[str, np.ndarray[Any, Any]],
    predicates: list[dict[str, Any]],
) -> dict[str, np.ndarray[Any, Any]]:
    left_count = len(next(iter(left.values()))) if left else 0
    right_count = len(next(iter(right.values()))) if right else 0
    step = 128
    left_indices: list[np.ndarray[Any, Any]] = []
    right_indices: list[np.ndarray[Any, Any]] = []
    for left_start in range(0, left_count, step):
        left_end = min(left_count, left_start + step)
        left_chunk = {
            name: np.repeat(values[left_start:left_end], right_count)
            for name, values in left.items()
        }
        right_chunk = {
            name: np.tile(values, left_end - left_start)
            for name, values in right.items()
        }
        mask = np.ones(left_chunk[next(iter(left_chunk))].shape[0], dtype=bool) if left_chunk else np.asarray([], dtype=bool)
        for predicate in predicates:
            mask = np.logical_and(mask, _predicate_mask(left_chunk, right_chunk, predicate))
        local_left = np.repeat(np.arange(left_start, left_end, dtype=np.int64), right_count)[mask]
        local_right = np.tile(np.arange(right_count, dtype=np.int64), left_end - left_start)[mask]
        if local_left.size:
            left_indices.append(local_left)
            right_indices.append(local_right)
    if left_indices:
        joined_left = np.concatenate(left_indices)
        joined_right = np.concatenate(right_indices)
    else:
        joined_left = np.asarray([], dtype=np.int64)
        joined_right = np.asarray([], dtype=np.int64)
    return _materialize(left, right, joined_left, joined_right, predicates)


JOIN_IMPLEMENTATIONS = {
    "hash": _hash_join,
    "sort_merge": _sort_merge_join,
    "nested_loop": _nested_loop_join,
}


def _aggregate_result(
    frame: dict[str, np.ndarray[Any, Any]],
    projections: list[dict[str, Any]],
    group_by: list[str],
    aggregates: list[dict[str, Any]],
    joined_rows: int,
) -> dict[str, Any]:
    projection_aliases = [item["alias"] for item in projections]
    aggregate_aliases = [item["alias"] for item in aggregates]
    columns = projection_aliases + aggregate_aliases
    if not aggregates:
        row_count = len(next(iter(frame.values()))) if frame else 0
        rows: list[list[int]] = []
        if row_count:
            for index in range(row_count):
                rows.append([int(frame[item["column"]][index]) for item in projections])
        rows.sort(key=lambda row: tuple(row))
        return {"columns": columns, "rows": rows}

    group_keys = group_by
    states: dict[tuple[int, ...], dict[str, Any]] = {}
    if not group_keys:
        states[()] = {"rows": 0}
        for aggregate in aggregates:
            function = aggregate["function"]
            if function == "count_star":
                states[()][aggregate["alias"]] = 0
            elif function == "count":
                states[()][aggregate["alias"]] = 0
            elif function == "sum":
                states[()][aggregate["alias"]] = 0
            elif function == "min":
                states[()][aggregate["alias"]] = None
            elif function == "max":
                states[()][aggregate["alias"]] = None
            else:
                raise ValueError(f"unsupported aggregate {function}")
    else:
        group_arrays = [frame[column] for column in group_keys]
        for index, key_values in enumerate(zip(*[array.tolist() for array in group_arrays], strict=True)):
            key = tuple(int(item) for item in key_values)
            state = states.get(key)
            if state is None:
                state = {"rows": 0}
                for aggregate in aggregates:
                    function = aggregate["function"]
                    if function in ("count_star", "count", "sum"):
                        state[aggregate["alias"]] = 0
                    elif function in ("min", "max"):
                        state[aggregate["alias"]] = None
                    else:
                        raise ValueError(f"unsupported aggregate {function}")
                states[key] = state
            _accumulate_state(state, frame, index, aggregates)

    if not group_keys and not frame:
        for aggregate in aggregates:
            alias = aggregate["alias"]
            if aggregate["function"] in ("count_star", "count"):
                states[()][alias] = 0
            elif aggregate["function"] == "sum":
                states[()][alias] = 0
            elif aggregate["function"] == "min":
                states[()][alias] = 0
            elif aggregate["function"] == "max":
                states[()][alias] = 0
    elif not group_keys and frame:
        group_arrays = []
        row_count = len(next(iter(frame.values()))) if frame else 0
        for index in range(row_count):
            _accumulate_state(states[()], frame, index, aggregates)

    rows_by_key: list[tuple[tuple[int, ...], list[int]]] = []
    for key, state in states.items():
        row = list(key) + [int(state[aggregate["alias"]]) for aggregate in aggregates]
        rows_by_key.append((key, row))
    rows_by_key.sort(key=lambda item: item[0])
    return {"columns": columns, "rows": [row for _, row in rows_by_key]}


def _accumulate_state(
    state: dict[str, Any],
    frame: dict[str, np.ndarray[Any, Any]],
    index: int,
    aggregates: list[dict[str, Any]],
) -> None:
    state["rows"] += 1
    for aggregate in aggregates:
        function = aggregate["function"]
        alias = aggregate["alias"]
        if function == "count_star":
            state[alias] += 1
        elif function == "count":
            state[alias] += 1
        elif function == "sum":
            state[alias] += int(frame[aggregate["input"]][index])
        elif function == "min":
            value = int(frame[aggregate["input"]][index])
            state[alias] = value if state[alias] is None else min(state[alias], value)
        elif function == "max":
            value = int(frame[aggregate["input"]][index])
            state[alias] = value if state[alias] is None else max(state[alias], value)
        else:
            raise ValueError(f"unsupported aggregate {function}")


def evaluate_plan_summary(plan_summary: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, Any]:
    database = batch_to_numpy(batch)
    scans: dict[str, dict[str, np.ndarray[Any, Any]]] = {}
    frame: dict[str, np.ndarray[Any, Any]] | None = None
    joined_rows = 0
    operators = plan_summary.get("operators")
    if not isinstance(operators, list) or not operators:
        raise ValueError("plan summary must contain operators")

    for operator in operators:
        kind = operator.get("kind")
        if kind == "scan":
            table = operator["table"]
            if table not in database:
                raise ValueError(f"batch is missing table {table}")
            qualified = {
                f"{table}.{column}": values
                for column, values in database[table].items()
            }
            filtered = apply_scan_filters(qualified, operator.get("filters", []))
            scans[table] = filtered
            if frame is None:
                frame = filtered
                joined_rows = len(next(iter(filtered.values()))) if filtered else 0
            continue
        if kind == "join_right":
            if frame is None:
                raise ValueError("join appeared before initial scan")
            table = operator["table"]
            right = scans.get(table)
            if right is None:
                raise ValueError(f"join table {table} was not scanned")
            algorithm = operator["algorithm"]
            try:
                implementation = JOIN_IMPLEMENTATIONS[algorithm]
            except KeyError as exc:
                raise ValueError(f"unsupported join algorithm {algorithm}") from exc
            frame = implementation(frame, right, operator.get("predicates", []))
            joined_rows = len(next(iter(frame.values()))) if frame else 0
            continue
        raise ValueError(f"unsupported operator kind {kind}")

    if frame is None:
        raise ValueError("plan did not produce a frame")
    result = _aggregate_result(
        frame,
        plan_summary.get("projections", []),
        plan_summary.get("group_by", []),
        plan_summary.get("aggregates", []),
        joined_rows,
    )
    return {
        "joined_rows": joined_rows,
        "result": result,
        "result_digest": digest_json(result),
        "row_count": len(result["rows"]),
    }


def merge_aggregate_results(
    results: list[dict[str, Any]],
    plan_summary: Mapping[str, Any],
) -> dict[str, Any]:
    projections = plan_summary.get("projections", [])
    group_by = plan_summary.get("group_by", [])
    aggregates = plan_summary.get("aggregates", [])
    if not aggregates:
        rows = [row for result in results for row in result["result"]["rows"]]
        rows.sort(key=lambda row: tuple(row))
        merged = {"columns": [item["alias"] for item in projections], "rows": rows}
        return {"result": merged, "result_digest": digest_json(merged), "row_count": len(rows)}

    group_count = len(group_by)
    aggregate_indexes = {
        aggregate["alias"]: group_count + index
        for index, aggregate in enumerate(aggregates)
    }
    merged_states: dict[tuple[int, ...], dict[str, int | None]] = {}
    for shard_result in results:
        for row in shard_result["result"]["rows"]:
            key = tuple(int(value) for value in row[:group_count]) if group_count else ()
            state = merged_states.get(key)
            if state is None:
                state = {}
                for aggregate in aggregates:
                    function = aggregate["function"]
                    state[aggregate["alias"]] = 0 if function in ("count_star", "count", "sum") else None
                merged_states[key] = state
            for aggregate in aggregates:
                alias = aggregate["alias"]
                value = int(row[aggregate_indexes[alias]])
                function = aggregate["function"]
                if function in ("count_star", "count", "sum"):
                    state[alias] = int(state[alias]) + value
                elif function == "min":
                    state[alias] = value if state[alias] is None else min(int(state[alias]), value)
                elif function == "max":
                    state[alias] = value if state[alias] is None else max(int(state[alias]), value)
                else:
                    raise ValueError(f"unsupported aggregate {function}")
    rows_by_key: list[tuple[tuple[int, ...], list[int]]] = []
    for key, state in merged_states.items():
        row = list(key) + [int(state[aggregate["alias"]]) for aggregate in aggregates]
        rows_by_key.append((key, row))
    rows_by_key.sort(key=lambda item: item[0])
    merged = {
        "columns": [item["alias"] for item in projections] + [item["alias"] for item in aggregates],
        "rows": [row for _, row in rows_by_key],
    }
    return {"result": merged, "result_digest": digest_json(merged), "row_count": len(merged["rows"])}
