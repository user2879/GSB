from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_bytes, canonical_json, sha256_bytes, sha256_obj
from .model import AggregateSpec, Column, Dataset, Query, Table


TASK_OVERHEAD_BYTES = 64


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    table_name: str
    table_version: int
    table_digest: str
    row_start: int
    row_end: int
    referenced_columns: tuple[str, ...]
    split_eligible: bool
    parent_task_id: str | None
    split_source: str | None
    estimated_bytes: int

    @property
    def rows_scanned(self) -> int:
        return self.row_end - self.row_start


@dataclass(frozen=True)
class Plan:
    tasks: tuple[TaskSpec, ...]
    plan_digest: str
    referenced_columns: tuple[str, ...]

    def payload(self) -> list[dict[str, Any]]:
        return [
            {
                "task_id": task.task_id,
                "table": task.table_name,
                "table_version": task.table_version,
                "table_digest": task.table_digest,
                "row_start": task.row_start,
                "row_end": task.row_end,
                "referenced_columns": list(task.referenced_columns),
                "split_eligible": task.split_eligible,
                "parent_task_id": task.parent_task_id,
                "split_source": task.split_source,
                "estimated_bytes": task.estimated_bytes,
            }
            for task in self.tasks
        ]


@dataclass
class ShardOutcome:
    task: TaskSpec
    rows_scanned: int
    rows_completed: int
    aggregate_map: dict[tuple[Any, ...], dict[str, dict[str, int]]]
    group_counts: dict[tuple[Any, ...], int]
    aggregate_state: list[dict[str, Any]]
    digest: str


def estimate_task_bytes(table: Table, referenced_columns: Sequence[str], row_count: int) -> int:
    columns = table.column_map()
    row_width = sum(columns[name].memory_width_bytes() for name in referenced_columns)
    return TASK_OVERHEAD_BYTES + row_count * row_width


def make_plan(dataset: Dataset, row_group_size: int) -> Plan:
    referenced = dataset.query.referenced_columns
    tasks: list[TaskSpec] = []
    tables = dataset.table_map()
    for table_name in dataset.query.from_tables:
        table = tables[table_name]
        for row_start in range(0, table.row_count, row_group_size):
            row_end = min(table.row_count, row_start + row_group_size)
            task_id = (
                f"{table.name}@v{table.version}"
                f":r{row_start:012d}-{row_end:012d}"
            )
            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    table_name=table.name,
                    table_version=table.version,
                    table_digest=table.digest,
                    row_start=row_start,
                    row_end=row_end,
                    referenced_columns=referenced,
                    split_eligible=True,
                    parent_task_id=None,
                    split_source=None,
                    estimated_bytes=estimate_task_bytes(table, referenced, row_end - row_start),
                )
            )
    tasks.sort(key=lambda task: task.task_id)
    payload = [
        {
            "task_id": task.task_id,
            "table": task.table_name,
            "table_version": task.table_version,
            "table_digest": task.table_digest,
            "row_start": task.row_start,
            "row_end": task.row_end,
            "referenced_columns": list(task.referenced_columns),
        }
        for task in tasks
        if task.parent_task_id is None
    ]
    return Plan(
        tasks=tuple(tasks),
        plan_digest=sha256_obj(payload),
        referenced_columns=referenced,
    )


def _column_types_for_groups(query: Query, dataset: Dataset) -> tuple[str, ...]:
    first_table = dataset.table(dataset.query.from_tables[0])
    columns = first_table.column_map()
    return tuple(columns[name].type_name for name in query.group_by)


def group_sort_key(key: tuple[Any, ...], group_types: Sequence[str]) -> tuple[Any, ...]:
    components: list[tuple[int, Any]] = []
    for value, type_name in zip(key, group_types, strict=True):
        if value is None:
            components.append((0, 0))
        elif type_name in {"int", "fixed"}:
            components.append((1, int(value)))
        else:
            components.append((1, str(value)))
    return tuple(components)


def serialize_aggregate_map(
    aggregate_map: Mapping[tuple[Any, ...], Mapping[str, Mapping[str, int]]],
    query: Query,
    group_types: Sequence[str],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for key in sorted(aggregate_map, key=lambda item: group_sort_key(item, group_types)):
        metrics = aggregate_map[key]
        groups.append(
            {
                "key": list(key),
                "metrics": {
                    aggregate.name: {name: int(value) for name, value in metrics[aggregate.name].items()}
                    for aggregate in query.aggregates
                },
            }
        )
    return groups


def _evaluate_filter(node: Mapping[str, Any], columns: Mapping[str, Column], row: int) -> bool:
    if node.get("op") == "and":
        return all(_evaluate_filter(term, columns, row) for term in node["terms"])
    value = columns[str(node["column"])].value_at(row)
    if value is None:
        return False
    literal = node["value"]
    op = node["op"]
    if op == "eq":
        return value == literal
    if op == "gt":
        return value > literal
    if op == "ge":
        return value >= literal
    if op == "lt":
        return value < literal
    if op == "le":
        return value <= literal
    raise AssertionError(f"unknown filter op: {op}")


def _new_metric(aggregate: AggregateSpec) -> dict[str, int]:
    if aggregate.op == "count":
        return {"count": 0}
    return {"count": 0, "sum": 0}


def execute_task(dataset: Dataset, query: Query, task: TaskSpec) -> ShardOutcome:
    table = dataset.table(task.table_name)
    columns = table.column_map()
    group_columns = [columns[name] for name in query.group_by]
    aggregate_columns: list[Column | None] = [
        None if aggregate.column == "*" else columns[aggregate.column]
        for aggregate in query.aggregates
    ]
    group_types = tuple(column.type_name for column in group_columns)
    aggregate_map: dict[tuple[Any, ...], dict[str, dict[str, int]]] = {}
    group_counts: dict[tuple[Any, ...], int] = {}
    rows_completed = 0
    for row in range(task.row_start, task.row_end):
        if query.where is not None and not _evaluate_filter(query.where, columns, row):
            continue
        rows_completed += 1
        key = tuple(column.value_at(row) for column in group_columns)
        group_counts[key] = group_counts.get(key, 0) + 1
        metrics = aggregate_map.get(key)
        if metrics is None:
            metrics = {aggregate.name: _new_metric(aggregate) for aggregate in query.aggregates}
            aggregate_map[key] = metrics
        for index, aggregate in enumerate(query.aggregates):
            metric = metrics[aggregate.name]
            if aggregate.op == "count" and aggregate.column == "*":
                metric["count"] += 1
                continue
            column = aggregate_columns[index]
            assert column is not None
            value = column.value_at(row)
            if value is None:
                continue
            metric["count"] += 1
            if aggregate.op in {"sum", "avg"}:
                metric["sum"] += int(value)
    aggregate_state = serialize_aggregate_map(aggregate_map, query, group_types)
    digest_payload = {
        "task_id": task.task_id,
        "table": task.table_name,
        "table_version": task.table_version,
        "table_digest": task.table_digest,
        "row_start": task.row_start,
        "row_end": task.row_end,
        "rows_scanned": task.rows_scanned,
        "rows_completed": rows_completed,
        "aggregate": aggregate_state,
    }
    return ShardOutcome(
        task=task,
        rows_scanned=task.rows_scanned,
        rows_completed=rows_completed,
        aggregate_map=aggregate_map,
        group_counts=group_counts,
        aggregate_state=aggregate_state,
        digest=sha256_obj(digest_payload),
    )


def merge_aggregate_maps(
    outcomes: Iterable[ShardOutcome],
    query: Query,
    group_types: Sequence[str] | None = None,
) -> dict[tuple[Any, ...], dict[str, dict[str, int]]]:
    merged: dict[tuple[Any, ...], dict[str, dict[str, int]]] = {}
    for outcome in sorted(outcomes, key=lambda item: item.task.task_id):
        if group_types is None:
            ordered_keys = sorted(
                outcome.aggregate_map,
                key=lambda item: canonical_json(list(item)),
            )
        else:
            ordered_keys = sorted(
                outcome.aggregate_map,
                key=lambda item: group_sort_key(item, group_types),
            )
        for key in ordered_keys:
            source_metrics = outcome.aggregate_map[key]
            target_metrics = merged.get(key)
            if target_metrics is None:
                target_metrics = {
                    aggregate.name: _new_metric(aggregate) for aggregate in query.aggregates
                }
                merged[key] = target_metrics
            for aggregate in query.aggregates:
                source = source_metrics[aggregate.name]
                target = target_metrics[aggregate.name]
                target["count"] += source["count"]
                if "sum" in source:
                    target["sum"] += source["sum"]
    return merged


def _floor_divide(numerator: int, denominator: int) -> int:
    return numerator // denominator


def build_final_result(
    query: Query,
    group_types: Sequence[str],
    aggregate_map: Mapping[tuple[Any, ...], Mapping[str, Mapping[str, int]]],
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for key in sorted(aggregate_map, key=lambda item: group_sort_key(item, group_types)):
        source_metrics = aggregate_map[key]
        values: dict[str, int | None] = {}
        for aggregate in query.aggregates:
            metric = source_metrics[aggregate.name]
            if aggregate.op == "count":
                values[aggregate.name] = int(metric["count"])
            elif aggregate.op == "sum":
                values[aggregate.name] = (
                    int(metric["sum"]) if metric["count"] > 0 else None
                )
            elif aggregate.op == "avg":
                values[aggregate.name] = (
                    _floor_divide(int(metric["sum"]), int(metric["count"]))
                    if metric["count"] > 0
                    else None
                )
            else:
                raise AssertionError(f"unknown aggregate op: {aggregate.op}")
        groups.append({"key": list(key), "metrics": values})
    schema: list[dict[str, Any]] = []
    for aggregate in query.aggregates:
        item: dict[str, Any] = {
            "name": aggregate.name,
            "op": aggregate.op,
            "column": aggregate.column,
            "value_representation": (
                "integer" if aggregate.op == "count" else "integer_scaled"
            ),
            "scale": aggregate.scale,
            "aggregation_error_upper_bound_scaled_units": 0,
        }
        if aggregate.op == "avg":
            item["avg_rounding"] = "floor"
            item["aggregation_error_upper_bound_scaled_units"] = 1
        schema.append(item)
    return {
        "query_id": query.query_id,
        "group_by": list(query.group_by),
        "aggregate_schema": schema,
        "precision": {
            "storage": "integer_fixed_point",
            "sum_error_scaled_units": 0,
            "avg_rounding": "floor",
            "avg_error_upper_bound_scaled_units": 1,
            "error_bound_meaning": "0 <= exact_average - emitted_average < 1 scaled unit",
        },
        "groups": groups,
    }


def result_bytes(result: Mapping[str, Any], *, newline: bool = True) -> bytes:
    return canonical_bytes(result, newline=newline)


def result_digest(result: Mapping[str, Any]) -> str:
    return sha256_bytes(result_bytes(result))


def aggregate_state_json(outcome: ShardOutcome) -> str:
    return canonical_json(outcome.aggregate_state)


def hot_groups(
    outcome: ShardOutcome,
    threshold: int,
) -> list[tuple[tuple[Any, ...], int]]:
    candidates = [
        (key, count) for key, count in outcome.group_counts.items() if count >= threshold
    ]
    candidates.sort(key=lambda item: (-item[1], canonical_json(list(item[0]))))
    return candidates
