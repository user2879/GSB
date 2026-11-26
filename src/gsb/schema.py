from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import digest_json
from .errors import SchemaError

WORKLOAD_FORMAT = "gsb-workload-v1"
SUPPORTED_SCALES = (1, 8, 10, 32, 64, 100, 1000, 10000)
SUPPORTED_ENCODINGS = ("int64", "int32", "dictionary", "bitpacked")
SUPPORTED_JOIN_OPS = ("=", "!=", "<", "<=", ">", ">=")
SUPPORTED_FILTER_OPS = ("=", "!=", "<", "<=", ">", ">=", "in")
SUPPORTED_AGGREGATES = ("count_star", "count", "sum", "min", "max")
ENCODING_WIDTHS = {"int64": 8, "int32": 4, "dictionary": 4, "bitpacked": 2}


@dataclass(frozen=True, order=True)
class ColumnRef:
    block_group: str
    column: str

    @property
    def qualified(self) -> str:
        return f"{self.block_group}.{self.column}"

    def as_dict(self) -> dict[str, str]:
        return {"block_group": self.block_group, "column": self.column}


@dataclass(frozen=True)
class Histogram:
    bounds: tuple[int, ...]
    counts: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"bounds": list(self.bounds), "counts": list(self.counts)}


@dataclass(frozen=True)
class ColumnDecl:
    name: str
    encoding: str
    histogram: Histogram

    @property
    def width_bytes(self) -> int:
        return ENCODING_WIDTHS[self.encoding]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "encoding": self.encoding,
            "histogram": self.histogram.as_dict(),
        }


@dataclass(frozen=True)
class BlockGroupDecl:
    id: str
    row_count: int
    seed: int
    columns: tuple[ColumnDecl, ...]

    def column(self, name: str) -> ColumnDecl:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    @property
    def row_width_bytes(self) -> int:
        return sum(column.width_bytes for column in self.columns)

    @property
    def fingerprint(self) -> str:
        return digest_json(
            {
                "id": self.id,
                "row_count": self.row_count,
                "seed": self.seed,
                "columns": [column.as_dict() for column in self.columns],
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "row_count": self.row_count,
            "seed": self.seed,
            "columns": [column.as_dict() for column in self.columns],
        }


@dataclass(frozen=True)
class JoinPredicate:
    left: ColumnRef
    right: ColumnRef
    operator: str

    @property
    def is_equality(self) -> bool:
        return self.operator == "="

    def normalized(self) -> "JoinPredicate":
        if self.operator == "=" and self.right < self.left:
            return JoinPredicate(self.right, self.left, self.operator)
        return self

    def as_dict(self) -> dict[str, str]:
        return {
            "left": self.left.qualified,
            "right": self.right.qualified,
            "operator": self.operator,
        }


@dataclass(frozen=True)
class FilterPredicate:
    column: ColumnRef
    operator: str
    value: int | tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        value: int | list[int]
        value = list(self.value) if isinstance(self.value, tuple) else self.value
        return {
            "column": self.column.qualified,
            "operator": self.operator,
            "value": value,
        }


@dataclass(frozen=True)
class Projection:
    column: ColumnRef
    alias: str

    def as_dict(self) -> dict[str, str]:
        return {"column": self.column.qualified, "alias": self.alias}


@dataclass(frozen=True)
class Aggregate:
    function: str
    input: ColumnRef | None
    alias: str

    def as_dict(self) -> dict[str, str | None]:
        return {
            "function": self.function,
            "input": self.input.qualified if self.input else None,
            "alias": self.alias,
        }


@dataclass(frozen=True)
class QueryDecl:
    id: str
    joins: tuple[JoinPredicate, ...]
    filters: tuple[FilterPredicate, ...]
    projections: tuple[Projection, ...]
    group_by: tuple[ColumnRef, ...]
    aggregates: tuple[Aggregate, ...]
    tables: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "joins": [item.as_dict() for item in self.joins],
            "filters": [item.as_dict() for item in self.filters],
            "projections": [item.as_dict() for item in self.projections],
            "group_by": [item.qualified for item in self.group_by],
            "aggregates": [item.as_dict() for item in self.aggregates],
        }

    @property
    def fingerprint(self) -> str:
        return digest_json(self.as_dict())


@dataclass(frozen=True)
class CostModel:
    access_row_weight: int
    comparison_weight: int
    memory_byte_weight: int
    intermediate_row_weight: int
    fixed_scale: int
    batch_rows: int
    error_band: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": {
                "access_row": self.access_row_weight,
                "comparison": self.comparison_weight,
                "memory_byte": self.memory_byte_weight,
                "intermediate_row": self.intermediate_row_weight,
            },
            "scale_factors": {
                "fixed": self.fixed_scale,
                "batch_rows": self.batch_rows,
            },
            "error_band": self.error_band,
        }

    @property
    def fingerprint(self) -> str:
        return digest_json(self.as_dict())

    @property
    def arithmetic_contract(self) -> dict[str, str]:
        return {
            "integer_domain": "non-negative signed 64-bit",
            "rounding": "multiply/divide truncates toward zero (floor)",
            "batch_quantization": "row estimates are rounded upward to batch_rows",
            "overflow": "saturate at 9223372036854775807",
        }


@dataclass(frozen=True)
class ExecutorSpec:
    id: str
    host: str
    port: int
    concurrency: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "concurrency": self.concurrency,
        }


@dataclass(frozen=True)
class SchedulerConfig:
    global_inflight: int
    lease_seconds: float
    max_lease_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_inflight": self.global_inflight,
            "lease_seconds": self.lease_seconds,
            "max_lease_seconds": self.max_lease_seconds,
        }


@dataclass(frozen=True)
class PartitionSpec:
    driver: str
    co_partition: tuple[tuple[str, ColumnRef], ...]
    replicate: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "driver": self.driver,
            "co_partition": [
                {"table": table, "column": ref.qualified}
                for table, ref in self.co_partition
            ],
            "replicate": list(self.replicate),
        }


@dataclass(frozen=True)
class ExecutionSpec:
    shard_count: int
    partition: PartitionSpec

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard_count": self.shard_count,
            "partition": self.partition.as_dict(),
        }


@dataclass(frozen=True)
class Workload:
    format: str
    block_groups: tuple[BlockGroupDecl, ...]
    query: QueryDecl
    cost_model: CostModel
    executors: tuple[ExecutorSpec, ...]
    scheduler: SchedulerConfig | None
    execution: ExecutionSpec | None
    source: dict[str, Any]

    def block_group(self, block_group_id: str) -> BlockGroupDecl:
        for group in self.block_groups:
            if group.id == block_group_id:
                return group
        raise KeyError(block_group_id)

    def query_block_groups(self) -> tuple[BlockGroupDecl, ...]:
        return tuple(self.block_group(table_id) for table_id in self.query.tables)

    @property
    def fingerprint(self) -> str:
        return digest_json(
            {
                "format": self.format,
                "block_groups": [group.as_dict() for group in self.block_groups],
                "query": self.query.as_dict(),
                "cost_model": self.cost_model.as_dict(),
            }
        )


def load_workload(path: str | Path) -> Workload:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError(
            f"invalid JSON: {exc.msg}",
            field=f"line:{exc.lineno}:column:{exc.colno}",
        ) from exc
    except OSError as exc:
        raise SchemaError(f"cannot read workload: {exc}", field=str(path)) from exc
    return parse_workload(raw)


def parse_workload(raw: Any) -> Workload:
    if not isinstance(raw, dict):
        raise SchemaError("workload must be a JSON object", field="$")
    if raw.get("format") != WORKLOAD_FORMAT:
        raise SchemaError(
            f"format must be {WORKLOAD_FORMAT}",
            field="format",
        )

    raw_groups = raw.get("block_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise SchemaError("block_groups must be a non-empty array", field="block_groups")

    groups_by_id: dict[str, dict[str, Any]] = {}
    for ordinal, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, dict):
            raise SchemaError("block group must be an object", field=f"block_groups[{ordinal}]")
        group_id = raw_group.get("id")
        if not isinstance(group_id, str) or not group_id:
            raise SchemaError(
                "block group id must be a non-empty string",
                field=f"block_groups[{ordinal}].id",
            )
        if group_id in groups_by_id:
            raise SchemaError(
                "block group id must be unique",
                field="block_groups[].id",
                block_group_id=group_id,
            )
        groups_by_id[group_id] = raw_group

    groups = tuple(_parse_block_group(group_id, groups_by_id[group_id]) for group_id in sorted(groups_by_id))
    if not groups:
        raise SchemaError("at least one block group is required", field="block_groups")

    raw_query = raw.get("query")
    if not isinstance(raw_query, dict):
        raise SchemaError("query must be an object", field="query")
    query = _parse_query(raw_query, groups)

    raw_cost = raw.get("cost_model")
    if not isinstance(raw_cost, dict):
        raise SchemaError("cost_model must be an object", field="cost_model")
    cost_model = _parse_cost_model(raw_cost)

    executors: tuple[ExecutorSpec, ...] = ()
    scheduler: SchedulerConfig | None = None
    execution: ExecutionSpec | None = None
    if "executors" in raw:
        executors = _parse_executors(raw["executors"], groups)
    if "scheduler" in raw:
        scheduler = _parse_scheduler(raw["scheduler"])
    if "execution" in raw:
        execution = _parse_execution(raw["execution"], groups, query)
    if executors and scheduler is None:
        raise SchemaError("scheduler is required when executors are declared", field="scheduler")

    return Workload(
        format=WORKLOAD_FORMAT,
        block_groups=groups,
        query=query,
        cost_model=cost_model,
        executors=executors,
        scheduler=scheduler,
        execution=execution,
        source=raw,
    )


def _require_int(
    value: Any,
    *,
    field: str,
    block_group_id: str | None = None,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(
            "value must be an integer",
            field=field,
            block_group_id=block_group_id,
        )
    if minimum is not None and value < minimum:
        raise SchemaError(
            f"value must be >= {minimum}",
            field=field,
            block_group_id=block_group_id,
        )
    return value


def _parse_block_group(group_id: str, raw: dict[str, Any]) -> BlockGroupDecl:
    row_count = _require_int(
        raw.get("row_count"),
        field="row_count",
        block_group_id=group_id,
        minimum=1,
    )
    seed = _require_int(raw.get("seed"), field="seed", block_group_id=group_id)

    raw_columns = raw.get("columns")
    if not isinstance(raw_columns, list) or not raw_columns:
        raise SchemaError(
            "columns must be a non-empty array",
            field="columns",
            block_group_id=group_id,
        )

    seen_columns: set[str] = set()
    columns: list[ColumnDecl] = []
    for ordinal, raw_column in enumerate(raw_columns):
        if not isinstance(raw_column, dict):
            raise SchemaError(
                "column must be an object",
                field=f"columns[{ordinal}]",
                block_group_id=group_id,
            )
        name = raw_column.get("name")
        if not isinstance(name, str) or not name:
            raise SchemaError(
                "column name must be a non-empty string",
                field=f"columns[{ordinal}].name",
                block_group_id=group_id,
            )
        if name in seen_columns:
            raise SchemaError(
                "column name must be unique within a block group",
                field="columns[].name",
                block_group_id=group_id,
                details={"column": name},
            )
        seen_columns.add(name)

        encoding = raw_column.get("encoding")
        if encoding not in SUPPORTED_ENCODINGS:
            raise SchemaError(
                f"encoding must be one of {list(SUPPORTED_ENCODINGS)}",
                field=f"columns[{name}].encoding",
                block_group_id=group_id,
            )

        raw_histogram = raw_column.get("histogram")
        if not isinstance(raw_histogram, dict):
            raise SchemaError(
                "histogram must be an object",
                field=f"columns[{name}].histogram",
                block_group_id=group_id,
            )
        histogram = _parse_histogram(
            raw_histogram,
            row_count=row_count,
            block_group_id=group_id,
            column_name=name,
        )
        if encoding == "int32":
            if histogram.bounds[0] < -(2**31) or histogram.bounds[-1] - 1 > 2**31 - 1:
                raise SchemaError(
                    "histogram range does not fit int32 encoding",
                    field=f"columns[{name}].histogram.bounds",
                    block_group_id=group_id,
                )
        columns.append(ColumnDecl(name=name, encoding=encoding, histogram=histogram))

    columns.sort(key=lambda column: column.name)
    return BlockGroupDecl(
        id=group_id,
        row_count=row_count,
        seed=seed,
        columns=tuple(columns),
    )


def _parse_histogram(
    raw: dict[str, Any],
    *,
    row_count: int,
    block_group_id: str,
    column_name: str,
) -> Histogram:
    raw_bounds = raw.get("bounds")
    raw_counts = raw.get("counts")
    field_prefix = f"columns[{column_name}].histogram"
    if not isinstance(raw_bounds, list) or len(raw_bounds) < 2:
        raise SchemaError(
            "histogram bounds must contain at least two integers",
            field=f"{field_prefix}.bounds",
            block_group_id=block_group_id,
        )
    if not isinstance(raw_counts, list) or not raw_counts:
        raise SchemaError(
            "histogram counts must be a non-empty array",
            field=f"{field_prefix}.counts",
            block_group_id=block_group_id,
        )
    if len(raw_counts) != len(raw_bounds) - 1:
        raise SchemaError(
            "histogram counts length must equal bounds length minus one",
            field=f"{field_prefix}.counts",
            block_group_id=block_group_id,
        )
    bounds = tuple(
        _require_int(value, field=f"{field_prefix}.bounds[{index}]", block_group_id=block_group_id)
        for index, value in enumerate(raw_bounds)
    )
    for index in range(len(bounds) - 1):
        if bounds[index] >= bounds[index + 1]:
            raise SchemaError(
                "histogram bounds must be strictly increasing and non-overlapping",
                field=f"{field_prefix}.bounds[{index}]",
                block_group_id=block_group_id,
            )
    counts = tuple(
        _require_int(
            value,
            field=f"{field_prefix}.counts[{index}]",
            block_group_id=block_group_id,
            minimum=0,
        )
        for index, value in enumerate(raw_counts)
    )
    if sum(counts) != row_count:
        raise SchemaError(
            "histogram counts must sum to row_count",
            field=f"{field_prefix}.counts",
            block_group_id=block_group_id,
            details={"sum": sum(counts), "row_count": row_count},
        )
    return Histogram(bounds=bounds, counts=counts)


def parse_column_ref(value: Any, *, field: str) -> ColumnRef:
    if not isinstance(value, str) or value.count(".") != 1:
        raise SchemaError(
            "column reference must have form 'block_group.column'",
            field=field,
        )
    block_group, column = value.split(".", 1)
    if not block_group or not column:
        raise SchemaError(
            "column reference must have form 'block_group.column'",
            field=field,
        )
    return ColumnRef(block_group, column)


def _check_ref(
    ref: ColumnRef,
    groups: dict[str, BlockGroupDecl],
    *,
    field: str,
) -> None:
    if ref.block_group not in groups:
        raise SchemaError(
            "column reference names an unknown block group",
            field=field,
            block_group_id=ref.block_group,
            details={"column": ref.column},
        )
    try:
        groups[ref.block_group].column(ref.column)
    except KeyError as exc:
        raise SchemaError(
            "column reference names an unknown column",
            field=field,
            block_group_id=ref.block_group,
            details={"column": ref.column},
        ) from exc


def _parse_query(raw: dict[str, Any], groups: tuple[BlockGroupDecl, ...]) -> QueryDecl:
    groups_by_id = {group.id: group for group in groups}
    query_id = raw.get("id")
    if not isinstance(query_id, str) or not query_id:
        raise SchemaError("query id must be a non-empty string", field="query.id")

    raw_joins = raw.get("joins", [])
    raw_filters = raw.get("filters", [])
    raw_projections = raw.get("projections", [])
    raw_group_by = raw.get("group_by", [])
    raw_aggregates = raw.get("aggregates", [])
    for name, value in (
        ("query.joins", raw_joins),
        ("query.filters", raw_filters),
        ("query.projections", raw_projections),
        ("query.group_by", raw_group_by),
        ("query.aggregates", raw_aggregates),
    ):
        if not isinstance(value, list):
            raise SchemaError(f"{name} must be an array", field=name)

    joins: list[JoinPredicate] = []
    referenced_tables: set[str] = set()
    equality_edges: list[tuple[str, str]] = []
    for ordinal, raw_join in enumerate(raw_joins):
        if not isinstance(raw_join, dict):
            raise SchemaError("join predicate must be an object", field=f"query.joins[{ordinal}]")
        left = parse_column_ref(raw_join.get("left"), field=f"query.joins[{ordinal}].left")
        right = parse_column_ref(raw_join.get("right"), field=f"query.joins[{ordinal}].right")
        _check_ref(left, groups_by_id, field=f"query.joins[{ordinal}].left")
        _check_ref(right, groups_by_id, field=f"query.joins[{ordinal}].right")
        if left.block_group == right.block_group:
            raise SchemaError(
                "join predicate must connect two different block groups",
                field=f"query.joins[{ordinal}]",
                block_group_id=left.block_group,
            )
        operator = raw_join.get("operator", "=")
        if operator not in SUPPORTED_JOIN_OPS:
            raise SchemaError(
                f"join operator must be one of {list(SUPPORTED_JOIN_OPS)}",
                field=f"query.joins[{ordinal}].operator",
                block_group_id=left.block_group,
            )
        normalized = JoinPredicate(left, right, operator).normalized()
        joins.append(normalized)
        referenced_tables.update((left.block_group, right.block_group))
        if normalized.is_equality:
            equality_edges.append((normalized.left.block_group, normalized.right.block_group))

    filters: list[FilterPredicate] = []
    for ordinal, raw_filter in enumerate(raw_filters):
        if not isinstance(raw_filter, dict):
            raise SchemaError("filter predicate must be an object", field=f"query.filters[{ordinal}]")
        column = parse_column_ref(raw_filter.get("column"), field=f"query.filters[{ordinal}].column")
        _check_ref(column, groups_by_id, field=f"query.filters[{ordinal}].column")
        operator = raw_filter.get("operator")
        if operator not in SUPPORTED_FILTER_OPS:
            raise SchemaError(
                f"filter operator must be one of {list(SUPPORTED_FILTER_OPS)}",
                field=f"query.filters[{ordinal}].operator",
                block_group_id=column.block_group,
            )
        raw_value = raw_filter.get("value")
        if operator == "in":
            if not isinstance(raw_value, list) or not raw_value:
                raise SchemaError(
                    "in filter value must be a non-empty integer array",
                    field=f"query.filters[{ordinal}].value",
                    block_group_id=column.block_group,
                )
            values = tuple(
                _require_int(
                    value,
                    field=f"query.filters[{ordinal}].value[{index}]",
                    block_group_id=column.block_group,
                )
                for index, value in enumerate(raw_value)
            )
            value: int | tuple[int, ...] = values
        else:
            value = _require_int(
                raw_value,
                field=f"query.filters[{ordinal}].value",
                block_group_id=column.block_group,
            )
        filters.append(FilterPredicate(column, operator, value))
        referenced_tables.add(column.block_group)

    projections: list[Projection] = []
    for ordinal, raw_projection in enumerate(raw_projections):
        if not isinstance(raw_projection, dict):
            raise SchemaError(
                "projection must be an object",
                field=f"query.projections[{ordinal}]",
            )
        column = parse_column_ref(
            raw_projection.get("column"),
            field=f"query.projections[{ordinal}].column",
        )
        _check_ref(column, groups_by_id, field=f"query.projections[{ordinal}].column")
        alias = raw_projection.get("alias", column.column)
        if not isinstance(alias, str) or not alias:
            raise SchemaError(
                "projection alias must be a non-empty string",
                field=f"query.projections[{ordinal}].alias",
                block_group_id=column.block_group,
            )
        projections.append(Projection(column, alias))
        referenced_tables.add(column.block_group)

    group_by: list[ColumnRef] = []
    for ordinal, raw_ref in enumerate(raw_group_by):
        column = parse_column_ref(raw_ref, field=f"query.group_by[{ordinal}]")
        _check_ref(column, groups_by_id, field=f"query.group_by[{ordinal}]")
        group_by.append(column)
        referenced_tables.add(column.block_group)

    aggregates: list[Aggregate] = []
    for ordinal, raw_aggregate in enumerate(raw_aggregates):
        if not isinstance(raw_aggregate, dict):
            raise SchemaError(
                "aggregate must be an object",
                field=f"query.aggregates[{ordinal}]",
            )
        function = raw_aggregate.get("function")
        if function not in SUPPORTED_AGGREGATES:
            raise SchemaError(
                f"aggregate function must be one of {list(SUPPORTED_AGGREGATES)}",
                field=f"query.aggregates[{ordinal}].function",
            )
        input_value = raw_aggregate.get("input")
        if function == "count_star":
            if input_value is not None:
                raise SchemaError(
                    "count_star does not accept an input column",
                    field=f"query.aggregates[{ordinal}].input",
                )
            input_ref = None
        else:
            input_ref = parse_column_ref(
                input_value,
                field=f"query.aggregates[{ordinal}].input",
            )
            _check_ref(input_ref, groups_by_id, field=f"query.aggregates[{ordinal}].input")
            referenced_tables.add(input_ref.block_group)
        alias = raw_aggregate.get("alias")
        if not isinstance(alias, str) or not alias:
            raise SchemaError(
                "aggregate alias must be a non-empty string",
                field=f"query.aggregates[{ordinal}].alias",
            )
        aggregates.append(Aggregate(function, input_ref, alias))

    if not joins and not filters and not projections and not aggregates:
        raise SchemaError("query must select at least one table", field="query")

    all_aliases = [item.alias for item in projections] + [item.alias for item in aggregates]
    if len(set(all_aliases)) != len(all_aliases):
        raise SchemaError("projection and aggregate aliases must be unique", field="query.aliases")

    if not aggregates and group_by:
        raise SchemaError("group_by requires at least one aggregate", field="query.group_by")
    if aggregates and not group_by:
        group_by = []
    if aggregates:
        allowed_group_columns = set(group_by)
        for projection in projections:
            if projection.column not in allowed_group_columns:
                raise SchemaError(
                    "non-aggregate projection must appear in group_by",
                    field="query.projections",
                    block_group_id=projection.column.block_group,
                    details={"column": projection.column.qualified},
                )

    tables = tuple(sorted(referenced_tables))
    if not tables:
        raise SchemaError("query references no block groups", field="query")

    _check_equality_connectivity(tables, equality_edges)
    joins.sort(key=lambda item: (item.left.block_group, item.left.column, item.right.block_group, item.right.column, item.operator))
    filters.sort(key=lambda item: (item.column.block_group, item.column.column, item.operator, _value_sort_key(item.value)))
    group_by.sort()
    return QueryDecl(
        id=query_id,
        joins=tuple(joins),
        filters=tuple(filters),
        projections=tuple(projections),
        group_by=tuple(group_by),
        aggregates=tuple(aggregates),
        tables=tables,
    )


def _value_sort_key(value: int | tuple[int, ...]) -> tuple[int, ...]:
    return value if isinstance(value, tuple) else (value,)


def _check_equality_connectivity(tables: tuple[str, ...], edges: Iterable[tuple[str, str]]) -> None:
    if len(tables) <= 1:
        return
    adjacency: dict[str, set[str]] = {table: set() for table in tables}
    for left, right in edges:
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)
    visited = {tables[0]}
    frontier = [tables[0]]
    while frontier:
        current = frontier.pop()
        for neighbor in sorted(adjacency[current]):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    if visited != set(tables):
        missing = sorted(set(tables) - visited)
        raise SchemaError(
            "query join graph is disconnected without equality predicates",
            field="query.joins",
            block_group_id=missing[0],
            details={"unreachable_block_groups": missing},
        )


def _parse_cost_model(raw: dict[str, Any]) -> CostModel:
    weights = raw.get("weights")
    if not isinstance(weights, dict):
        raise SchemaError("cost_model.weights must be an object", field="cost_model.weights")
    weight_values = {}
    for name in ("access_row", "comparison", "memory_byte", "intermediate_row"):
        weight_values[name] = _require_int(
            weights.get(name),
            field=f"cost_model.weights.{name}",
            minimum=0,
        )

    scales = raw.get("scale_factors")
    if not isinstance(scales, dict):
        raise SchemaError(
            "cost_model.scale_factors must be an object",
            field="cost_model.scale_factors",
        )
    fixed_scale = _require_int(
        scales.get("fixed"),
        field="cost_model.scale_factors.fixed",
        minimum=1,
    )
    batch_rows = _require_int(
        scales.get("batch_rows"),
        field="cost_model.scale_factors.batch_rows",
        minimum=1,
    )
    for name, value in (("fixed", fixed_scale), ("batch_rows", batch_rows)):
        if value not in SUPPORTED_SCALES:
            raise SchemaError(
                f"scale factor must be one of {list(SUPPORTED_SCALES)}",
                field=f"cost_model.scale_factors.{name}",
                details={"value": value},
            )

    error_band = _require_int(
        raw.get("error_band", 0),
        field="cost_model.error_band",
        minimum=0,
    )
    if error_band > 2**63 - 1:
        raise SchemaError("error_band is outside signed 64-bit range", field="cost_model.error_band")

    return CostModel(
        access_row_weight=weight_values["access_row"],
        comparison_weight=weight_values["comparison"],
        memory_byte_weight=weight_values["memory_byte"],
        intermediate_row_weight=weight_values["intermediate_row"],
        fixed_scale=fixed_scale,
        batch_rows=batch_rows,
        error_band=error_band,
    )


def _parse_executors(raw: Any, groups: tuple[BlockGroupDecl, ...]) -> tuple[ExecutorSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise SchemaError("executors must be a non-empty array", field="executors")
    specs: list[ExecutorSpec] = []
    seen: set[str] = set()
    for ordinal, raw_spec in enumerate(raw):
        if not isinstance(raw_spec, dict):
            raise SchemaError("executor must be an object", field=f"executors[{ordinal}]")
        executor_id = raw_spec.get("id")
        if not isinstance(executor_id, str) or not executor_id:
            raise SchemaError("executor id must be a non-empty string", field=f"executors[{ordinal}].id")
        if executor_id in seen:
            raise SchemaError("executor id must be unique", field="executors[].id")
        seen.add(executor_id)
        host = raw_spec.get("host", "127.0.0.1")
        if host != "127.0.0.1":
            raise SchemaError("executor host must be 127.0.0.1", field=f"executors[{ordinal}].host")
        port = _require_int(raw_spec.get("port"), field=f"executors[{ordinal}].port", minimum=0)
        if port > 65535:
            raise SchemaError("executor port must be <= 65535", field=f"executors[{ordinal}].port")
        concurrency = _require_int(
            raw_spec.get("concurrency"),
            field=f"executors[{ordinal}].concurrency",
            minimum=1,
        )
        specs.append(ExecutorSpec(executor_id, host, port, concurrency))
    specs.sort(key=lambda item: item.id)
    return tuple(specs)


def _parse_scheduler(raw: Any) -> SchedulerConfig:
    if not isinstance(raw, dict):
        raise SchemaError("scheduler must be an object", field="scheduler")
    global_inflight = _require_int(
        raw.get("global_inflight"),
        field="scheduler.global_inflight",
        minimum=1,
    )
    max_lease_seconds = raw.get("max_lease_seconds")
    if isinstance(max_lease_seconds, bool) or not isinstance(max_lease_seconds, (int, float)):
        raise SchemaError("max_lease_seconds must be a number", field="scheduler.max_lease_seconds")
    if not 0 < float(max_lease_seconds) <= 3600:
        raise SchemaError("max_lease_seconds must be in (0, 3600]", field="scheduler.max_lease_seconds")
    lease_seconds = raw.get("lease_seconds", max_lease_seconds)
    if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float)):
        raise SchemaError("lease_seconds must be a number", field="scheduler.lease_seconds")
    if not 0 < float(lease_seconds) <= float(max_lease_seconds):
        raise SchemaError("lease_seconds must be in (0, max_lease_seconds]", field="scheduler.lease_seconds")
    return SchedulerConfig(
        global_inflight=global_inflight,
        lease_seconds=float(lease_seconds),
        max_lease_seconds=float(max_lease_seconds),
    )


def _parse_execution(
    raw: Any,
    groups: tuple[BlockGroupDecl, ...],
    query: QueryDecl,
) -> ExecutionSpec:
    if not isinstance(raw, dict):
        raise SchemaError("execution must be an object", field="execution")
    shard_count = _require_int(raw.get("shard_count"), field="execution.shard_count", minimum=1)
    raw_partition = raw.get("partition")
    if not isinstance(raw_partition, dict):
        raise SchemaError("execution.partition must be an object", field="execution.partition")
    driver = raw_partition.get("driver")
    if not isinstance(driver, str) or driver not in query.tables:
        raise SchemaError("partition driver must be a query block group", field="execution.partition.driver")
    raw_co = raw_partition.get("co_partition")
    if not isinstance(raw_co, list) or not raw_co:
        raise SchemaError("co_partition must be a non-empty array", field="execution.partition.co_partition")
    co: list[tuple[str, ColumnRef]] = []
    for ordinal, item in enumerate(raw_co):
        if not isinstance(item, dict):
            raise SchemaError("co_partition entry must be an object", field=f"execution.partition.co_partition[{ordinal}]")
        table = item.get("table")
        if not isinstance(table, str) or table not in query.tables:
            raise SchemaError("co_partition table must be a query block group", field=f"execution.partition.co_partition[{ordinal}].table")
        ref = parse_column_ref(item.get("column"), field=f"execution.partition.co_partition[{ordinal}].column")
        _check_ref(ref, {group.id: group for group in groups}, field=f"execution.partition.co_partition[{ordinal}].column")
        if ref.block_group != table:
            raise SchemaError("co_partition column must belong to its table", field=f"execution.partition.co_partition[{ordinal}].column")
        co.append((table, ref))
    raw_replicate = raw_partition.get("replicate", [])
    if not isinstance(raw_replicate, list):
        raise SchemaError("replicate must be an array", field="execution.partition.replicate")
    replicate: list[str] = []
    for ordinal, table in enumerate(raw_replicate):
        if not isinstance(table, str) or table not in query.tables:
            raise SchemaError("replicate table must be a query block group", field=f"execution.partition.replicate[{ordinal}]")
        replicate.append(table)
    co_tables = {table for table, _ in co}
    if co_tables & set(replicate):
        raise SchemaError("a table cannot be both co-partitioned and replicated", field="execution.partition")
    if co_tables | set(replicate) != set(query.tables):
        raise SchemaError("partition must cover every query table", field="execution.partition")
    return ExecutionSpec(
        shard_count=shard_count,
        partition=PartitionSpec(
            driver=driver,
            co_partition=tuple(sorted(co, key=lambda item: item[0])),
            replicate=tuple(sorted(replicate)),
        ),
    )
