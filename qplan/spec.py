from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import content_digest, normalize
from .errors import QPlanError, spec_error


SUPPORTED_TYPES = {"integer", "real", "float"}
SUPPORTED_OPS = {"eq", "ne", "lt", "le", "gt", "ge", "between", "in"}
RESERVED_TABLE_NAMES = {"qplan_meta"}


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    type: str
    selectivity_hint: float
    nullable: bool
    null_fraction: float
    generator: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "selectivity_hint": self.selectivity_hint,
            "nullable": self.nullable,
            "null_fraction": self.null_fraction,
            "generator": normalize(self.generator),
        }


@dataclass(frozen=True)
class IndexSpec:
    name: str
    table: str
    columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "table": self.table, "columns": list(self.columns)}


@dataclass(frozen=True)
class TableSpec:
    name: str
    row_count: int
    columns: tuple[ColumnSpec, ...]
    candidate_indexes: tuple[IndexSpec, ...]

    def column(self, name: str) -> ColumnSpec:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "row_count": self.row_count,
            "columns": [column.to_dict() for column in self.columns],
            "candidate_indexes": [index.to_dict() for index in self.candidate_indexes],
        }


@dataclass(frozen=True)
class ColumnRef:
    table: str
    column: str

    def to_dict(self) -> dict[str, str]:
        return {"table": self.table, "column": self.column}


@dataclass(frozen=True)
class FilterSpec:
    table: str
    column: str
    op: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return {"table": self.table, "column": self.column, "op": self.op, "value": normalize(self.value)}


@dataclass(frozen=True)
class JoinSpec:
    left_table: str
    left_column: str
    right_table: str
    right_column: str

    def columns_for(self, table: str) -> tuple[str, ...]:
        result = []
        if self.left_table == table:
            result.append(self.left_column)
        if self.right_table == table:
            result.append(self.right_column)
        return tuple(result)

    def other_table(self, table: str) -> str:
        if table == self.left_table:
            return self.right_table
        if table == self.right_table:
            return self.left_table
        raise KeyError(table)

    def to_dict(self) -> dict[str, str]:
        return {
            "left_table": self.left_table,
            "left_column": self.left_column,
            "right_table": self.right_table,
            "right_column": self.right_column,
        }


@dataclass(frozen=True)
class QuerySpec:
    id: str
    tables: tuple[str, ...]
    filters: tuple[FilterSpec, ...]
    joins: tuple[JoinSpec, ...]
    select: tuple[ColumnRef, ...]

    def filters_for(self, table: str) -> tuple[FilterSpec, ...]:
        return tuple(item for item in self.filters if item.table == table)

    def joins_for(self, table: str) -> tuple[JoinSpec, ...]:
        return tuple(item for item in self.joins if item.left_table == table or item.right_table == table)

    def needed_columns(self, table: str, spec: "Spec") -> tuple[str, ...]:
        names: list[str] = []
        for item in self.filters_for(table):
            if item.column not in names:
                names.append(item.column)
        for join in self.joins_for(table):
            for column in join.columns_for(table):
                if column not in names:
                    names.append(column)
        for ref in self.select:
            if ref.table == table and ref.column not in names:
                names.append(ref.column)
        return tuple(names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tables": list(self.tables),
            "filters": [item.to_dict() for item in self.filters],
            "joins": [item.to_dict() for item in self.joins],
            "select": [item.to_dict() for item in self.select],
        }


@dataclass(frozen=True)
class CostWeights:
    rows: int = 1
    comparisons: int = 1
    lookups: int = 4
    io_pages: int = 2

    def to_dict(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "comparisons": self.comparisons,
            "lookups": self.lookups,
            "io_pages": self.io_pages,
        }


@dataclass(frozen=True)
class StatsConfig:
    version: str = "stats-v1"
    row_threshold: int = 5000
    sample_size: int = 2048
    confidence: float = 0.95
    histogram_buckets: int = 16
    value_count_limit: int = 4096
    page_rows: int = 100
    random_seed: int = 20261123

    @property
    def caliber_version(self) -> str:
        return (
            f"{self.version}:full<{self.row_threshold},sample={self.sample_size},"
            f"ci={self.confidence:g},hist={self.histogram_buckets},seed={self.random_seed}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "row_threshold": self.row_threshold,
            "sample_size": self.sample_size,
            "confidence": self.confidence,
            "histogram_buckets": self.histogram_buckets,
            "value_count_limit": self.value_count_limit,
            "page_rows": self.page_rows,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True)
class SearchConfig:
    exhaustive_max_candidates: int = 6
    oracle_max_candidates: int = 12

    def to_dict(self) -> dict[str, int]:
        return {
            "exhaustive_max_candidates": self.exhaustive_max_candidates,
            "oracle_max_candidates": self.oracle_max_candidates,
        }


@dataclass(frozen=True)
class ExecutionConfig:
    warmup_runs: int = 1
    repetitions: int = 2

    def to_dict(self) -> dict[str, int]:
        return {"warmup_runs": self.warmup_runs, "repetitions": self.repetitions}


@dataclass(frozen=True)
class BiasConfig:
    threshold: float = 1.75

    def to_dict(self) -> dict[str, float]:
        return {"threshold": self.threshold}


@dataclass(frozen=True)
class Spec:
    name: str
    tables: tuple[TableSpec, ...]
    queries: tuple[QuerySpec, ...]
    cost_weights: CostWeights
    stats: StatsConfig
    search: SearchConfig
    execution: ExecutionConfig
    bias: BiasConfig
    spec_version: int
    seed: int

    def table(self, name: str) -> TableSpec:
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(name)

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(table.name for table in self.tables)

    @property
    def candidate_indexes(self) -> tuple[IndexSpec, ...]:
        return tuple(index for table in self.tables for index in table.candidate_indexes)

    def query(self, query_id: str) -> QuerySpec:
        for query in self.queries:
            if query.id == query_id:
                return query
        raise KeyError(query_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "seed": self.seed,
            "tables": [table.to_dict() for table in self.tables],
            "queries": [query.to_dict() for query in self.queries],
            "cost_weights": self.cost_weights.to_dict(),
            "stats": self.stats.to_dict(),
            "search": self.search.to_dict(),
            "execution": self.execution.to_dict(),
            "bias": self.bias.to_dict(),
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)


def _require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise spec_error(location, "must be a JSON object")
    return value


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise spec_error(location, "must be a JSON array")
    return value


def _require_string(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise spec_error(location, "must be a string")
    return value


def _safe_ident(value: Any, location: str, *, query_id: bool = False) -> str:
    text = _require_string(value, location)
    if not text:
        raise spec_error(location, "must not be empty")
    if "\x00" in text or any(ord(character) < 32 for character in text):
        raise spec_error(location, "must not contain NUL or control characters")
    return text


def _integer(value: Any, location: str, *, minimum: int | None = None) -> int:
    if not _is_int(value):
        raise spec_error(location, "must be an integer")
    if minimum is not None and value < minimum:
        raise spec_error(location, f"must be >= {minimum}")
    return value


def _number(value: Any, location: str) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise spec_error(location, "must be a finite number")
    return float(value)


def _check_type_value(value: Any, column_type: str, location: str) -> Any:
    if column_type == "integer":
        if not _is_int(value):
            raise spec_error(location, "must be an integer for an integer column")
        return value
    if not _is_number(value) or not math.isfinite(float(value)):
        raise spec_error(location, "must be a finite number for a float/real column")
    return value


def _validate_generator(value: Any, column_type: str, location: str) -> dict[str, Any]:
    if value is None:
        return {}
    generator = _require_object(value, location)
    kind = _require_string(generator.get("kind"), f"{location}.kind")
    if kind == "unique_int":
        start = generator.get("start", 1)
        _integer(start, f"{location}.start")
        if column_type != "integer":
            raise spec_error(location, "unique_int generator requires an integer column")
    elif kind == "uniform_int":
        low = _integer(generator.get("min"), f"{location}.min")
        high = _integer(generator.get("max"), f"{location}.max")
        if low > high:
            raise spec_error(f"{location}.min", "must be <= max")
        if column_type != "integer":
            raise spec_error(location, "uniform_int generator requires an integer column")
    elif kind == "categorical":
        values = _require_list(generator.get("values"), f"{location}.values")
        if not values:
            raise spec_error(f"{location}.values", "must not be empty")
        for index, item in enumerate(values):
            _check_type_value(item, column_type, f"{location}.values[{index}]")
        weights = generator.get("weights")
        if weights is not None:
            weight_values = _require_list(weights, f"{location}.weights")
            if len(weight_values) != len(values):
                raise spec_error(f"{location}.weights", "must have the same length as values")
            for index, item in enumerate(weight_values):
                weight = _number(item, f"{location}.weights[{index}]")
                if weight < 0:
                    raise spec_error(f"{location}.weights[{index}]", "must be >= 0")
            if sum(float(item) for item in weight_values) <= 0:
                raise spec_error(f"{location}.weights", "sum must be > 0")
    elif kind == "uniform_float":
        low = _number(generator.get("min"), f"{location}.min")
        high = _number(generator.get("max"), f"{location}.max")
        if low > high:
            raise spec_error(f"{location}.min", "must be <= max")
        decimals = generator.get("decimals", 6)
        _integer(decimals, f"{location}.decimals", minimum=0)
        if decimals > 12:
            raise spec_error(f"{location}.decimals", "must be <= 12")
        if column_type == "integer":
            raise spec_error(location, "uniform_float generator requires a float/real column")
    elif kind == "foreign_key":
        _safe_ident(generator.get("table"), f"{location}.table")
        _safe_ident(generator.get("column"), f"{location}.column")
        if column_type != "integer":
            raise spec_error(location, "foreign_key generator requires an integer column")
    elif kind == "constant":
        _check_type_value(generator.get("value"), column_type, f"{location}.value")
    else:
        raise spec_error(f"{location}.kind", f"unsupported generator kind {kind!r}")
    return normalize(generator)


def _parse_weights(raw: Any) -> CostWeights:
    if raw is None:
        return CostWeights()
    obj = _require_object(raw, "cost_weights")
    values: dict[str, int] = {}
    for key in ("rows", "comparisons", "lookups", "io_pages"):
        values[key] = _integer(obj.get(key, getattr(CostWeights(), key)), f"cost_weights.{key}", minimum=0)
    return CostWeights(**values)


def _parse_stats(raw: Any) -> StatsConfig:
    if raw is None:
        return StatsConfig()
    obj = _require_object(raw, "stats")
    version = _require_string(obj.get("version", "stats-v1"), "stats.version")
    if not version:
        raise spec_error("stats.version", "must not be empty")
    row_threshold = _integer(obj.get("row_threshold", 5000), "stats.row_threshold", minimum=1)
    sample_size = _integer(obj.get("sample_size", 2048), "stats.sample_size", minimum=1)
    confidence = _number(obj.get("confidence", 0.95), "stats.confidence")
    if not 0.5 < confidence < 1.0:
        raise spec_error("stats.confidence", "must be between 0.5 and 1.0 (exclusive)")
    histogram_buckets = _integer(obj.get("histogram_buckets", 16), "stats.histogram_buckets", minimum=1)
    value_count_limit = _integer(obj.get("value_count_limit", 4096), "stats.value_count_limit", minimum=1)
    page_rows = _integer(obj.get("page_rows", 100), "stats.page_rows", minimum=1)
    random_seed = _integer(obj.get("random_seed", 20261123), "stats.random_seed", minimum=0)
    return StatsConfig(
        version=version,
        row_threshold=row_threshold,
        sample_size=sample_size,
        confidence=confidence,
        histogram_buckets=histogram_buckets,
        value_count_limit=value_count_limit,
        page_rows=page_rows,
        random_seed=random_seed,
    )


def _parse_search(raw: Any) -> SearchConfig:
    if raw is None:
        return SearchConfig()
    obj = _require_object(raw, "search")
    exhaustive = _integer(obj.get("exhaustive_max_candidates", 6), "search.exhaustive_max_candidates", minimum=0)
    oracle = _integer(obj.get("oracle_max_candidates", 12), "search.oracle_max_candidates", minimum=0)
    if oracle < exhaustive:
        raise spec_error("search.oracle_max_candidates", "must be >= exhaustive_max_candidates")
    return SearchConfig(exhaustive_max_candidates=exhaustive, oracle_max_candidates=oracle)


def _parse_execution(raw: Any) -> ExecutionConfig:
    if raw is None:
        return ExecutionConfig()
    obj = _require_object(raw, "execution")
    return ExecutionConfig(
        warmup_runs=_integer(obj.get("warmup_runs", 1), "execution.warmup_runs", minimum=0),
        repetitions=_integer(obj.get("repetitions", 2), "execution.repetitions", minimum=1),
    )


def _parse_bias(raw: Any) -> BiasConfig:
    if raw is None:
        return BiasConfig()
    obj = _require_object(raw, "bias")
    threshold = _number(obj.get("threshold", 1.75), "bias.threshold")
    if threshold < 1.0:
        raise spec_error("bias.threshold", "must be >= 1.0")
    return BiasConfig(threshold=threshold)


def parse_spec(raw: Any) -> Spec:
    root = _require_object(raw, "$")
    name = _require_string(root.get("name", "local-workload"), "name")
    spec_version = _integer(root.get("spec_version", 1), "spec_version", minimum=1)
    seed = _integer(root.get("seed", 20261123), "seed", minimum=0)
    tables_raw = _require_list(root.get("tables"), "tables")
    if not tables_raw:
        raise spec_error("tables", "must contain at least one table")

    tables: list[TableSpec] = []
    table_names: set[str] = set()
    index_names: set[str] = set()
    for table_index, table_value in enumerate(tables_raw):
        table_loc = f"tables[{table_index}]"
        table_obj = _require_object(table_value, table_loc)
        table_name = _safe_ident(table_obj.get("name"), f"{table_loc}.name")
        if table_name in table_names:
            raise spec_error(f"{table_loc}.name", f"duplicate table name {table_name!r}")
        if table_name in RESERVED_TABLE_NAMES or table_name.lower().startswith("sqlite_"):
            raise spec_error(f"{table_loc}.name", f"table name {table_name!r} is reserved by the platform or SQLite")
        table_names.add(table_name)
        row_count = _integer(table_obj.get("row_count"), f"{table_loc}.row_count", minimum=1)

        columns_raw = _require_list(table_obj.get("columns"), f"{table_loc}.columns")
        if not columns_raw:
            raise spec_error(f"{table_loc}.columns", "must contain at least one column")
        columns: list[ColumnSpec] = []
        column_names: set[str] = set()
        for column_index, column_value in enumerate(columns_raw):
            column_loc = f"{table_loc}.columns[{column_index}]"
            column_obj = _require_object(column_value, column_loc)
            column_name = _safe_ident(column_obj.get("name"), f"{column_loc}.name")
            if column_name in column_names:
                raise spec_error(f"{column_loc}.name", f"duplicate column name {column_name!r} in table {table_name!r}")
            column_names.add(column_name)
            column_type = _require_string(column_obj.get("type"), f"{column_loc}.type").lower()
            if column_type not in SUPPORTED_TYPES:
                raise spec_error(f"{column_loc}.type", f"unsupported type {column_type!r}; expected integer or float/real")
            if column_type == "float":
                column_type = "real"
            hint = _number(column_obj.get("selectivity_hint"), f"{column_loc}.selectivity_hint")
            if not 0.0 < hint <= 1.0:
                raise spec_error(f"{column_loc}.selectivity_hint", "must be > 0 and <= 1")
            nullable = column_obj.get("nullable", False)
            if not isinstance(nullable, bool):
                raise spec_error(f"{column_loc}.nullable", "must be a boolean")
            null_fraction = _number(column_obj.get("null_fraction", 0.0), f"{column_loc}.null_fraction")
            if not 0.0 <= null_fraction < 1.0:
                raise spec_error(f"{column_loc}.null_fraction", "must be >= 0 and < 1")
            if null_fraction > 0 and not nullable:
                raise spec_error(f"{column_loc}.null_fraction", "must be 0 when nullable is false")
            generator = _validate_generator(column_obj.get("generator"), column_type, f"{column_loc}.generator")
            columns.append(
                ColumnSpec(
                    name=column_name,
                    type=column_type,
                    selectivity_hint=hint,
                    nullable=nullable,
                    null_fraction=null_fraction,
                    generator=generator,
                )
            )

        indexes_raw = table_obj.get("candidate_indexes", [])
        indexes_list = _require_list(indexes_raw, f"{table_loc}.candidate_indexes")
        indexes: list[IndexSpec] = []
        for index_index, index_value in enumerate(indexes_list):
            index_loc = f"{table_loc}.candidate_indexes[{index_index}]"
            index_obj = _require_object(index_value, index_loc)
            index_name = _safe_ident(index_obj.get("name"), f"{index_loc}.name")
            if index_name in index_names:
                raise spec_error(f"{index_loc}.name", f"duplicate index name {index_name!r}")
            index_names.add(index_name)
            index_columns_raw = _require_list(index_obj.get("columns"), f"{index_loc}.columns")
            if not index_columns_raw:
                raise spec_error(f"{index_loc}.columns", "must contain at least one column")
            index_columns: list[str] = []
            for column_index, column_value in enumerate(index_columns_raw):
                column_name = _safe_ident(column_value, f"{index_loc}.columns[{column_index}]")
                if column_name not in column_names:
                    raise spec_error(
                        f"{index_loc}.columns[{column_index}]", f"unknown column {column_name!r} in table {table_name!r}"
                    )
                if column_name in index_columns:
                    raise spec_error(f"{index_loc}.columns[{column_index}]", f"duplicate index column {column_name!r}")
                index_columns.append(column_name)
            indexes.append(IndexSpec(name=index_name, table=table_name, columns=tuple(index_columns)))
        tables.append(TableSpec(name=table_name, row_count=row_count, columns=tuple(columns), candidate_indexes=tuple(indexes)))

    table_map = {table.name: table for table in tables}
    for table_index, table in enumerate(tables):
        for index_index, index in enumerate(table.candidate_indexes):
            if index.name in table_names:
                raise spec_error(
                    f"tables[{table_index}].candidate_indexes[{index_index}].name",
                    f"index name {index.name!r} conflicts with a table name",
                )
            if index.name in RESERVED_TABLE_NAMES or index.name.lower().startswith("sqlite_"):
                raise spec_error(
                    f"tables[{table_index}].candidate_indexes[{index_index}].name",
                    f"index name {index.name!r} is reserved by the platform or SQLite",
                )
    for table_index, table in enumerate(tables):
        for column_index, column in enumerate(table.columns):
            generator = column.generator
            if generator.get("kind") != "foreign_key":
                continue
            target_table = generator["table"]
            target_column = generator["column"]
            location = f"tables[{table_index}].columns[{column_index}].generator"
            if target_table not in table_map:
                raise spec_error(f"{location}.table", f"unknown foreign-key table {target_table!r}")
            target_spec = table_map[target_table]
            if target_column not in target_spec.column_names:
                raise spec_error(f"{location}.column", f"unknown foreign-key column {target_table}.{target_column}")
            target = target_spec.column(target_column)
            if target.type != "integer" or target.generator.get("kind") != "unique_int":
                raise spec_error(f"{location}.column", "foreign-key target must be an integer unique_int column")

    queries_raw = _require_list(root.get("queries"), "queries")
    if not queries_raw:
        raise spec_error("queries", "must contain at least one query")
    queries: list[QuerySpec] = []
    query_ids: set[str] = set()
    for query_index, query_value in enumerate(queries_raw):
        query_loc = f"queries[{query_index}]"
        query_obj = _require_object(query_value, query_loc)
        query_id = _safe_ident(query_obj.get("id"), f"{query_loc}.id", query_id=True)
        if query_id in query_ids:
            raise spec_error(f"{query_loc}.id", f"duplicate query id {query_id!r}")
        query_ids.add(query_id)
        tables_value = _require_list(query_obj.get("tables"), f"{query_loc}.tables")
        if not tables_value:
            raise spec_error(f"{query_loc}.tables", "must contain at least one table reference")
        query_tables: list[str] = []
        for table_index, table_value in enumerate(tables_value):
            table_name = _safe_ident(table_value, f"{query_loc}.tables[{table_index}]")
            if table_name not in table_map:
                raise spec_error(f"{query_loc}.tables[{table_index}]", f"unknown table {table_name!r}")
            if table_name in query_tables:
                raise spec_error(f"{query_loc}.tables[{table_index}]", f"duplicate table reference {table_name!r}")
            query_tables.append(table_name)
        query_table_set = set(query_tables)

        filters_raw = query_obj.get("filters", [])
        filters_list = _require_list(filters_raw, f"{query_loc}.filters")
        filters: list[FilterSpec] = []
        seen_filter_keys: set[tuple[str, str, str]] = set()
        for filter_index, filter_value in enumerate(filters_list):
            filter_loc = f"{query_loc}.filters[{filter_index}]"
            filter_obj = _require_object(filter_value, filter_loc)
            table_name = _safe_ident(filter_obj.get("table"), f"{filter_loc}.table")
            column_name = _safe_ident(filter_obj.get("column"), f"{filter_loc}.column")
            if table_name not in query_table_set:
                raise spec_error(f"{filter_loc}.table", f"table {table_name!r} is not in the query table list")
            column = table_map[table_name].column(column_name) if column_name in table_map[table_name].column_names else None
            if column is None:
                raise spec_error(f"{filter_loc}.column", f"unknown column {table_name}.{column_name}")
            op = _require_string(filter_obj.get("op"), f"{filter_loc}.op")
            if op not in SUPPORTED_OPS:
                raise spec_error(f"{filter_loc}.op", f"unsupported operator {op!r}; supported={sorted(SUPPORTED_OPS)}")
            if "value" not in filter_obj:
                raise spec_error(f"{filter_loc}.value", "is required")
            filter_value_data = filter_obj["value"]
            if op == "between":
                values = _require_list(filter_value_data, f"{filter_loc}.value")
                if len(values) != 2:
                    raise spec_error(f"{filter_loc}.value", "between requires exactly two values")
                for value_index, item in enumerate(values):
                    _check_type_value(item, column.type, f"{filter_loc}.value[{value_index}]")
                if values[0] > values[1]:
                    raise spec_error(f"{filter_loc}.value", "lower bound must be <= upper bound")
                normalized_value: Any = list(values)
            elif op == "in":
                values = _require_list(filter_value_data, f"{filter_loc}.value")
                if not values:
                    raise spec_error(f"{filter_loc}.value", "in requires at least one value")
                for value_index, item in enumerate(values):
                    _check_type_value(item, column.type, f"{filter_loc}.value[{value_index}]")
                normalized_value = list(values)
            else:
                _check_type_value(filter_value_data, column.type, f"{filter_loc}.value")
                normalized_value = filter_value_data
            key = (table_name, column_name, op)
            if key in seen_filter_keys:
                raise spec_error(f"{filter_loc}", f"duplicate filter {table_name}.{column_name} {op}")
            seen_filter_keys.add(key)
            filters.append(FilterSpec(table=table_name, column=column_name, op=op, value=normalized_value))

        joins_raw = query_obj.get("joins", [])
        joins_list = _require_list(joins_raw, f"{query_loc}.joins")
        joins: list[JoinSpec] = []
        seen_joins: set[tuple[str, str, str, str]] = set()
        for join_index, join_value in enumerate(joins_list):
            join_loc = f"{query_loc}.joins[{join_index}]"
            join_obj = _require_object(join_value, join_loc)
            left_table = _safe_ident(join_obj.get("left_table"), f"{join_loc}.left_table")
            left_column = _safe_ident(join_obj.get("left_column"), f"{join_loc}.left_column")
            right_table = _safe_ident(join_obj.get("right_table"), f"{join_loc}.right_table")
            right_column = _safe_ident(join_obj.get("right_column"), f"{join_loc}.right_column")
            if left_table not in query_table_set:
                raise spec_error(f"{join_loc}.left_table", f"table {left_table!r} is not in the query table list")
            if right_table not in query_table_set:
                raise spec_error(f"{join_loc}.right_table", f"table {right_table!r} is not in the query table list")
            if left_table == right_table:
                raise spec_error(f"{join_loc}.left_table", "self joins are not supported")
            if left_column not in table_map[left_table].column_names:
                raise spec_error(f"{join_loc}.left_column", f"unknown column {left_table}.{left_column}")
            if right_column not in table_map[right_table].column_names:
                raise spec_error(f"{join_loc}.right_column", f"unknown column {right_table}.{right_column}")
            key = (left_table, left_column, right_table, right_column)
            if key in seen_joins:
                raise spec_error(join_loc, "duplicate join condition")
            seen_joins.add(key)
            joins.append(JoinSpec(left_table, left_column, right_table, right_column))

        if len(query_tables) > 1:
            if not joins:
                raise spec_error(f"{query_loc}.joins", "multi-table queries require join conditions")
            adjacency: dict[str, set[str]] = {table: set() for table in query_tables}
            for join in joins:
                adjacency[join.left_table].add(join.right_table)
                adjacency[join.right_table].add(join.left_table)
            seen = {query_tables[0]}
            stack = [query_tables[0]]
            while stack:
                node = stack.pop()
                for other in adjacency[node]:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
            missing = [table for table in query_tables if table not in seen]
            if missing:
                raise spec_error(f"{query_loc}.joins", f"join graph does not connect table(s) {missing}")

        select_raw = query_obj.get("select")
        select: list[ColumnRef] = []
        if select_raw is None:
            for table_name in query_tables:
                for column in table_map[table_name].columns:
                    select.append(ColumnRef(table_name, column.name))
        else:
            select_list = _require_list(select_raw, f"{query_loc}.select")
            if not select_list:
                raise spec_error(f"{query_loc}.select", "must not be empty when provided")
            seen_selects: set[tuple[str, str]] = set()
            for select_index, select_value in enumerate(select_list):
                select_loc = f"{query_loc}.select[{select_index}]"
                select_obj = _require_object(select_value, select_loc)
                table_name = _safe_ident(select_obj.get("table"), f"{select_loc}.table")
                column_name = _safe_ident(select_obj.get("column"), f"{select_loc}.column")
                if table_name not in query_table_set:
                    raise spec_error(f"{select_loc}.table", f"table {table_name!r} is not in the query table list")
                if column_name not in table_map[table_name].column_names:
                    raise spec_error(f"{select_loc}.column", f"unknown column {table_name}.{column_name}")
                key = (table_name, column_name)
                if key in seen_selects:
                    raise spec_error(select_loc, f"duplicate selected column {table_name}.{column_name}")
                seen_selects.add(key)
                select.append(ColumnRef(table_name, column_name))
        queries.append(
            QuerySpec(
                id=query_id,
                tables=tuple(query_tables),
                filters=tuple(filters),
                joins=tuple(joins),
                select=tuple(select),
            )
        )

    return Spec(
        name=name,
        tables=tuple(tables),
        queries=tuple(queries),
        cost_weights=_parse_weights(root.get("cost_weights")),
        stats=_parse_stats(root.get("stats")),
        search=_parse_search(root.get("search")),
        execution=_parse_execution(root.get("execution")),
        bias=_parse_bias(root.get("bias")),
        spec_version=spec_version,
        seed=seed,
    )


def load_spec(path: str | Path) -> Spec:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise spec_error("$", f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    except OSError as exc:
        raise QPlanError("E_SPEC", f"cannot read specification: {exc}", "$") from exc
    return parse_spec(raw)


def apply_overrides(
    spec: Spec,
    *,
    weights: dict[str, int] | None = None,
    stats_row_threshold: int | None = None,
    sample_size: int | None = None,
    exhaustive_max_candidates: int | None = None,
    repetitions: int | None = None,
) -> Spec:
    """Apply session overrides while preserving strict validation."""

    if weights is not None:
        allowed = {"rows", "comparisons", "lookups", "io_pages"}
        unknown = sorted(set(weights) - allowed)
        if unknown:
            raise spec_error("cost_weights", f"unknown weight keys {unknown}")
        data = spec.cost_weights.to_dict()
        for key, value in weights.items():
            if not _is_int(value) or value < 0:
                raise spec_error(f"cost_weights.{key}", "must be a non-negative integer")
            data[key] = value
        cost_weights = CostWeights(**data)
    else:
        cost_weights = spec.cost_weights
    stats = spec.stats
    if stats_row_threshold is not None or sample_size is not None:
        if stats_row_threshold is not None and stats_row_threshold < 1:
            raise spec_error("--row-threshold", "must be >= 1")
        if sample_size is not None and sample_size < 1:
            raise spec_error("--sample-size", "must be >= 1")
        stats = StatsConfig(
            version=stats.version,
            row_threshold=stats_row_threshold if stats_row_threshold is not None else stats.row_threshold,
            sample_size=sample_size if sample_size is not None else stats.sample_size,
            confidence=stats.confidence,
            histogram_buckets=stats.histogram_buckets,
            value_count_limit=stats.value_count_limit,
            page_rows=stats.page_rows,
            random_seed=stats.random_seed,
        )
    search = spec.search
    if exhaustive_max_candidates is not None:
        if exhaustive_max_candidates < 0:
            raise spec_error("--exhaustive-max-candidates", "must be >= 0")
        if exhaustive_max_candidates > search.oracle_max_candidates:
            raise spec_error("--exhaustive-max-candidates", "must be <= oracle_max_candidates")
        search = SearchConfig(
            exhaustive_max_candidates=exhaustive_max_candidates,
            oracle_max_candidates=search.oracle_max_candidates,
        )
    execution = spec.execution
    if repetitions is not None:
        if repetitions < 1:
            raise spec_error("--repetitions", "must be >= 1")
        execution = ExecutionConfig(warmup_runs=spec.execution.warmup_runs, repetitions=repetitions)
    return Spec(
        name=spec.name,
        tables=spec.tables,
        queries=spec.queries,
        cost_weights=cost_weights,
        stats=stats,
        search=search,
        execution=execution,
        bias=spec.bias,
        spec_version=spec.spec_version,
        seed=spec.seed,
    )


def validate_spec_file(path: str | Path) -> dict[str, Any]:
    spec = load_spec(path)
    return {
        "status": "OK",
        "spec_digest": spec.digest,
        "tables": len(spec.tables),
        "queries": len(spec.queries),
        "candidate_indexes": len(spec.candidate_indexes),
        "stats_caliber_version": spec.stats.caliber_version,
    }


def iter_column_refs(query: QuerySpec) -> Iterable[ColumnRef]:
    yield from query.select
