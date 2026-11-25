from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .canonical import INT64_MAX, INT64_MIN, canonical_json, digest_value
from .errors import SpecError, SpecIssue


ENCODINGS = ("raw", "dict", "rle")
PREDICATE_OPS = ("eq", "ne", "lt", "le", "gt", "ge", "in", "is_null", "is_not_null")
AGGREGATE_OPS = ("count", "sum", "min", "max", "avg")
SUPPORTED_SCALES = (1, 10, 100, 1000, 10000)
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _issue(segment_id: str, field_name: str, message: str) -> None:
    raise SpecError(SpecIssue(segment_id, field_name, message))


def _require_object(value: Any, segment_id: str, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _issue(segment_id, field_name, f"{field_name} must be an object")
    return value


def _check_keys(
    value: dict[str, Any],
    *,
    allowed: Iterable[str],
    required: Iterable[str],
    segment_id: str,
    field_name: str,
) -> None:
    allowed_set = set(allowed)
    required_set = set(required)
    unknown = sorted(set(value) - allowed_set)
    if unknown:
        _issue(segment_id, unknown[0], f"unknown field {unknown[0]!r} in {field_name}")
    missing = sorted(required_set - set(value))
    if missing:
        _issue(segment_id, missing[0], f"missing field {missing[0]!r} in {field_name}")


def _require_name(value: Any, segment_id: str, field_name: str) -> str:
    if not isinstance(value, str) or not NAME_RE.fullmatch(value):
        _issue(segment_id, field_name, f"{field_name} must match {NAME_RE.pattern}")
    return value


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    encoding: str
    value_min: int = 0
    value_max: int = 1_000_000
    null_per_mille: int = 0
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "encoding": self.encoding,
            "value_min": self.value_min,
            "value_max": self.value_max,
            "null_per_mille": self.null_per_mille,
        }
        if self.seed is not None:
            result["seed"] = self.seed
        return result


@dataclass(frozen=True)
class SegmentSpec:
    id: str
    rows: int
    seed: int
    columns: tuple[ColumnSpec, ...]

    def column_map(self) -> dict[str, ColumnSpec]:
        return {column.name: column for column in self.columns}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rows": self.rows,
            "seed": self.seed,
            "columns": [column.to_dict() for column in sorted(self.columns, key=lambda c: c.name)],
        }


@dataclass(frozen=True)
class AggregateSpec:
    name: str
    op: str
    column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name, "op": self.op}
        if self.column is not None:
            result["column"] = self.column
        return result


@dataclass(frozen=True)
class QuerySpec:
    id: str
    segments: tuple[str, ...]
    project: tuple[str, ...]
    predicate: dict[str, Any] | None
    group_by: tuple[str, ...]
    aggregates: tuple[AggregateSpec, ...]
    predicate_columns: tuple[str, ...] = field(default=())

    def aggregate_columns(self) -> tuple[str, ...]:
        return tuple(
            sorted({agg.column for agg in self.aggregates if agg.column is not None})
        )

    def effective_projection(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.project) | set(self.aggregate_columns())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "segments": list(self.segments),
            "project": list(self.project),
            "predicate": self.predicate,
            "group_by": list(self.group_by),
            "aggregates": [agg.to_dict() for agg in self.aggregates],
        }


@dataclass(frozen=True)
class ExecutionConfig:
    batch_rows: int
    scale: int
    segment_decode_memory_bytes: int
    max_concurrency: int
    attribution_tolerance_ppm: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_rows": self.batch_rows,
            "scale": self.scale,
            "segment_decode_memory_bytes": self.segment_decode_memory_bytes,
            "max_concurrency": self.max_concurrency,
            "attribution_tolerance_ppm": self.attribution_tolerance_ppm,
        }


@dataclass(frozen=True)
class ScanSpec:
    format_version: int
    segments: tuple[SegmentSpec, ...]
    queries: tuple[QuerySpec, ...]
    execution: ExecutionConfig

    def segment_map(self) -> dict[str, SegmentSpec]:
        return {segment.id: segment for segment in self.segments}

    def query_map(self) -> dict[str, QuerySpec]:
        return {query.id: query for query in self.queries}

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "segments": [segment.to_dict() for segment in self.segments],
            "queries": [
                {
                    **query.to_dict(),
                    "segments": list(query.segments),
                    "project": list(query.project),
                    "group_by": list(query.group_by),
                    "aggregates": [agg.to_dict() for agg in query.aggregates],
                }
                for query in self.queries
            ],
            "execution": self.execution.to_dict(),
        }

    @property
    def digest(self) -> str:
        return digest_value(self.canonical_dict())

    def with_queries(self, queries: Iterable[QuerySpec]) -> "ScanSpec":
        return ScanSpec(
            self.format_version,
            self.segments,
            tuple(sorted(queries, key=lambda query: query.id)),
            self.execution,
        )

    def with_execution(self, execution: ExecutionConfig) -> "ScanSpec":
        return ScanSpec(self.format_version, self.segments, self.queries, execution)


def canonicalize_predicate(value: Any, segment_id: str, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    node = _require_object(value, segment_id, field_name)
    op = node.get("op")
    if not isinstance(op, str):
        _issue(segment_id, field_name, "predicate.op must be a string")

    if op in ("and", "or"):
        _check_keys(
            node,
            allowed=("op", "args"),
            required=("op", "args"),
            segment_id=segment_id,
            field_name=field_name,
        )
        args = node["args"]
        if not isinstance(args, list) or not args:
            _issue(segment_id, field_name, f"predicate.{op}.args must be a non-empty array")
        canonical_args: list[dict[str, Any]] = []
        for index, arg in enumerate(args):
            canonical = canonicalize_predicate(arg, segment_id, f"{field_name}.args[{index}]")
            if canonical is None:
                _issue(segment_id, field_name, "null predicate is only valid at the root")
            if canonical["op"] == op:
                canonical_args.extend(canonical["args"])
            else:
                canonical_args.append(canonical)
        by_text: dict[str, dict[str, Any]] = {
            canonical_json(arg): arg for arg in canonical_args
        }
        return {"op": op, "args": [by_text[key] for key in sorted(by_text)]}

    if op == "not":
        _check_keys(
            node,
            allowed=("op", "arg"),
            required=("op", "arg"),
            segment_id=segment_id,
            field_name=field_name,
        )
        arg = canonicalize_predicate(node["arg"], segment_id, f"{field_name}.arg")
        if arg is None:
            _issue(segment_id, field_name, "predicate.not.arg cannot be null")
        return {"op": "not", "arg": arg}

    if op not in PREDICATE_OPS:
        _issue(segment_id, field_name, f"unsupported predicate operator {op!r}")

    if op in ("is_null", "is_not_null"):
        _check_keys(
            node,
            allowed=("column", "op"),
            required=("column", "op"),
            segment_id=segment_id,
            field_name=field_name,
        )
        column = _require_name(node["column"], segment_id, f"{field_name}.column")
        return {"column": column, "op": op}

    if op == "in":
        _check_keys(
            node,
            allowed=("column", "op", "values"),
            required=("column", "op", "values"),
            segment_id=segment_id,
            field_name=field_name,
        )
        column = _require_name(node["column"], segment_id, f"{field_name}.column")
        values = node["values"]
        if not isinstance(values, list) or not values:
            _issue(segment_id, field_name, "predicate.in.values must be a non-empty array")
        if any(not _is_int(item) for item in values):
            _issue(segment_id, field_name, "predicate.in.values must contain integers")
        if any(item < INT64_MIN or item > INT64_MAX for item in values):
            _issue(segment_id, field_name, "predicate.in.values must fit signed int64")
        normalized = sorted(set(values))
        return {"column": column, "op": "in", "values": normalized}

    _check_keys(
        node,
        allowed=("column", "op", "value"),
        required=("column", "op", "value"),
        segment_id=segment_id,
        field_name=field_name,
    )
    column = _require_name(node["column"], segment_id, f"{field_name}.column")
    if not _is_int(node["value"]):
        _issue(segment_id, field_name, "predicate.value must be an integer")
    if node["value"] < INT64_MIN or node["value"] > INT64_MAX:
        _issue(segment_id, field_name, "predicate.value must fit signed int64")
    return {"column": column, "op": op, "value": node["value"]}


def predicate_columns(predicate: dict[str, Any] | None) -> tuple[str, ...]:
    if predicate is None:
        return ()
    op = predicate["op"]
    if op in ("and", "or"):
        result: set[str] = set()
        for arg in predicate["args"]:
            result.update(predicate_columns(arg))
        return tuple(sorted(result))
    if op == "not":
        return predicate_columns(predicate["arg"])
    return (predicate["column"],)


def predicate_leaf_count(predicate: dict[str, Any] | None) -> int:
    if predicate is None:
        return 0
    op = predicate["op"]
    if op in ("and", "or"):
        return sum(predicate_leaf_count(arg) for arg in predicate["args"])
    if op == "not":
        return predicate_leaf_count(predicate["arg"])
    return 1


def _parse_segment(value: Any, index: int) -> SegmentSpec:
    fallback = f"<segments[{index}]>"
    obj = _require_object(value, fallback, "segment")
    raw_id = obj.get("id")
    segment_id = raw_id if isinstance(raw_id, str) and raw_id else fallback
    _check_keys(
        obj,
        allowed=("id", "rows", "seed", "columns"),
        required=("id", "rows", "seed", "columns"),
        segment_id=segment_id,
        field_name="segment",
    )
    segment_id = _require_name(raw_id, segment_id, "id")
    rows = obj["rows"]
    if not _is_int(rows) or rows <= 0:
        _issue(segment_id, "rows", "rows must be a positive integer")
    seed = obj["seed"]
    if not _is_int(seed):
        _issue(segment_id, "seed", "seed must be an integer")
    columns_value = obj["columns"]
    if not isinstance(columns_value, list) or not columns_value:
        _issue(segment_id, "columns", "columns must be a non-empty array")

    columns: list[ColumnSpec] = []
    seen: set[str] = set()
    for column_index, column_value in enumerate(columns_value):
        column_obj = _require_object(
            column_value, segment_id, f"columns[{column_index}]"
        )
        raw_name = column_obj.get("name")
        column_name = raw_name if isinstance(raw_name, str) and raw_name else f"<columns[{column_index}]>"
        _check_keys(
            column_obj,
            allowed=(
                "name",
                "encoding",
                "value_min",
                "value_max",
                "null_per_mille",
                "seed",
            ),
            required=("name", "encoding"),
            segment_id=segment_id,
            field_name=column_name,
        )
        column_name = _require_name(raw_name, segment_id, "name")
        if column_name in seen:
            _issue(segment_id, "name", f"duplicate column name {column_name!r}")
        seen.add(column_name)
        encoding = column_obj["encoding"]
        if encoding not in ENCODINGS:
            _issue(segment_id, "encoding", f"unsupported column encoding {encoding!r}")
        value_min = column_obj.get("value_min", 0)
        value_max = column_obj.get("value_max", 1_000_000)
        null_per_mille = column_obj.get("null_per_mille", 0)
        column_seed = column_obj.get("seed")
        if not _is_int(value_min) or not _is_int(value_max):
            _issue(segment_id, "value_min", "value_min/value_max must be integers")
        if value_min < INT64_MIN or value_max > INT64_MAX or value_min > value_max:
            _issue(segment_id, "value_max", "value range must satisfy INT64_MIN <= value_min <= value_max <= INT64_MAX")
        if not _is_int(null_per_mille) or not 0 <= null_per_mille <= 1000:
            _issue(segment_id, "null_per_mille", "null_per_mille must be an integer in [0, 1000]")
        if column_seed is not None and not _is_int(column_seed):
            _issue(segment_id, "seed", "column seed must be an integer")
        columns.append(
            ColumnSpec(
                name=column_name,
                encoding=encoding,
                value_min=value_min,
                value_max=value_max,
                null_per_mille=null_per_mille,
                seed=column_seed,
            )
        )
    return SegmentSpec(
        id=segment_id,
        rows=rows,
        seed=seed,
        columns=tuple(sorted(columns, key=lambda column: column.name)),
    )


def _parse_query(value: Any, index: int, segment_map: dict[str, SegmentSpec]) -> QuerySpec:
    fallback = f"<queries[{index}]>"
    obj = _require_object(value, fallback, "query")
    raw_id = obj.get("id")
    query_id = raw_id if isinstance(raw_id, str) and raw_id else fallback
    _check_keys(
        obj,
        allowed=("id", "segments", "project", "predicate", "group_by", "aggregates"),
        required=("id", "project", "aggregates"),
        segment_id=query_id,
        field_name="query",
    )
    query_id = _require_name(raw_id, query_id, "id")

    raw_segments = obj.get("segments", sorted(segment_map))
    if not isinstance(raw_segments, list) or not raw_segments:
        _issue(query_id, "segments", "segments must be a non-empty array")
    if any(not isinstance(item, str) or not item for item in raw_segments):
        _issue(query_id, "segments", "segments must contain non-empty strings")
    duplicate_segment = next(
        (item for item in sorted(raw_segments) if raw_segments.count(item) > 1), None
    )
    if duplicate_segment is not None:
        _issue(duplicate_segment, "segments", f"duplicate query segment {duplicate_segment!r}")
    query_segments = tuple(sorted(raw_segments))
    for segment_id in query_segments:
        if segment_id not in segment_map:
            _issue(segment_id, "segments", f"query references unknown segment {segment_id!r}")

    raw_project = obj["project"]
    if not isinstance(raw_project, list) or not raw_project:
        _issue(query_id, "project", "project must be a non-empty array")
    if any(not isinstance(item, str) or not item for item in raw_project):
        _issue(query_id, "project", "project must contain column names")
    project = tuple(sorted(set(raw_project)))
    if len(project) != len(raw_project):
        _issue(query_id, "project", "project must not contain duplicate columns")

    predicate = canonicalize_predicate(
        obj.get("predicate"), query_id, "predicate"
    )
    refs = predicate_columns(predicate)

    raw_groups = obj.get("group_by", [])
    if not isinstance(raw_groups, list):
        _issue(query_id, "group_by", "group_by must be an array")
    if any(not isinstance(item, str) or not item for item in raw_groups):
        _issue(query_id, "group_by", "group_by must contain column names")
    group_by = tuple(sorted(set(raw_groups)))
    if len(group_by) != len(raw_groups):
        _issue(query_id, "group_by", "group_by must not contain duplicate columns")
    for column in group_by:
        if column not in project:
            _issue(query_id, "group_by", f"group column {column!r} must also be projected")

    raw_aggregates = obj["aggregates"]
    if not isinstance(raw_aggregates, list) or not raw_aggregates:
        _issue(query_id, "aggregates", "aggregates must be a non-empty array")
    aggregates: list[AggregateSpec] = []
    aggregate_names: set[str] = set()
    for aggregate_index, aggregate_value in enumerate(raw_aggregates):
        aggregate_obj = _require_object(
            aggregate_value, query_id, f"aggregates[{aggregate_index}]"
        )
        _check_keys(
            aggregate_obj,
            allowed=("name", "op", "column"),
            required=("name", "op"),
            segment_id=query_id,
            field_name=f"aggregates[{aggregate_index}]",
        )
        name = _require_name(aggregate_obj["name"], query_id, "name")
        if name in aggregate_names:
            _issue(query_id, "name", f"duplicate aggregate name {name!r}")
        aggregate_names.add(name)
        op = aggregate_obj["op"]
        if op not in AGGREGATE_OPS:
            _issue(query_id, "op", f"unsupported aggregate operator {op!r}")
        column = aggregate_obj.get("column")
        if op == "count" and column is None:
            pass
        elif column is None:
            _issue(query_id, "column", f"aggregate {name!r} requires a column")
        else:
            column = _require_name(column, query_id, "column")
        aggregates.append(AggregateSpec(name=name, op=op, column=column))

    referenced_columns = set(project) | set(refs) | {
        aggregate.column for aggregate in aggregates if aggregate.column is not None
    }
    for segment_id in query_segments:
        columns = segment_map[segment_id].column_map()
        missing = sorted(referenced_columns - set(columns))
        if missing:
            if missing[0] in project:
                issue_field = "project"
            elif missing[0] in refs:
                issue_field = "predicate"
            else:
                issue_field = "aggregates"
            _issue(
                segment_id,
                issue_field,
                f"query {query_id!r} references missing column {missing[0]!r}",
            )

    return QuerySpec(
        id=query_id,
        segments=query_segments,
        project=project,
        predicate=predicate,
        group_by=group_by,
        aggregates=tuple(sorted(aggregates, key=lambda agg: agg.name)),
        predicate_columns=refs,
    )


def parse_spec(data: Any) -> ScanSpec:
    if not isinstance(data, dict):
        _issue("<root>", "format_version", "spec root must be an object")
    _check_keys(
        data,
        allowed=("format_version", "segments", "queries", "execution"),
        required=("format_version", "segments", "queries", "execution"),
        segment_id="<root>",
        field_name="format_version",
    )
    format_version = data["format_version"]
    if not _is_int(format_version) or format_version != 1:
        _issue("<root>", "format_version", "format_version must be the integer 1")

    raw_segments = data["segments"]
    if not isinstance(raw_segments, list) or not raw_segments:
        _issue("<root>", "segments", "segments must be a non-empty array")
    segments: list[SegmentSpec] = []
    seen_segments: set[str] = set()
    for index, raw_segment in enumerate(raw_segments):
        segment = _parse_segment(raw_segment, index)
        if segment.id in seen_segments:
            _issue(segment.id, "id", f"duplicate segment id {segment.id!r}")
        seen_segments.add(segment.id)
        segments.append(segment)
    sorted_segments = tuple(sorted(segments, key=lambda segment: segment.id))
    segment_map = {segment.id: segment for segment in sorted_segments}

    raw_queries = data["queries"]
    if not isinstance(raw_queries, list) or not raw_queries:
        _issue("<root>", "queries", "queries must be a non-empty array")
    queries: list[QuerySpec] = []
    seen_queries: set[str] = set()
    for index, raw_query in enumerate(raw_queries):
        query = _parse_query(raw_query, index, segment_map)
        if query.id in seen_queries:
            _issue(query.id, "id", f"duplicate query id {query.id!r}")
        seen_queries.add(query.id)
        queries.append(query)

    execution_obj = _require_object(data["execution"], "<root>", "execution")
    _check_keys(
        execution_obj,
        allowed=(
            "batch_rows",
            "scale",
            "segment_decode_memory_bytes",
            "max_concurrency",
            "attribution_tolerance_ppm",
        ),
        required=(
            "batch_rows",
            "scale",
            "segment_decode_memory_bytes",
            "max_concurrency",
        ),
        segment_id="<root>",
        field_name="execution",
    )
    batch_rows = execution_obj["batch_rows"]
    if not _is_int(batch_rows) or batch_rows <= 0:
        _issue("<root>", "batch_rows", "batch_rows must be a positive integer")
    scale = execution_obj["scale"]
    if not _is_int(scale) or scale not in SUPPORTED_SCALES:
        _issue("<root>", "scale", f"scale must be one of {SUPPORTED_SCALES}")
    memory = execution_obj["segment_decode_memory_bytes"]
    if not _is_int(memory) or memory <= 0:
        _issue(
            "<root>",
            "segment_decode_memory_bytes",
            "segment_decode_memory_bytes must be a positive integer",
        )
    concurrency = execution_obj["max_concurrency"]
    if not _is_int(concurrency) or concurrency <= 0:
        _issue("<root>", "max_concurrency", "max_concurrency must be a positive integer")
    tolerance = execution_obj.get("attribution_tolerance_ppm", 2)
    if not _is_int(tolerance) or tolerance < 0:
        _issue(
            "<root>",
            "attribution_tolerance_ppm",
            "attribution_tolerance_ppm must be a non-negative integer",
        )
    execution = ExecutionConfig(
        batch_rows=batch_rows,
        scale=scale,
        segment_decode_memory_bytes=memory,
        max_concurrency=concurrency,
        attribution_tolerance_ppm=tolerance,
    )
    return ScanSpec(
        format_version=1,
        segments=sorted_segments,
        queries=tuple(sorted(queries, key=lambda query: query.id)),
        execution=execution,
    )


def load_spec(path: str | Path) -> ScanSpec:
    source = Path(path)
    def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=_closed_object)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SpecError(SpecIssue("<root>", "json", f"cannot load spec: {exc}")) from exc
    return parse_spec(data)


def write_spec(path: str | Path, spec: ScanSpec) -> None:
    from .canonical import atomic_write

    atomic_write(path, (canonical_json(spec.canonical_dict()) + "\n").encode("utf-8"))
