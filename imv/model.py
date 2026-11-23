from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import fail
from .fixed import FixedPointContext
from .util import canonical_json_text, digest_obj


COLUMN_TYPES = {"integer", "float", "text", "boolean"}
NUMERIC_COLUMN_TYPES = {"integer", "float"}
AGGREGATE_FUNCTIONS = {"count", "sum", "min", "max"}
REFRESH_MODES = {"incremental", "full"}
CHANGE_TYPES = {"insert", "update", "delete"}
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,95}$")


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    nullable: bool = True

    def payload(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type, "nullable": self.nullable}


@dataclass(frozen=True)
class AggregateSpec:
    func: str
    column: str
    alias: str

    def payload(self) -> dict[str, str]:
        return {"func": self.func, "column": self.column, "alias": self.alias}


@dataclass(frozen=True)
class ViewDef:
    view_id: str
    group_by: tuple[str, ...]
    aggregates: tuple[AggregateSpec, ...]
    filter: dict[str, Any] | None
    refresh_mode: str
    per_view_quota: int
    lease_ttl_ms: int
    lease_renew_ms: int
    payload: dict[str, Any]
    digest: str

    @property
    def aggregate_dicts(self) -> list[dict[str, Any]]:
        return [spec.payload() for spec in self.aggregates]


@dataclass(frozen=True)
class SessionConfig:
    scale_factor: int
    truncation: str
    overflow: str
    global_concurrency: int
    batch_size: int
    default_lease_ttl_ms: int
    default_lease_renew_ms: int

    @property
    def fixed(self) -> FixedPointContext:
        return FixedPointContext(self.scale_factor, self.truncation, self.overflow)

    def payload(self) -> dict[str, Any]:
        return {
            "scale_factor": self.scale_factor,
            "truncation": self.truncation,
            "overflow": self.overflow,
            "global_concurrency": self.global_concurrency,
            "batch_size": self.batch_size,
            "default_lease_ttl_ms": self.default_lease_ttl_ms,
            "default_lease_renew_ms": self.default_lease_renew_ms,
        }


@dataclass(frozen=True)
class Definitions:
    session: SessionConfig
    views: dict[str, ViewDef]
    fingerprint: str
    source_path: str

    @property
    def fixed(self) -> FixedPointContext:
        return self.session.fixed


@dataclass(frozen=True)
class BaseTable:
    table_name: str
    row_id_column: str
    columns: tuple[Column, ...]
    rows: dict[Any, dict[str, Any]]
    null_bitmap: dict[Any, tuple[str, ...]]
    digest: str
    source_path: str

    @property
    def column_map(self) -> dict[str, Column]:
        return {column.name: column for column in self.columns}

    @property
    def row_ids(self) -> tuple[Any, ...]:
        return tuple(sorted(self.rows, key=_row_sort_key))

    def canonical_payload(self) -> dict[str, Any]:
        return canonical_rows_payload(self.table_name, self.row_id_column, self.columns, self.rows)


@dataclass(frozen=True)
class Change:
    seq: int
    op: str
    row_id: Any
    row: dict[str, Any] | None
    line: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class ChangeStream:
    changes: tuple[Change, ...]
    digest: str
    source_path: str


@dataclass
class Dataset:
    base: BaseTable
    definitions: Definitions
    changes: ChangeStream


def _row_sort_key(row_id: Any) -> tuple[str, str]:
    return (type(row_id).__name__, canonical_json_text(row_id))


def canonical_rows_payload(
    table_name: str,
    row_id_column: str,
    columns: tuple[Column, ...],
    rows: dict[Any, dict[str, Any]],
) -> dict[str, Any]:
    ordered_rows = []
    for row_id in sorted(rows, key=_row_sort_key):
        ordered_rows.append(
            {
                "row_id": row_id,
                "values": {column.name: rows[row_id].get(column.name) for column in columns},
            }
        )
    return {
        "table": table_name,
        "row_id_column": row_id_column,
        "columns": [column.payload() for column in columns],
        "rows": ordered_rows,
    }


def rows_from_payload(payload: dict[str, Any]) -> tuple[str, str, tuple[Column, ...], dict[Any, dict[str, Any]]]:
    columns = tuple(
        Column(item["name"], item["type"], bool(item.get("nullable", True)))
        for item in payload["columns"]
    )
    rows = {
        item["row_id"]: {column.name: item["values"].get(column.name) for column in columns}
        for item in payload["rows"]
    }
    return payload["table"], payload["row_id_column"], columns, rows


def load_dataset(base_path: str | Path, views_path: str | Path, changes_path: str | Path) -> Dataset:
    base = load_base_table(base_path)
    definitions = load_definitions(views_path)
    validate_definitions_against_base(definitions, base)
    changes = load_changes(changes_path, base)
    return Dataset(base=base, definitions=definitions, changes=changes)


def _read_json(path: str | Path) -> tuple[Any, str]:
    source = str(Path(path))
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            return json.load(handle), source
    except FileNotFoundError:
        fail("E_SCHEMA", "input file does not exist", location={"file": source, "path": "$"})
    except PermissionError as exc:
        fail("E_SCHEMA", f"input file is not readable: {exc}", location={"file": source, "path": "$"})
    except json.JSONDecodeError as exc:
        fail(
            "E_SCHEMA",
            f"invalid JSON: {exc.msg}",
            location={"file": source, "line": exc.lineno, "column": exc.colno, "path": "$"},
        )
    except OSError as exc:
        fail("E_SCHEMA", f"cannot read input file: {exc}", location={"file": source, "path": "$"})


def _require_object(value: Any, *, source: str, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail("E_SCHEMA", "expected a JSON object", location={"file": source, "path": path})
    return value


def _require_list(value: Any, *, source: str, path: str) -> list[Any]:
    if not isinstance(value, list):
        fail("E_SCHEMA", "expected a JSON array", location={"file": source, "path": path})
    return value


def _positive_int(value: Any, *, source: str, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail("E_SCHEMA", "expected a positive integer", location={"file": source, "path": path})
    return value


def _identifier(value: Any, *, source: str, path: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.match(value):
        fail(
            "E_SCHEMA",
            "identifier must match [A-Za-z_][A-Za-z0-9_]{0,95}",
            location={"file": source, "path": path},
        )
    return value


def _coerce_row_id(value: Any, column: Column, *, source: str, path: str) -> Any:
    if column.type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            fail("E_SCHEMA", "row identifier must be an integer", location={"file": source, "path": path})
        return value
    if column.type == "text":
        if not isinstance(value, str) or not value:
            fail("E_SCHEMA", "row identifier must be a non-empty string", location={"file": source, "path": path})
        return value
    fail("E_SCHEMA", "row identifier column must be integer or text", location={"file": source, "path": path})


def _validate_cell(
    value: Any,
    column: Column,
    *,
    source: str,
    path: str,
    code: str = "E_SCHEMA",
    allow_null: bool = True,
) -> Any:
    if value is None:
        if not allow_null or not column.nullable:
            fail(code, f"column {column.name!r} does not allow null", location={"file": source, "path": path})
        return None
    if column.type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            fail(code, f"column {column.name!r} expects integer", location={"file": source, "path": path})
        return value
    if column.type == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail(code, f"column {column.name!r} expects float", location={"file": source, "path": path})
        numeric = float(value)
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            fail(code, f"column {column.name!r} expects a finite float", location={"file": source, "path": path})
        return numeric
    if column.type == "text":
        if not isinstance(value, str):
            fail(code, f"column {column.name!r} expects text", location={"file": source, "path": path})
        return value
    if column.type == "boolean":
        if not isinstance(value, bool):
            fail(code, f"column {column.name!r} expects boolean", location={"file": source, "path": path})
        return value
    fail("E_SCHEMA", f"unsupported column type {column.type!r}", location={"file": source, "path": path})


def _normalize_null_bitmap(
    bitmap: Any,
    *,
    source: str,
    row_id_column: Column,
) -> dict[Any, tuple[str, ...]]:
    if bitmap is None:
        return {}
    result: dict[Any, list[str]] = {}
    if isinstance(bitmap, dict):
        entries = []
        for key, value in bitmap.items():
            normalized_key: Any = key
            if row_id_column.type == "integer" and isinstance(key, str):
                try:
                    normalized_key = int(key)
                except ValueError:
                    fail(
                        "E_SCHEMA",
                        "null bitmap row key must be an integer",
                        location={"file": source, "path": f"$.null_bitmap.{key}"},
                    )
            entries.append({"row_id": normalized_key, "columns": value})
    elif isinstance(bitmap, list):
        entries = bitmap
    else:
        fail("E_SCHEMA", "null_bitmap must be an object or array", location={"file": source, "path": "$.null_bitmap"})
    for index, entry in enumerate(entries):
        obj = _require_object(entry, source=source, path=f"$.null_bitmap[{index}]")
        if "row_id" not in obj:
            fail("E_SCHEMA", "null bitmap entry is missing row_id", location={"file": source, "path": f"$.null_bitmap[{index}].row_id"})
        row_id = _coerce_row_id(obj["row_id"], row_id_column, source=source, path=f"$.null_bitmap[{index}].row_id")
        columns = obj.get("columns", obj.get("null_columns"))
        columns = _require_list(columns, source=source, path=f"$.null_bitmap[{index}].columns")
        normalized_columns: list[str] = []
        for col_index, name in enumerate(columns):
            if not isinstance(name, str):
                fail("E_SCHEMA", "null column name must be a string", location={"file": source, "path": f"$.null_bitmap[{index}].columns[{col_index}]"})
            if name in normalized_columns:
                fail("E_SCHEMA", "duplicate column in null bitmap", location={"file": source, "path": f"$.null_bitmap[{index}].columns[{col_index}]"})
            normalized_columns.append(name)
        if row_id in result:
            fail("E_SCHEMA", "duplicate row in null bitmap", location={"file": source, "path": f"$.null_bitmap[{index}].row_id"})
        result[row_id] = normalized_columns
    return {row_id: tuple(sorted(columns)) for row_id, columns in result.items()}


def load_base_table(path: str | Path) -> BaseTable:
    raw, source = _read_json(path)
    root = _require_object(raw, source=source, path="$")
    if root.get("format") not in (None, "imv.base.v1"):
        fail("E_SCHEMA", "unsupported base table format", location={"file": source, "path": "$.format"})
    table_name = root.get("table")
    if not isinstance(table_name, str) or not table_name.strip():
        fail("E_SCHEMA", "table must be a non-empty string", location={"file": source, "path": "$.table"})
    row_id_column_name = root.get("row_id_column")
    if not isinstance(row_id_column_name, str):
        fail("E_SCHEMA", "row_id_column must be a string", location={"file": source, "path": "$.row_id_column"})

    raw_columns = _require_list(root.get("columns"), source=source, path="$.columns")
    if not raw_columns:
        fail("E_SCHEMA", "at least one column is required", location={"file": source, "path": "$.columns"})
    columns: list[Column] = []
    seen_names: set[str] = set()
    for index, raw_column in enumerate(raw_columns):
        path_prefix = f"$.columns[{index}]"
        item = _require_object(raw_column, source=source, path=path_prefix)
        name = _identifier(item.get("name"), source=source, path=f"{path_prefix}.name")
        if name in seen_names:
            fail("E_SCHEMA", f"duplicate column name {name!r}", location={"file": source, "path": f"{path_prefix}.name"})
        seen_names.add(name)
        type_name = item.get("type")
        if type_name not in COLUMN_TYPES:
            fail("E_SCHEMA", "unsupported column type", location={"file": source, "path": f"{path_prefix}.type"})
        nullable = item.get("nullable", True)
        if not isinstance(nullable, bool):
            fail("E_SCHEMA", "nullable must be boolean", location={"file": source, "path": f"{path_prefix}.nullable"})
        columns.append(Column(name, type_name, nullable))
    column_map = {column.name: column for column in columns}
    if row_id_column_name not in column_map:
        fail("E_SCHEMA", "row_id_column is not declared in columns", location={"file": source, "path": "$.row_id_column"})
    row_id_column = column_map[row_id_column_name]
    if row_id_column.type not in {"integer", "text"}:
        fail("E_SCHEMA", "row identifier column must be integer or text", location={"file": source, "path": "$.row_id_column"})
    if row_id_column.nullable:
        fail("E_SCHEMA", "row identifier column must not be nullable", location={"file": source, "path": "$.row_id_column"})

    null_bitmap = _normalize_null_bitmap(root.get("null_bitmap", []), source=source, row_id_column=row_id_column)
    for row_id, names in null_bitmap.items():
        for column_name in names:
            if column_name not in column_map:
                fail(
                    "E_SCHEMA",
                    f"null bitmap references unknown column {column_name!r}",
                    location={"file": source, "path": "$.null_bitmap", "row_id": row_id, "column": column_name},
                )

    raw_rows = _require_list(root.get("rows"), source=source, path="$.rows")
    rows: dict[Any, dict[str, Any]] = {}
    known_bitmap_rows = set(null_bitmap)
    for index, raw_row in enumerate(raw_rows):
        prefix = f"$.rows[{index}]"
        item = _require_object(raw_row, source=source, path=prefix)
        if row_id_column_name not in item:
            fail("E_SCHEMA", "row is missing row identifier", location={"file": source, "path": f"{prefix}.{row_id_column_name}"})
        row_id = _coerce_row_id(item[row_id_column_name], row_id_column, source=source, path=f"{prefix}.{row_id_column_name}")
        if row_id in rows:
            fail("E_SCHEMA", f"duplicate row identifier {row_id!r}", location={"file": source, "path": f"{prefix}.{row_id_column_name}"})
        extra = sorted(set(item) - set(column_map))
        if extra:
            fail("E_SCHEMA", f"row contains unknown column {extra[0]!r}", location={"file": source, "path": f"{prefix}.{extra[0]}"})
        nulls = set(null_bitmap.get(row_id, ()))
        normalized: dict[str, Any] = {}
        for column in columns:
            value_path = f"{prefix}.{column.name}"
            if column.name in nulls:
                if column.name in item and item[column.name] is not None:
                    fail("E_SCHEMA", "null bitmap conflicts with a non-null row value", location={"file": source, "path": value_path})
                normalized[column.name] = None
            else:
                if column.name not in item:
                    fail("E_SCHEMA", "row is missing a non-null column", location={"file": source, "path": value_path})
                if item[column.name] is None:
                    fail("E_SCHEMA", "null value is missing from null_bitmap", location={"file": source, "path": value_path})
                normalized[column.name] = _validate_cell(item[column.name], column, source=source, path=value_path)
        if normalized[row_id_column_name] != row_id:
            fail("E_SCHEMA", "row identifier mismatch", location={"file": source, "path": f"{prefix}.{row_id_column_name}"})
        rows[row_id] = normalized
    stray_bitmap = sorted(known_bitmap_rows - set(rows), key=_row_sort_key)
    if stray_bitmap:
        fail("E_SCHEMA", "null bitmap references an unknown row", location={"file": source, "path": "$.null_bitmap", "row_id": stray_bitmap[0]})
    if not rows:
        fail("E_SCHEMA", "base table must contain at least one row", location={"file": source, "path": "$.rows"})

    column_tuple = tuple(columns)
    payload = canonical_rows_payload(table_name, row_id_column_name, column_tuple, rows)
    return BaseTable(
        table_name=table_name,
        row_id_column=row_id_column_name,
        columns=column_tuple,
        rows=rows,
        null_bitmap=null_bitmap,
        digest=digest_obj(payload),
        source_path=source,
    )


def _normalize_operator(operator: Any, *, source: str, path: str) -> str:
    aliases = {
        "=": "eq",
        "==": "eq",
        "!=": "ne",
        ">": "gt",
        ">=": "gte",
        "<": "lt",
        "<=": "lte",
    }
    normalized = aliases.get(operator, operator)
    supported = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "is_null", "is_not_null", "and", "or", "not"}
    if normalized not in supported:
        fail("E_SCHEMA", f"unsupported filter operator {operator!r}", location={"file": source, "path": path})
    return normalized


def _validate_filter(node: Any, column_map: dict[str, Column], *, source: str, path: str) -> dict[str, Any]:
    obj = _require_object(node, source=source, path=path)
    operator = _normalize_operator(obj.get("op"), source=source, path=f"{path}.op")
    if operator in {"and", "or"}:
        args = _require_list(obj.get("args"), source=source, path=f"{path}.args")
        if not args:
            fail("E_SCHEMA", "boolean filter requires at least one argument", location={"file": source, "path": f"{path}.args"})
        normalized_args = [_validate_filter(arg, column_map, source=source, path=f"{path}.args[{index}]") for index, arg in enumerate(args)]
        return {"op": operator, "args": normalized_args}
    if operator == "not":
        return {
            "op": "not",
            "arg": _validate_filter(obj.get("arg"), column_map, source=source, path=f"{path}.arg"),
        }
    column_name = obj.get("column")
    if not isinstance(column_name, str):
        fail("E_SCHEMA", "filter column must be a string", location={"file": source, "path": f"{path}.column"})
    column = column_map.get(column_name)
    if column is None:
        fail("E_SCHEMA", f"filter references unknown column {column_name!r}", location={"file": source, "path": f"{path}.column"})
    if operator in {"is_null", "is_not_null"}:
        return {"op": operator, "column": column_name}
    if "value" not in obj:
        fail("E_SCHEMA", "filter is missing value", location={"file": source, "path": f"{path}.value"})
    value = obj["value"]
    if operator in {"in", "not_in"}:
        values = _require_list(value, source=source, path=f"{path}.value")
        if not values:
            fail("E_SCHEMA", "filter value list must not be empty", location={"file": source, "path": f"{path}.value"})
        normalized_values = [
            _validate_cell(item, column, source=source, path=f"{path}.value[{index}]", allow_null=True)
            for index, item in enumerate(values)
        ]
        return {"op": operator, "column": column_name, "value": normalized_values}
    if operator in {"gt", "gte", "lt", "lte"} and column.type not in NUMERIC_COLUMN_TYPES:
        fail("E_SCHEMA", "range filter requires a numeric column", location={"file": source, "path": f"{path}.column"})
    normalized_value = _validate_cell(value, column, source=source, path=f"{path}.value", allow_null=True)
    return {"op": operator, "column": column_name, "value": normalized_value}


def _load_session(root: dict[str, Any], *, source: str) -> SessionConfig:
    raw = root.get("session", {})
    session = _require_object(raw, source=source, path="$.session")
    scale_factor = _positive_int(session.get("scale_factor", 100), source=source, path="$.session.scale_factor")
    truncation = session.get("truncation", "toward_zero")
    overflow = session.get("overflow", "saturate_int64")
    if truncation not in {"toward_zero", "floor", "ceil"}:
        fail("E_SCHEMA", "unsupported truncation mode", location={"file": source, "path": "$.session.truncation"})
    if overflow != "saturate_int64":
        fail("E_SCHEMA", "unsupported overflow mode", location={"file": source, "path": "$.session.overflow"})
    global_concurrency = _positive_int(session.get("global_concurrency", 2), source=source, path="$.session.global_concurrency")
    batch_size = _positive_int(session.get("batch_size", 4), source=source, path="$.session.batch_size")
    ttl = _positive_int(session.get("default_lease_ttl_ms", 800), source=source, path="$.session.default_lease_ttl_ms")
    renew = _positive_int(session.get("default_lease_renew_ms", 200), source=source, path="$.session.default_lease_renew_ms")
    if renew >= ttl:
        fail("E_SCHEMA", "lease renewal interval must be shorter than lease TTL", location={"file": source, "path": "$.session.default_lease_renew_ms"})
    return SessionConfig(scale_factor, truncation, overflow, global_concurrency, batch_size, ttl, renew)


def load_definitions(path: str | Path) -> Definitions:
    raw, source = _read_json(path)
    root = _require_object(raw, source=source, path="$")
    if root.get("format") not in (None, "imv.views.v1"):
        fail("E_SCHEMA", "unsupported views format", location={"file": source, "path": "$.format"})
    session = _load_session(root, source=source)
    raw_views = _require_list(root.get("views"), source=source, path="$.views")
    if not raw_views:
        fail("E_SCHEMA", "at least one view is required", location={"file": source, "path": "$.views"})
    views: dict[str, ViewDef] = {}
    for index, raw_view in enumerate(raw_views):
        prefix = f"$.views[{index}]"
        item = _require_object(raw_view, source=source, path=prefix)
        view_id = _identifier(item.get("id"), source=source, path=f"{prefix}.id")
        if view_id in views:
            fail("E_SCHEMA", f"duplicate view id {view_id!r}", location={"file": source, "path": f"{prefix}.id"})
        group_by_raw = _require_list(item.get("group_by"), source=source, path=f"{prefix}.group_by")
        if not group_by_raw:
            fail("E_SCHEMA", "group_by must not be empty", location={"file": source, "path": f"{prefix}.group_by"})
        group_by: list[str] = []
        for group_index, name in enumerate(group_by_raw):
            if not isinstance(name, str):
                fail("E_SCHEMA", "group key column must be a string", location={"file": source, "path": f"{prefix}.group_by[{group_index}]"})
            if name in group_by:
                fail("E_SCHEMA", "duplicate group key column", location={"file": source, "path": f"{prefix}.group_by[{group_index}]"})
            group_by.append(name)
        raw_aggregates = _require_list(item.get("aggregates"), source=source, path=f"{prefix}.aggregates")
        if not raw_aggregates:
            fail("E_SCHEMA", "at least one aggregate is required", location={"file": source, "path": f"{prefix}.aggregates"})
        aggregates: list[AggregateSpec] = []
        aliases: set[str] = set()
        for aggregate_index, raw_aggregate in enumerate(raw_aggregates):
            agg_prefix = f"{prefix}.aggregates[{aggregate_index}]"
            aggregate = _require_object(raw_aggregate, source=source, path=agg_prefix)
            func = aggregate.get("func")
            if func not in AGGREGATE_FUNCTIONS:
                fail("E_SCHEMA", "unsupported aggregate function", location={"file": source, "path": f"{agg_prefix}.func"})
            column = aggregate.get("column", "*")
            if not isinstance(column, str):
                fail("E_SCHEMA", "aggregate column must be a string", location={"file": source, "path": f"{agg_prefix}.column"})
            alias = aggregate.get("alias", f"{func}_{column.replace('*', 'rows')}")
            alias = _identifier(alias, source=source, path=f"{agg_prefix}.alias")
            if alias in aliases:
                fail("E_SCHEMA", f"duplicate aggregate alias {alias!r}", location={"file": source, "path": f"{agg_prefix}.alias"})
            aliases.add(alias)
            aggregates.append(AggregateSpec(func, column, alias))
        refresh_mode = item.get("refresh_mode", "incremental")
        if refresh_mode not in REFRESH_MODES:
            fail("E_SCHEMA", "unsupported refresh_mode", location={"file": source, "path": f"{prefix}.refresh_mode"})
        per_view_quota = _positive_int(item.get("per_view_quota", 1), source=source, path=f"{prefix}.per_view_quota")
        ttl = _positive_int(item.get("lease_ttl_ms", session.default_lease_ttl_ms), source=source, path=f"{prefix}.lease_ttl_ms")
        renew = _positive_int(item.get("lease_renew_ms", session.default_lease_renew_ms), source=source, path=f"{prefix}.lease_renew_ms")
        if renew >= ttl:
            fail("E_SCHEMA", "lease renewal interval must be shorter than lease TTL", location={"file": source, "path": f"{prefix}.lease_renew_ms"})
        filter_raw = item.get("filter")
        filter_obj: dict[str, Any] | None = None
        if filter_raw is not None:
            if not isinstance(filter_raw, dict):
                fail("E_SCHEMA", "filter must be a JSON object", location={"file": source, "path": f"{prefix}.filter"})
            filter_obj = filter_raw
        payload = {
            "id": view_id,
            "group_by": group_by,
            "aggregates": [aggregate.payload() for aggregate in aggregates],
            "filter": filter_obj,
            "refresh_mode": refresh_mode,
            "per_view_quota": per_view_quota,
            "lease_ttl_ms": ttl,
            "lease_renew_ms": renew,
        }
        views[view_id] = ViewDef(
            view_id=view_id,
            group_by=tuple(group_by),
            aggregates=tuple(aggregates),
            filter=filter_obj,
            refresh_mode=refresh_mode,
            per_view_quota=per_view_quota,
            lease_ttl_ms=ttl,
            lease_renew_ms=renew,
            payload=payload,
            digest=digest_obj(payload),
        )
    fingerprint_payload = {
        "scale_factor": session.scale_factor,
        "truncation": session.truncation,
        "overflow": session.overflow,
        "views": [views[view_id].payload for view_id in sorted(views)],
    }
    return Definitions(session=session, views=views, fingerprint=digest_obj(fingerprint_payload), source_path=source)


def validate_definitions_against_base(definitions: Definitions, base: BaseTable) -> None:
    """Validate all column references after both documents are available."""

    source = definitions.source_path
    column_map = base.column_map
    for view_index, view_id in enumerate(sorted(definitions.views)):
        view = definitions.views[view_id]
        prefix = f"$.views[{view_index}]"
        for group_index, name in enumerate(view.group_by):
            if name not in column_map:
                fail("E_SCHEMA", f"group key references unknown column {name!r}", location={"file": source, "path": f"{prefix}.group_by[{group_index}]", "view_id": view_id})
        for aggregate_index, aggregate in enumerate(view.aggregates):
            if aggregate.column == "*":
                if aggregate.func != "count":
                    fail("E_SCHEMA", "only count supports column '*'", location={"file": source, "path": f"{prefix}.aggregates[{aggregate_index}].column", "view_id": view_id})
                continue
            column = column_map.get(aggregate.column)
            if column is None:
                fail("E_SCHEMA", f"aggregate references unknown column {aggregate.column!r}", location={"file": source, "path": f"{prefix}.aggregates[{aggregate_index}].column", "view_id": view_id})
            if aggregate.func in {"sum", "min", "max"} and column.type not in NUMERIC_COLUMN_TYPES:
                fail("E_SCHEMA", "numeric aggregate requires a numeric column", location={"file": source, "path": f"{prefix}.aggregates[{aggregate_index}].column", "view_id": view_id})
        if view.filter is not None:
            view_payload = dict(view.payload)
            view_payload["filter"] = _validate_filter(view.filter, column_map, source=source, path=f"{prefix}.filter")
            normalized = dict(view.payload)
            normalized["filter"] = view_payload["filter"]
            # Rebuild the frozen definition with the fully validated filter.
            object.__setattr__(view, "filter", view_payload["filter"])
            object.__setattr__(view, "payload", normalized)
            object.__setattr__(view, "digest", digest_obj(normalized))

    fingerprint_payload = {
        "scale_factor": definitions.session.scale_factor,
        "truncation": definitions.session.truncation,
        "overflow": definitions.session.overflow,
        "views": [definitions.views[view_id].payload for view_id in sorted(definitions.views)],
    }
    object.__setattr__(definitions, "fingerprint", digest_obj(fingerprint_payload))


def load_changes(path: str | Path, base: BaseTable) -> ChangeStream:
    source = str(Path(path))
    changes: list[Change] = []
    effective_rows = {row_id: dict(row) for row_id, row in base.rows.items()}
    expected_seq = 1
    column_map = base.column_map
    try:
        handle = open(path, "r", encoding="utf-8-sig")
    except FileNotFoundError:
        fail("E_SCHEMA", "change stream does not exist", location={"file": source, "line": 0, "path": "$"})
    except OSError as exc:
        fail("E_SCHEMA", f"cannot read change stream: {exc}", location={"file": source, "line": 0, "path": "$"})
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(
                    "E_SCHEMA",
                    f"invalid NDJSON: {exc.msg}",
                    location={"file": source, "line": line_number, "column": exc.colno, "path": "$"},
                )
            if not isinstance(raw, dict):
                fail("E_SCHEMA", "change must be a JSON object", location={"file": source, "line": line_number, "path": "$"})
            seq = raw.get("seq", raw.get("sequence"))
            if isinstance(seq, bool) or not isinstance(seq, int):
                fail("E_SCHEMA", "change seq must be an integer", location={"file": source, "line": line_number, "path": "$.seq"})
            if seq != expected_seq:
                reason = "duplicate" if seq < expected_seq else "non-contiguous"
                fail(
                    "E_SCHEMA",
                    f"change sequence is {reason}; expected {expected_seq}, got {seq}",
                    location={"file": source, "line": line_number, "path": "$.seq", "sequence": seq},
                )
            op = raw.get("op", raw.get("type"))
            if op not in CHANGE_TYPES:
                fail("E_SCHEMA", "change type must be insert, update or delete", location={"file": source, "line": line_number, "path": "$.op"})
            row_id_value = raw.get("row_id", raw.get(base.row_id_column))
            row_id_column = column_map[base.row_id_column]
            row_id = _coerce_row_id(row_id_value, row_id_column, source=source, path=f"line {line_number}:$.row_id")
            row_raw = raw.get("row", raw.get("after"))
            normalized_row: dict[str, Any] | None = None
            if op in {"insert", "update"}:
                if row_raw is None:
                    fail("E_SCHEMA", f"{op} change requires row data", location={"file": source, "line": line_number, "path": "$.row"})
                row_obj = _require_object(row_raw, source=source, path=f"line {line_number}:$.row")
                extra = sorted(set(row_obj) - set(column_map))
                if extra:
                    fail("E_SCHEMA", f"change row contains unknown column {extra[0]!r}", location={"file": source, "line": line_number, "path": f"$.row.{extra[0]}"})
                normalized_row = {}
                for column in base.columns:
                    if column.name not in row_obj:
                        fail("E_SCHEMA", "change row is missing a column", location={"file": source, "line": line_number, "path": f"$.row.{column.name}"})
                    normalized_row[column.name] = _validate_cell(
                        row_obj[column.name],
                        column,
                        source=source,
                        path=f"line {line_number}:$.row.{column.name}",
                    )
                if normalized_row[base.row_id_column] != row_id:
                    fail("E_SCHEMA", "change row identifier does not match row_id", location={"file": source, "line": line_number, "path": f"$.row.{base.row_id_column}"})
            if op == "insert" and row_id in effective_rows:
                fail("E_SCHEMA", f"insert row identifier {row_id!r} already exists", location={"file": source, "line": line_number, "path": "$.row_id", "row_id": row_id})
            if op in {"update", "delete"} and row_id not in effective_rows:
                fail("E_SCHEMA", f"{op} row identifier {row_id!r} does not exist in the base table", location={"file": source, "line": line_number, "path": "$.row_id", "row_id": row_id})
            if op == "delete":
                effective_rows.pop(row_id)
            else:
                assert normalized_row is not None
                effective_rows[row_id] = normalized_row
            payload = {"seq": seq, "op": op, "row_id": row_id, "row": normalized_row}
            changes.append(Change(seq, op, row_id, normalized_row, line_number, payload))
            expected_seq += 1
    digest = digest_obj([change.payload for change in changes])
    return ChangeStream(tuple(changes), digest, source)


def definition_summary(definitions: Definitions) -> dict[str, Any]:
    return {
        "fingerprint": definitions.fingerprint,
        "scale_factor": definitions.session.scale_factor,
        "truncation": definitions.session.truncation,
        "overflow": definitions.session.overflow,
        "view_count": len(definitions.views),
        "views": {
            view_id: {
                "digest": definitions.views[view_id].digest,
                "group_by": list(definitions.views[view_id].group_by),
                "aggregates": [spec.alias for spec in definitions.views[view_id].aggregates],
                "refresh_mode": definitions.views[view_id].refresh_mode,
            }
            for view_id in sorted(definitions.views)
        },
    }
