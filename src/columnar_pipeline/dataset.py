from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_json, sha256_obj
from .errors import data_error, digest_error, skew_error
from .model import (
    NUMERIC_COLUMN_TYPES,
    SUPPORTED_COLUMN_TYPES,
    Column,
    Dataset,
    Query,
    Table,
)
from .query import IDENTIFIER_RE, parse_query


def _expect_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise data_error("expected a JSON object", location)
    return value


def _expect_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise data_error("expected a JSON list", location)
    return value


def _expect_positive_int(value: Any, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise data_error("expected an integer >= 1", location)
    return value


def _expect_nonnegative_int(value: Any, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise data_error("expected an integer >= 0", location)
    return value


def _validate_value(value: Any, type_name: str, location: str) -> None:
    if type_name in NUMERIC_COLUMN_TYPES:
        if not isinstance(value, int) or isinstance(value, bool):
            raise data_error(f"value must be an integer for type {type_name}", location)
    elif type_name == "string":
        if not isinstance(value, str):
            raise data_error("value must be a string for type string", location)


def _parse_column(raw: Any, row_count: int, location: str) -> Column:
    doc = _expect_mapping(raw, location)
    name = doc.get("name")
    if not isinstance(name, str) or not IDENTIFIER_RE.match(name):
        raise data_error("column name must be a valid identifier", f"{location}.name")
    type_name = doc.get("type")
    if not isinstance(type_name, str) or type_name not in SUPPORTED_COLUMN_TYPES:
        raise data_error(
            f"unsupported column type: {type_name!r}",
            f"{location}.type",
            details={"supported": sorted(SUPPORTED_COLUMN_TYPES)},
        )
    scale = doc.get("scale", 0)
    if type_name == "fixed":
        if not isinstance(scale, int) or isinstance(scale, bool) or scale < 0:
            raise data_error("fixed column scale must be an integer >= 0", f"{location}.scale")
    elif scale not in (0, None):
        raise data_error("scale is only valid for fixed columns", f"{location}.scale")
    else:
        scale = 0
    null_bitmap = doc.get("null_bitmap")
    if not isinstance(null_bitmap, list):
        raise data_error("null_bitmap must be a list", f"{location}.null_bitmap")
    if len(null_bitmap) != row_count:
        raise data_error(
            "null_bitmap length does not equal row_count",
            f"{location}.null_bitmap",
            details={"expected": row_count, "actual": len(null_bitmap)},
        )
    for row, bit in enumerate(null_bitmap):
        if not isinstance(bit, int) or isinstance(bit, bool) or bit not in (0, 1):
            raise data_error(
                "null bitmap entries must be 0 or 1",
                f"{location}.null_bitmap[{row}]",
            )
    values = doc.get("values")
    if not isinstance(values, list):
        raise data_error("values must be a list", f"{location}.values")
    if len(values) != row_count:
        raise data_error(
            "values length does not equal row_count",
            f"{location}.values",
            details={"expected": row_count, "actual": len(values)},
        )
    for row, (value, is_null) in enumerate(zip(values, null_bitmap, strict=True)):
        if is_null:
            if value is not None:
                raise data_error(
                    "values at null bitmap positions must be JSON null",
                    f"{location}.values[{row}]",
                )
            continue
        _validate_value(value, type_name, f"{location}.values[{row}]")
    return Column(
        name=name,
        type_name=type_name,
        values=tuple(values),
        null_bitmap=tuple(null_bitmap),
        scale=int(scale),
    )


def _parse_table(raw: Any, location: str) -> Table:
    doc = _expect_mapping(raw, location)
    name = doc.get("name")
    if not isinstance(name, str) or not IDENTIFIER_RE.match(name):
        raise data_error("table name must be a valid identifier", f"{location}.name")
    version = _expect_positive_int(doc.get("version"), f"{location}.version")
    row_count = _expect_nonnegative_int(doc.get("row_count"), f"{location}.row_count")
    raw_columns = _expect_list(doc.get("columns"), f"{location}.columns")
    if not raw_columns:
        raise data_error("table must contain at least one column", f"{location}.columns")
    columns: list[Column] = []
    column_names: set[str] = set()
    for index, raw_column in enumerate(raw_columns):
        column = _parse_column(raw_column, row_count, f"{location}.columns[{index}]")
        if column.name in column_names:
            raise data_error(
                f"duplicate column name {column.name!r}",
                f"{location}.columns[{index}].name",
            )
        column_names.add(column.name)
        columns.append(column)
    digest_payload = {
        "name": name,
        "version": version,
        "row_count": row_count,
        "columns": [
            {
                "name": column.name,
                "type": column.type_name,
                "scale": column.scale,
                "null_bitmap": list(column.null_bitmap),
                "values": list(column.values),
            }
            for column in columns
        ],
    }
    digest = sha256_obj(digest_payload)
    declared_digest = doc.get("digest")
    if declared_digest is not None:
        if not isinstance(declared_digest, str):
            raise data_error("table digest must be a string", f"{location}.digest")
        if declared_digest != digest:
            raise digest_error(
                f"declared digest for table {name!r} does not match table content",
                location=f"{location}.digest",
                details={"declared": declared_digest, "actual": digest, "table": name},
            )
    return Table(
        name=name,
        version=version,
        row_count=row_count,
        columns=tuple(columns),
        digest=digest,
    )


def _dataset_digest(tables: Sequence[Table]) -> str:
    return sha256_obj(
        {
            "tables": [
                {
                    "name": table.name,
                    "version": table.version,
                    "row_count": table.row_count,
                    "digest": table.digest,
                }
                for table in tables
            ]
        }
    )


def _rebuild_table(table: Table, duplicate_rows: Sequence[int], multiplier: int) -> Table:
    if multiplier <= 1 or not duplicate_rows:
        return table
    repeat_count = multiplier - 1
    new_values: list[tuple[Any, ...]] = []
    new_bitmaps: list[tuple[int, ...]] = []
    for column in table.columns:
        values = list(column.values)
        bitmap = list(column.null_bitmap)
        for _ in range(repeat_count):
            for row in duplicate_rows:
                values.append(column.values[row])
                bitmap.append(column.null_bitmap[row])
        new_values.append(tuple(values))
        new_bitmaps.append(tuple(bitmap))
    columns = tuple(
        Column(
            name=column.name,
            type_name=column.type_name,
            values=new_values[index],
            null_bitmap=new_bitmaps[index],
            scale=column.scale,
        )
        for index, column in enumerate(table.columns)
    )
    row_count = table.row_count + len(duplicate_rows) * repeat_count
    payload = {
        "name": table.name,
        "version": table.version,
        "row_count": row_count,
        "columns": [
            {
                "name": column.name,
                "type": column.type_name,
                "scale": column.scale,
                "null_bitmap": list(column.null_bitmap),
                "values": list(column.values),
            }
            for column in columns
        ],
    }
    return Table(
        name=table.name,
        version=table.version,
        row_count=row_count,
        columns=columns,
        digest=sha256_obj(payload),
    )


def _passes_filter(query: Query, table: Table, row: int) -> bool:
    column_map = table.column_map()

    def evaluate(node: Mapping[str, Any]) -> bool:
        if node.get("op") == "and":
            return all(evaluate(term) for term in node["terms"])
        value = column_map[str(node["column"])].value_at(row)
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

    return True if query.where is None else evaluate(query.where)


def _global_group_counts(dataset: Dataset) -> dict[tuple[Any, ...], int]:
    counts: dict[tuple[Any, ...], int] = {}
    for table in dataset.tables:
        if table.name not in dataset.query.from_tables:
            continue
        columns = table.column_map()
        group_columns = [columns[name] for name in dataset.query.group_by]
        for row in range(table.row_count):
            if not _passes_filter(dataset.query, table, row):
                continue
            key = tuple(column.value_at(row) for column in group_columns)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _hottest_key(counts: Mapping[tuple[Any, ...], int]) -> tuple[tuple[Any, ...], int]:
    if not counts:
        raise data_error("skew amplification found no rows after filtering", "injection.skew")
    return min(counts.items(), key=lambda item: (-item[1], canonical_json(list(item[0]))))


def apply_skew_amplification(dataset: Dataset, multiplier: int, safety_limit: int) -> Dataset:
    if multiplier <= 1:
        return dataset
    counts = _global_group_counts(dataset)
    hot_key, hot_count = _hottest_key(counts)
    tables: list[Table] = []
    rows_added = 0
    for table in dataset.tables:
        duplicate_rows: list[int] = []
        if table.name in dataset.query.from_tables:
            columns = table.column_map()
            group_columns = [columns[name] for name in dataset.query.group_by]
            for row in range(table.row_count):
                if not _passes_filter(dataset.query, table, row):
                    continue
                key = tuple(column.value_at(row) for column in group_columns)
                if key == hot_key:
                    duplicate_rows.append(row)
        tables.append(_rebuild_table(table, duplicate_rows, multiplier))
        rows_added += len(duplicate_rows) * (multiplier - 1)
    amplified_count = hot_count * multiplier
    injection = {
        "kind": "skew_amplification",
        "multiplier": multiplier,
        "group_key": list(hot_key),
        "original_hot_count": hot_count,
        "amplified_hot_count": amplified_count,
        "rows_added": rows_added,
        "safety_limit": safety_limit,
    }
    result = Dataset(
        tables=tuple(tables),
        query=dataset.query,
        data_digest=_dataset_digest(tables),
        injection=injection,
    )
    if amplified_count > safety_limit:
        raise skew_error(
            "injected skew amplification exceeded the configured safety limit",
            location="injection.skew",
            details=injection,
        )
    return result


def load_dataset(path: str | Path) -> Dataset:
    dataset_path = Path(path)
    try:
        text = dataset_path.read_text(encoding="utf-8")
    except OSError as error:
        raise data_error(f"cannot read dataset file: {error}", str(dataset_path)) from error
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise data_error(
            f"invalid JSON: {error.msg}",
            f"{dataset_path}:{error.lineno}:{error.colno}",
        ) from error
    root = _expect_mapping(document, "$")
    format_version = root.get("format_version", 1)
    if format_version != 1:
        raise data_error(
            f"unsupported format_version: {format_version!r}",
            "$.format_version",
        )
    raw_tables = _expect_list(root.get("tables"), "$.tables")
    if not raw_tables:
        raise data_error("dataset must contain at least one table", "$.tables")
    tables: list[Table] = []
    table_names: set[str] = set()
    for index, raw_table in enumerate(raw_tables):
        table = _parse_table(raw_table, f"$.tables[{index}]")
        if table.name in table_names:
            raise data_error(
                f"duplicate table name {table.name!r}",
                f"$.tables[{index}].name",
            )
        table_names.add(table.name)
        tables.append(table)
    tables.sort(key=lambda table: table.name)
    query = parse_query(_expect_mapping(root.get("query"), "$.query"), tuple(tables))
    return Dataset(
        tables=tuple(tables),
        query=query,
        data_digest=_dataset_digest(tables),
        injection={},
    )


def clone_dataset_with_query(dataset: Dataset, query: Query) -> Dataset:
    return Dataset(
        tables=copy.deepcopy(dataset.tables),
        query=query,
        data_digest=dataset.data_digest,
        injection=copy.deepcopy(dataset.injection),
    )
