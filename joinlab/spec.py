from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .canonical import canonical_json, canonical_rows, digest_json, row_tokens
from .errors import JoinLabError, fail


TYPE_ALIASES: dict[str, str] = {
    "integer": "integer",
    "int": "integer",
    "int64": "integer",
    "float": "float",
    "float64": "float",
    "double": "float",
    "real": "float",
}
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
FILTER_OPS = {
    "eq": "eq",
    "=": "eq",
    "==": "eq",
    "ne": "ne",
    "!=": "ne",
    "<>": "ne",
    "lt": "lt",
    "<": "lt",
    "le": "le",
    "<=": "le",
    "gt": "gt",
    ">": "gt",
    "ge": "ge",
    ">=": "ge",
    "is_null": "is_null",
    "is not null": "not_null",
    "not_null": "not_null",
    "is_not_null": "not_null",
}


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    type: str


@dataclass(frozen=True)
class FilterSpec:
    column: str
    op: str
    value: Any = None


@dataclass
class TableSpec:
    name: str
    columns: tuple[ColumnSpec, ...]
    row_count: int
    skew: bool
    rows: list[dict[str, Any]]
    structure_version: str
    data_digest: str
    source_path: str

    @property
    def column_types(self) -> tuple[str, ...]:
        return tuple(column.type for column in self.columns)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> ColumnSpec | None:
        for column in self.columns:
            if column.name == name:
                return column
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "row_count": self.row_count,
            "columns": [{"name": c.name, "type": c.type} for c in self.columns],
            "skew": self.skew,
            "structure_version": self.structure_version,
            "data_digest": self.data_digest,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class JoinSpec:
    left_table: str
    left_key: str
    right_table: str
    right_key: str


@dataclass
class WorkspaceSpec:
    workspace_id: str
    source_path: str
    tables: dict[str, TableSpec]
    join: JoinSpec
    filters: dict[str, tuple[FilterSpec, ...]]
    default_params: dict[str, Any]
    attribution_k: int

    def table(self, name: str) -> TableSpec:
        try:
            return self.tables[name]
        except KeyError:
            fail("E_TABLE", f"未知表：{name}", "join", table=name)
            raise AssertionError("unreachable")


def _need(condition: bool, path: str, message: str, **details: Any) -> None:
    if not condition:
        fail("E_TABLE", message, path, **details)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_json(path: Path, path_name: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        fail("E_TABLE", "文件不存在", path_name, file=str(path))
    except json.JSONDecodeError as exc:
        fail(
            "E_TABLE",
            "JSON 解析失败",
            path_name,
            line=exc.lineno,
            column=exc.colno,
            reason=exc.msg,
        )
    except OSError as exc:
        fail("E_TABLE", "文件无法读取", path_name, file=str(path), reason=str(exc))
    raise AssertionError("unreachable")


def _normalise_column(raw: Any, path: str) -> ColumnSpec:
    _need(isinstance(raw, dict), path, "列定义必须是对象")
    name = raw.get("name")
    _need(isinstance(name, str) and bool(name), path + ".name", "列名必须是非空字符串")
    _need(not name.startswith("__"), path + ".name", "以 __ 开头的列名保留给系统")
    raw_type = raw.get("type")
    _need(isinstance(raw_type, str) and bool(raw_type), path + ".type", "列类型必须声明")
    resolved = TYPE_ALIASES.get(raw_type.strip().lower())
    if resolved is None:
        fail(
            "E_TYPE",
            "联接分析仅支持 integer/float 数值列",
            path + ".type",
            declared_type=raw_type,
        )
    return ColumnSpec(name=name, type=resolved)


def _normalise_cell(value: Any, column: ColumnSpec, path: str) -> Any:
    if value is None:
        return None
    if column.type == "integer":
        if not _is_int(value):
            fail("E_TABLE", "整型列中出现非整数", path, column=column.name, value=value)
        if value < INT64_MIN or value > INT64_MAX:
            fail("E_TABLE", "整数超出 SQLite 64 位范围", path, column=column.name, value=value)
        return int(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail("E_TABLE", "浮点列中出现非数值", path, column=column.name, value=value)
    converted = float(value)
    if not math.isfinite(converted):
        fail("E_TABLE", "浮点列不允许 NaN/Inf", path, column=column.name)
    return converted


def _extract_rows_payload(document: Any, data_path: str) -> tuple[Any, str]:
    if isinstance(document, dict):
        _need("rows" in document, data_path, "行数据对象缺少 rows")
        return document["rows"], data_path + ".rows"
    return document, data_path


def _normalise_rows(
    payload: Any,
    columns: Sequence[ColumnSpec],
    row_path: str,
) -> list[list[Any]]:
    _need(isinstance(payload, list), row_path, "rows 必须是数组")
    names = [column.name for column in columns]
    expected_names = set(names)
    result: list[list[Any]] = []
    for row_index, raw_row in enumerate(payload):
        path = f"{row_path}[{row_index}]"
        if isinstance(raw_row, dict):
            missing = [name for name in names if name not in raw_row]
            _need(not missing, path, "行缺少列", missing=missing)
            extra = sorted(set(raw_row) - expected_names)
            _need(not extra, path, "行包含未声明列", extra=extra)
            values = [raw_row[name] for name in names]
            value_paths = [f"{path}.{name}" for name in names]
        elif isinstance(raw_row, (list, tuple)):
            _need(
                len(raw_row) == len(columns),
                path,
                "行长度与列数不一致",
                expected=len(columns),
                actual=len(raw_row),
            )
            values = list(raw_row)
            value_paths = [f"{path}[{index}]" for index in range(len(columns))]
        else:
            fail("E_TABLE", "行必须是数组或对象", path)
        result.append(
            [
                _normalise_cell(value, columns[index], value_paths[index])
                for index, value in enumerate(values)
            ]
        )
    return result


def _load_table(raw: Any, index: int, base_dir: Path) -> TableSpec:
    path = f"tables[{index}]"
    _need(isinstance(raw, dict), path, "表定义必须是对象")
    name = raw.get("name")
    _need(isinstance(name, str) and bool(name.strip()), path + ".name", "表名必须非空")
    name = name.strip()
    _need("\x00" not in name, path + ".name", "表名不能含 NUL")
    row_count = raw.get("row_count")
    _need(_is_int(row_count) and row_count > 0, path + ".row_count", "row_count 必须是正整数")
    skew = raw.get("skew", False)
    _need(isinstance(skew, bool), path + ".skew", "skew 必须是布尔值")
    raw_columns = raw.get("columns")
    _need(isinstance(raw_columns, list) and bool(raw_columns), path + ".columns", "columns 必须是非空数组")
    columns = tuple(
        _normalise_column(column, f"{path}.columns[{column_index}]")
        for column_index, column in enumerate(raw_columns)
    )
    column_names = [column.name for column in columns]
    duplicates = sorted({n for n in column_names if column_names.count(n) > 1})
    _need(not duplicates, path + ".columns", "列名必须唯一", duplicates=duplicates)

    if "rows" in raw:
        payload = raw["rows"]
        row_path = path + ".rows"
        source_path = "inline:rows"
    else:
        data_file = raw.get("data_file")
        _need(isinstance(data_file, str) and bool(data_file), path + ".data_file", "必须提供 rows 或 data_file")
        data_path = (base_dir / data_file).resolve()
        document = _read_json(data_path, path + ".data_file")
        payload, row_path = _extract_rows_payload(document, path + ".data_file")
        source_path = str(data_path)
    raw_rows = _normalise_rows(payload, columns, row_path)
    _need(
        len(raw_rows) == row_count,
        path + ".row_count",
        "样本行数与声明规模不一致",
        declared=row_count,
        actual=len(raw_rows),
    )

    column_types = tuple(column.type for column in columns)
    canonical = canonical_rows(name, raw_rows, column_types)
    rows = [
        {
            "__rid": item["__rid"],
            "__tokens": item["__tokens"],
            **{column.name: value for column, value in zip(columns, item["values"])},
        }
        for item in canonical
    ]
    structure_payload = {
        "name": name,
        "columns": [{"name": c.name, "type": c.type} for c in columns],
        "row_count": row_count,
    }
    structure_version = digest_json(structure_payload)
    data_payload = {
        "table": name,
        "columns": [{"name": c.name, "type": c.type} for c in columns],
        "rows": [item["__tokens"] for item in canonical],
    }
    data_digest = digest_json(data_payload)
    return TableSpec(
        name=name,
        columns=columns,
        row_count=row_count,
        skew=skew,
        rows=rows,
        structure_version=structure_version,
        data_digest=data_digest,
        source_path=source_path,
    )


def _normalise_filters(raw: Any, tables: dict[str, TableSpec]) -> dict[str, tuple[FilterSpec, ...]]:
    if raw is None:
        return {}
    _need(isinstance(raw, dict), "filters", "filters 必须是对象")
    result: dict[str, tuple[FilterSpec, ...]] = {}
    for table_name, raw_filters in raw.items():
        path = f"filters.{table_name}"
        _need(table_name in tables, path, "过滤谓词引用了未知表")
        table = tables[table_name]
        items = raw_filters if isinstance(raw_filters, list) else [raw_filters]
        filters: list[FilterSpec] = []
        for index, item in enumerate(items):
            item_path = f"{path}[{index}]"
            _need(isinstance(item, dict), item_path, "过滤谓词必须是对象")
            column_name = item.get("column")
            _need(isinstance(column_name, str) and bool(column_name), item_path + ".column", "过滤列必须声明")
            column = table.column(column_name)
            _need(column is not None, item_path + ".column", "过滤列不存在", column=column_name)
            raw_op = item.get("op")
            _need(isinstance(raw_op, str), item_path + ".op", "过滤操作符必须声明")
            op = FILTER_OPS.get(raw_op.strip().lower())
            _need(op is not None, item_path + ".op", "不支持的过滤操作符", op=raw_op)
            value = item.get("value")
            if (
                op not in {"is_null", "not_null"}
                and value is not None
                and (isinstance(value, bool) or not isinstance(value, (int, float)))
            ):
                fail("E_TABLE", "过滤值必须是数值或 null", item_path + ".value")
            if isinstance(value, float) and not math.isfinite(value):
                fail("E_TABLE", "过滤值不允许 NaN/Inf", item_path + ".value")
            filters.append(FilterSpec(column=column_name, op=op, value=value))
        result[table_name] = tuple(filters)
    return result


def _normalise_join(raw: Any, tables: dict[str, TableSpec]) -> JoinSpec:
    _need(isinstance(raw, dict), "join", "join 必须是对象")
    fields = ("left_table", "left_key", "right_table", "right_key")
    for field in fields:
        _need(isinstance(raw.get(field), str) and bool(raw[field]), f"join.{field}", "联接字段必须声明")
    join = JoinSpec(
        left_table=raw["left_table"],
        left_key=raw["left_key"],
        right_table=raw["right_table"],
        right_key=raw["right_key"],
    )
    _need(join.left_table in tables, "join.left_table", "左表不存在", table=join.left_table)
    _need(join.right_table in tables, "join.right_table", "右表不存在", table=join.right_table)
    _need(join.left_table != join.right_table, "join.right_table", "等值联接需要两张不同基表")
    left = tables[join.left_table]
    right = tables[join.right_table]
    left_column = left.column(join.left_key)
    _need(left_column is not None, "join.left_key", "左联接键列不存在", column=join.left_key)
    right_column = right.column(join.right_key)
    _need(right_column is not None, "join.right_key", "右联接键列不存在", column=join.right_key)
    numeric = {"integer", "float"}
    if left_column.type not in numeric or right_column.type not in numeric:
        fail(
            "E_TYPE",
            "联接键类型不可比较",
            "join",
            left_type=left_column.type,
            right_type=right_column.type,
        )
    return join


def _check_skew(join: JoinSpec, tables: dict[str, TableSpec], threshold: float) -> None:
    for table_name, key in (
        (join.left_table, join.left_key),
        (join.right_table, join.right_key),
    ):
        table = tables[table_name]
        if not table.skew:
            continue
        values = [row[key] for row in table.rows if row[key] is not None]
        if not values:
            continue
        top_key, top_count = Counter(values).most_common(1)[0]
        ratio = top_count / len(values)
        if ratio >= threshold:
            fail(
                "E_SKEW",
                "倾斜标记命中，联接键集中度超过阈值",
                f"tables[{list(tables).index(table_name)}].skew",
                table=table_name,
                join_key=key,
                top_key=top_key,
                top_count=top_count,
                non_null_rows=len(values),
                ratio=f"{ratio:.6f}",
                threshold=f"{threshold:.6f}",
            )


def _extract_defaults(document: dict[str, Any]) -> tuple[dict[str, Any], int]:
    params = document.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        fail("E_TABLE", "params 必须是对象", "params")
    raw_k = document.get("attribution_k", params.get("attribution_k", 5))
    if not _is_int(raw_k) or raw_k <= 0:
        fail("E_TABLE", "attribution_k 必须是正整数", "attribution_k")
    return dict(params), raw_k


def load_workspace(path: str | Path, skew_threshold: float = 0.65) -> WorkspaceSpec:
    source = Path(path).resolve()
    document = _read_json(source, "$")
    _need(isinstance(document, dict), "$", "工作区 JSON 顶层必须是对象")
    raw_tables = document.get("tables")
    _need(isinstance(raw_tables, list) and bool(raw_tables), "tables", "tables 必须是非空数组")
    tables: dict[str, TableSpec] = {}
    for index, raw_table in enumerate(raw_tables):
        table = _load_table(raw_table, index, source.parent)
        if table.name in tables:
            fail(
                "E_TABLE",
                "表名重复",
                f"tables[{index}].name",
                table=table.name,
            )
        tables[table.name] = table
    join = _normalise_join(document.get("join"), tables)
    filters = _normalise_filters(document.get("filters"), tables)
    defaults, attribution_k = _extract_defaults(document)
    _check_skew(join, tables, skew_threshold)
    workspace_id = document.get("workspace_id")
    _need(isinstance(workspace_id, str) and bool(workspace_id), "workspace_id", "workspace_id 必须声明")
    return WorkspaceSpec(
        workspace_id=workspace_id,
        source_path=str(source),
        tables=tables,
        join=join,
        filters=filters,
        default_params=defaults,
        attribution_k=attribution_k,
    )


def sqlite_table_name(table_name: str) -> str:
    return "jl_" + hashlib.sha256(table_name.encode("utf-8")).hexdigest()[:16]


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def ingest_workspace(spec: WorkspaceSpec, db_path: str | Path) -> dict[str, Any]:
    """Atomically build the local SQLite mirror and verify it after reopening."""

    target = Path(db_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=target.stem + ".", suffix=".tmp.sqlite", dir=str(target.parent)
    )
    os.close(file_descriptor)
    temp_path = Path(temporary_name)
    try:
        connection = sqlite3.connect(str(temp_path))
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE jl_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            for table in spec.tables.values():
                physical_name = sqlite_table_name(table.name)
                column_defs = ", ".join(
                    [f"{_quote_identifier('__rid')} TEXT PRIMARY KEY"]
                    + [f"{_quote_identifier(column.name)} {('INTEGER' if column.type == 'integer' else 'REAL')}" for column in table.columns]
                )
                connection.execute(f"CREATE TABLE {_quote_identifier(physical_name)} ({column_defs})")
                placeholders = ", ".join("?" for _ in range(len(table.columns) + 1))
                insert_sql = (
                    f"INSERT INTO {_quote_identifier(physical_name)} VALUES ({placeholders})"
                )
                data = [
                    [row["__rid"]] + [row[column.name] for column in table.columns]
                    for row in table.rows
                ]
                connection.executemany(insert_sql, data)
                connection.execute(
                    "INSERT INTO jl_meta(key,value) VALUES(?,?)",
                    (f"table:{table.name}:logical", table.name),
                )
                connection.execute(
                    "INSERT INTO jl_meta(key,value) VALUES(?,?)",
                    (f"table:{table.name}:physical", physical_name),
                )
                connection.execute(
                    "INSERT INTO jl_meta(key,value) VALUES(?,?)",
                    (f"table:{table.name}:structure_version", table.structure_version),
                )
                connection.execute(
                    "INSERT INTO jl_meta(key,value) VALUES(?,?)",
                    (f"table:{table.name}:data_digest", table.data_digest),
                )
                connection.execute(
                    "INSERT INTO jl_meta(key,value) VALUES(?,?)",
                    (f"table:{table.name}:row_count", str(table.row_count)),
                )
            connection.commit()
        finally:
            connection.close()
        os.replace(temp_path, target)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise

    verify: dict[str, Any] = {}
    for table in spec.tables.values():
        rows = load_table_rows(target, table)
        if len(rows) != table.row_count:
            fail(
                "E_TABLE",
                "SQLite 回读行数与声明不一致",
                f"tables[{list(spec.tables).index(table.name)}].row_count",
                declared=table.row_count,
                actual=len(rows),
            )
        ordered_tokens = sorted(
            row_tokens([row[c.name] for c in table.columns], table.column_types)
            for row in rows
        )
        digest_payload = {
            "table": table.name,
            "columns": [{"name": c.name, "type": c.type} for c in table.columns],
            "rows": ordered_tokens,
        }
        readback_digest = digest_json(digest_payload)
        if readback_digest != table.data_digest:
            fail(
                "E_TABLE",
                "SQLite 回读行摘要与源 JSON 不一致",
                f"tables[{list(spec.tables).index(table.name)}].data_file",
                expected=table.data_digest,
                actual=readback_digest,
            )
        verify[table.name] = {
            "row_count": len(rows),
            "data_digest": readback_digest,
            "structure_version": table.structure_version,
        }
    return {
        "db_path": str(target),
        "tables": verify,
        "data_digest": digest_json(verify),
    }


def load_table_rows(db_path: str | Path, table: TableSpec) -> list[dict[str, Any]]:
    path = Path(db_path)
    if not path.exists():
        fail("E_TABLE", "SQLite 样本文件不存在", "tables", file=str(path))
    physical_name = sqlite_table_name(table.name)
    columns = [column.name for column in table.columns]
    selected = ", ".join([_quote_identifier("__rid")] + [_quote_identifier(name) for name in columns])
    try:
        connection = sqlite3.connect(str(path))
        try:
            cursor = connection.execute(
                f"SELECT {selected} FROM {_quote_identifier(physical_name)} ORDER BY {_quote_identifier('__rid')}"
            )
            rows: list[dict[str, Any]] = []
            for values in cursor.fetchall():
                row = {"__rid": values[0]}
                row.update({name: values[index + 1] for index, name in enumerate(columns)})
                rows.append(row)
            return rows
        finally:
            connection.close()
    except sqlite3.Error as exc:
        fail("E_TABLE", "SQLite 样本加载失败", "tables", table=table.name, reason=str(exc))
    raise AssertionError("unreachable")


def table_structure_payload(spec: WorkspaceSpec) -> dict[str, Any]:
    return {
        name: {
            "structure_version": table.structure_version,
            "data_digest": table.data_digest,
            "row_count": table.row_count,
        }
        for name, table in sorted(spec.tables.items())
    }
