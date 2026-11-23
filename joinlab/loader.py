from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from .errors import LabError, schema_error
from .model import (
    Column,
    ColumnRef,
    FilterPredicate,
    Histogram,
    HistogramBin,
    Query,
    Table,
    Tuning,
    Workload,
    canonical_key,
)
from .sqlparse import parse_query


FORMAT_VERSION = 1
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED = {
    "select",
    "from",
    "where",
    "inner",
    "join",
    "on",
    "group",
    "by",
    "order",
    "as",
    "and",
    "or",
    "is",
    "not",
    "null",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "asc",
    "desc",
}


def load_workload(path: str | Path) -> Workload:
    file_path = Path(path)
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except FileNotFoundError as exc:
        raise LabError("E_SCHEMA", f"工作负载文件不存在: {file_path}", field="file") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise schema_error(
            f"JSON 语法错误: {getattr(exc, 'msg', str(exc))}",
            field="json",
            line=getattr(exc, "lineno", None),
            column=getattr(exc, "colno", None),
        ) from exc
    except UnicodeDecodeError as exc:
        raise schema_error("工作负载文件不是有效 UTF-8", field="file") from exc
    return load_workload_dict(raw, source=str(file_path))


def load_workload_dict(raw: Any, *, source: str = "<memory>") -> Workload:
    root = _require_dict(raw, "根对象", field="workload")
    format_version = _require_int(root.get("format_version"), "format_version", field="format_version")
    if format_version != FORMAT_VERSION:
        raise schema_error(
            f"不支持的格式版本 {format_version}",
            field="format_version",
            supported=FORMAT_VERSION,
        )
    tuning = _parse_tuning(root.get("tuning", {}))
    table_values = root.get("tables")
    if not isinstance(table_values, list) or not table_values:
        raise schema_error("tables 必须是非空数组", field="tables")

    tables: list[Table] = []
    seen_tables: set[str] = set()
    for table_index, table_raw in enumerate(table_values):
        table = _parse_table(table_raw, table_index)
        if table.name.lower() in seen_tables:
            raise schema_error(
                f"重复表名 {table.name!r}",
                table=table.name,
                field="tables.name",
                table_index=table_index,
            )
        seen_tables.add(table.name.lower())
        tables.append(table)
    tables.sort(key=lambda table: table.name)
    table_map = {table.name: table for table in tables}

    query_raw = root.get("query")
    if not isinstance(query_raw, dict):
        raise schema_error("query 必须是对象", field="query")
    sql = query_raw.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        raise schema_error("query.sql 必须是非空字符串", field="query.sql")
    try:
        query = parse_query(sql)
    except LabError as exc:
        if exc.code == "E_SCHEMA":
            raise
        raise
    _resolve_query(query, table_map)
    workload = Workload(
        format_version=format_version,
        tables=tuple(tables),
        query=query,
        tuning=tuning,
        digest="",
    )
    digest = _workload_digest(workload)
    return replace(workload, digest=digest)


def _parse_tuning(raw: Any) -> Tuning:
    if not isinstance(raw, dict):
        raise schema_error("tuning 必须是对象", field="tuning")
    defaults = Tuning()

    def integer(name: str, minimum: int) -> int:
        value = raw.get(name, getattr(defaults, name))
        return _require_int(value, name, field=f"tuning.{name}", minimum=minimum)

    def number(name: str, minimum: float, *, strict: bool = False) -> float:
        value = raw.get(name, getattr(defaults, name))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise schema_error(f"tuning.{name} 必须是数值", field=f"tuning.{name}")
        if not math.isfinite(float(value)) or value < minimum or (strict and value == minimum):
            raise schema_error(
                f"tuning.{name} 必须{'大于' if strict else '大于等于'} {minimum}",
                field=f"tuning.{name}",
            )
        return float(value)

    confidence = number("confidence_level", 0.0, strict=True)
    if confidence >= 1.0:
        raise schema_error("tuning.confidence_level 必须小于 1", field="tuning.confidence_level")
    return Tuning(
        seed=integer("seed", 0),
        fixed_point_scale=integer("fixed_point_scale", 1),
        exact_scan_threshold_rows=integer("exact_scan_threshold_rows", 1),
        sample_size=integer("sample_size", 1),
        confidence_level=confidence,
        max_candidates=integer("max_candidates", 1),
        page_size_bytes=integer("page_size_bytes", 1),
        row_overhead_bytes=integer("row_overhead_bytes", 0),
        global_concurrency=integer("global_concurrency", 1),
        per_table_scan_quota=integer("per_table_scan_quota", 1),
        memory_budget_bytes=integer("memory_budget_bytes", 1),
        bias_threshold_ratio=number("bias_threshold_ratio", 1.0, strict=True),
        comparison_weight=integer("comparison_weight", 1),
        io_page_weight=integer("io_page_weight", 1),
        memory_weight=integer("memory_weight", 1),
    )


def _parse_table(raw: Any, table_index: int) -> Table:
    if not isinstance(raw, dict):
        raise schema_error("表声明必须是对象", field="tables", table_index=table_index)
    name = raw.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name) or name.lower() in _RESERVED:
        raise schema_error(
            "表名必须是合法且非 SQL 保留字的标识符",
            table=name if isinstance(name, str) else None,
            field="tables.name",
            table_index=table_index,
        )

    column_values = raw.get("columns")
    if not isinstance(column_values, list) or not column_values:
        raise schema_error("columns 必须是非空数组", table=name, field="columns")
    columns: list[Column] = []
    seen_columns: set[str] = set()
    for column_index, col_raw in enumerate(column_values):
        if not isinstance(col_raw, dict):
            raise schema_error(
                "列声明必须是对象",
                table=name,
                field=f"columns[{column_index}]",
            )
        col_name = col_raw.get("name")
        if not isinstance(col_name, str) or not _NAME_RE.match(col_name) or col_name.lower() in _RESERVED:
            raise schema_error(
                "列名必须是合法且非 SQL 保留字的标识符",
                table=name,
                column=col_name if isinstance(col_name, str) else None,
                field=f"columns[{column_index}].name",
            )
        if col_name.lower() in seen_columns:
            raise schema_error(
                f"重复列名 {col_name!r}",
                table=name,
                column=col_name,
                field=f"columns[{column_index}].name",
            )
        col_type = col_raw.get("type")
        if col_type not in {"integer", "float"}:
            raise schema_error(
                "列类型只能是 integer 或 float",
                table=name,
                column=col_name,
                field=f"columns[{column_index}].type",
            )
        seen_columns.add(col_name.lower())
        columns.append(Column(col_name, col_type))
    column_map = {column.name: column for column in columns}

    rows_raw = raw.get("rows")
    if not isinstance(rows_raw, list):
        raise schema_error("rows 必须是数组", table=name, field="rows")
    rows: list[tuple[Any, ...]] = []
    for row_index, row in enumerate(rows_raw):
        if not isinstance(row, list) or len(row) != len(columns):
            raise schema_error(
                f"第 {row_index} 行长度必须为 {len(columns)}",
                table=name,
                field=f"rows[{row_index}]",
            )
        typed_row: list[Any] = []
        for column_index, value in enumerate(row):
            column = columns[column_index]
            if value is None:
                typed_row.append(None)
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise schema_error(
                    "单元格必须是数值或 null",
                    table=name,
                    column=column.name,
                    field=f"rows[{row_index}][{column_index}]",
                )
            if not _finite_number(value):
                raise schema_error(
                    "单元格必须是有限数值",
                    table=name,
                    column=column.name,
                    field=f"rows[{row_index}][{column_index}]",
                )
            if column.type == "integer" and not isinstance(value, int):
                raise schema_error(
                    "integer 列不能包含浮点样本值",
                    table=name,
                    column=column.name,
                    field=f"rows[{row_index}][{column_index}]",
                )
            typed_row.append(value)
        rows.append(tuple(typed_row))

    bitmap_raw = raw.get("null_bitmap")
    if not isinstance(bitmap_raw, list) or len(bitmap_raw) != len(rows):
        raise schema_error(
            "null_bitmap 长度必须等于行数",
            table=name,
            field="null_bitmap",
            expected=len(rows),
            actual=len(bitmap_raw) if isinstance(bitmap_raw, list) else None,
        )
    null_bitmap: list[list[int]] = []
    for row_index, bits in enumerate(bitmap_raw):
        if not isinstance(bits, list) or len(bits) != len(columns):
            raise schema_error(
                f"null_bitmap[{row_index}] 长度必须为 {len(columns)}",
                table=name,
                field=f"null_bitmap[{row_index}]",
            )
        typed_bits: list[int] = []
        for column_index, bit in enumerate(bits):
            if isinstance(bit, bool) or not isinstance(bit, int) or bit not in (0, 1):
                raise schema_error(
                    "空值位必须是 0 或 1",
                    table=name,
                    column=columns[column_index].name,
                    field=f"null_bitmap[{row_index}][{column_index}]",
                )
            value = rows[row_index][column_index]
            if bit == 1 and value is not None:
                raise schema_error(
                    "空值位为 1 时样本值必须为 null",
                    table=name,
                    column=columns[column_index].name,
                    field=f"rows[{row_index}][{column_index}]",
                )
            if bit == 0 and value is None:
                raise schema_error(
                    "空值位为 0 时样本值不能为 null",
                    table=name,
                    column=columns[column_index].name,
                    field=f"null_bitmap[{row_index}][{column_index}]",
                )
            typed_bits.append(int(bit))
        null_bitmap.append(typed_bits)

    hist_raw = raw.get("histograms")
    if not isinstance(hist_raw, dict):
        raise schema_error("histograms 必须是对象", table=name, field="histograms")
    histograms: dict[str, Histogram] = {}
    for column_index, column in enumerate(columns):
        if column.name not in hist_raw:
            raise schema_error(
                "缺少列直方图",
                table=name,
                column=column.name,
                field=f"histograms.{column.name}",
            )
        histogram = _parse_histogram(
            hist_raw[column.name],
            table=name,
            column=column,
            nonnull_count=sum(1 for row in rows if row[column_index] is not None),
        )
        histograms[column.name] = histogram
    extra = sorted(set(hist_raw) - set(column_map))
    if extra:
        raise schema_error(
            f"直方图引用未知列 {extra[0]!r}",
            table=name,
            column=extra[0],
            field=f"histograms.{extra[0]}",
        )

    return Table(name, tuple(columns), rows, null_bitmap, histograms)


def _parse_histogram(raw: Any, *, table: str, column: Column, nonnull_count: int) -> Histogram:
    if not isinstance(raw, dict):
        raise schema_error(
            "列直方图必须是对象", table=table, column=column.name, field=f"histograms.{column.name}"
        )
    version = _require_int(raw.get("version"), "version", table=table, column=column.name, field=f"histograms.{column.name}.version", minimum=1)
    bins_raw = raw.get("bins")
    if not isinstance(bins_raw, list) or not bins_raw:
        raise schema_error(
            "直方图 bins 必须是非空数组",
            table=table,
            column=column.name,
            field=f"histograms.{column.name}.bins",
        )
    bins: list[HistogramBin] = []
    previous_hi: Decimal | None = None
    for index, bin_raw in enumerate(bins_raw):
        if not isinstance(bin_raw, dict):
            raise schema_error(
                "直方图区间必须是对象",
                table=table,
                column=column.name,
                field=f"histograms.{column.name}.bins[{index}]",
            )
        lo = _decimal(bin_raw.get("lo"))
        hi = _decimal(bin_raw.get("hi"))
        count = _require_int(
            bin_raw.get("count"),
            "count",
            table=table,
            column=column.name,
            field=f"histograms.{column.name}.bins[{index}].count",
            minimum=0,
        )
        if lo is None or hi is None or not lo.is_finite() or not hi.is_finite():
            raise schema_error(
                "直方图区间端点必须有限",
                table=table,
                column=column.name,
                field=f"histograms.{column.name}.bins[{index}]",
            )
        if lo >= hi:
            raise schema_error(
                "直方图区间必须满足 lo < hi",
                table=table,
                column=column.name,
                field=f"histograms.{column.name}.bins[{index}]",
            )
        if previous_hi is not None and lo != previous_hi:
            kind = "重叠" if lo < previous_hi else "存在空隙"
            raise schema_error(
                f"直方图区间{kind}：前区间 hi={previous_hi}，当前 lo={lo}",
                table=table,
                column=column.name,
                field=f"histograms.{column.name}.bins[{index}].lo",
            )
        if column.type == "integer" and (lo != lo.to_integral_value() or hi != hi.to_integral_value()):
            raise schema_error(
                "integer 列直方图端点必须是整数",
                table=table,
                column=column.name,
                field=f"histograms.{column.name}.bins[{index}]",
            )
        bins.append(HistogramBin(lo, hi, count))
        previous_hi = hi
    total = sum(item.count for item in bins)
    if total != nonnull_count:
        raise schema_error(
            f"直方图计数之和 {total} 必须等于非空行数 {nonnull_count}",
            table=table,
            column=column.name,
            field=f"histograms.{column.name}.bins.count",
        )
    canonical = [
        {"lo": str(item.lo), "hi": str(item.hi), "count": item.count} for item in bins
    ]
    digest = _sha256_json({"version": version, "bins": canonical})
    return Histogram(version, tuple(bins), digest)


def _resolve_query(query: Query, table_map: dict[str, Table]) -> None:
    if len(query.tables) < 2:
        raise schema_error("多表连接查询至少需要两张表", field="query.tables")
    for table_name in query.source_order:
        if table_name not in table_map:
            raise schema_error(
                f"查询引用未知表 {table_name!r}",
                table=table_name,
                field="query.tables",
            )

    def resolve_ref(ref: ColumnRef, field: str) -> Column:
        if ref.table not in query.tables:
            raise schema_error(
                f"查询列引用表 {ref.table!r} 不在 FROM/JOIN 范围中",
                table=ref.table,
                column=ref.column,
                field=field,
            )
        table = table_map.get(ref.table)
        if table is None:
            raise schema_error(
                f"查询引用未知表 {ref.table!r}",
                table=ref.table,
                column=ref.column,
                field=field,
            )
        try:
            return table.column(ref.column)
        except KeyError as exc:
            raise schema_error(
                f"查询引用未知列 {ref.key!r}",
                table=ref.table,
                column=ref.column,
                field=field,
            ) from exc

    seen_edges: set[tuple[str, str, str, str]] = set()
    for predicate in query.joins:
        left_col = resolve_ref(predicate.left, "query.joins.left")
        right_col = resolve_ref(predicate.right, "query.joins.right")
        if predicate.left.table == predicate.right.table:
            raise schema_error(
                "连接谓词必须连接两张不同的表",
                table=predicate.left.table,
                column=predicate.left.column,
                field="query.joins",
            )
        if not _types_comparable(left_col.type, right_col.type):
            raise schema_error(
                "连接谓词列类型不可比较",
                table=predicate.left.table,
                column=predicate.left.column,
                field="query.joins.type",
            )
        edge = (
            predicate.left.table,
            predicate.left.column,
            predicate.right.table,
            predicate.right.column,
        )
        if edge in seen_edges:
            raise schema_error("重复连接谓词", table=predicate.left.table, column=predicate.left.column, field="query.joins")
        seen_edges.add(edge)

    for predicate in query.filters:
        column = resolve_ref(predicate.column, "query.filters.column")
        if predicate.is_null_test:
            continue
        if predicate.value is None:
            raise schema_error(
                "非 IS NULL 谓词不能使用 NULL 字面量",
                table=predicate.column.table,
                column=predicate.column.column,
                field="query.filters.value",
            )
        if isinstance(predicate.value, bool) or not _finite_number(predicate.value):
            raise schema_error(
                "过滤谓词字面量必须是有限数值",
                table=predicate.column.table,
                column=predicate.column.column,
                field="query.filters.value",
            )
        if column.type == "integer" and isinstance(predicate.value, float) and not predicate.value.is_integer():
            # Integer/float comparisons are legal; this branch documents that no
            # implicit narrowing occurs and keeps all values as supplied.
            pass

    alias_set: set[str] = set()
    group_keys = {ref.key for ref in query.group_by}
    for ref in query.group_by:
        resolve_ref(ref, "query.group_by")
    for item in query.select:
        if item.alias in alias_set:
            raise schema_error(
                f"投影别名重复 {item.alias!r}", field="query.select.alias"
            )
        alias_set.add(item.alias)
        if item.column is not None:
            resolve_ref(item.column, "query.select.column")
        if item.aggregate is None and (item.column is None or item.column.key not in group_keys):
            raise schema_error(
                "非聚合投影列必须出现在 GROUP BY 中",
                table=item.column.table if item.column else None,
                column=item.column.column if item.column else None,
                field="query.select",
            )
    if not any(item.aggregate for item in query.select):
        raise schema_error("查询必须包含至少一个聚合投影", field="query.select.aggregate")
    for ref in query.order_by:
        resolve_ref(ref, "query.order_by")

    adjacency = {table: set() for table in query.tables}
    for predicate in query.joins:
        left, right = predicate.left.table, predicate.right.table
        adjacency[left].add(right)
        adjacency[right].add(left)
    visited: set[str] = set()
    stack = [query.tables[0]]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(adjacency[current] - visited)
    if visited != set(query.tables):
        missing = sorted(set(query.tables) - visited)
        raise schema_error(
            f"连接图不连通，缺少含等值谓词的表 {missing}",
            field="query.joins",
            tables=missing,
        )


def _types_comparable(left: str, right: str) -> bool:
    return left in {"integer", "float"} and right in {"integer", "float"}


def _require_dict(value: Any, label: str, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise schema_error(f"{label}必须是对象", field=field)
    return value


def _require_int(
    value: Any,
    label: str,
    *,
    field: str,
    table: str | None = None,
    column: str | None = None,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise schema_error(
            f"{label} 必须是整数",
            table=table,
            column=column,
            field=field,
        )
    if minimum is not None and value < minimum:
        raise schema_error(
            f"{label} 必须大于等于 {minimum}",
            table=table,
            column=column,
            field=field,
        )
    return value


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _reject_constant(value: str) -> None:
    raise ValueError(f"不支持非有限数值 {value}")


def _sha256_json(value: Any) -> str:
    encoded = canonical_key(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _workload_digest(workload: Workload) -> str:
    canonical = workload.canonical(include_rows=True)
    canonical["query"] = workload.query.canonical()
    canonical["query_source_order"] = list(workload.query.source_order)
    canonical["tables"] = sorted(canonical["tables"], key=lambda item: item["name"])
    return _sha256_json(canonical)


def rebuild_workload(tables: Iterable[Table], query: Query, tuning: Tuning) -> Workload:
    workload = Workload(
        format_version=FORMAT_VERSION,
        tables=tuple(sorted(tables, key=lambda table: table.name)),
        query=query,
        tuning=tuning,
        digest="",
    )
    return replace(workload, digest=_workload_digest(workload))


def validate_histogram_override(
    raw: Any,
    table: Table,
    column_name: str,
) -> Histogram:
    column_index = table.column_index(column_name)
    nonnull_count = sum(1 for row in table.rows if row[column_index] is not None)
    return _parse_histogram(
        raw,
        table=table.name,
        column=table.columns[column_index],
        nonnull_count=nonnull_count,
    )
