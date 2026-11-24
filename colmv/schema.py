"""Strict loading and validation for base-table catalogs and view definitions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, NoReturn

from .canonical import canonical_json, object_digest
from .errors import ColmvError
from .model import BaseTable, ColumnDef, Filter, PageMeta, RuntimeConfig, ViewDefinition
from .storage import read_page_file, validate_page_columns


SUPPORTED_AGGREGATES = frozenset({"count", "sum", "avg", "mean"})
SUPPORTED_FILTER_OPS = frozenset(
    {"eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "is_null", "is_not_null"}
)
SUPPORTED_REFRESH_MODES = frozenset(
    {"auto", "resident", "rescan", "incremental", "page_rescan"}
)
FORMAT_VERSION = 1


def _view_error(
    view_id: str,
    field: str,
    message: str,
    **details: Any,
) -> NoReturn:
    raise ColmvError("E_VIEW", message, view_id=view_id, field=field, details=details)


def _read_json(path: Path, *, view_id: str, field: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        _view_error(view_id, field, f"cannot read JSON file {path}: {exc}")
    except json.JSONDecodeError as exc:
        _view_error(
            view_id,
            field,
            f"JSON syntax error in {path}",
            line=exc.lineno,
            column=exc.colno,
            reason=exc.msg,
        )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_refresh_mode(raw: Any, view_id: str) -> str:
    if not isinstance(raw, str) or raw not in SUPPORTED_REFRESH_MODES:
        _view_error(
            view_id,
            "refresh_mode",
            f"unsupported refresh_mode {raw!r}",
            supported=sorted(SUPPORTED_REFRESH_MODES),
        )
    if raw == "incremental":
        return "resident"
    if raw == "page_rescan":
        return "rescan"
    return raw


def _parse_filter(raw: Any, *, view_id: str, index: int, columns: Mapping[str, ColumnDef]) -> Filter:
    field = f"filters[{index}]"
    if not isinstance(raw, dict):
        _view_error(view_id, field, "filter must be an object")
    column = raw.get("column")
    op = raw.get("op")
    if not isinstance(column, str) or column not in columns:
        _view_error(view_id, f"{field}.column", f"unknown filter column {column!r}")
    if not isinstance(op, str) or op not in SUPPORTED_FILTER_OPS:
        _view_error(
            view_id,
            f"{field}.op",
            f"unsupported filter operator {op!r}",
            supported=sorted(SUPPORTED_FILTER_OPS),
        )
    if op in ("is_null", "is_not_null"):
        if "value" in raw:
            _view_error(view_id, f"{field}.value", f"{op} filter must not contain value")
        return Filter(column=column, op=op)
    if "value" not in raw:
        _view_error(view_id, f"{field}.value", f"{op} filter requires a value")
    value = raw["value"]
    column_type = columns[column].type
    if op in ("in", "not_in"):
        if not isinstance(value, list):
            _view_error(view_id, f"{field}.value", f"{op} filter value must be a list")
        for item_index, item in enumerate(value):
            _validate_filter_scalar(item, column_type, view_id, f"{field}.value[{item_index}]")
    else:
        _validate_filter_scalar(value, column_type, view_id, f"{field}.value")
    return Filter(column=column, op=op, value=value)


def _validate_filter_scalar(value: Any, column_type: str, view_id: str, field: str) -> None:
    if value is None:
        return
    if column_type == "int":
        if not _is_int(value):
            _view_error(view_id, field, "integer column filter value must be an integer or null")
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _view_error(view_id, field, "float column filter value must be numeric or null")
        if not math.isfinite(float(value)):
            _view_error(view_id, field, "float column filter value must be finite")


def _parse_views_object(
    raw: Any,
    columns: Mapping[str, ColumnDef],
) -> tuple[tuple[ViewDefinition, ...], int, dict[str, Any]]:
    default_view = "<views>"
    if not isinstance(raw, dict):
        _view_error(default_view, "document", "views JSON root must be an object")
    if not _is_int(raw.get("format_version")) or raw.get("format_version") != FORMAT_VERSION:
        _view_error(
            default_view,
            "format_version",
            f"views format_version must be integer {FORMAT_VERSION}",
        )
    scale_power = raw.get("scale_power")
    if not _is_int(scale_power) or not 0 <= scale_power <= 18:
        _view_error(default_view, "scale_power", "scale_power must be an integer in [0, 18]")
    raw_views = raw.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        _view_error(default_view, "views", "views must be a non-empty list")

    parsed: list[ViewDefinition] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_views):
        fallback_id = f"<views[{index}]>"
        if not isinstance(item, dict):
            _view_error(fallback_id, "view_id", "view definition must be an object")
        view_id = item.get("view_id")
        if not isinstance(view_id, str) or not view_id.strip():
            _view_error(fallback_id, "view_id", "view_id must be a non-empty string")
        if view_id in seen_ids:
            _view_error(view_id, "view_id", f"duplicate view_id {view_id!r}")
        seen_ids.add(view_id)

        group_by_raw = item.get("group_by")
        if not isinstance(group_by_raw, list) or not group_by_raw:
            _view_error(view_id, "group_by", "group_by must be a non-empty list")
        group_by: list[str] = []
        for group_index, column in enumerate(group_by_raw):
            if not isinstance(column, str):
                _view_error(view_id, f"group_by[{group_index}]", "group_by entries must be strings")
            if column not in columns:
                _view_error(view_id, f"group_by[{group_index}]", f"unknown group column {column!r}")
            if column in group_by:
                _view_error(view_id, f"group_by[{group_index}]", f"duplicate group column {column!r}")
            group_by.append(column)

        aggregates_raw = item.get("aggregates")
        if not isinstance(aggregates_raw, list) or not aggregates_raw:
            _view_error(view_id, "aggregates", "aggregates must be a non-empty list")
        aggregates: list[str] = []
        for aggregate_index, aggregate in enumerate(aggregates_raw):
            if not isinstance(aggregate, str) or aggregate not in SUPPORTED_AGGREGATES:
                _view_error(
                    view_id,
                    f"aggregates[{aggregate_index}]",
                    f"unsupported aggregate {aggregate!r}",
                    supported=sorted(SUPPORTED_AGGREGATES),
                )
            normalized = "avg" if aggregate == "mean" else aggregate
            if normalized in aggregates:
                _view_error(view_id, f"aggregates[{aggregate_index}]", f"duplicate aggregate {aggregate!r}")
            aggregates.append(normalized)

        value_column_raw = item.get("value_column")
        needs_value = any(name in ("sum", "avg") for name in aggregates)
        if value_column_raw is None:
            if needs_value:
                _view_error(
                    view_id,
                    "value_column",
                    "value_column is required for sum/avg aggregates",
                )
            value_column: str | None = None
        else:
            if not isinstance(value_column_raw, str) or value_column_raw not in columns:
                _view_error(view_id, "value_column", f"unknown value_column {value_column_raw!r}")
            if columns[value_column_raw].type not in ("int", "float"):
                _view_error(view_id, "value_column", "value_column must be numeric")
            value_column = value_column_raw

        filters_raw = item.get("filters", [])
        if filters_raw is None:
            filters_raw = []
        if not isinstance(filters_raw, list):
            _view_error(view_id, "filters", "filters must be a list")
        filters = tuple(
            _parse_filter(filter_raw, view_id=view_id, index=filter_index, columns=columns)
            for filter_index, filter_raw in enumerate(filters_raw)
        )

        refresh_mode = _normalize_refresh_mode(item.get("refresh_mode", "auto"), view_id)
        parsed.append(
            ViewDefinition(
                view_id=view_id,
                group_by=tuple(group_by),
                aggregates=tuple(aggregates),
                value_column=value_column,
                filters=filters,
                refresh_mode=refresh_mode,
            )
        )

    normalized = {
        "format_version": FORMAT_VERSION,
        "scale_power": scale_power,
        "views": [view.to_dict() for view in parsed],
    }
    return tuple(parsed), scale_power, normalized


def _parse_columns(raw: Any, first_view: str) -> tuple[tuple[ColumnDef, ...], dict[str, ColumnDef]]:
    if not isinstance(raw, list) or not raw:
        _view_error(first_view, "base.columns", "table.columns must be a non-empty list")
    columns: list[ColumnDef] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        field = f"base.columns[{index}]"
        if not isinstance(item, dict):
            _view_error(first_view, field, "column definition must be an object")
        name = item.get("name")
        column_type = item.get("type")
        if not isinstance(name, str) or not name:
            _view_error(first_view, f"{field}.name", "column name must be a non-empty string")
        if name in seen:
            _view_error(first_view, f"{field}.name", f"duplicate column {name!r}")
        seen.add(name)
        if column_type not in ("int", "float"):
            _view_error(first_view, f"{field}.type", f"unsupported column type {column_type!r}")
        columns.append(ColumnDef(name=name, type=column_type))
    return tuple(columns), {column.name: column for column in columns}


def _parse_catalog_views(
    raw_views: Any,
    columns: Mapping[str, ColumnDef],
) -> tuple[tuple[ViewDefinition, ...], int, dict[str, Any]]:
    return _parse_views_object(raw_views, columns)


def _peek_first_view_id(raw_views: Any) -> str:
    if isinstance(raw_views, dict):
        items = raw_views.get("views")
        if isinstance(items, list) and items:
            first = items[0]
            if isinstance(first, dict) and isinstance(first.get("view_id"), str) and first["view_id"].strip():
                return first["view_id"]
            return "<views[0]>"
    return "<views>"


def load_config(
    catalog_path: str | Path,
    views_path: str | Path,
    *,
    inject_truncate: bool = False,
) -> RuntimeConfig:
    catalog_file = Path(catalog_path)
    views_file = Path(views_path)

    views_raw = _read_json(views_file, view_id="<views>", field="document")
    first_declared_view = _peek_first_view_id(views_raw)
    catalog_raw = _read_json(catalog_file, view_id=first_declared_view, field="base.document")

    if not isinstance(catalog_raw, dict):
        _view_error(first_declared_view, "base.document", "catalog JSON root must be an object")
    if not _is_int(catalog_raw.get("format_version")) or catalog_raw.get("format_version") != FORMAT_VERSION:
        _view_error(
            first_declared_view,
            "base.format_version",
            f"catalog format_version must be integer {FORMAT_VERSION}",
        )
    table_raw = catalog_raw.get("table")
    if not isinstance(table_raw, dict):
        _view_error(first_declared_view, "base.table", "catalog.table must be an object")
    columns, column_map = _parse_columns(table_raw.get("columns"), first_declared_view)

    # Views are parsed before page contents so the first invalid view/field is stable.
    views, scale_power, normalized_views = _parse_catalog_views(views_raw, column_map)
    first_view = views[0].view_id

    table_name = table_raw.get("name", "base")
    if not isinstance(table_name, str) or not table_name:
        _view_error(first_view, "base.table.name", "table.name must be a non-empty string")
    row_count = table_raw.get("row_count")
    if not _is_int(row_count) or row_count < 0:
        _view_error(first_view, "base.table.row_count", "row_count must be a non-negative integer")
    row_key = table_raw.get("row_key")
    if not isinstance(row_key, str) or row_key not in column_map:
        _view_error(first_view, "base.table.row_key", f"unknown row_key {row_key!r}")
    if column_map[row_key].type != "int":
        _view_error(first_view, "base.table.row_key", "row_key column must have integer type")

    pages_raw = catalog_raw.get("pages")
    if not isinstance(pages_raw, list) or (row_count > 0 and not pages_raw):
        _view_error(first_view, "base.pages", "pages must be a non-empty list for non-empty table")
    pages: list[PageMeta] = []
    seen_page_ids: set[str] = set()
    cursor = 0
    for index, item in enumerate(pages_raw):
        field = f"base.pages[{index}]"
        if not isinstance(item, dict):
            _view_error(first_view, field, "page directory entry must be an object")
        page_id = item.get("page_id")
        file_name = item.get("file")
        start_row = item.get("start_row")
        end_row = item.get("end_row")
        if not isinstance(page_id, str) or not page_id:
            _view_error(first_view, f"{field}.page_id", "page_id must be a non-empty string")
        if page_id == "delta":
            _view_error(first_view, f"{field}.page_id", "page_id 'delta' is reserved for inserts")
        if page_id in seen_page_ids:
            _view_error(first_view, f"{field}.page_id", f"duplicate page_id {page_id!r}")
        seen_page_ids.add(page_id)
        if not isinstance(file_name, str) or not file_name:
            _view_error(first_view, f"{field}.file", "page file must be a non-empty string")
        if not _is_int(start_row) or not _is_int(end_row):
            _view_error(first_view, field, "start_row and end_row must be integers")
        if start_row != cursor:
            reason = "overlap" if start_row < cursor else "gap"
            _view_error(
                first_view,
                f"{field}.start_row",
                "page row intervals must be contiguous, non-overlapping, and start at zero",
                reason=reason,
                expected_start_row=cursor,
                actual_start_row=start_row,
            )
        if end_row <= start_row:
            _view_error(
                first_view,
                f"{field}.end_row",
                "page row interval must contain at least one row",
            )
        declared_count = item.get("row_count")
        if declared_count is not None and (not _is_int(declared_count) or declared_count != end_row - start_row):
            _view_error(
                first_view,
                f"{field}.row_count",
                "page row_count must equal end_row - start_row",
                declared=declared_count,
                actual=end_row - start_row,
            )
        pages.append(PageMeta(page_id=page_id, file=file_name, start_row=start_row, end_row=end_row))
        cursor = end_row
    if cursor != row_count:
        _view_error(
            first_view,
            "base.table.row_count",
            "last page end_row must equal table.row_count",
            page_directory_end=cursor,
            row_count=row_count,
        )

    page_by_id = {page.page_id: page for page in pages}
    table = BaseTable(
        name=table_name,
        row_count=row_count,
        row_key=row_key,
        columns=columns,
        pages=tuple(pages),
        base_dir=catalog_file.parent,
        page_by_id=page_by_id,
    )

    expected_columns = [{"name": column.name, "type": column.type} for column in columns]
    row_keys: set[int] = set()
    for page_index, page_meta in enumerate(pages):
        page_path = Path(page_meta.file)
        if not page_path.is_absolute():
            page_path = catalog_file.parent / page_path
        try:
            page = read_page_file(
                page_path,
                expected_page_id=page_meta.page_id,
                expected_start_row=page_meta.start_row,
                expected_end_row=page_meta.end_row,
                inject_truncate=bool(inject_truncate and page_index == 0),
                semantic_view_id=first_view,
            )
        except ColmvError as exc:
            # Physical file integrity has a dedicated error code.  Metadata
            # mismatch remains E_VIEW, pinned to the first declared view.
            if exc.code == "E_INTEGRITY":
                raise
            raise
        validate_page_columns(page, expected_columns, view_id=first_view, page_id=page_meta.page_id)
        if page.row_count != page_meta.row_count:
            _view_error(
                first_view,
                f"base.pages[{page_index}].row_count",
                "page header row_count differs from the page directory",
                expected=page_meta.row_count,
                actual=page.row_count,
            )
        for column in columns:
            nulls = page.nulls.get(column.name)
            values = page.values.get(column.name)
            if nulls is None or values is None:
                _view_error(
                    first_view,
                    f"base.pages[{page_index}].columns",
                    f"page is missing column {column.name!r}",
                )
            if len(nulls) != page.row_count:
                _view_error(
                    first_view,
                    f"base.pages[{page_index}].null_bitmap.{column.name}",
                    "null bitmap length must equal page row_count",
                    expected=page.row_count,
                    actual=len(nulls),
                )
            if column.type == "float":
                for offset, value in enumerate(values):
                    if value is not None and not math.isfinite(float(value)):
                        _view_error(
                            first_view,
                            f"base.pages[{page_index}].columns.{column.name}[{offset}]",
                            "float column value must be finite",
                        )
        for offset, row in enumerate(page.rows()):
            key_value = row.get(row_key)
            if key_value is None:
                _view_error(
                    first_view,
                    f"base.pages[{page_index}].row_key",
                    "row_key value must not be null",
                    page_row=offset,
                )
            if key_value in row_keys:
                _view_error(
                    first_view,
                    f"base.pages[{page_index}].row_key",
                    f"duplicate row_key value {key_value!r}",
                )
            row_keys.add(key_value)

    catalog_digest = object_digest(
        {
            "format_version": FORMAT_VERSION,
            "table": table_raw,
            "pages": pages_raw,
        }
    )
    views_digest = object_digest(normalized_views)
    return RuntimeConfig(
        table=table,
        views=views,
        scale_power=scale_power,
        catalog_digest=catalog_digest,
        views_digest=views_digest,
        source_catalog_path=str(catalog_file),
        source_views_path=str(views_file),
    )


def load_config_with_views_object(
    catalog_path: str | Path,
    views_object: Mapping[str, Any],
    *,
    source_views_path: str = "<ledger>",
) -> RuntimeConfig:
    """Load a catalog while taking the exact normalized views object from a ledger."""

    import tempfile

    with tempfile.TemporaryDirectory(prefix="colmv-views-") as temp_dir:
        views_path = Path(temp_dir) / "views.json"
        views_path.write_text(canonical_json(views_object), encoding="utf-8")
        config = load_config(catalog_path, views_path)
    return RuntimeConfig(
        table=config.table,
        views=config.views,
        scale_power=config.scale_power,
        catalog_digest=config.catalog_digest,
        views_digest=config.views_digest,
        source_catalog_path=str(Path(catalog_path)),
        source_views_path=source_views_path,
    )


def load_changes(path: str | Path) -> list[Any]:
    """Import lazily to keep schema imports light."""

    from .changes import load_changes as _load_changes

    return _load_changes(path)
