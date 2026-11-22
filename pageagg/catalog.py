"""Page catalog loading and strong E_PAGE/E_NULLMAP validation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import digest_value
from .errors import EPage
from .models import ColumnDef, PageCatalog, PageData
from .page_format import decode_page

PAGE_CATALOG_FORMAT_VERSION = 1
ALLOWED_TYPES = {"int", "float"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def catalog_digest(raw: dict[str, Any]) -> str:
    return digest_value(raw)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_catalog_json(path: Path) -> dict[str, Any]:
    try:
        raw_bytes = path.read_bytes()
    except FileNotFoundError as exc:
        raise EPage(
            "page catalog does not exist",
            field="catalog.path",
            details={"path": str(path)},
        ) from exc
    except OSError as exc:
        raise EPage(
            "page catalog cannot be read",
            field="catalog.path",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EPage(
            "page catalog is not valid UTF-8; it may be truncated",
            field="catalog.encoding",
            details={"error": str(exc), "byte_length": len(raw_bytes)},
        ) from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EPage(
            "page catalog JSON is invalid or truncated",
            field="catalog.json",
            details={
                "error": exc.msg,
                "line": exc.lineno,
                "column": exc.colno,
                "byte_length": len(raw_bytes),
            },
        ) from exc
    if not isinstance(value, dict):
        raise EPage("page catalog root must be an object", field="catalog")
    return value


def _parse_columns(raw: dict[str, Any]) -> tuple[ColumnDef, ...]:
    columns_raw = raw.get("columns")
    if not isinstance(columns_raw, list) or not columns_raw:
        raise EPage("catalog.columns must be a non-empty list", field="columns")
    columns: list[ColumnDef] = []
    names: set[str] = set()
    for index, item in enumerate(columns_raw):
        if not isinstance(item, dict):
            raise EPage("column definition must be an object", field=f"columns[{index}]")
        name = item.get("name")
        column_type = item.get("type")
        if not isinstance(name, str) or not name:
            raise EPage("column name must be a non-empty string", field=f"columns[{index}].name")
        if column_type not in ALLOWED_TYPES:
            raise EPage(
                "column type must be int or float",
                field=f"columns[{name}].type",
                details={"type": column_type},
            )
        if name in names:
            raise EPage("column name is duplicated", field=f"columns[{name}].name")
        names.add(name)
        columns.append(ColumnDef(name=name, column_type=column_type))
    return tuple(columns)


def _safe_page_path(catalog_path: Path, relative: Any, page_id: int | None) -> Path:
    if not isinstance(relative, str) or not relative:
        raise EPage("page file name must be a non-empty string", page_id=page_id, field="file")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise EPage("page file name must be relative to the catalog", page_id=page_id, field="file")
    base = catalog_path.parent.resolve()
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise EPage(
            "page file escapes the catalog directory",
            page_id=page_id,
            field="file",
            details={"file": relative},
        ) from exc
    return resolved


def load_catalog(path: str | Path, *, group_key_override: str | None = None) -> PageCatalog:
    catalog_path = Path(path).resolve()
    raw = _read_catalog_json(catalog_path)

    format_version = raw.get("format_version")
    if format_version != PAGE_CATALOG_FORMAT_VERSION:
        raise EPage(
            "unsupported page catalog format version",
            field="format_version",
            details={"actual": format_version, "expected": PAGE_CATALOG_FORMAT_VERSION},
        )
    dataset_id = raw.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise EPage("dataset_id must be a non-empty string", field="dataset_id")

    rows_per_page = raw.get("rows_per_page")
    if not _is_int(rows_per_page) or rows_per_page <= 0:
        raise EPage("rows_per_page must be a positive integer", field="rows_per_page")
    batch_size = raw.get("batch_size")
    if not _is_int(batch_size) or batch_size <= 0:
        raise EPage("batch_size must be a positive integer", field="batch_size")
    threshold = raw.get("overflow_ratio_threshold_ppm")
    if not _is_int(threshold) or not 0 <= threshold <= 1_000_000:
        raise EPage(
            "overflow_ratio_threshold_ppm must be an integer in [0, 1000000]",
            field="overflow_ratio_threshold_ppm",
        )
    rows_reference = raw.get("rows_reference")
    if not _is_int(rows_reference) or rows_reference <= 0:
        raise EPage("rows_reference must be a positive integer", field="rows_reference")
    boundary_margin = raw.get("boundary_margin_ppm")
    if not _is_int(boundary_margin) or boundary_margin < 0:
        raise EPage("boundary_margin_ppm must be a non-negative integer", field="boundary_margin_ppm")

    columns = _parse_columns(raw)
    column_names = {column.name for column in columns}
    group_key = group_key_override if group_key_override is not None else raw.get("group_key")
    if not isinstance(group_key, str) or not group_key:
        raise EPage("group_key must be a non-empty string", field="group_key")
    if group_key not in column_names:
        raise EPage("group_key does not name a catalog column", field="group_key", details={"group_key": group_key})
    group_column = next(column for column in columns if column.name == group_key)
    if group_column.column_type != "int":
        raise EPage(
            "group_key column must have integer type",
            field=f"columns[{group_key}].type",
            details={"type": group_column.column_type},
        )

    pages_raw = raw.get("pages")
    if not isinstance(pages_raw, list) or not pages_raw:
        raise EPage("catalog.pages must be a non-empty list", field="pages")
    page_entries: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for index, item in enumerate(pages_raw):
        if not isinstance(item, dict):
            raise EPage("page entry must be an object", field=f"pages[{index}]")
        page_id = item.get("page_id")
        if not _is_int(page_id) or not (-(1 << 63) <= page_id <= (1 << 63) - 1):
            raise EPage(
                "page_id must be a signed 64-bit integer",
                page_id=page_id if _is_int(page_id) else None,
                field=f"pages[{index}].page_id",
            )
        if page_id in seen_ids:
            raise EPage("page_id is duplicated in the catalog", page_id=page_id, field="page_id")
        seen_ids.add(page_id)
        page_entries.append(item)

    scan_sequence_raw = raw.get("scan_sequence")
    if not isinstance(scan_sequence_raw, list) or not scan_sequence_raw:
        raise EPage("scan_sequence must be a non-empty list", field="scan_sequence")
    scan_sequence: list[int] = []
    for index, page_id in enumerate(scan_sequence_raw):
        if not _is_int(page_id):
            raise EPage(
                "scan_sequence entries must be integers",
                field=f"scan_sequence[{index}]",
            )
        if page_id not in seen_ids:
            raise EPage(
                "scan_sequence references an undeclared page",
                page_id=page_id,
                field=f"scan_sequence[{index}]",
            )
        scan_sequence.append(page_id)

    pages: dict[int, PageData] = {}
    for item in sorted(page_entries, key=lambda entry: entry["page_id"]):
        page_id = item["page_id"]
        entry_rows = item.get("rows")
        if entry_rows != rows_per_page:
            raise EPage(
                "catalog page row count does not match rows_per_page",
                page_id=page_id,
                field="rows",
                details={"page_rows": entry_rows, "rows_per_page": rows_per_page},
            )
        expected_digest = item.get("content_sha256")
        if not isinstance(expected_digest, str) or not SHA256_RE.match(expected_digest):
            raise EPage(
                "content_sha256 must be a lowercase SHA-256 hex digest",
                page_id=page_id,
                field="content_sha256",
                details={"value": expected_digest},
            )
        page_path = _safe_page_path(catalog_path, item.get("file"), page_id)
        page, actual_digest = decode_page(
            page_path,
            expected_page_id=page_id,
            expected_rows=rows_per_page,
            expected_columns=columns,
        )
        if actual_digest != expected_digest:
            raise EPage(
                "page content digest does not match the catalog",
                page_id=page_id,
                field="content_sha256",
                details={"catalog": expected_digest, "computed": actual_digest},
            )
        group_values = page.columns[group_key]
        group_nulls = page.null_maps[group_key]
        if bool(np.any(group_nulls != 0)):
            bad_index = int(np.flatnonzero(group_nulls != 0)[0])
            raise EPage(
                "group key contains a null value",
                page_id=page_id,
                field=f"columns[{group_key}].null_bitmap",
                details={"row_index": bad_index},
            )
        pages[page_id] = page

    digest = catalog_digest(raw)
    return PageCatalog(
        path=catalog_path,
        raw=raw,
        digest=digest,
        dataset_id=dataset_id,
        rows_per_page=rows_per_page,
        group_key=group_key,
        batch_size=batch_size,
        overflow_ratio_threshold_ppm=threshold,
        rows_reference=rows_reference,
        boundary_margin_ppm=boundary_margin,
        columns=columns,
        pages=pages,
        scan_sequence=tuple(scan_sequence),
    )


def update_page_digest(catalog_path: str | Path, page_id: int, new_digest: str) -> None:
    """Test/fault helper: update one page digest without changing other metadata."""
    target = Path(catalog_path)
    raw = _read_catalog_json(target)
    found = False
    for item in raw.get("pages", []):
        if item.get("page_id") == page_id:
            item["content_sha256"] = new_digest
            found = True
            break
    if not found:
        raise KeyError(f"page {page_id} not found in catalog")
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(canonical_catalog_text(raw), encoding="utf-8", newline="\n")
    os.replace(temporary, target)


def canonical_catalog_text(raw: dict[str, Any]) -> str:
    return json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
