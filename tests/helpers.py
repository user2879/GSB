from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pageagg.canonical import sha256_bytes, write_text_atomic
from pageagg.catalog import canonical_catalog_text
from pageagg.models import ColumnDef
from pageagg.page_format import encode_page


def write_custom_dataset(
    root: Path,
    *,
    groups_by_page: dict[int, list[int]],
    values_by_page: dict[int, list[int]] | None = None,
    scan_sequence: list[int] | None = None,
    batch_size: int = 2,
) -> Path:
    rows = {len(groups) for groups in groups_by_page.values()}
    if len(rows) != 1:
        raise ValueError("all custom pages must have the same row count")
    rows_per_page = rows.pop()
    columns = [ColumnDef("group_id", "int"), ColumnDef("value_int", "int")]
    root.mkdir(parents=True, exist_ok=True)
    pages_dir = root / "pages"
    pages_dir.mkdir(exist_ok=True)
    entries: list[dict[str, Any]] = []
    for page_id in sorted(groups_by_page):
        groups = np.asarray(groups_by_page[page_id], dtype="<i8")
        values = np.asarray(
            (values_by_page or {}).get(page_id, list(range(rows_per_page))),
            dtype="<i8",
        )
        nulls = {
            "group_id": np.zeros(rows_per_page, dtype=np.uint8),
            "value_int": np.zeros(rows_per_page, dtype=np.uint8),
        }
        payload = encode_page(
            page_id,
            rows_per_page,
            columns,
            {"group_id": groups, "value_int": values},
            nulls,
        )
        relative = f"pages/page_{page_id:04d}.cpg"
        (root / relative).write_bytes(payload)
        entries.append(
            {
                "page_id": page_id,
                "file": relative,
                "rows": rows_per_page,
                "content_sha256": sha256_bytes(payload),
            }
        )
    catalog = {
        "format_version": 1,
        "dataset_id": "custom-evolution-sample-v1",
        "rows_per_page": rows_per_page,
        "group_key": "group_id",
        "batch_size": batch_size,
        "overflow_ratio_threshold_ppm": 400000,
        "rows_reference": max(1, rows_per_page),
        "boundary_margin_ppm": 50000,
        "columns": [column.payload() for column in columns],
        "scan_sequence": scan_sequence or sorted(groups_by_page),
        "pages": entries,
    }
    catalog_path = root / "catalog.json"
    write_text_atomic(catalog_path, canonical_catalog_text(catalog))
    return catalog_path


def rewrite_catalog_json(catalog_path: Path, value: dict[str, Any]) -> None:
    write_text_atomic(catalog_path, canonical_catalog_text(value))


def catalog_object(catalog_path: Path) -> dict[str, Any]:
    return json.loads(catalog_path.read_text(encoding="utf-8"))

