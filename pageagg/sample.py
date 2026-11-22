"""Deterministic local samples and fault injection."""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import canonical_json, sha256_bytes, write_json_atomic, write_text_atomic
from .catalog import canonical_catalog_text, catalog_digest
from .models import ColumnDef
from .page_format import HEADER, MAGIC, PAGE_FORMAT_VERSION, encode_page

SAMPLE_COLUMNS = [
    ColumnDef("group_id", "int"),
    ColumnDef("value_int", "int"),
    ColumnDef("value_float", "float"),
    ColumnDef("flag", "int"),
]


def _sample_arrays(page_id: int, rows: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    row = np.arange(rows, dtype=np.int64)
    group = 1 + ((row + page_id * 2) % 5)
    value_int = ((page_id * 137 + row * 19) % 10007) - 5000
    value_float = (page_id * 1009 + row * 37 - 5000) / 7.0
    value_float = value_float + page_id * 0.0000007
    flag = (page_id * 3 + row) % 2
    values = {
        "group_id": group.astype("<i8"),
        "value_int": value_int.astype("<i8"),
        "value_float": value_float.astype("<f8"),
        "flag": flag.astype("<i8"),
    }
    nulls = {
        "group_id": np.zeros(rows, dtype=np.uint8),
        "value_int": ((row + page_id) % 7 == 0).astype(np.uint8),
        "value_float": ((row * 2 + page_id) % 6 == 0).astype(np.uint8),
        "flag": ((row * 5 + page_id) % 11 == 0).astype(np.uint8),
    }
    return values, nulls


def generate_sample(
    output_dir: str | Path,
    *,
    page_count: int = 8,
    rows_per_page: int = 24,
    scan_sequence: list[int] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if page_count <= 0 or rows_per_page <= 0:
        raise ValueError("page_count and rows_per_page must be positive")
    target = Path(output_dir).resolve()
    if target.exists() and any(target.iterdir()) and not force:
        raise FileExistsError(f"output directory is not empty: {target}; use --force")
    target.mkdir(parents=True, exist_ok=True)
    pages_dir = target / "pages"
    if pages_dir.exists():
        shutil.rmtree(pages_dir)
    pages_dir.mkdir(parents=True)

    sequence = scan_sequence
    if sequence is None:
        # Revisits create deterministic cache hits and replacement pressure.
        if page_count == 8:
            sequence = [1, 2, 3, 1, 4, 2, 3, 5, 6, 1, 7, 8]
        else:
            base = list(range(1, page_count + 1))
            sequence = base + base[: min(4, page_count)]

    page_entries: list[dict[str, Any]] = []
    for page_id in range(1, page_count + 1):
        values, nulls = _sample_arrays(page_id, rows_per_page)
        payload = encode_page(page_id, rows_per_page, SAMPLE_COLUMNS, values, nulls)
        relative = f"pages/page_{page_id:04d}.cpg"
        page_path = target / relative
        page_path.write_bytes(payload)
        page_entries.append(
            {
                "page_id": page_id,
                "file": relative,
                "rows": rows_per_page,
                "content_sha256": sha256_bytes(payload),
            }
        )

    catalog = {
        "format_version": 1,
        "dataset_id": "deterministic-page-sample-v1",
        "rows_per_page": rows_per_page,
        "group_key": "group_id",
        "batch_size": 3,
        "overflow_ratio_threshold_ppm": 400000,
        "rows_reference": 32,
        "boundary_margin_ppm": 50000,
        "columns": [column.payload() for column in SAMPLE_COLUMNS],
        "scan_sequence": list(sequence),
        "pages": page_entries,
    }
    catalog_path = target / "catalog.json"
    write_text_atomic(catalog_path, canonical_catalog_text(catalog))
    digest = catalog_digest(catalog)
    manifest = {
        "dataset_id": catalog["dataset_id"],
        "catalog_file": "catalog.json",
        "catalog_sha256": digest,
        "page_count": page_count,
        "access_count": len(sequence),
        "rows_per_page": rows_per_page,
        "group_key": catalog["group_key"],
        "scan_sequence": list(sequence),
        "pages": [
            {
                "page_id": entry["page_id"],
                "file": entry["file"],
                "content_sha256": entry["content_sha256"],
            }
            for entry in page_entries
        ],
        "deterministic_generator": "pageagg.sample.generate_sample/v1",
    }
    write_json_atomic(target / "dataset_manifest.json", manifest)
    return {
        "output_dir": str(target),
        "catalog_path": str(catalog_path),
        "catalog_digest": digest,
        "page_count": page_count,
        "access_count": len(sequence),
        "rows_per_page": rows_per_page,
        "scan_sequence": list(sequence),
    }


def _copy_dataset(catalog_path: Path, output_dir: Path) -> Path:
    source_root = catalog_path.parent.resolve()
    target_root = output_dir.resolve()
    if target_root.exists():
        shutil.rmtree(target_root)
    shutil.copytree(source_root, target_root)
    return target_root / catalog_path.name


def _nullmap_offset(payload: bytes, target_column: str) -> tuple[int, int, bytes]:
    magic, version, page_id, rows, ncols = HEADER.unpack_from(payload, 0)
    if magic != MAGIC or version != PAGE_FORMAT_VERSION:
        raise ValueError("not a supported page")
    offset = HEADER.size
    for _ in range(ncols):
        name_len = struct.unpack_from("<H", payload, offset)[0]
        offset += 2
        name = payload[offset : offset + name_len].decode("utf-8")
        offset += name_len
        offset += 1  # type
        offset += rows * 8
        bitmap_start = offset
        bitmap = payload[bitmap_start : bitmap_start + rows]
        offset += rows
        offset += 4  # existing crc
        if name == target_column:
            return bitmap_start, rows, bitmap
    raise KeyError(f"column {target_column} not found")


def inject_fault(
    catalog_file: str | Path,
    output_dir: str | Path,
    kind: str,
    *,
    page_id: int = 2,
    column: str = "value_float",
) -> dict[str, Any]:
    source_catalog = Path(catalog_file).resolve()
    target_root = Path(output_dir).resolve()
    target_catalog = _copy_dataset(source_catalog, target_root)
    raw = json.loads(target_catalog.read_text(encoding="utf-8"))
    if kind == "catalog_truncate":
        payload = target_catalog.read_bytes()
        target_catalog.write_bytes(payload[: max(8, len(payload) // 2)])
        return {
            "kind": kind,
            "catalog_path": str(target_catalog),
            "expected_code": "E_PAGE",
            "page_id": None,
            "field": "catalog.json",
        }
    if kind != "nullmap_misalign":
        raise ValueError("fault kind must be catalog_truncate or nullmap_misalign")

    entry = next((item for item in raw["pages"] if item["page_id"] == page_id), None)
    if entry is None:
        raise KeyError(f"page {page_id} not found")
    page_path = target_root / entry["file"]
    payload = bytearray(page_path.read_bytes())
    bitmap_offset, rows, old_bitmap = _nullmap_offset(bytes(payload), column)
    shifted = old_bitmap[1:] + old_bitmap[:1]
    if shifted == old_bitmap:
        changed = bytearray(old_bitmap)
        changed[0] = 1 - changed[0]
        shifted = bytes(changed)
    payload[bitmap_offset : bitmap_offset + rows] = shifted
    page_path.write_bytes(payload)
    entry["content_sha256"] = sha256_bytes(bytes(payload))
    write_text_atomic(target_catalog, canonical_catalog_text(raw))
    return {
        "kind": kind,
        "catalog_path": str(target_catalog),
        "expected_code": "E_NULLMAP",
        "page_id": page_id,
        "field": f"columns[{column}].null_bitmap_crc32",
        "column": column,
        "note": "bitmap shifted by one row; catalog digest updated; null-map CRC remains stale",
    }
