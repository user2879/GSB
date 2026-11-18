from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pagecache.canonical import canonical_bytes
from pagecache.fsutil import remove_tree
from pagecache.manifest import write_manifest
from pagecache.model import CURRENT_FORMAT_VERSION, ColumnSummary, PageEntry
from pagecache.pagefile import page_checksum, read_page_file, summaries_for_page, write_page
from pagecache.model import ColumnKind, PageColumn, PageData
from pagecache.segment import write_segment


DEFAULT_SEED = 20261118
DEFAULT_ROWS = 12
DEFAULT_PAGES = 8
SCAN_ORDER = [
    "p0005",
    "p0001",
    "p0004",
    "p0002",
    "p0001",
    "p0006",
    "p0003",
    "p0005",
    "p0008",
    "p0002",
    "p0007",
    "p0001",
]


def generate_samples(output_dir: str | Path, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    output = Path(output_dir)
    remove_tree(output)
    pages_dir = output / "pages"
    pages_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)

    pages: list[PageData] = []
    for page_index in range(DEFAULT_PAGES):
        pages.append(_build_page(page_index, rng))
    written_pages: list[PageData] = []
    for page in pages:
        write_page(pages_dir / f"{page.page_id}.bin", page)
        written_pages.append(read_page_file(pages_dir / f"{page.page_id}.bin", expected_page_id=page.page_id))
    pages = written_pages

    version_rows: dict[str, Any] = {}
    for version in (1, 2, 3):
        version_dir = output / "versions" / f"v{version}"
        overflow_dir = version_dir / "overflow"
        overflow_dir.mkdir(parents=True)
        entries = _entries_for_version(pages, version)
        manifest_path = version_dir / "manifest.pcm"
        write_manifest(manifest_path, entries, format_version=version)
        for index, page in enumerate(pages):
            compressed = version == 3 and index % 2 == 0
            write_segment(
                overflow_dir / f"{page.page_id}.seg",
                page_id=page.page_id,
                row_count=page.row_count,
                payload=page.raw_bytes,
                compressed=compressed,
                version=version,
            )
        version_rows[f"v{version}"] = {
            "manifest": str(manifest_path.relative_to(output)).replace("\\", "/"),
            "overflow_dir": str(overflow_dir.relative_to(output)).replace("\\", "/"),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "record_versions": sorted({entry.record_version for entry in entries}),
            "segment_versions": [version],
            "segments": sorted(path.name for path in overflow_dir.glob("*.seg")),
        }

    meta = {
        "seed": seed,
        "page_count": len(pages),
        "row_count": DEFAULT_ROWS,
        "columns": [
            {"name": "group", "kind": "int"},
            {"name": "value", "kind": "float"},
            {"name": "qty", "kind": "int"},
            {"name": "score", "kind": "float"},
        ],
        "scan_order": list(SCAN_ORDER),
        "group_column": "group",
        "value_column": "value",
        "page_size": 512,
        "full_budget_bytes": 65536,
        "small_budget_bytes": 1536,
        "versions": version_rows,
        "page_sha256": {
            page.page_id: hashlib.sha256(page.raw_bytes).hexdigest() for page in sorted(pages, key=lambda item: item.page_id)
        },
    }
    (output / "sample_meta.json").write_bytes(canonical_bytes(meta))
    return meta


def _build_page(page_index: int, rng: np.random.Generator) -> PageData:
    page_id = f"p{page_index + 1:04d}"
    group_values = rng.integers(1, 4, size=DEFAULT_ROWS)
    value_values = np.round(rng.normal(10.0, 4.0, size=DEFAULT_ROWS), 3)
    qty_values = rng.integers(-20, 50, size=DEFAULT_ROWS)
    score_values = np.round(rng.uniform(0.0, 1.0, size=DEFAULT_ROWS), 3)
    group_nulls = rng.random(DEFAULT_ROWS) < 0.08
    value_nulls = rng.random(DEFAULT_ROWS) < 0.12
    qty_nulls = rng.random(DEFAULT_ROWS) < 0.10
    score_nulls = rng.random(DEFAULT_ROWS) < 0.15
    # Guarantee at least one interesting null and one valid group per page.
    group_nulls[page_index % DEFAULT_ROWS] = False
    value_nulls[(page_index + 3) % DEFAULT_ROWS] = True

    columns = (
        _column("group", ColumnKind.INT, group_values.tolist(), group_nulls.tolist()),
        _column("value", ColumnKind.FLOAT, value_values.tolist(), value_nulls.tolist()),
        _column("qty", ColumnKind.INT, qty_values.tolist(), qty_nulls.tolist()),
        _column("score", ColumnKind.FLOAT, score_values.tolist(), score_nulls.tolist()),
    )
    return PageData(page_id=page_id, row_count=DEFAULT_ROWS, columns=columns)


def _column(
    name: str,
    kind: ColumnKind,
    values: list[int | float],
    nulls: list[bool],
) -> PageColumn:
    typed_values: tuple[int | float | None, ...] = tuple(None if is_null else value for value, is_null in zip(values, nulls, strict=True))
    bitmap = bytearray((len(typed_values) + 7) // 8)
    for index, value in enumerate(typed_values):
        if value is None:
            bitmap[index // 8] |= 1 << (index % 8)
    return PageColumn(name=name, kind=kind, values=typed_values, null_bitmap=bytes(bitmap))


def _entries_for_version(pages: list[PageData], version: int) -> list[PageEntry]:
    entries: list[PageEntry] = []
    for index, page in enumerate(pages):
        raw = page.raw_bytes
        if version == 1:
            entries.append(PageEntry(page_id=page.page_id, row_count=page.row_count, record_version=1))
            continue
        summaries = summaries_for_page(page)
        checksum = page_checksum(raw)
        if version == 2:
            entries.append(
                PageEntry(
                    page_id=page.page_id,
                    row_count=page.row_count,
                    record_version=2,
                    column_summaries=summaries,
                    payload_checksum=checksum,
                )
            )
            continue
        overflow_ref = f"{page.page_id}.seg" if index % 2 == 0 else None
        entries.append(
            PageEntry(
                page_id=page.page_id,
                row_count=page.row_count,
                record_version=CURRENT_FORMAT_VERSION,
                column_summaries=summaries,
                payload_checksum=checksum,
                compressed=index % 2 == 0,
                overflow_ref=overflow_ref,
            )
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic page-cache samples")
    parser.add_argument("--out", default="samples/data", help="output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    meta = generate_samples(args.out, seed=args.seed)
    print(json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
