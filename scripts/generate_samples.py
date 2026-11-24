"""Regenerate the deterministic sample catalog, columnar pages, views, and stream."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
BASE = SAMPLES / "base"
PAGES = BASE / "pages"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


COLUMNS = [
    {"name": "row_id", "type": "int"},
    {"name": "region_id", "type": "int"},
    {"name": "product_id", "type": "int"},
    {"name": "amount", "type": "float"},
    {"name": "qty", "type": "int"},
    {"name": "status", "type": "int"},
]


ROWS: list[dict[str, Any]] = [
    {"row_id": 1, "region_id": 10, "product_id": 100, "amount": 12.3456, "qty": 2, "status": 1},
    {"row_id": 2, "region_id": 10, "product_id": 101, "amount": 7.5, "qty": 1, "status": 1},
    {"row_id": 3, "region_id": 11, "product_id": 100, "amount": 3.3333, "qty": 4, "status": 1},
    {"row_id": 4, "region_id": None, "product_id": 102, "amount": 8.0, "qty": 2, "status": 1},
    {"row_id": 5, "region_id": 12, "product_id": 101, "amount": None, "qty": 3, "status": 1},
    {"row_id": 6, "region_id": 11, "product_id": 102, "amount": 19.99, "qty": 1, "status": 0},
    {"row_id": 7, "region_id": 12, "product_id": 100, "amount": 1.2345, "qty": 5, "status": 1},
    {"row_id": 8, "region_id": 10, "product_id": 101, "amount": 2.0, "qty": 2, "status": None},
    {"row_id": 9, "region_id": 11, "product_id": 100, "amount": 5.5555, "qty": 1, "status": 1},
    {"row_id": 10, "region_id": 12, "product_id": 102, "amount": 4.4444, "qty": 3, "status": 1},
    {"row_id": 11, "region_id": 10, "product_id": 102, "amount": 9.9999, "qty": 2, "status": 0},
    {"row_id": 12, "region_id": 11, "product_id": 101, "amount": 6.25, "qty": 4, "status": 1},
    {"row_id": 13, "region_id": None, "product_id": 100, "amount": 7.7777, "qty": 2, "status": 1},
    {"row_id": 14, "region_id": 12, "product_id": 101, "amount": 3.1415, "qty": 1, "status": 1},
    {"row_id": 15, "region_id": 10, "product_id": 100, "amount": None, "qty": 3, "status": 1},
    {"row_id": 16, "region_id": 11, "product_id": 102, "amount": 11.11, "qty": 2, "status": 1},
    {"row_id": 17, "region_id": 12, "product_id": 100, "amount": 0.5, "qty": 1, "status": 1},
    {"row_id": 18, "region_id": 10, "product_id": 101, "amount": 3.0, "qty": 5, "status": 0},
    {"row_id": 19, "region_id": 11, "product_id": 100, "amount": 8.8888, "qty": 2, "status": 1},
    {"row_id": 20, "region_id": 12, "product_id": 102, "amount": 2.2222, "qty": 4, "status": None},
    {"row_id": 21, "region_id": 10, "product_id": 102, "amount": 6.6666, "qty": 1, "status": 1},
    {"row_id": 22, "region_id": None, "product_id": 101, "amount": 9.0909, "qty": 2, "status": 1},
    {"row_id": 23, "region_id": 11, "product_id": 101, "amount": 4.0, "qty": 3, "status": 1},
    {"row_id": 24, "region_id": 12, "product_id": 100, "amount": 5.4321, "qty": 1, "status": 0},
]


VIEWS = {
    "format_version": 1,
    "scale_power": 3,
    "views": [
        {
            "view_id": "sales_by_region",
            "group_by": ["region_id"],
            "aggregates": ["count", "sum", "avg"],
            "value_column": "amount",
            "filters": [{"column": "status", "op": "eq", "value": 1}],
            "refresh_mode": "auto",
        },
        {
            "view_id": "units_by_product",
            "group_by": ["product_id"],
            "aggregates": ["count", "sum", "avg"],
            "value_column": "qty",
            "filters": [{"column": "qty", "op": "gte", "value": 1}],
            "refresh_mode": "auto",
        },
        {
            "view_id": "sales_region_product_live",
            "group_by": ["region_id", "product_id"],
            "aggregates": ["count", "sum"],
            "value_column": "amount",
            "filters": [
                {"column": "status", "op": "is_not_null"},
                {"column": "status", "op": "ne", "value": 0},
            ],
            "refresh_mode": "auto",
        },
    ],
}


VIEWS_V2 = {
    "format_version": 1,
    "scale_power": 3,
    "views": [
        {
            "view_id": "sales_by_region",
            "group_by": ["region_id", "product_id"],
            "aggregates": ["count", "sum", "avg"],
            "value_column": "amount",
            "filters": [{"column": "status", "op": "eq", "value": 1}],
            "refresh_mode": "auto",
        },
        {
            "view_id": "units_by_product",
            "group_by": ["product_id"],
            "aggregates": ["count", "sum", "avg"],
            "value_column": "qty",
            "filters": [{"column": "qty", "op": "gte", "value": 1}],
            "refresh_mode": "auto",
        },
        {
            "view_id": "sales_region_product_live",
            "group_by": ["region_id", "product_id"],
            "aggregates": ["count", "sum"],
            "value_column": "amount",
            "filters": [{"column": "status", "op": "eq", "value": 1}],
            "refresh_mode": "auto",
        },
    ],
}


def _row_copy(row_id: int, **changes: Any) -> dict[str, Any]:
    source = next(row for row in ROWS if row["row_id"] == row_id)
    result = dict(source)
    result.update(changes)
    return result


CHANGES = [
    {"seq": 1, "op": "update", "row_id": 2, "row": _row_copy(2, amount=8.25)},
    {"seq": 2, "op": "update", "row_id": 3, "row": _row_copy(3, region_id=12, amount=4.0)},
    {"seq": 3, "op": "delete", "row_id": 6},
    {"seq": 4, "op": "insert", "row_id": 1001, "row": {"row_id": 1001, "region_id": 10, "product_id": 100, "amount": 13.579, "qty": 2, "status": 1}},
    {"seq": 5, "op": "update", "row_id": 5, "row": _row_copy(5, amount=5.0, status=1)},
    {"seq": 6, "op": "update", "row_id": 8, "row": _row_copy(8, region_id=11)},
    {"seq": 7, "op": "delete", "row_id": 11},
    {"seq": 8, "op": "update", "row_id": 14, "row": _row_copy(14, region_id=13, amount=3.3333)},
    {"seq": 9, "op": "insert", "row_id": 1002, "row": {"row_id": 1002, "region_id": 11, "product_id": 103, "amount": None, "qty": 3, "status": 1}},
    {"seq": 10, "op": "update", "row_id": 20, "row": _row_copy(20, amount=2.5, status=1)},
    {"seq": 11, "op": "delete", "row_id": 1},
    {"seq": 12, "op": "update", "row_id": 4, "row": _row_copy(4, region_id=10, amount=8.5)},
    {"seq": 13, "op": "insert", "row_id": 1003, "row": {"row_id": 1003, "region_id": None, "product_id": 100, "amount": 4.0, "qty": 1, "status": 1}},
    {"seq": 14, "op": "update", "row_id": 17, "row": _row_copy(17, amount=0.75)},
    {"seq": 15, "op": "delete", "row_id": 13},
    {"seq": 16, "op": "update", "row_id": 21, "row": _row_copy(21, region_id=None)},
    {"seq": 17, "op": "insert", "row_id": 1004, "row": {"row_id": 1004, "region_id": 13, "product_id": 104, "amount": 6.789, "qty": 1, "status": 0}},
    {"seq": 18, "op": "update", "row_id": 9, "row": _row_copy(9, amount=6.0001)},
    {"seq": 19, "op": "delete", "row_id": 10},
    {"seq": 20, "op": "update", "row_id": 23, "row": _row_copy(23, amount=4.4444)},
]


def main() -> None:
    if tuple(int(part) for part in np.__version__.split(".")[:2]) < (1, 24):
        raise SystemExit(f"NumPy >= 1.24 is required, found {np.__version__}")
    BASE.mkdir(parents=True, exist_ok=True)
    PAGES.mkdir(parents=True, exist_ok=True)

    from colmv.storage import write_page_file

    page_size = 8
    pages_meta: list[dict[str, Any]] = []
    for page_index, start in enumerate(range(0, len(ROWS), page_size)):
        end = min(start + page_size, len(ROWS))
        page_id = f"p{page_index:03d}"
        relative_file = f"pages/{page_id}.col"
        page_rows = ROWS[start:end]
        values = {column["name"]: [row[column["name"]] for row in page_rows] for column in COLUMNS}
        digest = write_page_file(
            BASE / relative_file,
            page_id=page_id,
            start_row=start,
            end_row=end,
            columns=COLUMNS,
            values=values,
        )
        pages_meta.append(
            {
                "page_id": page_id,
                "file": relative_file,
                "start_row": start,
                "end_row": end,
                "row_count": end - start,
                "content_sha256": digest,
            }
        )

    catalog = {
        "format_version": 1,
        "table": {
            "name": "sales_events",
            "row_count": len(ROWS),
            "row_key": "row_id",
            "columns": COLUMNS,
        },
        "pages": pages_meta,
    }
    (BASE / "catalog.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (SAMPLES / "views.json").write_text(json.dumps(VIEWS, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (SAMPLES / "views_v2.json").write_text(
        json.dumps(VIEWS_V2, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    changes_path = SAMPLES / "changes.ndjson"
    changes_path.write_text(
        "".join(json.dumps(change, ensure_ascii=False, separators=(",", ":")) + "\n" for change in CHANGES),
        encoding="utf-8",
    )
    print(f"wrote {len(ROWS)} rows in {len(pages_meta)} pages, {len(VIEWS['views'])} views, {len(CHANGES)} changes")
    print(f"numpy={np.__version__} python={sys.version.split()[0]}")


if __name__ == "__main__":
    main()
