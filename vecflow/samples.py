"""Deterministic local sample generation used by CLI and tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .service import apply_injection


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def build_data() -> dict[str, Any]:
    order_rows = 24
    order_qty = [1 + (index * 5) % 9 for index in range(order_rows)]
    order_price = [round(10.5 + index * 1.25, 2) for index in range(order_rows)]
    order_discount = [float((index * 3) % 5) for index in range(order_rows)]
    order_discount[3] = -1.0
    order_active = [index % 3 != 1 for index in range(order_rows)]
    order_qty_null = [0] * order_rows
    order_qty_null[5] = 1
    order_qty_null[17] = 1
    order_price_null = [0] * order_rows
    order_price_null[11] = 1
    order_discount_null = [0] * order_rows
    order_discount_null[9] = 1
    order_active_null = [0] * order_rows
    order_active_null[21] = 1

    inventory_rows = 12
    stock = [20 + index * 3 for index in range(inventory_rows)]
    reserve = [index % 5 for index in range(inventory_rows)]
    safety = [round(0.5 + index * 0.25, 2) for index in range(inventory_rows)]
    safety[4] = -1.0
    enabled = [index % 2 == 0 for index in range(inventory_rows)]
    stock_null = [0] * inventory_rows
    stock_null[7] = 1
    safety_null = [0] * inventory_rows
    safety_null[10] = 1

    return {
        "format_version": 1,
        "blocks": [
            {
                "id": "orders",
                "row_count": order_rows,
                "columns": [
                    {"name": "order_id", "type": "integer", "values": list(range(1, order_rows + 1)), "null_bitmap": [0] * order_rows},
                    {"name": "qty", "type": "integer", "values": order_qty, "null_bitmap": order_qty_null},
                    {"name": "unit_price", "type": "float", "values": order_price, "null_bitmap": order_price_null},
                    {"name": "discount", "type": "float", "values": order_discount, "null_bitmap": order_discount_null},
                    {"name": "active", "type": "boolean", "values": order_active, "null_bitmap": order_active_null},
                ],
            },
            {
                "id": "inventory",
                "row_count": inventory_rows,
                "columns": [
                    {"name": "stock", "type": "integer", "values": stock, "null_bitmap": stock_null},
                    {"name": "reserve", "type": "integer", "values": reserve, "null_bitmap": [0] * inventory_rows},
                    {"name": "safety", "type": "float", "values": safety, "null_bitmap": safety_null},
                    {"name": "enabled", "type": "boolean", "values": enabled, "null_bitmap": [0] * inventory_rows},
                ],
            },
        ],
    }


def build_tasks() -> dict[str, Any]:
    return {
        "format_version": 1,
        "parameter_version": 1,
        "global_budget": 4,
        "threads": 4,
        "tasks": [
            {
                "id": "revenue_guard",
                "block": "orders",
                "expression": "if(active, (unit_price * qty) / (discount + 1.0), 0.0)",
                "quota": 2,
                "aggregation": "sum",
                "delay_ms": 45,
                "shards": [
                    {"start": 0, "end": 5, "resources": 2},
                    {"start": 5, "end": 10, "resources": 3},
                    {"start": 10, "end": 15, "resources": 2},
                    {"start": 15, "end": 20, "resources": 3},
                    {"start": 20, "end": 24, "resources": 2},
                ],
            },
            {
                "id": "inventory_margin",
                "block": "inventory",
                "expression": "((stock - reserve) * safety) / (safety + 1.0)",
                "quota": 1,
                "aggregation": "concat",
                "delay_ms": 55,
                "shard_rows": 4,
                "resources_per_shard": [2, 1, 2],
            },
            {
                "id": "risk_rows",
                "block": "orders",
                "expression": "qty >= 8 and active ? order_id : 0",
                "quota": 2,
                "aggregation": "count",
                "delay_ms": 35,
                "shards": [
                    {"start": 0, "end": 8, "resources": 1},
                    {"start": 8, "end": 16, "resources": 2},
                    {"start": 16, "end": 24, "resources": 1},
                ],
            },
        ],
    }


def generate_samples(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    data = build_data()
    tasks = build_tasks()
    _write(root / "data_blocks.json", data)
    _write(root / "tasks.json", tasks)
    for kind in ("range", "nullmap", "budget"):
        injected = apply_injection(data, tasks, kind)
        target = root / f"inject_{kind}"
        _write(target / "data_blocks.json", injected.data_payload)
        _write(target / "tasks.json", injected.tasks_payload)
    readme = """# VecFlow local samples

All files are generated locally by `python -m vecflow generate-samples` or
`python tools/generate_samples.py`.

- `data_blocks.json`: deterministic columnar blocks with null bitmaps.
- `tasks.json`: arithmetic, comparison, logic, ternary, explicit and generated
  shards, plus sum/concat/count aggregation.
- `inject_range/`: shard end beyond `row_count` (must yield `E_RANGE`).
- `inject_nullmap/`: shortened null bitmap (must yield `E_NULLMAP`).
- `inject_budget/`: global budget 0 (must yield `E_BUDGET`).

Runtime files (`ledger.sqlite`, `snapshots/`, `logs/`) are created by the app
and can be cleared before a repeat demonstration.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    return {"samples": str(root.resolve()), "tasks": len(tasks["tasks"]), "blocks": len(data["blocks"])}
