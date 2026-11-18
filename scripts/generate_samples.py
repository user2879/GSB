from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PARAMS = DATA / "params"


QUERY = {
    "from": ["sales_a", "sales_b"],
    "where": "(channel = 'online') AND ((amount >= 100) AND (amount <= 900000))",
    "group_by": ["region"],
    "aggregates": [
        {"name": "row_count", "op": "count", "column": "*"},
        {"name": "amount_sum", "op": "sum", "column": "amount"},
        {"name": "amount_avg", "op": "avg", "column": "amount"},
    ],
}


def table(name: str, rows: list[tuple[str, str, int]], version: int = 1) -> dict[str, Any]:
    amount_bitmap: list[int] = []
    amount_values: list[int | None] = []
    for index, (_, _, amount) in enumerate(rows):
        is_null = index in {7, 23, 41} or (name == "sales_b" and index == 13)
        amount_bitmap.append(1 if is_null else 0)
        amount_values.append(None if is_null else amount)
    return {
        "name": name,
        "version": version,
        "row_count": len(rows),
        "columns": [
            {
                "name": "row_id",
                "type": "int",
                "null_bitmap": [0] * len(rows),
                "values": list(range(1, len(rows) + 1)),
            },
            {
                "name": "region",
                "type": "string",
                "null_bitmap": [0] * len(rows),
                "values": [row[0] for row in rows],
            },
            {
                "name": "channel",
                "type": "string",
                "null_bitmap": [0] * len(rows),
                "values": [row[1] for row in rows],
            },
            {
                "name": "amount",
                "type": "fixed",
                "scale": 2,
                "null_bitmap": amount_bitmap,
                "values": amount_values,
            },
            {
                "name": "note",
                "type": "string",
                "null_bitmap": [0] * len(rows),
                "values": [f"row-{index + 1:03d}" for index in range(len(rows))],
            },
        ],
    }


def small_rows(count: int, seed: int) -> list[tuple[str, str, int]]:
    regions = ["north", "south", "east", "west"]
    rows: list[tuple[str, str, int]] = []
    for index in range(count):
        region = regions[(index * 3 + seed) % len(regions)]
        channel = "online" if (index + seed) % 3 != 0 else "store"
        amount = 100 + ((index * 7919 + seed * 137) % 700000)
        rows.append((region, channel, amount))
    return rows


def skewed_rows() -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    regions = ["north", "south", "east", "west"]
    rows_a: list[tuple[str, str, int]] = []
    for index in range(96):
        if index < 72:
            region = "east"
            channel = "online"
        else:
            region = regions[(index - 72) % len(regions)]
            channel = "online" if index % 2 == 0 else "store"
        amount = 100 + ((index * 3571 + 900) % 800000)
        rows_a.append((region, channel, amount))
    rows_b: list[tuple[str, str, int]] = []
    for index in range(24):
        region = ["north", "west", "south"][index % 3]
        channel = "online" if index % 3 != 0 else "store"
        amount = 250 + ((index * 1237 + 17) % 600000)
        rows_b.append((region, channel, amount))
    return rows_a, rows_b


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def table_digest(raw_table: dict[str, Any]) -> str:
    payload = {
        "name": raw_table["name"],
        "version": raw_table["version"],
        "row_count": raw_table["row_count"],
        "columns": [
            {
                "name": column["name"],
                "type": column["type"],
                "scale": column.get("scale", 0),
                "null_bitmap": column["null_bitmap"],
                "values": column["values"],
            }
            for column in raw_table["columns"]
        ],
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def write_dataset(path: Path, tables: list[dict[str, Any]]) -> None:
    document = {
        "format_version": 1,
        "tables": tables,
        "query": QUERY,
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_params() -> None:
    PARAMS.mkdir(parents=True, exist_ok=True)
    params = {
        "small.json": {
            "parallelism": 2,
            "memory_budget_bytes": 2048,
            "budget_version": 1,
            "split_threshold": 24,
            "split_parts": 2,
            "row_group_size": 16,
            "ledger_batch_size": 2,
            "split_strategy": "row_range",
            "skew_safety_limit": 100000,
        },
        "skew.json": {
            "parallelism": 3,
            "memory_budget_bytes": 4096,
            "budget_version": 2,
            "split_threshold": 8,
            "split_parts": 4,
            "row_group_size": 32,
            "ledger_batch_size": 3,
            "split_strategy": "row_range",
            "skew_safety_limit": 100000,
        },
        "tight_budget.json": {
            "parallelism": 1,
            "memory_budget_bytes": 900,
            "budget_version": 3,
            "split_threshold": 8,
            "split_parts": 4,
            "row_group_size": 32,
            "ledger_batch_size": 2,
            "split_strategy": "row_range",
            "skew_safety_limit": 100000,
        },
        "inject_skew.json": {
            "parallelism": 2,
            "memory_budget_bytes": 2048,
            "budget_version": 11,
            "split_threshold": 8,
            "split_parts": 4,
            "row_group_size": 32,
            "ledger_batch_size": 2,
            "split_strategy": "row_range",
            "skew_safety_limit": 512,
            "inject_skew_amplify": 8,
        },
        "inject_budget.json": {
            "parallelism": 2,
            "memory_budget_bytes": 1024,
            "budget_version": 12,
            "split_threshold": 8,
            "split_parts": 4,
            "row_group_size": 32,
            "ledger_batch_size": 2,
            "split_strategy": "row_range",
            "skew_safety_limit": 100000,
            "inject_budget_shortfall": True,
        },
        "inject_digest.json": {
            "parallelism": 2,
            "memory_budget_bytes": 2048,
            "budget_version": 13,
            "split_threshold": 8,
            "split_parts": 4,
            "row_group_size": 32,
            "ledger_batch_size": 2,
            "split_strategy": "row_range",
            "skew_safety_limit": 100000,
            "inject_digest_task": "first",
        },
    }
    for name, payload in params.items():
        (PARAMS / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    small_tables = [
        table("sales_a", small_rows(48, 1)),
        table("sales_b", small_rows(36, 2)),
    ]
    skew_a, skew_b = skewed_rows()
    skew_tables = [
        table("sales_a", skew_a),
        table("sales_b", skew_b),
    ]
    write_dataset(DATA / "small.json", small_tables)
    write_dataset(DATA / "skew_amplified.json", skew_tables)
    write_params()
    print("generated", DATA / "small.json")
    print("generated", DATA / "skew_amplified.json")
    print("generated parameter files in", PARAMS)
    print("table digests", {item["name"]: table_digest(item) for item in small_tables})


if __name__ == "__main__":
    main()
