"""Generate deterministic local sample data, queries, fault parameters and an
old-version snapshot database.  No network, no randomness: every value is a
pure function of the row index."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .engine import ScanEngine
from .model import BudgetParams
from .snapshot import SnapshotStore

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(ROOT, "sample_data")
DATA_DIR = os.path.join(ROOT, "data")


def _sales_definition() -> Dict[str, Any]:
    row_group_rows = [12, 10, 14]
    total = sum(row_group_rows)
    region_pattern = (1, 1, 2, 2, 2, 3, 3, 4, 4, 1, 2, 3)
    order_id = [1000 + i for i in range(total)]
    region_id = [region_pattern[i % len(region_pattern)] for i in range(total)]
    amount = [50 + 25 * ((i * 7) % 5) for i in range(total)]
    return {
        "name": "sales",
        "format_version": 1,
        "columns": [
            {"name": "order_id", "dtype": "int32", "compressed": False},
            {"name": "region_id", "dtype": "int32", "compressed": True},
            {"name": "amount", "dtype": "int64", "compressed": False},
        ],
        "row_group_rows": row_group_rows,
        "values": {"order_id": order_id, "region_id": region_id, "amount": amount},
        "block_specs": [
            {"row_group": 0, "column": "order_id", "encoding": "raw"},
            {"row_group": 1, "column": "region_id", "encoding": "rle"},
        ],
    }


def _inventory_definition() -> Dict[str, Any]:
    row_group_rows = [10, 10]
    total = sum(row_group_rows)
    sku_id = [500 + i for i in range(total)]
    warehouse_runs = [7] * 8 + [8] * 7 + [9] * 5
    qty = [(i * 13) % 9 + 1 for i in range(total)]
    return {
        "name": "inventory",
        "format_version": 1,
        "columns": [
            {"name": "sku_id", "dtype": "int32", "compressed": False},
            {"name": "warehouse_id", "dtype": "int32", "compressed": True},
            {"name": "qty", "dtype": "int64", "compressed": False},
        ],
        "row_group_rows": row_group_rows,
        "values": {"sku_id": sku_id, "warehouse_id": warehouse_runs, "qty": qty},
        "block_specs": [
            {"row_group": 1, "column": "qty", "encoding": "dictionary"}
        ],
    }


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def prepare() -> Dict[str, str]:
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    paths: Dict[str, str] = {}

    sales_path = os.path.join(SAMPLE_DIR, "segment_sales.json")
    inventory_path = os.path.join(SAMPLE_DIR, "segment_inventory.json")
    _write_json(sales_path, _sales_definition())
    _write_json(inventory_path, _inventory_definition())
    paths["sales_definition"] = sales_path
    paths["inventory_definition"] = inventory_path

    queries = {
        "query_eq.json": {
            "select": ["region_id", "amount"],
            "filter": {"op": "eq", "column": "region_id", "value": 2},
        },
        "query_and.json": {
            "select": ["order_id", "amount", "region_id"],
            "filter": {
                "op": "and",
                "args": [
                    {"op": "ge", "column": "amount", "value": 100},
                    {"op": "eq", "column": "region_id", "value": 2},
                ],
            },
        },
        # Same semantics as query_and.json: swapped AND order, extra nested
        # grouping and compact formatting -> identical canonical query id.
        "query_and_variant.json": {
            "select": ["order_id", "amount", "region_id"],
            "filter": {
                "op": "and",
                "args": [
                    {"op": "eq", "column": "region_id", "value": 2},
                    {"op": "ge", "column": "amount", "value": 100},
                ],
            },
        },
        "query_in.json": {
            "select": ["order_id", "region_id"],
            "filter": {"op": "in", "column": "region_id", "value": [1, 3]},
        },
        "query_invalid.json": {
            "select": ["order_id"],
            "filter": {"op": "contains", "column": "order_id", "value": 1},
        },
    }
    for name, payload in queries.items():
        path = os.path.join(SAMPLE_DIR, name)
        _write_json(path, payload)
        paths[name[:-5]] = path

    fault_params = {
        "payload_truncation": {
            "inject": "payload_truncation",
            "segment": "sales",
            "description": "truncate one byte of the last block payload",
        },
        "encoding_mismatch": {
            "inject": "encoding_mismatch",
            "segment": "sales",
            "description": "flip the block encoding marker but keep payload bytes",
        },
        "zero_budget": {
            "inject": "zero_budget",
            "max_parallelism": 4,
            "memory_budget_bytes": 0,
            "per_segment_concurrency": 2,
        },
    }
    fault_path = os.path.join(SAMPLE_DIR, "fault_params.json")
    _write_json(fault_path, fault_params)
    paths["fault_params"] = fault_path

    _build_old_snapshot()
    paths["old_snapshot"] = os.path.join(SAMPLE_DIR, "snapshots", "old_session.db")

    manifest = {
        "sample_dir": SAMPLE_DIR,
        "data_dir": DATA_DIR,
        "paths": paths,
        "deterministic": True,
    }
    manifest_path = os.path.join(SAMPLE_DIR, "manifest.json")
    _write_json(manifest_path, manifest)
    paths["manifest"] = manifest_path
    return paths


def _build_old_snapshot() -> None:
    """Persist an old-format sample snapshot at a fixed path."""
    from .model import SegmentDefinition

    engine = ScanEngine(DATA_DIR, BudgetParams(max_parallelism=2, memory_budget_bytes=4096,
                                               per_segment_concurrency=1))
    for filename in ("segment_sales.json", "segment_inventory.json"):
        with open(os.path.join(SAMPLE_DIR, filename), "r", encoding="utf-8") as handle:
            engine.register_definition(SegmentDefinition.from_dict(json.load(handle)))
    engine.build_segments()
    with open(os.path.join(SAMPLE_DIR, "query_eq.json"), "r", encoding="utf-8") as handle:
        query = json.load(handle)
    result = engine.scan(query)

    snapshot_dir = os.path.join(SAMPLE_DIR, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    db_path = os.path.join(snapshot_dir, "old_session.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    store = SnapshotStore(db_path)
    summaries = [engine.segments[name].file_summary() for name in engine.segments]
    store.save_session(
        snapshot_id="old-session-v1",
        result=result,
        budget=engine.budget,
        segment_summaries=summaries,
        encoding_version=engine.encoding_selection_version,
        current_state={
            "budget": engine.budget.to_dict(),
            "encoding_selection_version": engine.encoding_selection_version,
        },
    )
    store.close()


if __name__ == "__main__":
    generated = prepare()
    print(json.dumps(generated, indent=2))
