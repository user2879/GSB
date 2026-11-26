from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .ledger import LEDGER_FORMAT_VERSION, Ledger


def _unit_histogram(low: int, count: int) -> dict[str, Any]:
    return {
        "bounds": list(range(low, low + count + 1)),
        "counts": [1] * count,
    }


def _uniform_buckets(low: int, high: int, buckets: int, counts_per_bucket: list[int]) -> dict[str, Any]:
    bounds = [low + (high - low) * index // buckets for index in range(buckets)] + [high]
    return {"bounds": bounds, "counts": counts_per_bucket}


def sample_workload_dict() -> dict[str, Any]:
    customers_columns = [
        {"name": "customer_id", "encoding": "int32", "histogram": _unit_histogram(1, 64)},
        {
            "name": "region_id",
            "encoding": "bitpacked",
            "histogram": _uniform_buckets(1, 5, 4, [16, 16, 16, 16]),
        },
        {
            "name": "score",
            "encoding": "int32",
            "histogram": _uniform_buckets(0, 100, 4, [16, 16, 16, 16]),
        },
    ]
    orders_columns = [
        {"name": "order_id", "encoding": "int64", "histogram": _unit_histogram(1, 256)},
        {
            "name": "customer_id",
            "encoding": "int32",
            "histogram": _uniform_buckets(1, 65, 64, [4] * 64),
        },
        {
            "name": "product_id",
            "encoding": "int32",
            "histogram": _uniform_buckets(1, 33, 32, [8] * 32),
        },
        {
            "name": "amount",
            "encoding": "int32",
            "histogram": _uniform_buckets(0, 100, 4, [64, 64, 64, 64]),
        },
        {
            "name": "status",
            "encoding": "bitpacked",
            "histogram": _uniform_buckets(0, 2, 2, [128, 128]),
        },
    ]
    products_columns = [
        {"name": "product_id", "encoding": "int32", "histogram": _unit_histogram(1, 32)},
        {
            "name": "category_id",
            "encoding": "bitpacked",
            "histogram": _uniform_buckets(1, 9, 8, [4] * 8),
        },
        {
            "name": "price",
            "encoding": "int32",
            "histogram": _uniform_buckets(0, 100, 4, [8, 8, 8, 8]),
        },
    ]
    return {
        "format": "gsb-workload-v1",
        "block_groups": [
            {"id": "customers", "row_count": 64, "seed": 104729, "columns": customers_columns},
            {"id": "orders", "row_count": 256, "seed": 130363, "columns": orders_columns},
            {"id": "products", "row_count": 32, "seed": 155921, "columns": products_columns},
        ],
        "query": {
            "id": "regional_product_revenue",
            "joins": [
                {
                    "left": "orders.customer_id",
                    "right": "customers.customer_id",
                    "operator": "=",
                },
                {
                    "left": "orders.product_id",
                    "right": "products.product_id",
                    "operator": "=",
                },
            ],
            "filters": [
                {"column": "orders.status", "operator": "=", "value": 1},
                {"column": "products.category_id", "operator": "=", "value": 6},
            ],
            "projections": [{"column": "customers.region_id", "alias": "region"}],
            "group_by": ["customers.region_id"],
            "aggregates": [
                {"function": "count_star", "input": None, "alias": "order_count"},
                {
                    "function": "sum",
                    "input": "orders.amount",
                    "alias": "total_amount",
                },
            ],
        },
        "cost_model": {
            "weights": {
                "access_row": 1000,
                "comparison": 1,
                "memory_byte": 3,
                "intermediate_row": 100,
            },
            "scale_factors": {"fixed": 1000, "batch_rows": 64},
            "error_band": 8,
        },
        "executors": [
            {"id": "exec-a", "host": "127.0.0.1", "port": 19101, "concurrency": 2},
            {"id": "exec-b", "host": "127.0.0.1", "port": 19102, "concurrency": 2},
            {"id": "exec-c", "host": "127.0.0.1", "port": 19103, "concurrency": 2},
        ],
        "scheduler": {
            "global_inflight": 4,
            "lease_seconds": 2.5,
            "max_lease_seconds": 4.0,
        },
        "execution": {
            "shard_count": 6,
            "partition": {
                "driver": "orders",
                "co_partition": [
                    {"table": "customers", "column": "customers.customer_id"},
                    {"table": "orders", "column": "orders.customer_id"},
                ],
                "replicate": ["products"],
            },
        },
    }


def write_sample(directory: str | Path) -> dict[str, Path]:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_path = output_dir / "workload.json"
    temporary = workload_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(sample_workload_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(workload_path)

    ledger_path = output_dir / "ledger.sqlite"
    with Ledger(ledger_path) as ledger:
        ledger.ensure_snapshot(
            version=0,
            format_version=LEDGER_FORMAT_VERSION,
            payload={
                "session": "deterministic-sample",
                "plan_version": 0,
                "workload_digest": __import__("gsb.canonical", fromlist=["digest_json"]).digest_json(sample_workload_dict()),
            },
        )
    manifest = {
        "workload": str(workload_path),
        "ledger": str(ledger_path),
        "workload_digest": __import__("gsb.canonical", fromlist=["digest_json"]).digest_json(sample_workload_dict()),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return {
        "workload": workload_path,
        "ledger": ledger_path,
        "manifest": manifest_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic GSB sample")
    parser.add_argument("--out", default="samples")
    args = parser.parse_args(argv)
    paths = write_sample(args.out)
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
