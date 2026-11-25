"""按固定整数种子生成块组声明、查询、权重与会话配置，并物化列式样本。

用法：python -m scripts.generate_sample --workspace workspace
不访问网络，所有数据由 SplitMix64 与 NumPy 确定性生成。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coljoin.canonical import canonical_bytes
from coljoin.loader import load_block_groups, load_query
from coljoin.storage import write_workspace

SEED = 20261126

BLOCK_GROUPS = [
    {
        "bg_id": "orders",
        "rows": 600,
        "seed": SEED,
        "null_bitmap": [0] * 600,
        "columns": [
            {"name": "order_id", "dtype": "integer", "encoding": "plain", "null_count": 0, "ndv": 600},
            {"name": "cust_id", "dtype": "integer", "encoding": "dictionary", "null_count": 0, "ndv": 120},
            {"name": "amount", "dtype": "integer", "encoding": "rle", "null_count": 0, "ndv": 300},
        ],
        "value_ranges": {
            "order_id": {"low": 1, "high": 600},
            "cust_id": {"low": 1, "high": 120},
            "amount": {"low": 10, "high": 900},
        },
    },
    {
        "bg_id": "customers",
        "rows": 120,
        "seed": SEED + 1,
        "null_bitmap": [0] * 120,
        "columns": [
            {"name": "cust_id", "dtype": "integer", "encoding": "dictionary", "null_count": 0, "ndv": 120},
            {"name": "region_id", "dtype": "integer", "encoding": "plain", "null_count": 0, "ndv": 8},
        ],
        "value_ranges": {
            "cust_id": {"low": 1, "high": 120},
            "region_id": {"low": 1, "high": 8},
        },
    },
    {
        "bg_id": "regions",
        "rows": 8,
        "seed": SEED + 2,
        "null_bitmap": [0] * 8,
        "columns": [
            {"name": "region_id", "dtype": "integer", "encoding": "plain", "null_count": 0, "ndv": 8},
            {"name": "zone_score", "dtype": "float", "encoding": "plain", "null_count": 0, "ndv": 8},
        ],
        "value_ranges": {
            "region_id": {"low": 1, "high": 8},
            "zone_score": {"low": 0.0, "high": 100.0},
        },
    },
    {
        "bg_id": "payments",
        "rows": 450,
        "seed": SEED + 3,
        "null_bitmap": [0] * 450,
        "columns": [
            {"name": "payment_id", "dtype": "integer", "encoding": "plain", "null_count": 0, "ndv": 450},
            {"name": "order_id", "dtype": "integer", "encoding": "dictionary", "null_count": 0, "ndv": 300},
            {"name": "pay_amount", "dtype": "integer", "encoding": "rle", "null_count": 0, "ndv": 200},
        ],
        "value_ranges": {
            "payment_id": {"low": 1, "high": 450},
            "order_id": {"low": 1, "high": 300},
            "pay_amount": {"low": 5, "high": 500},
        },
    },
]

QUERY = {
    "query_id": "q-join-demo",
    "joins": [
        {"left_bg": "orders", "left_col": "cust_id", "right_bg": "customers", "right_col": "cust_id"},
        {"left_bg": "customers", "left_col": "region_id", "right_bg": "regions", "right_col": "region_id"},
        {"left_bg": "orders", "left_col": "order_id", "right_bg": "payments", "right_col": "order_id"},
    ],
    "filters": [
        {"bg": "orders", "column": "amount", "op": "ge", "value": 100},
        {"bg": "customers", "column": "region_id", "op": "le", "value": 6},
    ],
    "projections": [
        {"bg": "orders", "column": "order_id"},
        {"bg": "regions", "column": "region_id"},
    ],
    "group_by": [{"bg": "regions", "column": "region_id"}],
    "aggregates": [
        {"func": "count", "bg": "orders", "column": None, "alias": "order_cnt"},
        {"func": "sum", "bg": "payments", "column": "pay_amount", "alias": "pay_sum"},
        {"func": "avg", "bg": "orders", "column": "amount", "alias": "amount_avg"},
    ],
}

WEIGHTS = {
    "scan": 2,
    "compare": 3,
    "build": 5,
    "probe": 4,
    "memory": 1,
    "weight_sum_min": 1,
    "weight_sum_max": 1000000,
}

SESSION = {
    "scale": 1000,
    "block_granularity": 128,
    "stats_version": "stats-v1",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--null-rate", type=float, default=0.0)
    args = parser.parse_args()
    config_dir = os.path.join(args.workspace, "configs")
    data_dir = os.path.join(args.workspace, "data")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    block_path = os.path.join(config_dir, "block_groups.json")
    query_path = os.path.join(config_dir, "query.json")
    weights_path = os.path.join(config_dir, "weights.json")
    session_path = os.path.join(config_dir, "session.json")
    with open(block_path, "wb") as handle:
        handle.write(canonical_bytes({"block_groups": BLOCK_GROUPS}))
    with open(query_path, "wb") as handle:
        handle.write(canonical_bytes(QUERY))
    with open(weights_path, "wb") as handle:
        handle.write(canonical_bytes(WEIGHTS))
    with open(session_path, "wb") as handle:
        handle.write(canonical_bytes(SESSION))

    groups = load_block_groups(block_path)
    query = load_query(query_path, groups)
    manifest = write_workspace(groups, data_dir, null_rate=args.null_rate)
    print(
        json.dumps(
            {
                "block_groups": [group.bg_id for group in groups],
                "query": query.query_id,
                "plans_inputs": "validated",
                "data_dir": data_dir,
                "manifest_entries": len(manifest["block_groups"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
