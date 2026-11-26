"""Build the local deterministic diagnostic sample."""

from __future__ import annotations

import copy
from typing import Any

from .codec import actual_index_stats, generate_column_values


def _column(name: str, seed: int, modulus: int, encoding: str, compression: str, null_every: int = 0) -> dict[str, Any]:
    return {
        "name": name,
        "seed": seed,
        "modulus": modulus,
        "offset": 0,
        "null_every": null_every,
        "encoding": encoding,
        "compression": compression,
    }


def build_sample_spec(*, faulted: bool = True) -> dict[str, Any]:
    groups = [
        {
            "id": "BG-A01",
            "row_count": 12,
            "columns": [
                _column("amount", 303, 101, "rle", "zlib-6", 6),
                _column("flag", 404, 2, "rle", "zlib-1", 5),
                _column("id", 101, 1000, "raw", "none"),
                _column("region", 202, 4, "dict", "zlib-1"),
            ],
        },
        {
            "id": "BG-A02",
            "row_count": 10,
            "columns": [
                _column("amount", 313, 101, "dict", "zlib-1"),
                _column("flag", 414, 2, "raw", "zlib-6", 4),
                _column("id", 111, 1000, "dict", "zlib-1"),
                _column("region", 212, 4, "raw", "zlib-6"),
            ],
        },
        {
            "id": "BG-A03",
            "row_count": 14,
            "columns": [
                _column("amount", 323, 101, "raw", "zlib-6", 6),
                _column("flag", 424, 2, "dict", "zlib-1", 4),
                _column("id", 121, 1000, "rle", "zlib-6"),
                _column("region", 222, 4, "dict", "zlib-1", 7),
            ],
        },
        {
            "id": "BG-A04",
            "row_count": 8,
            "columns": [
                _column("amount", 333, 101, "rle", "zlib-6", 1),
                _column("flag", 434, 2, "rle", "zlib-6", 4),
                _column("id", 131, 1000, "raw", "zlib-6"),
                _column("region", 232, 4, "raw", "zlib-1"),
            ],
        },
    ]

    for group in groups:
        index = {}
        for column in group["columns"]:
            values = generate_column_values(column, group["row_count"])
            stats = actual_index_stats(values)
            index[column["name"]] = {
                "min": 0 if stats["min"] is None else stats["min"],
                "max": 0 if stats["max"] is None else stats["max"],
                "null_count": stats["null_count"],
            }
        group["skip_index"] = index

    if faulted:
        # Deliberately distort the amount range so q_amount_high false-skips.
        groups[1]["skip_index"]["amount"].update({"min": 0, "max": 30, "null_count": 0})

    queries = [
        {
            "id": "q_amount_high",
            "predicates": [
                {"column": "amount", "op": ">=", "value": 60},
                {"column": "region", "op": "in", "value": [2, 1]},
            ],
            "projection": ["id", "region", "amount"],
            "aggregations": [
                {"function": "count_star", "alias": "rows"},
                {"function": "sum", "column": "amount", "alias": "sum_amount"},
            ],
        },
        {
            "id": "q_id_values",
            "predicates": [{"column": "id", "op": ">=", "value": 500}],
            "projection": ["id"],
            "aggregations": [
                {"function": "max", "column": "id", "alias": "max_id"},
                {"function": "count_star", "alias": "rows"},
            ],
        },
        {
            "id": "q_null_amount",
            "predicates": [{"column": "amount", "op": "is_null"}],
            "projection": ["id", "amount"],
            "aggregations": [{"function": "count_star", "alias": "rows"}],
        },
        {
            "id": "q_region_3",
            "predicates": [{"column": "region", "op": "=", "value": 3}],
            "projection": ["id", "region", "amount"],
            "aggregations": [
                {"function": "count", "column": "amount", "alias": "non_null_amount"},
                {"function": "sum", "column": "amount", "alias": "sum_amount"},
            ],
        },
    ]

    return {
        "format_version": 1,
        "version": 1,
        "session_id": "demo",
        "verification": {"mode": "full"},
        "scheduler": {
            "global_decode_budget": 3,
            "per_column_lease": 2,
            "socket_timeout_ms": 4000,
            "worker_response_delay_ms": 0,
            "worker_start_timeout_ms": 6000,
        },
        "block_groups": groups,
        "queries": queries,
    }


def build_changed_sample_spec() -> dict[str, Any]:
    spec = copy.deepcopy(build_sample_spec(faulted=True))
    amount = next(column for column in spec["block_groups"][1]["columns"] if column["name"] == "amount")
    amount["seed"] = 314
    amount["compression"] = "zlib-9"
    return spec
