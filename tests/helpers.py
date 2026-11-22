from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from qplan.spec import parse_spec


def small_spec_dict() -> dict[str, Any]:
    return {
        "spec_version": 1,
        "name": "unit-small",
        "seed": 424242,
        "stats": {
            "version": "stats-test-v1",
            "row_threshold": 100,
            "sample_size": 32,
            "confidence": 0.95,
            "histogram_buckets": 8,
            "value_count_limit": 256,
            "page_rows": 20,
            "random_seed": 424242,
        },
        "cost_weights": {"rows": 1, "comparisons": 1, "lookups": 4, "io_pages": 2},
        "search": {"exhaustive_max_candidates": 2, "oracle_max_candidates": 8},
        "execution": {"warmup_runs": 0, "repetitions": 1},
        "bias": {"threshold": 1.75},
        "tables": [
            {
                "name": "users",
                "row_count": 40,
                "columns": [
                    {
                        "name": "id",
                        "type": "integer",
                        "selectivity_hint": 0.025,
                        "generator": {"kind": "unique_int", "start": 1},
                    },
                    {
                        "name": "region",
                        "type": "integer",
                        "selectivity_hint": 0.25,
                        "generator": {
                            "kind": "categorical",
                            "values": [1, 2, 3, 4],
                            "weights": [0.4, 0.3, 0.2, 0.1],
                        },
                    },
                    {
                        "name": "score",
                        "type": "real",
                        "selectivity_hint": 0.2,
                        "generator": {"kind": "uniform_float", "min": 0, "max": 100, "decimals": 3},
                    },
                ],
                "candidate_indexes": [
                    {"name": "idx_users_region", "columns": ["region"]},
                    {"name": "idx_users_region_id", "columns": ["region", "id"]},
                ],
            },
            {
                "name": "events",
                "row_count": 120,
                "columns": [
                    {
                        "name": "id",
                        "type": "integer",
                        "selectivity_hint": 0.0083333333,
                        "generator": {"kind": "unique_int", "start": 1},
                    },
                    {
                        "name": "user_id",
                        "type": "integer",
                        "selectivity_hint": 0.025,
                        "generator": {"kind": "foreign_key", "table": "users", "column": "id"},
                    },
                    {
                        "name": "status",
                        "type": "integer",
                        "selectivity_hint": 0.25,
                        "generator": {
                            "kind": "categorical",
                            "values": [0, 1, 2, 3],
                            "weights": [0.4, 0.3, 0.2, 0.1],
                        },
                    },
                    {
                        "name": "amount",
                        "type": "real",
                        "selectivity_hint": 0.1,
                        "generator": {"kind": "uniform_float", "min": 0, "max": 100, "decimals": 2},
                    },
                ],
                "candidate_indexes": [
                    {"name": "idx_events_status", "columns": ["status"]},
                    {"name": "idx_events_status_user", "columns": ["status", "user_id"]},
                    {
                        "name": "idx_events_status_user_amount_id",
                        "columns": ["status", "user_id", "amount", "id"],
                    },
                    {"name": "idx_events_user", "columns": ["user_id"]},
                ],
            },
        ],
        "queries": [
            {
                "id": "Q_events",
                "tables": ["events", "users"],
                "filters": [
                    {"table": "events", "column": "status", "op": "eq", "value": 1},
                    {"table": "users", "column": "region", "op": "eq", "value": 2},
                ],
                "joins": [
                    {
                        "left_table": "events",
                        "left_column": "user_id",
                        "right_table": "users",
                        "right_column": "id",
                    }
                ],
                "select": [
                    {"table": "events", "column": "id"},
                    {"table": "events", "column": "amount"},
                    {"table": "users", "column": "id"},
                ],
            },
            {
                "id": "Q_users",
                "tables": ["users"],
                "filters": [
                    {"table": "users", "column": "region", "op": "eq", "value": 2},
                    {"table": "users", "column": "score", "op": "between", "value": [20, 30]},
                ],
                "joins": [],
                "select": [
                    {"table": "users", "column": "id"},
                    {"table": "users", "column": "score"},
                    {"table": "users", "column": "region"},
                ],
            },
        ],
    }


def write_spec(path: Path, data: dict[str, Any] | None = None) -> Path:
    path.write_text(json.dumps(data or small_spec_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def make_spec(data: dict[str, Any] | None = None):
    return parse_spec(copy.deepcopy(data or small_spec_dict()))


def reorder_candidate_indexes(data: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(data)
    for table in result["tables"]:
        table["candidate_indexes"] = list(reversed(table["candidate_indexes"]))
    return result

