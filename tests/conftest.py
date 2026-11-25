from __future__ import annotations

import copy
from typing import Any

import pytest

from vscan.spec import parse_spec


def base_spec_dict() -> dict[str, Any]:
    return {
        "format_version": 1,
        "segments": [
            {
                "id": "s1",
                "rows": 12,
                "seed": 7,
                "columns": [
                    {
                        "name": "g",
                        "encoding": "dict",
                        "value_min": 0,
                        "value_max": 2,
                        "null_per_mille": 80,
                    },
                    {
                        "name": "v",
                        "encoding": "raw",
                        "value_min": 0,
                        "value_max": 20,
                        "null_per_mille": 100,
                    },
                    {
                        "name": "r",
                        "encoding": "rle",
                        "value_min": 0,
                        "value_max": 3,
                        "null_per_mille": 50,
                    },
                    {
                        "name": "unused",
                        "encoding": "dict",
                        "value_min": 0,
                        "value_max": 1,
                        "null_per_mille": 0,
                    },
                ],
            }
        ],
        "queries": [
            {
                "id": "q1",
                "project": ["g", "v", "r"],
                "predicate": {
                    "op": "and",
                    "args": [
                        {"op": "ge", "column": "v", "value": 5},
                        {"op": "ne", "column": "g", "value": 1},
                    ],
                },
                "group_by": ["g"],
                "aggregates": [
                    {"name": "n", "op": "count"},
                    {"name": "sum_v", "op": "sum", "column": "v"},
                    {"name": "min_v", "op": "min", "column": "v"},
                    {"name": "max_v", "op": "max", "column": "v"},
                    {"name": "avg_v", "op": "avg", "column": "v"},
                ],
            }
        ],
        "execution": {
            "batch_rows": 4,
            "scale": 1000,
            "segment_decode_memory_bytes": 100_000,
            "max_concurrency": 3,
            "attribution_tolerance_ppm": 2,
        },
    }


@pytest.fixture
def spec_dict() -> dict[str, Any]:
    return base_spec_dict()


@pytest.fixture
def spec(spec_dict):
    return parse_spec(copy.deepcopy(spec_dict))
