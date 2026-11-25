from __future__ import annotations

import pytest

from cross_coding.config import load_bundle_from_docs


@pytest.fixture
def feature_doc() -> dict:
    return {
        "seed": 424242,
        "row_count": 24,
        "columns": [
            {"name": "a", "type": "categorical", "categories": ["x", "y", "z"], "missing_rate": 0.08},
            {"name": "b", "type": "categorical", "categories": ["p", "q", "r", "s"], "missing_rate": 0.04},
            {"name": "c", "type": "categorical", "categories": ["north", "south", "east"], "missing_rate": 0.0},
            {"name": "d", "type": "categorical", "categories": ["u", "v"], "missing_rate": 0.1},
            {"name": "score", "type": "numeric", "min": 0, "max": 50, "integer": False, "missing_rate": 0.05},
        ],
    }


@pytest.fixture
def policy_doc() -> dict:
    return {
        "candidate_pairs": [["a", "b"], ["b", "c"], ["c", "d"]],
        "max_order": 3,
        "combination_enumeration": "connected_subsets",
        "cardinality_budget": 12,
        "over_cardinality_strategy": "truncate_hash_buckets",
        "encoding_width_bits": 8,
        "fixed_point_bits": 16,
        "quantization_levels": 16,
        "encoding_point_budget": 10_000_000,
        "max_workers": 3,
        "snapshot_commit_interval": 1,
    }


@pytest.fixture
def bundle(feature_doc, policy_doc):
    loaded, report = load_bundle_from_docs(feature_doc, policy_doc)
    assert report.ok, report.errors
    return loaded
