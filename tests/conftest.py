from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def base_input() -> dict:
    return {
        "name": "x",
        "seed": 12345,
        "shape": [2, 3],
        "dtype": "float64",
        "distribution": {"kind": "uniform", "low": -2.0, "high": 2.0},
        "special_values": [
            {"index": [0, 0], "value": 0.0312500004},
            {"index": [0, 1], "value": "nan"},
            {"index": [1, 2], "value": "inf"},
        ],
    }


@pytest.fixture
def relu_perm_chain() -> dict:
    return {
        "input": "x",
        "outputs": ["perm1"],
        "nodes": [
            {"id": "relu1", "op": "relu", "params": {}, "inplace": False},
            {"id": "perm1", "op": "permute", "params": {"perm": [1, 0]}, "inplace": False},
        ],
        "edges": [
            {"from": "x", "to": "relu1"},
            {"from": "relu1", "to": "perm1"},
        ],
    }


@pytest.fixture
def base_config() -> dict:
    return {
        "precision_tier": "fp64",
        "quantization_tier": "q8",
        "thread_pool_size": 2,
        "recompute_budget": 100,
        "snapshot_interval": 1,
        "tolerance": 1e-9,
        "fusion_strategy": "auto",
    }


@pytest.fixture
def sample_chain() -> dict:
    return json.loads((ROOT / "samples" / "chain.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_input() -> dict:
    return json.loads((ROOT / "samples" / "input.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_config() -> dict:
    return json.loads((ROOT / "samples" / "config.json").read_text(encoding="utf-8"))

