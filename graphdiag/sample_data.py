"""Deterministic sample graph / query generation (NumPy seeded RNG).

Two artifacts are produced under ``data/``:

* ``sample_graph.json`` - a declaration rich enough to exercise quantization
  remainders, pruning, tie-breaking and cap indistinguishability.
* ``tiny_graph.json`` - a hand-verifiable graph used by the demo and tests.

All weights are integer micro units; RNG draws come from a fixed-seed NumPy
``default_rng`` so regenerating yields byte-identical files.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .core.models import canonical_json

SEED = 20261122


def _write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False))
        handle.write("\n")


def build_tiny() -> dict:
    # Hand-tuned so scale exponent 0 (ulp = 1_000_000 micro) truncates every
    # fractional weight and visibly propagates error along multi-edge paths.
    nodes = [
        {"node_id": 0, "label": 100},
        {"node_id": 1, "label": 101},
        {"node_id": 2, "label": 102},
        {"node_id": 3, "label": 103},
        {"node_id": 4, "label": 104},
        {"node_id": 5, "label": 105},
    ]
    edges = [
        {"source": 0, "target": 1, "weight_micro": 1_500_000},
        {"source": 0, "target": 2, "weight_micro": 2_400_000},
        {"source": 1, "target": 2, "weight_micro": 700_000},
        {"source": 1, "target": 3, "weight_micro": 3_300_000},
        {"source": 2, "target": 3, "weight_micro": 1_900_000},
        {"source": 2, "target": 4, "weight_micro": 2_600_000},
        {"source": 3, "target": 4, "weight_micro": 600_000},
        {"source": 3, "target": 5, "weight_micro": 4_200_000},
        {"source": 4, "target": 5, "weight_micro": 1_300_000},
        # A deliberately heavy edge that must be pruned by the cap.
        {"source": 0, "target": 5, "weight_micro": 9_800_000},
    ]
    queries = [
        {
            "query_id": "q-small-cap",
            "source": 0,
            "targets": [3, 4, 5],
            "cap_micro": 4_500_000,
        },
        {
            "query_id": "q-wide-cap",
            "source": 0,
            "targets": [2, 4, 5],
            "cap_micro": 9_000_000,
        },
        {
            # Cap placed in the error band at scale 0 to force an
            # indeterminate error-cross classification.
            "query_id": "q-indistinct",
            "source": 0,
            "targets": [4],
            "cap_micro": 4_899_999,
        },
    ]
    return {
        "config": {"scale_exponent": 0},
        "nodes": nodes,
        "edges": edges,
        "queries": queries,
    }


def build_large() -> dict:
    rng = np.random.default_rng(SEED)
    node_count = 18
    nodes = [
        {"node_id": idx, "label": int(1000 + idx * 7)} for idx in range(node_count)
    ]
    edges: list[dict] = []
    for source in range(node_count):
        # Sparse forward-ish DAG plus a few backward links, fixed out-degree 3.
        candidates = [
            int(v) % node_count
            for v in rng.choice(node_count, size=4, replace=False)
        ]
        for target in candidates:
            if target == source:
                continue
            # Weights span integer micro values, many not divisible by 10^6 so
            # truncation remainders are exercised at every scale.
            weight = int(rng.integers(250_000, 6_500_000))
            edges.append(
                {"source": source, "target": target, "weight_micro": weight}
            )
    # Guarantee reachability to the last nodes.
    for source in range(node_count - 1):
        edges.append(
            {
                "source": source,
                "target": source + 1,
                "weight_micro": int(rng.integers(300_000, 2_000_000)),
            }
        )
    queries = [
        {
            "query_id": f"q-{idx:02d}",
            "source": 0,
            "targets": [
                int(node_count - 1),
                int(node_count // 2),
                int(rng.integers(1, node_count - 1)),
            ],
            "cap_micro": int(rng.choice([3_000_000, 6_000_000, 12_000_000])),
        }
        for idx in range(6)
    ]
    return {
        "config": {"scale_exponent": 3},
        "nodes": nodes,
        "edges": edges,
        "queries": queries,
    }


def generate(data_dir: str = "data") -> dict[str, str]:
    tiny_path = os.path.join(data_dir, "tiny_graph.json")
    large_path = os.path.join(data_dir, "sample_graph.json")
    _write(tiny_path, build_tiny())
    _write(large_path, build_large())
    return {"tiny": tiny_path, "large": large_path}


if __name__ == "__main__":
    paths = generate()
    for name, path in paths.items():
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        print(f"{name}: {path} digest={canonical_json(payload)[:0] or ''}")
        print(
            f"  nodes={len(payload['nodes'])} edges={len(payload['edges'])} "
            f"queries={len(payload['queries'])} scale=1e{payload['config']['scale_exponent']}"
        )
