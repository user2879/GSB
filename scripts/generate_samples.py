#!/usr/bin/env python3
"""Generate deterministic local sample directories.

All randomness uses a fixed integer seed and JSON is emitted with sorted keys,
UTF-8, LF newlines.  No timestamps or absolute paths are written.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

FEATURE_ORDER = [
    "recall.bm25",
    "recall.ann",
    "dense.0",
    "dense.1",
    "dense.2",
    "dense.3",
    "sparse.brand",
    "sparse.color",
]
WEIGHTS = [0.15, 0.20, 0.40, 0.30, 0.20, 0.10, 0.05, 0.10]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def _dense_cluster(rng: random.Random) -> list[int]:
    base = [rng.randint(-40, 60) for _ in range(4)]
    return base


def make_candidates(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for request_number in range(1, 5):
        request_id = f"r-{request_number:03d}"
        clusters = [_dense_cluster(rng), _dense_cluster(rng)]
        for candidate_number in range(1, 6):
            item_id = f"c-{candidate_number:03d}"
            dense = list(clusters[0] if candidate_number in (1, 2, 5) else clusters[1])
            if candidate_number == 5:
                # Keep one candidate similar but not identical to the first cluster.
                dense[3] += 2
            rows.append(
                {
                    "request_id": request_id,
                    "item_id": item_id,
                    "recall_scores": {
                        "bm25": rng.randint(20, 100),
                        "ann": rng.randint(20, 100),
                    },
                    "dense_features": dense,
                    "sparse_features": {
                        "brand": rng.randint(0, 4),
                        "color": rng.randint(0, 4),
                    },
                }
            )
    return rows


def make_config(global_budget: int = 2000) -> dict[str, Any]:
    return {
        "feature_weights": WEIGHTS,
        "normalization": "minmax",
        "fusion_temperature": 1.0,
        "score_clip_interval": [-0.4, 0.75],
        "fixed_decimal_places": 4,
        "quantization_step": 1,
        "rerank_window_size": 4,
        "diversity_penalty": 0.15,
        "concurrency": 3,
        "global_mac_budget": global_budget,
        "per_request_quota": 500,
        "cache_residency_budget": 30,
        "snapshot_commit_interval_commands": 2,
        "interaction_latency_budget_ms": 10,
        "min_command_batch": 1,
        "latency_units_per_ms": 1000,
        "similarity_threshold": 0.8,
    }


def make_groups() -> dict[str, Any]:
    return {
        "feature_order": FEATURE_ORDER,
        "groups": [
            {
                "name": "recall",
                "features": ["recall.bm25", "recall.ann"],
                "conflict": True,
            },
            {
                "name": "dense",
                "features": ["dense.0", "dense.1", "dense.2", "dense.3"],
                "conflict": False,
            },
            {
                "name": "sparse",
                "features": ["sparse.brand", "sparse.color"],
                "conflict": True,
            },
        ],
        "similarity_features": ["dense.0", "dense.1", "dense.2", "dense.3"],
        "similarity_threshold": 0.8,
    }


def make_commands(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    added = {
        "request_id": "r-002",
        "item_id": "c-099",
        "recall_scores": {"bm25": 88, "ann": 91},
        # Deliberately close to r-002/c-001's dense cluster.
        "dense_features": [-5, 20, 10, 30],
        "sparse_features": {"brand": 1, "color": 3},
    }
    return [
        {
            "command": "set_weights",
            "weights": [0.12, 0.18, 0.42, 0.28, 0.22, 0.08, 0.04, 0.12],
        },
        {"command": "set_normalization", "normalization": "zscore"},
        {
            "command": "set_quantization",
            "fixed_decimal_places": 3,
            "quantization_step": 2,
        },
        {
            "command": "set_rerank",
            "rerank_window_size": 3,
            "diversity_penalty": 0.20,
        },
        {"command": "add_candidate", "candidate": added},
        {
            "command": "set_weights",
            "weights": {
                "recall.bm25": 0.11,
                "recall.ann": 0.17,
                "dense.0": 0.41,
                "dense.1": 0.27,
                "dense.2": 0.21,
                "dense.3": 0.09,
                "sparse.brand": 0.05,
                "sparse.color": 0.13,
            },
        },
        {"command": "remove_candidate", "request_id": "r-001", "item_id": "c-002"},
        {
            "command": "set_quantization",
            "fixed_decimal_places": 4,
            "quantization_step": 1,
        },
    ]


def make_injections() -> dict[str, Any]:
    return {
        "injections": [
            {
                "id": "INJ-001",
                "request_id": "r-002",
                "stage": "load",
                "fault": "non_finite_feature",
            },
            {
                "id": "INJ-002",
                "request_id": "r-003",
                "stage": "fusion",
                "fault": "weight_length_mismatch",
            },
            {
                "id": "INJ-003",
                "request_id": "r-004",
                "stage": "budget",
                "fault": "quota_below_minimum",
            },
        ]
    }


def write_sample(
    target: Path,
    seed: int,
    with_injections: bool,
    *,
    global_budget: int = 2000,
    budget_mode: bool = False,
) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    candidates = make_candidates(seed)
    commands = (
        [
            {
                "command": "set_rerank",
                "rerank_window_size": 4,
                "diversity_penalty": 0.18,
            },
            {
                "command": "set_weights",
                "weights": [0.11, 0.17, 0.41, 0.27, 0.21, 0.09, 0.05, 0.13],
            },
        ]
        if budget_mode
        else make_commands(candidates)
    )
    (target / "candidates.jsonl").write_text(
        _jsonl(candidates), encoding="utf-8", newline="\n"
    )
    (target / "config.json").write_text(
        _json(make_config(global_budget)), encoding="utf-8", newline="\n"
    )
    (target / "feature_groups.json").write_text(
        _json(make_groups()), encoding="utf-8", newline="\n"
    )
    (target / "commands.jsonl").write_text(
        _jsonl(commands), encoding="utf-8", newline="\n"
    )
    manifest = {
        "generator": "scripts/generate_samples.py",
        "seed": seed,
        "request_count": 4,
        "candidate_count": len(candidates),
        "command_count": len(commands),
        "with_injections": with_injections,
        "global_mac_budget": global_budget,
        "budget_mode": budget_mode,
    }
    (target / "sample_manifest.json").write_text(
        _json(manifest), encoding="utf-8", newline="\n"
    )
    if with_injections:
        (target / "injections.json").write_text(
            _json(make_injections()), encoding="utf-8", newline="\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--default-dir", default="samples/default")
    parser.add_argument("--injection-dir", default="samples/injection")
    parser.add_argument("--budget-dir", default="samples/budget")
    args = parser.parse_args()
    write_sample(Path(args.default_dir), args.seed, with_injections=False)
    write_sample(Path(args.injection_dir), args.seed, with_injections=True)
    write_sample(
        Path(args.budget_dir),
        args.seed,
        with_injections=False,
        global_budget=100,
        budget_mode=True,
    )
    print(
        json.dumps(
            {
                "seed": args.seed,
                "default_dir": str(Path(args.default_dir)),
                "injection_dir": str(Path(args.injection_dir)),
                "budget_dir": str(Path(args.budget_dir)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
