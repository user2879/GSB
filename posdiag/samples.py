from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import canonical_json


SAMPLE_LAYOUT = (
    (4, (0, 1, 2, 3, 4)),
    (12, (0, 3, 6, 9, 11)),
    (31, (0, 8, 16, 24, 31)),
    (32, (0, 16, 24, 31, 32)),
    (40, (0, 10, 20, 30, 40)),
    (63, (1, 15, 31, 47, 63)),
    (96, (0, 32, 64, 80, 96)),
    (127, (2, 31, 64, 95, 127)),
    (18, ()),
    (40, (8,)),
    (20, (19,)),
    (70, (1, 33, 35, 69)),
    (5, (0, 8, 16, 24, 32)),
    (120, (31, 32, 63, 64)),
)


def _sample(index: int, query_position: int, key_positions: tuple[int, ...], seed: int, *, sample_id: str | None = None) -> dict:
    rng = np.random.default_rng(seed + index * 1009)
    dimension = 16
    query_vector = rng.normal(0.0, 1.0, size=dimension).astype(np.float64)
    key_vectors = [rng.normal(0.0, 1.0, size=dimension).astype(np.float64) for _ in key_positions]
    return {
        "sample_id": sample_id or f"s{index:04d}",
        "query_position": int(query_position),
        "key_positions": [int(value) for value in key_positions],
        "query_vector": [float(value) for value in query_vector.tolist()],
        "key_vectors": [[float(value) for value in vector.tolist()] for vector in key_vectors],
    }


def generate_sample_directory(output: str | Path, *, seed: int = 20261124) -> dict[str, Path]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    config = {
        "scheme": "rope_standard",
        "base_frequency": 10000.0,
        "training_context_length": 32,
        "extrapolation_target_length": 128,
        "dimension": 16,
        "scaling_factor": 2.0,
        "precision": "strict",
        "batch_point_budget": 64,
        "global_mac_budget": 200000,
        "thread_pool_concurrency": 4,
        "snapshot_commit_interval": 4,
        "tolerance": 0.35,
        "global_version": 1,
        "comparison_schemes": [],
    }
    config_path = output_path / "position_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    samples = [
        _sample(index, query, keys, seed)
        for index, (query, keys) in enumerate(SAMPLE_LAYOUT, start=1)
    ]
    samples_path = output_path / "attention_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(canonical_json(sample) + "\n")

    replacement = _sample(7, 100, (1, 33, 65, 97, 121), seed + 70000)
    combined_plan = {
        "description": "Switch scheme, adjust base frequency, replace one sample, and change precision.",
        "operations": [
            {"op": "switch_scheme", "scheme": "rope_ntk"},
            {"op": "adjust_base_frequency", "factor": 1.5},
            {"op": "change_precision", "precision": "compact"},
            {"op": "replace_sample", "sample_id": "s0007", "sample": replacement},
        ],
    }
    plans = {
        "edit_plan.json": combined_plan,
        "edit_plan_sample_only.json": {
            "description": "Replace one sample only; all other sample cache entries must be reused.",
            "operations": [{"op": "replace_sample", "sample_id": "s0007", "sample": replacement}],
        },
        "edit_plan_scheme_only.json": {
            "description": "Switch from standard RoPE to NTK-aware scaling.",
            "operations": [{"op": "switch_scheme", "scheme": "rope_ntk"}],
        },
        "edit_plan_frequency_only.json": {
            "description": "Adjust the RoPE base frequency.",
            "operations": [{"op": "adjust_base_frequency", "factor": 1.5}],
        },
        "edit_plan_precision_only.json": {
            "description": "Change the report precision gear.",
            "operations": [{"op": "change_precision", "precision": "compact"}],
        },
    }
    plan_paths: dict[str, Path] = {}
    for name, plan in plans.items():
        path = output_path / name
        path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        plan_paths[name] = path

    return {
        "config": config_path,
        "samples": samples_path,
        "edit_plan": plan_paths["edit_plan.json"],
        **plan_paths,
    }
