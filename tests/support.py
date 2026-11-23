from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def config_dict(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "schema_version": 1,
        "global_version": 1,
        "schemes": ["rope", "linear_interpolation", "ntk_aware"],
        "base": 10000.0,
        "training_context_length": 4,
        "extrapolation_target_length": 16,
        "dimension": 8,
        "scaling_factor": 4.0,
        "precision": "float64",
        "quantization": "none",
        "dimension_chunk_size": 4,
        "point_budget": 100000,
        "concurrency": 2,
        "snapshot_interval": 2,
        "error_tolerances": {
            "position_vector": 0.01,
            "score": 0.5,
            "attention": 0.01,
        },
        "sensitivity": {
            "base_step": 1000.0,
            "scale_step": 0.5,
            "radius": 1,
        },
    }
    config.update(overrides)
    return config


def pruning_dict() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policies": [
            {
                "policy_id": "full",
                "description": "all pairs",
                "pair_ranges": [{"start_ratio": 0.0, "end_ratio": 1.0}],
                "pair_stride": 1,
            },
            {
                "policy_id": "balanced_75",
                "description": "first 75 percent",
                "pair_ranges": [{"start_ratio": 0.0, "end_ratio": 0.75}],
                "pair_stride": 1,
            },
            {
                "policy_id": "low_rank_50",
                "description": "first half",
                "pair_ranges": [{"start_ratio": 0.0, "end_ratio": 0.5}],
                "pair_stride": 1,
            },
            {
                "policy_id": "even_pairs",
                "description": "every second pair",
                "pair_ranges": [{"start_ratio": 0.0, "end_ratio": 1.0}],
                "pair_stride": 2,
            },
        ],
    }


def injection_dict(items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"schema_version": 1, "injections": items or []}


def sample_rows(
    *,
    count: int = 4,
    dimension: int = 8,
    seed: int = 42,
    no_extrapolation: bool = False,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        if no_extrapolation:
            query_position = 3
            positions = [0, 1, 2, 3]
        else:
            query_position = 6 + index * 2
            positions = sorted(
                set(
                    int(value)
                    for value in np.linspace(
                        max(0, query_position - 8), query_position, num=7, dtype=np.int64
                    ).tolist()
                )
            )
        query_features = rng.normal(size=dimension)
        key_features = rng.normal(size=(len(positions), dimension))
        rows.append(
            {
                "sample_id": f"s{index + 1:04d}",
                "query_position": query_position,
                "key_positions": positions,
                "query_features": [float(value) for value in query_features.tolist()],
                "key_features": [
                    [float(value) for value in row] for row in key_features.tolist()
                ],
            }
        )
    return rows


def write_case(
    root: Path,
    *,
    config: dict[str, Any] | None = None,
    samples: list[dict[str, Any]] | None = None,
    injections: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    data_dir = root / "sample_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_payload = config if config is not None else config_dict()
    rows = samples if samples is not None else sample_rows(dimension=int(config_payload["dimension"]))
    config_path = data_dir / "config.json"
    samples_path = data_dir / "samples.jsonl"
    pruning_path = data_dir / "pruning_strategies.json"
    injections_path = data_dir / "injections.json"
    config_path.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pruning_path.write_text(
        json.dumps(pruning_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    injections_path.write_text(
        json.dumps(injection_dict(injections), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with samples_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "config": config_path,
        "samples": samples_path,
        "pruning": pruning_path,
        "injections": injections_path,
        "root": root,
    }


def rewrite_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

