from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from posdiag.models import AttentionSample, PositionConfig, load_config, load_samples, parse_config_dict


BASE_CONFIG: dict[str, Any] = {
    "scheme": "rope_standard",
    "base_frequency": 10000.0,
    "training_context_length": 32,
    "extrapolation_target_length": 128,
    "dimension": 8,
    "scaling_factor": 2.0,
    "precision": "strict",
    "batch_point_budget": 64,
    "global_mac_budget": 200000,
    "thread_pool_concurrency": 2,
    "snapshot_commit_interval": 3,
    "tolerance": 0.35,
    "global_version": 1,
    "comparison_schemes": [],
}


def make_config(**overrides: Any) -> PositionConfig:
    raw = dict(BASE_CONFIG)
    raw.update(overrides)
    return parse_config_dict(raw)


def make_sample(
    sample_id: str,
    *,
    dimension: int = 8,
    query_position: int = 20,
    key_positions: tuple[int, ...] = (1, 10, 20),
) -> AttentionSample:
    base = (float(sample_id[-3:]) + 1.0) / 1000.0
    query = tuple(base + index * 0.01 for index in range(dimension))
    keys = tuple(
        tuple(base + pair * 0.013 + index * 0.017 for index in range(dimension))
        for pair in range(len(key_positions))
    )
    return AttentionSample(
        sample_id=sample_id,
        query_position=query_position,
        key_positions=key_positions,
        query_vector=query,
        key_vectors=keys,
    )


def write_custom_inputs(
    root: Path,
    config_overrides: dict[str, Any],
    samples: list[dict[str, Any]],
) -> tuple[Path, Path, PositionConfig, tuple[AttentionSample, ...]]:
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    samples_path = root / "samples.jsonl"
    raw_config = dict(BASE_CONFIG)
    raw_config.update(config_overrides)
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
    config = load_config(config_path)
    loaded = load_samples(samples_path, dimension=config.dimension)
    return config_path, samples_path, config, loaded.samples


def report_without_metadata(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "run_metadata"}
