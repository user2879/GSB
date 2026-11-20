#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np


SEED = 20261121


BASE_CONFIG: dict[str, Any] = {
    "scheme": "standard_rope",
    "base": 10000.0,
    "training_context_length": 32,
    "extrapolation_target_length": 128,
    "dimension": 32,
    "scale_factor": 4.0,
    "precision": "float64",
    "dimension_chunk_size": 4,
    "interaction_budget_ms": 3000,
    "concurrency": 4,
    "snapshot_commit_interval": 4,
    "pruning_tolerance": 0.05,
    "pruning_policy": "energy",
    "error_tolerance": 0.02,
}


def _feature_vector(rng: np.random.Generator, dimension: int) -> list[float]:
    scale = 1.0 / np.sqrt(float(dimension))
    return [float(value) for value in rng.normal(0.0, scale, size=dimension)]


def _sample(
    *,
    sample_id: str,
    query_position: int,
    distances: list[int],
    rng: np.random.Generator,
    dimension: int,
) -> dict[str, Any]:
    unique_distances = sorted({distance for distance in distances if 0 <= distance <= query_position})
    key_positions = [query_position - distance for distance in unique_distances]
    return {
        "sample_id": sample_id,
        "query_position": query_position,
        "key_positions": key_positions,
        "query_features": _feature_vector(rng, dimension),
        "key_features": [
            _feature_vector(rng, dimension) for _ in range(len(key_positions))
        ],
    }


def build_samples(count: int, dimension: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    query_positions = [8, 12, 16, 24, 31, 32, 33, 40, 48, 56, 64, 80, 96, 112, 127]
    candidates = [0, 1, 2, 3, 4, 7, 8, 15, 16, 23, 24, 30, 31, 32, 33, 40, 47, 48, 55, 63, 64, 79, 95, 111, 126]
    samples: list[dict[str, Any]] = []
    for index in range(count):
        query_position = query_positions[(index * 7 + 3) % len(query_positions)]
        allowed = [distance for distance in candidates if distance <= query_position]
        required = {0}
        if query_position >= 31:
            required.add(31)
        if query_position >= 32:
            required.add(32)
        if query_position >= 33:
            required.add(33)
        if query_position >= 40:
            required.add(40)
        optional_pool = [distance for distance in allowed if distance not in required]
        choose_count = min(len(optional_pool), 4 + (index % 3))
        chosen = sorted(required | set(int(value) for value in rng.choice(optional_pool, choose_count, replace=False)))
        samples.append(
            _sample(
                sample_id=f"s{index:03d}",
                query_position=query_position,
                distances=chosen,
                rng=rng,
                dimension=dimension,
            )
        )
    return samples


def build_commands(samples: list[dict[str, Any]], dimension: int) -> list[dict[str, Any]]:
    replacement_rng = np.random.default_rng(SEED + 99)
    replacement = _sample(
        sample_id="s007",
        query_position=120,
        distances=[0, 1, 7, 31, 32, 33, 47, 63, 95, 119],
        rng=replacement_rng,
        dimension=dimension,
    )
    return [
        {"command_id": "c01", "type": "set_scheme", "scheme": "linear_interpolation"},
        {"command_id": "c02", "type": "poll"},
        {"command_id": "c03", "type": "set_base", "base": 20000.0},
        {"command_id": "c04", "type": "set_scale", "scale_factor": 2.5},
        {"command_id": "c05", "type": "set_training_context_length", "training_context_length": 24},
        {
            "command_id": "c06",
            "type": "set_pruning",
            "pruning_policy": "frequency",
            "pruning_tolerance": 0.10,
        },
        {"command_id": "c07", "type": "set_scheme", "scheme": "ntk_aware"},
        {"command_id": "c08", "type": "poll"},
        {"command_id": "c09", "type": "update", "base": 12000.0, "scale_factor": 3.25},
        {"command_id": "c10", "type": "set_training_context_length", "training_context_length": 28},
        {
            "command_id": "c11",
            "type": "set_pruning",
            "pruning_policy": "energy",
            "pruning_tolerance": 0.08,
        },
        {"command_id": "c12", "type": "replace_sample", "sample": replacement},
        {"command_id": "c13", "type": "poll"},
        {"command_id": "c14", "type": "set_scheme", "scheme": "standard_rope"},
        {"command_id": "c15", "type": "set_base", "base": 10000.0},
        {"command_id": "c16", "type": "set_scale", "scale_factor": 4.0},
        {"command_id": "c17", "type": "set_training_context_length", "training_context_length": 32},
        {
            "command_id": "c18",
            "type": "set_pruning",
            "pruning_policy": "energy",
            "pruning_tolerance": 0.05,
        },
        {"command_id": "c19", "type": "poll"},
    ]


def _sample_line(value: dict[str, Any], *, allow_nan: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=allow_nan,
    )


def write_fault_inputs(output: Path, dimension: int) -> None:
    rng = np.random.default_rng(SEED + 7)
    valid_one = _sample(
        sample_id="f001",
        query_position=40,
        distances=[0, 7, 31, 32, 33, 39],
        rng=rng,
        dimension=dimension,
    )
    valid_continue = _sample(
        sample_id="f007",
        query_position=48,
        distances=[0, 16, 32, 33, 47],
        rng=rng,
        dimension=dimension,
    )
    nonfinite = _sample(
        sample_id="f002",
        query_position=20,
        distances=[0, 5, 19],
        rng=rng,
        dimension=dimension,
    )
    nonfinite["query_features"][0] = float("nan")
    mismatch = _sample(
        sample_id="f003",
        query_position=20,
        distances=[0, 5, 19],
        rng=rng,
        dimension=dimension,
    )
    mismatch["query_features"] = mismatch["query_features"][:-1]
    out_of_bounds = _sample(
        sample_id="f004",
        query_position=10,
        distances=[0, 5],
        rng=rng,
        dimension=dimension,
    )
    out_of_bounds["key_positions"].append(11)
    out_of_bounds["key_features"].append(_feature_vector(rng, dimension))
    duplicate = copy.deepcopy(valid_one)
    lines = [
        _sample_line(valid_one),
        "{this-is-not-json",
        _sample_line(nonfinite, allow_nan=True),
        _sample_line(mismatch),
        _sample_line(out_of_bounds),
        _sample_line(duplicate),
        _sample_line(valid_continue),
    ]
    (output / "fault_samples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "empty_samples.jsonl").write_text("", encoding="utf-8")

    base_fault = copy.deepcopy(BASE_CONFIG)
    base_fault["base"] = 0
    (output / "config_fault_base.json").write_text(
        json.dumps(base_fault, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    scale_fault = copy.deepcopy(BASE_CONFIG)
    scale_fault["scale_factor"] = 0
    (output / "config_fault_scale.json").write_text(
        json.dumps(scale_fault, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "config_invalid_json.json").write_text("{invalid-config-json\n", encoding="utf-8")
    manifest = {
        "seed": SEED,
        "dimension": dimension,
        "faults": [
            {
                "fault_id": "invalid_json_line",
                "file": "fault_samples.jsonl",
                "line": 2,
                "expected_code": "E_SAMPLE_INVALID",
            },
            {
                "fault_id": "nonfinite_feature",
                "file": "fault_samples.jsonl",
                "line": 3,
                "expected_code": "E_SAMPLE_INVALID",
            },
            {
                "fault_id": "dimension_length_mismatch",
                "file": "fault_samples.jsonl",
                "line": 4,
                "expected_code": "E_DIM_MISMATCH",
            },
            {
                "fault_id": "key_position_out_of_bounds",
                "file": "fault_samples.jsonl",
                "line": 5,
                "expected_code": "E_SAMPLE_INVALID",
            },
            {
                "fault_id": "duplicate_sample_id",
                "file": "fault_samples.jsonl",
                "line": 6,
                "expected_code": "E_SAMPLE_INVALID",
            },
            {
                "fault_id": "empty_sample_file",
                "file": "empty_samples.jsonl",
                "expected_code": "E_SAMPLE_INVALID",
            },
            {
                "fault_id": "zero_base",
                "file": "config_fault_base.json",
                "expected_code": "E_BASE_INVALID",
            },
            {
                "fault_id": "nonpositive_scale",
                "file": "config_fault_scale.json",
                "expected_code": "E_SCALE_INVALID",
            },
            {
                "fault_id": "invalid_config_json",
                "file": "config_invalid_json.json",
                "expected_code": "E_CONFIG_INVALID",
            },
        ],
    }
    (output / "fault_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    dimension = int(BASE_CONFIG["dimension"])
    samples = build_samples(64, dimension)
    commands = build_commands(samples, dimension)
    large_config = copy.deepcopy(BASE_CONFIG)
    small_config = copy.deepcopy(BASE_CONFIG)
    small_config["interaction_budget_ms"] = 20
    (output / "config_large_budget.json").write_text(
        json.dumps(large_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "config_small_budget.json").write_text(
        json.dumps(small_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "samples.jsonl").write_text(
        "\n".join(_sample_line(sample) for sample in samples) + "\n",
        encoding="utf-8",
    )
    (output / "commands.json").write_text(
        json.dumps({"seed": SEED, "commands": commands}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_fault_inputs(output, dimension)
    (output / "generation_manifest.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "sample_count": len(samples),
                "command_count": len(commands),
                "dimension": dimension,
                "training_context_length": BASE_CONFIG["training_context_length"],
                "extrapolation_target_length": BASE_CONFIG["extrapolation_target_length"],
                "relative_distance_coverage": {
                    "in_training": [0, 7, 31],
                    "boundary": [32],
                    "extrapolation": [33, 40, 47, 63, 95, 119],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic RoPE diagnostic fixtures")
    parser.add_argument("--output", default="samples", help="output sample directory")
    args = parser.parse_args()
    generate(Path(args.output))


if __name__ == "__main__":
    main()
