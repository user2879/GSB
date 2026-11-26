"""Deterministic local workspace generation. No network or external dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def default_config() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "seed": 388,
        "dimension": 8,
        "sample_count": 6,
        "key_count": 5,
        "train_length": 8,
        "target_length": 20,
        "point_budget": 240,
        "sample_quota": 80,
        "concurrency": 3,
        "extrapolation_bins": [2, 4, 8, 16],
        "schemes": [
            {
                "name": "rope_standard",
                "base_frequency": 10000.0,
                "angle_scale": 1.0,
                "block_size": 2,
                "frequency_correction": 0,
            },
            {
                "name": "scaled_ntk",
                "base_frequency": 50000.0,
                "angle_scale": 0.75,
                "block_size": 2,
                "frequency_correction": 1,
            },
            {
                "name": "block4_corrected",
                "base_frequency": 10000.0,
                "angle_scale": 1.125,
                "block_size": 4,
                "frequency_correction": 2,
            },
        ],
        "fixed_point": {
            "signed_bits": 32,
            "fraction_bits": 12,
            "accumulator_bits": 64,
            "quant_level": "nearest",
        },
        "tolerances": {
            "position": {"abs": 0.00005, "rel": 0.002},
            "rotation": {"abs": 0.002, "rel": 0.02},
            "dot": {"abs": 0.05, "rel": 0.08},
            "normalization": {"abs": 0.01, "rel": 0.05},
        },
    }


def _round_vector(values: Any) -> list[float]:
    return [round(float(value), 6) for value in values]


def make_sample(config: dict[str, Any], sample_id: str, index: int, *, replacement: bool = False) -> dict[str, Any]:
    dimension = int(config["dimension"])
    key_count = int(config["key_count"])
    target_length = int(config["target_length"])
    seed = int(config["seed"]) * 1000 + index + (70000 if replacement else 0)
    rng = np.random.default_rng(seed)

    query_positions = [0, 3, 6, 9, 12, 5, 14, 7]
    train_markers = [8, 8, 7, 9, 8, 6, 10, 8]
    query_position = query_positions[index % len(query_positions)]
    train_length = train_markers[index % len(train_markers)]
    if replacement:
        query_position = int(query_position) + 1
        train_length = max(4, int(train_length) - 1)

    if index == 0 and not replacement:
        key_positions = [0, 4, 8, 11, 15][:key_count]
        query_vector = _round_vector(rng.normal(0.0, 0.45, dimension))
        query_vector[0] = 0.0
        query_vector[1] = 1.0
        # An exact half-step at the default fraction width (2^12), ensuring
        # nearest-away and ties-to-even quantization gears produce different integers.
        query_vector[2] = 0.5 / float(1 << 12)
    else:
        choices = np.arange(0, target_length + 1, dtype=np.int64)
        key_positions = [int(value) for value in rng.choice(choices, size=key_count, replace=False)]
        query_vector = _round_vector(rng.normal(0.0, 0.55, dimension))

    key_vectors = [
        _round_vector(rng.normal(0.0, 0.5, dimension))
        for _ in range(key_count)
    ]
    return {
        "sample_id": sample_id,
        "generator_seed": seed,
        "query_position": int(query_position),
        "train_length": int(train_length),
        "query_vector": query_vector,
        "key_positions": [int(value) for value in key_positions],
        "key_vectors": key_vectors,
    }


def generate_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [make_sample(config, f"S{index:03d}", index) for index in range(int(config["sample_count"]))]


def generate_commands(config: dict[str, Any], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replacement = make_sample(config, "S002", 2, replacement=True)
    return [
        {
            "command_id": "CMD-01-ADD-SCHEME",
            "op": "add_scheme",
            "scheme": {
                "name": "experimental_block8",
                "base_frequency": 32000.0,
                "angle_scale": 0.9,
                "block_size": 8,
                "frequency_correction": 3,
            },
        },
        {
            "command_id": "CMD-02-BASE",
            "op": "update_scheme",
            "name": "rope_standard",
            "params": {"base_frequency": 20000.0},
        },
        {
            "command_id": "CMD-03-SCALE",
            "op": "update_scheme",
            "name": "scaled_ntk",
            "params": {"angle_scale": 1.25},
        },
        {"command_id": "CMD-04-TRAIN", "op": "set_train_length", "train_length": 10},
        {"command_id": "CMD-05-TARGET", "op": "set_target_length", "target_length": 24},
        {"command_id": "CMD-06-REPLACE", "op": "replace_sample", "sample": replacement},
        {
            "command_id": "CMD-07-FIXED",
            "op": "set_fixed_point",
            "fixed_point": {
                "signed_bits": 16,
                "fraction_bits": 8,
                "accumulator_bits": 64,
                "quant_level": "ties_even",
            },
        },
        {
            "command_id": "CMD-08-BUDGET",
            "op": "set_scheduling",
            "point_budget": 400,
            "sample_quota": 120,
            "concurrency": 4,
        },
        {"command_id": "CMD-09-DELETE-SAMPLE", "op": "delete_sample", "sample_id": "S005"},
        {"command_id": "CMD-10-REMOVE-SCHEME", "op": "remove_scheme", "name": "experimental_block8"},
    ]


def generate_injections() -> list[dict[str, Any]]:
    return [
        {
            "injection_id": "INJ-DUP-01",
            "kind": "duplicate_key_positions",
            "target_sample": "S001",
            "stage": "validation",
            "description": "Append a copy of an existing key position.",
        },
        {
            "injection_id": "INJ-BLOCK-02",
            "kind": "block_incompatible",
            "target_scheme": "block4_corrected",
            "stage": "validation",
            "block_size": 3,
            "description": "Set block_size=3 for dimension=8.",
        },
        {
            "injection_id": "INJ-WIDTH-03",
            "kind": "width_out_of_range",
            "stage": "validation",
            "fraction_bits": 99,
            "description": "Set fraction_bits outside the declared enum.",
        },
        {
            "injection_id": "INJ-NAN-04",
            "kind": "nonfinite_feature",
            "target_sample": "S002",
            "stage": "rotation",
            "path": "query_vector[3]",
            "fault_value": "NaN",
            "description": "Materialize float('nan') in a query feature.",
        },
        {
            "injection_id": "INJ-SCHEME-05",
            "kind": "undeclared_scheme",
            "target_scheme": "ghost_rope",
            "stage": "position",
            "description": "Reference a scheme absent from config.schemes.",
        },
    ]


def make_flip_attribution_cases() -> list[dict[str, Any]]:
    """Fixed-seed cases selected to exercise each ranking-flip cause."""
    definitions = [
        {
            "case_id": "FLIP-ANGLE-01",
            "expected_source": "angle_quantization",
            "seed": 9,
            "query_position": 31,
            "key_positions": [32, 45],
            "scale": 0.8,
            "fraction_bits": 4,
            "accumulator_bits": 64,
        },
        {
            "case_id": "FLIP-ROTATION-02",
            "expected_source": "rotation_rounding",
            "seed": 5,
            "query_position": 0,
            "key_positions": [1, 10],
            "scale": 0.8,
            "fraction_bits": 4,
            "accumulator_bits": 64,
        },
        {
            "case_id": "FLIP-OVERFLOW-03",
            "expected_source": "accumulator_overflow",
            "seed": 2,
            "query_position": 100,
            "key_positions": [101, 130],
            "scale": 2.0,
            "fraction_bits": 12,
            "accumulator_bits": 16,
        },
    ]
    cases = []
    dimension = 8
    for definition in definitions:
        rng = np.random.default_rng(int(definition["seed"]))
        scale = float(definition["scale"])
        sample = {
            "sample_id": definition["case_id"].replace("-", "_"),
            "generator_seed": definition["seed"],
            "query_position": definition["query_position"],
            "train_length": 0,
            "query_vector": [round(float(value), 6) for value in rng.normal(0.0, scale, dimension)],
            "key_positions": list(definition["key_positions"]),
            "key_vectors": [
                [round(float(value), 6) for value in rng.normal(0.0, scale, dimension)]
                for _ in definition["key_positions"]
            ],
        }
        cases.append({**definition, "sample": sample})
    return cases


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {key: value for key, value in row.items() if key != "_line"}
            handle.write(json.dumps(clean, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def initialize_workspace(root: str | Path = ".") -> dict[str, list[dict[str, Any]]]:
    root_path = Path(root)
    config = default_config()
    samples = generate_samples(config)
    commands = generate_commands(config, samples)
    injections = generate_injections()

    write_json(root_path / "configs/lab_config.json", config)
    write_jsonl(root_path / "data/samples.jsonl", samples)
    write_jsonl(root_path / "commands/edit_commands.jsonl", commands)
    write_json(root_path / "injections/injections.json", {"injections": injections})
    return {"config": config, "samples": samples, "commands": commands, "injections": injections}
