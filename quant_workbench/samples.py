"""Deterministic, fully local sample generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SEED = 20261124
LAYER_SPECS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("layer.0", (6, 8)),
    ("layer.1", (8, 6)),
    ("layer.2", (6, 5)),
    ("layer.3", (5, 4)),
    ("layer.4", (4, 3)),
)


def _dump_yaml_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_dump_yaml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(f"{key}: {_dump_yaml_value(item)}" for key, item in value.items())
            + "}"
        )
    return json.dumps(str(value), ensure_ascii=False)


def _yaml_lines(mapping: dict[str, Any], indent: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in mapping.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(_yaml_lines(value, indent + 2))
        else:
            lines.append(f"{prefix}{key}: {_dump_yaml_value(value)}")
    return lines


def _weight_pack(seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for index, (layer, shape) in enumerate(LAYER_SPECS):
        weights = rng.normal(0.0, 0.32, size=shape).astype(np.float64)
        flat = weights.reshape(-1)
        # Deterministic outliers exercise saturation without using external data.
        if flat.size >= 5:
            flat[1] = 2.75 + 0.15 * index
            flat[flat.size - 2] = -(2.40 + 0.11 * index)
            flat[3] = 0.73 * (index + 1)
        records.append(
            {
                "layer": layer,
                "version": index + 1,
                "shape": list(shape),
                "weights": weights.tolist(),
            }
        )
    return records


def _calibration_records(seed: int, count: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for sample_index in range(count):
        activations: dict[str, list[float]] = {}
        for layer_index, (layer, shape) in enumerate(LAYER_SPECS):
            values = rng.normal(
                loc=0.08 * ((sample_index % 3) - 1),
                scale=0.55 + 0.03 * layer_index,
                size=shape[-1],
            )
            activations[layer] = [float(value) for value in values]
        records.append({"sample_id": f"cal-{sample_index:03d}", "activations": activations})
    return records


def generate_samples(
    directory: str | Path = "samples", *, seed: int = DEFAULT_SEED
) -> dict[str, Path]:
    """Create all local deterministic fixtures and return their paths."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / "weights.jsonl"
    calibration_path = destination / "calibration.jsonl"
    calibration_alt_path = destination / "calibration_alt.jsonl"
    config_path = destination / "config.yaml"
    commands_path = destination / "commands.jsonl"

    weight_records = _weight_pack(seed)
    with weights_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in weight_records:
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")

    calibration = _calibration_records(seed + 1, 10)
    with calibration_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in calibration:
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")

    calibration_alt = _calibration_records(seed + 2, 9)
    with calibration_alt_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in calibration_alt:
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")

    config: dict[str, Any] = {
        "snapshot_version": 1,
        "allowed_bit_widths": [2, 3, 4, 5, 6, 8, 16],
        "latency_budget_ms": 20,
        "max_threads": 3,
        "thread_limit": 8,
        "point_batch_budget": 64,
        "snapshot_commit_interval": 2,
        "distillation_temperature": 1.25,
        "significant_digits": 7,
        "worker_delay_ms": 1.5,
        "defaults": {
            "bits": 8,
            "group_size": 4,
            "fractional_bits": 4,
            "scheme": "symmetric",
        },
        "layers": {
            "layer.0": {
                "bits": 8,
                "group_size": 4,
                "fractional_bits": 5,
                "scheme": "symmetric",
            },
            "layer.1": {
                "bits": 8,
                "group_size": 4,
                "fractional_bits": 4,
                "scheme": "symmetric",
            },
            "layer.2": {
                "bits": 6,
                "group_size": 2,
                "fractional_bits": 3,
                "scheme": "asymmetric",
            },
            "layer.3": {
                "bits": 8,
                "group_size": 4,
                "fractional_bits": 4,
                "scheme": "symmetric",
            },
            "layer.4": {
                "bits": 4,
                "group_size": 2,
                "fractional_bits": 2,
                "scheme": "symmetric",
            },
        },
    }
    config_path.write_text("\n".join(_yaml_lines(config)) + "\n", encoding="utf-8")

    replacement_rng = np.random.default_rng(seed + 77)
    replacement = replacement_rng.normal(0.0, 0.28, size=LAYER_SPECS[-1][1])
    replacement.flat[0] = 1.9
    replacement.flat[-1] = -1.7
    commands = [
        {"cmd": "set_bits", "layer": "layer.1", "bits": 4},
        {"cmd": "set_group_size", "layer": "layer.2", "group_size": 2},
        {"cmd": "set_scheme", "layer": "layer.3", "scheme": "asymmetric"},
        {
            "cmd": "replace_weight",
            "layer": "layer.4",
            "version": 6,
            "shape": list(LAYER_SPECS[-1][1]),
            "weights": replacement.tolist(),
        },
        {"cmd": "set_temperature", "temperature": 0.85},
        {"cmd": "replace_calibration", "path": "calibration_alt.jsonl"},
    ]
    with commands_path.open("w", encoding="utf-8", newline="\n") as handle:
        for command in commands:
            handle.write(json.dumps(command, separators=(",", ":"), allow_nan=False) + "\n")

    return {
        "weights": weights_path,
        "calibration": calibration_path,
        "calibration_alt": calibration_alt_path,
        "config": config_path,
        "commands": commands_path,
    }
