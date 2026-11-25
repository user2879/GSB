"""Deterministic synthetic seasonal-sales fixtures."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from .jsonutil import canonical_json, write_text_atomic


def _series(
    series_id: str,
    *,
    seed: int,
    length: int,
    level: float,
    slope: float,
    harmonic_1: float,
    harmonic_2: float,
    phase: float,
    missing_indices: tuple[int, ...] = (),
    trailing_missing: int = 0,
) -> dict[str, Any]:
    rng = random.Random(seed)
    values: list[float | None] = []
    for index in range(length):
        if trailing_missing and index >= length - trailing_missing:
            values.append(None)
            continue
        angle = 2.0 * math.pi * index / 12.0
        noise = rng.uniform(-1.75, 1.75)
        value = (
            level
            + slope * index
            + harmonic_1 * math.sin(angle + phase)
            + harmonic_2 * math.cos(2.0 * angle - 0.35)
            + noise
        )
        values.append(round(value, 3))
    for index in missing_indices:
        values[index] = None
    return {
        "series_id": series_id,
        "timestamps": [1000 + index for index in range(length)],
        "values": values,
    }


def example_series() -> list[dict[str, Any]]:
    specs = [
        ("sku-alpha", 11, 36, 180.0, 0.62, 14.0, 6.0, 0.10, (8, 23), 0),
        ("sku-bravo", 17, 30, 96.0, -0.41, 9.5, 3.2, 0.75, (3, 4, 20), 0),
        ("sku-charlie", 23, 42, 240.0, 0.08, 18.0, 9.0, 0.25, (0, 17, 31), 0),
        ("sku-delta", 29, 28, 140.0, 1.10, 8.0, 4.0, 0.60, (13,), 0),
        ("sku-echo", 31, 33, 72.0, -0.95, 12.0, 2.5, 1.20, (6, 7, 25), 0),
        ("sku-foxtrot", 37, 26, 310.0, -0.12, 6.0, 5.5, 2.00, (11,), 0),
        ("sku-golf", 41, 38, 205.0, 0.33, 15.0, 7.0, 0.45, (4, 19, 27), 0),
        ("sku-hotel", 43, 24, 118.0, 0.05, 7.0, 3.0, 1.55, (0, 12), 3),
    ]
    result = []
    for number, spec in enumerate(specs):
        (
            series_id,
            seed,
            length,
            level,
            slope,
            harmonic_1,
            harmonic_2,
            phase,
            missing,
            trailing,
        ) = spec
        result.append(
            _series(
                series_id,
                seed=20261125 + seed + number,
                length=length,
                level=level,
                slope=slope,
                harmonic_1=harmonic_1,
                harmonic_2=harmonic_2,
                phase=phase,
                missing_indices=missing,
                trailing_missing=trailing,
            )
        )
    return result


def appended_series() -> dict[str, Any]:
    return _series(
        "sku-india",
        seed=20261125 + 101,
        length=25,
        level=158.0,
        slope=0.27,
        harmonic_1=11.0,
        harmonic_2=4.5,
        phase=0.9,
        missing_indices=(5, 16),
    )


CONFIG_YAML = """\
# Deterministic seasonal fit demo configuration
version: 1

model:
  period: 12
  harmonic_order: 3
  regularization_strength: 0.05

fixed_point:
  bits: 48
  quantization_shift: 16
  pivot_threshold: 1000

tolerance:
  relative: 0.0001
  absolute: 0.0001

budget:
  scale_profile: standard
  global_fit_points: 4096
  per_series_quota: 768
  batch_observation_points: 1024

runtime:
  thread_pool_workers: 2

diagnostics:
  comparison_bit_widths: [32, 48]
"""


def example_commands() -> list[dict[str, Any]]:
    appended = appended_series()
    return [
        {"op": "append_series", "series": appended},
        {"op": "replace_observation", "series_id": "sku-alpha", "index": 5, "value": 209.125},
        {"op": "delete_series", "series_id": "sku-hotel"},
        {"op": "replace_observation", "series_id": "sku-not-present", "index": 2, "value": 1.0},
        {"op": "set_harmonic_order", "value": 2},
        {"op": "set_regularization_strength", "value": 0.5},
        {"op": "set_fixed_point", "bits": 32, "quantization_shift": 12},
        {"op": "set_scale_profile", "profile": "small"},
        {"op": "set_thread_pool_workers", "value": 2},
    ]


def generate_example_workspace(directory: str | Path, *, force: bool = False) -> dict[str, str]:
    root = Path(directory)
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(f"{root} is not empty; pass force=True to replace fixtures")
    root.mkdir(parents=True, exist_ok=True)
    series_path = root / "series.jsonl"
    config_path = root / "config.yaml"
    commands_path = root / "commands.jsonl"
    series_text = "".join(canonical_json(item) + "\n" for item in example_series())
    command_text = "".join(canonical_json(item) + "\n" for item in example_commands())
    write_text_atomic(series_path, series_text)
    write_text_atomic(config_path, CONFIG_YAML)
    write_text_atomic(commands_path, command_text)
    return {
        "series": str(series_path),
        "config": str(config_path),
        "commands": str(commands_path),
    }
