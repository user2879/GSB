"""Generate deterministic local weight, calibration, quantization, and injection fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .util import atomic_write_text, pretty_json


DEFAULT_SEED = 20261124


LAYER_SPECS = (
    ("block0.attn.k", (6, 8), 4),
    ("block0.attn.q", (6, 8), 4),
    ("block0.mlp.up", (8, 6), 3),
    ("block0.mlp.down", (6, 8), 4),
    ("block1.attn.k", (4, 10), 5),
    ("block1.attn.q", (4, 10), 5),
    ("block1.mlp.up", (8, 4), 4),
    ("block1.mlp.down", (8, 4), 4),
)


def build_quantization_config(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    return {
        "version": 1,
        "seed": seed,
        "quantization": {
            "active_profile": "q8",
            "profiles": [
                {"name": "q6", "bits": 6, "fractional_bits": 6},
                {"name": "q8", "bits": 8, "fractional_bits": 8},
                {"name": "q10", "bits": 10, "fractional_bits": 10},
            ],
            "allowed_bit_widths": list(range(2, 17)),
            "scheme": "symmetric",
            "group_size": 4,
            "min_group_size": 3,
            "saturation": "saturate",
            "rounding": "nearest_even",
            "percentile_low": 0.05,
            "percentile_high": 0.95,
            "interpolation": "linear",
            "tie_break": "lower_index",
        },
        "distillation": {
            "temperature": 1.5,
            "soft_label_weight": 0.35,
        },
        "runtime": {
            "max_workers": 4,
            "point_budget": 1000,
            "layer_point_quota": 1000,
            "snapshot_interval": 2,
        },
        "tolerances": {
            "absolute": 0.015,
            "relative": 0.12,
        },
        "layers": {
            "block0.attn.k": {"group_size": 4},
            "block0.attn.q": {"group_size": 2},
            "block0.mlp.up": {"group_size": 3},
            "block0.mlp.down": {"group_size": 4},
            "block1.attn.k": {"group_size": 5},
            "block1.attn.q": {"group_size": 5},
            "block1.mlp.up": {"group_size": 4},
            "block1.mlp.down": {"group_size": 4},
        },
    }


def build_injection_config(include_all: bool = False) -> dict[str, Any]:
    if not include_all:
        return {"version": 1, "injections": []}
    return {
        "version": 1,
        "injections": [
            {
                "id": "INJ-001",
                "type": "non_finite_weight",
                "layer": "block0.attn.k",
                "stage": "validation",
            },
            {
                "id": "INJ-002",
                "type": "group_indivisible",
                "layer": "block0.attn.q",
                "stage": "validation",
            },
            {
                "id": "INJ-003",
                "type": "bit_width_out_of_range",
                "layer": "block0.mlp.up",
                "stage": "config",
                "params": {"target_bits": 99},
            },
            {
                "id": "INJ-004",
                "type": "calibration_dimension_mismatch",
                "layer": "block1.attn.k",
                "stage": "calibration",
            },
            {
                "id": "INJ-005",
                "type": "zero_temperature",
                "stage": "distillation",
            },
            {
                "id": "INJ-006",
                "type": "duplicate_layer_name",
                "layer": "block1.mlp.down",
                "stage": "validation",
            },
        ],
    }


def generate_samples(
    sample_dir: str | Path,
    config_dir: str | Path | None = None,
    *,
    seed: int = DEFAULT_SEED,
    sample_count: int = 4,
) -> dict[str, str]:
    """Write byte-for-byte deterministic fixtures and return their paths."""

    sample_dir = Path(sample_dir)
    config_dir = Path(config_dir) if config_dir is not None else sample_dir.parent / "configs"
    sample_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    config = build_quantization_config(seed)

    layer_weights: dict[str, np.ndarray] = {}
    layer_lines: list[str] = []
    for version, (name, shape, group_size) in enumerate(LAYER_SPECS, start=1):
        weights = rng.normal(0.0, 0.38, size=int(np.prod(shape))).astype(np.float64)
        # Deliberate exact zero and outliers exercise the zero-truth branch and clipping.
        weights[0] = 0.0
        weights[1] = 7.5 if version % 2 else -7.5
        weights[-1] = -5.75 if version % 2 else 5.75
        layer_weights[name] = weights
        record = {
            "layer": name,
            "shape": list(shape),
            "weights": weights.tolist(),
            "version": version,
            "group_size": group_size,
        }
        layer_lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    samples: list[dict[str, Any]] = []
    for sample_index in range(sample_count):
        inputs: dict[str, list[float]] = {}
        references: dict[str, list[float]] = {}
        for name, shape, _group_size in LAYER_SPECS:
            vector = rng.normal(0.0, 1.0, size=shape[-1]).astype(np.float64)
            vector[0] = 0.0 if sample_index == 0 else vector[0]
            matrix = layer_weights[name].reshape((-1, shape[-1]))
            output = matrix @ vector
            inputs[name] = vector.tolist()
            references[name] = output.tolist()
        samples.append(
            {
                "id": f"s{sample_index:04d}",
                "inputs": inputs,
                "reference_outputs": references,
            }
        )

    calibration = {"version": 1, "seed": seed, "samples": samples}
    weight_path = sample_dir / "weights.jsonl"
    calibration_path = sample_dir / "calibration.json"
    config_path = sample_dir / "config.yaml"
    injections_empty_path = sample_dir / "injections.yaml"
    injections_all_path = sample_dir / "injections_all.yaml"
    config_standalone = config_dir / "quantization.yaml"
    injections_config_empty = config_dir / "injections_empty.yaml"
    injections_config_all = config_dir / "injections_all.yaml"

    atomic_write_text(weight_path, "\n".join(layer_lines) + "\n")
    atomic_write_text(calibration_path, pretty_json(calibration))
    config_text = pretty_json(config)
    atomic_write_text(config_path, config_text)
    atomic_write_text(config_standalone, config_text)
    empty_text = pretty_json(build_injection_config(False))
    all_text = pretty_json(build_injection_config(True))
    atomic_write_text(injections_empty_path, empty_text)
    atomic_write_text(injections_all_path, all_text)
    atomic_write_text(injections_config_empty, empty_text)
    atomic_write_text(injections_config_all, all_text)
    return {
        "weights": str(weight_path),
        "calibration": str(calibration_path),
        "config": str(config_path),
        "injections": str(injections_empty_path),
        "injections_all": str(injections_all_path),
        "config_dir": str(config_dir),
    }

