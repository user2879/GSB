"""Generate the checked-in deterministic sample corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_SEED = 20261121


def _rounded(values: np.ndarray) -> list[float]:
    return [float(f"{value:.10f}") for value in values.reshape(-1, order="C")]


def generate_samples(output_dir: str | Path, seed: int = DEFAULT_SEED) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(seed))

    specifications = [
        ("embed", (6, 8), 1),
        ("encoder.attn.k", (8, 8), 2),
        ("encoder.attn.q", (8, 8), 3),
        ("encoder.attn.v", (8, 8), 4),
        ("encoder.norm", (8,), 5),
        ("head", (4, 8), 6),
    ]
    layers: list[dict[str, Any]] = []
    layer_arrays: dict[str, np.ndarray] = {}
    for index, (name, shape, version) in enumerate(specifications):
        scale = 0.35 + 0.04 * index
        weights = np.round(rng.normal(0.0, scale, size=shape), 10)
        # Keep one quarter-row exactly zero to exercise the documented
        # zero-scale -> whole-layer fallback rule.
        if name == "encoder.attn.v":
            weights[2, 0:4] = 0.0
        layer_arrays[name] = np.asarray(weights, dtype=np.float64)
        layers.append(
            {
                "name": name,
                "shape": list(shape),
                "weights": _rounded(layer_arrays[name]),
                "content_version": version,
            }
        )

    weights_path = root / "weights.jsonl"
    with weights_path.open("w", encoding="utf-8") as stream:
        for layer in layers:
            stream.write(
                json.dumps(layer, ensure_ascii=False, separators=(",", ":"))
            )
            stream.write("\n")

    samples: list[dict[str, Any]] = []
    for sample_index in range(5):
        inputs = np.round(rng.normal(0.0, 0.5, size=8), 10)
        reference_parts = [
            layer_arrays[name].reshape(-1, layer_arrays[name].shape[-1]) @ inputs
            for name, _, _ in specifications
        ]
        reference = np.concatenate(reference_parts)
        samples.append(
            {
                "id": f"cal-{sample_index:02d}",
                "input": _rounded(inputs),
                "reference_output": _rounded(reference),
                "layers": ["all"],
            }
        )
    calibration_path = root / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "description": "Deterministic local calibration samples",
                "samples": samples,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    config = {
        "schema_version": 1,
        "quantization": {
            "default_bits": 8,
            "scheme": "symmetric",
            "group_size": 4,
            "group_axis": -1,
            "per_channel_scaling": True,
            "scale_significant_digits": 7,
        },
        "layers": {
            "embed": {"bits": 8, "group_size": 4},
            "encoder.attn.k": {"bits": 4, "group_size": 4},
            "encoder.attn.q": {"bits": 8, "group_size": 8},
            "encoder.attn.v": {"bits": 4, "group_size": 4},
            "encoder.norm": {"bits": 8, "group_size": 4},
            "head": {"bits": 4, "group_size": 8},
        },
        "full_precision_layers": ["encoder.norm"],
        "deployment": {
            "package_format_version": "1.0.0",
            "artifact_name": "demo-transformer-slice",
        },
        "runtime": {
            "interaction_latency_budget_ms": 20,
            "concurrency": 4,
            "point_budget_per_batch": 65536,
            "snapshot_commit_interval": 1,
            "stage_delay_ms": 0,
        },
        "distillation": {
            "temperature": 2.0,
            "soft_label_weight": 0.35,
            "correction_gain": 1.0,
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return {
        "weights": str(weights_path),
        "calibration": str(calibration_path),
        "config": str(config_path),
    }
