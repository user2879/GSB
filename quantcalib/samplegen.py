"""Deterministic local acceptance samples."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml


SEED = 20261125


def _weights(count: int, rng: random.Random, scale: float) -> list[float]:
    values = [round(rng.uniform(-scale, scale), 7) for _ in range(count)]
    # Deterministic tails exercise percentile clipping and saturation.
    values[0] = round(scale * 2.5, 7)
    values[-1] = round(-scale * 2.25, 7)
    return values


def generate_samples(root: Path) -> dict[str, Path]:
    root = Path(root)
    receipts_dir = root / "receipts"
    root.mkdir(parents=True, exist_ok=True)
    receipts_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    layers = [
        {
            "layer_name": "embedding.proj",
            "shape": [4, 4],
            "weights": _weights(16, rng, 1.0),
            "content_version": 1,
            "group_granularity": 4,
        },
        {
            "layer_name": "encoder.block",
            "shape": [4, 3],
            "weights": _weights(12, rng, 1.6),
            "content_version": 1,
            "group_granularity": 4,
        },
        {
            "layer_name": "retrieval.head",
            "shape": [2, 4],
            "weights": _weights(8, rng, 0.8),
            "content_version": 1,
            "group_granularity": 4,
        },
    ]
    weights_path = root / "weights.jsonl"
    weights_path.write_text(
        "".join(json.dumps(layer, ensure_ascii=False) + "\n" for layer in layers),
        encoding="utf-8",
        newline="\n",
    )

    calibration = {
        "input_dim": 4,
        "output_dim": 3,
        "samples": [
            {
                "id": "cal-000",
                "input": [0.25, -0.5, 0.75, 1.0],
                "reference_output": [0.1, -0.2, 0.3],
            },
            {
                "id": "cal-001",
                "input": [-1.0, 0.5, 0.0, 0.25],
                "reference_output": [-0.4, 0.2, 0.0],
            },
            {
                "id": "cal-002",
                "input": [0.0, 1.25, -0.75, 0.5],
                "reference_output": [0.0, 0.7, -0.1],
            },
        ],
    }
    calibration_path = root / "calibration.json"
    calibration_path.write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    config: dict[str, Any] = {
        "quantization": {
            "scheme": "asymmetric",
            "bit_width": 8,
            "group_granularity": 4,
            "percentile_lower_permille": 100,
            "percentile_upper_permille": 900,
            "fractional_bits": 8,
            "quantization_levels": 1,
            "scale_effective_digits": 15,
        },
        "layers": {
            "embedding.proj": {"bit_width": 8, "fractional_bits": 8},
            "encoder.block": {"bit_width": 8, "fractional_bits": 8},
            "retrieval.head": {"bit_width": 8, "fractional_bits": 8},
        },
        "distillation": {
            "temperature": 2.0,
            "steps": 3,
            "soft_label_weight": 0.5,
            "tolerance": 0.02,
        },
        "scheduling": {
            "thread_pool_workers": 3,
            "global_calibration_point_budget": 64,
            "per_layer_calibration_quota": 24,
            "batch_quantization_point_budget": 8,
            "snapshot_commit_interval_batches": 2,
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )

    q1_lines = [
        {"layer_name": "embedding.proj", "compression_permille": 437},
        {"layer_name": "encoder.block", "compression_permille": 452},
        {"layer_name": "retrieval.head", "compression_permille": 468},
    ]
    q2_lines = [
        {
            "layer_name": "embedding.proj",
            "compression_permille": 425,
            "parameter_version": 2,
            "elapsed_ms": 17,
        },
        {
            "layer_name": "encoder.block",
            "compression_permille": 441,
            "parameter_version": 2,
            "elapsed_ms": 23,
        },
        {
            "layer_name": "retrieval.head",
            "compression_permille": 459,
            "parameter_version": 2,
            "elapsed_ms": 11,
        },
    ]
    q3_lines = [
        {
            "layer_name": "embedding.proj",
            "compression_permille": 421,
            "parameter_version": 3,
            "elapsed_ms": 9,
            "group_scale_digest": "sha256:embedding-group-scales-v3",
            "supersedes": ["q2.jsonl"],
        },
        {
            "layer_name": "encoder.block",
            "compression_permille": 438,
            "parameter_version": 3,
            "elapsed_ms": 14,
            "group_scale_digest": "sha256:encoder-group-scales-v3",
            "supersedes": ["q2.jsonl"],
        },
        {
            "layer_name": "retrieval.head",
            "compression_permille": 455,
            "parameter_version": 3,
            "elapsed_ms": 7,
            "group_scale_digest": "sha256:retrieval-group-scales-v3",
            "supersedes": [],
        },
    ]
    q1_path = receipts_dir / "q1.jsonl"
    q2_path = receipts_dir / "q2.jsonl"
    q3_path = receipts_dir / "q3.jsonl"
    for path, records in ((q1_path, q1_lines), (q2_path, q2_lines), (q3_path, q3_lines)):
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
            newline="\n",
        )

    orphan_path = receipts_dir / "orphan.jsonl"
    orphan_path.write_text(
        json.dumps({"layer_name": "missing.layer", "compression_permille": 500}) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    replacement = _weights(8, random.Random(SEED + 99), 0.9)
    commands_path = root / "commands.jsonl"
    commands_path.write_text(
        json.dumps(
            {
                "op": "replace_layer",
                "layer": "retrieval.head",
                "shape": [2, 4],
                "weights": replacement,
                "content_version": 2,
                "group_granularity": 4,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    faults_path = root / "faults.yaml"
    faults = {
        "injections": [
            {
                "id": "F01_NONFINITE",
                "kind": "nonfinite_weight",
                "layer": "embedding.proj",
            },
            {
                "id": "F02_GROUP_INDIVISIBLE",
                "kind": "group_indivisible",
                "layer": "encoder.block",
                "value": 5,
            },
            {
                "id": "F03_BIT_WIDTH",
                "kind": "bit_width_out_of_range",
                "layer": "retrieval.head",
                "value": 32,
            },
            {
                "id": "F04_CALIBRATION_DIM",
                "kind": "calibration_dimension_mismatch",
                "sample_index": 0,
            },
        ]
    }
    faults_path.write_text(
        yaml.safe_dump(faults, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )

    return {
        "root": root,
        "weights": weights_path,
        "calibration": calibration_path,
        "config": config_path,
        "receipts": receipts_dir,
        "q1": q1_path,
        "q2": q2_path,
        "q3": q3_path,
        "orphan_receipts": orphan_path,
        "commands": commands_path,
        "faults": faults_path,
    }
