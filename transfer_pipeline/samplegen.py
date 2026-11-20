"""Fixed-seed sample/input generation without external downloads."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .canonical import atomic_write_text, canonical_json, pretty_json


CLASSES = ("alpha", "beta", "gamma", "delta")
LAYER_NAMES = ("layer_1", "layer_2", "layer_3")


def _write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, pretty_json(value))


def _encoder_layer(
    rng: np.random.Generator,
    name: str,
    version: str,
    input_dim: int,
    output_dim: int,
) -> dict[str, Any]:
    weights = rng.normal(0.0, math.sqrt(2.0 / input_dim), size=(input_dim, output_dim))
    bias = rng.normal(0.0, 0.05, size=output_dim)
    return {
        "name": name,
        "version": version,
        "weights": weights.round(12).tolist(),
        "bias": bias.round(12).tolist(),
    }


def generate_sample(root: str | Path, *, seed: int = 20261120, force: bool = True) -> dict[str, Any]:
    root = Path(root)
    if root.exists() and force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    feature_dim = 6
    class_centers = {
        label: rng.normal(0.0, 1.35, size=feature_dim)
        for label in CLASSES
    }

    # 160 samples provide generous, disjoint class-specific ranges. Some ranges
    # deliberately overlap between tasks so sample edits exercise precise
    # per-task invalidation rather than family-wide invalidation.
    features: dict[str, dict[str, Any]] = {}
    by_class: dict[str, list[str]] = {label: [] for label in CLASSES}
    for index in range(160):
        label = CLASSES[index % len(CLASSES)]
        sample_id = f"s{index + 1:04d}"
        vector = class_centers[label] + rng.normal(0.0, 0.62, size=feature_dim)
        features[sample_id] = {
            "sample_id": sample_id,
            "features": vector.round(12).tolist(),
            "label": label,
        }
        by_class[label].append(sample_id)

    def pick(label: str, start: int, stop: int) -> list[str]:
        return list(by_class[label][start:stop])

    tasks: list[dict[str, Any]] = [
        {
            "name": "vision_alpha",
            "classes": ["alpha", "beta"],
            "train_ids": pick("alpha", 0, 10) + pick("beta", 0, 8),
            "validation_ids": pick("alpha", 32, 35) + pick("beta", 32, 35),
            "target_layer": "layer_1",
            "target_layer_whitelist": ["layer_1", "layer_2", "layer_3"],
        },
        {
            "name": "vision_gamma",
            "classes": ["gamma", "delta"],
            "train_ids": pick("gamma", 0, 10) + pick("delta", 0, 8),
            "validation_ids": pick("gamma", 32, 35) + pick("delta", 32, 35),
            "target_layer": "layer_2",
            "target_layer_whitelist": ["layer_1", "layer_2", "layer_3"],
            "_injection_id": "INJ-006",
        },
        {
            "name": "signals_abg",
            "classes": ["alpha", "beta", "gamma"],
            # alpha[:6] and vision_alpha share the same real samples.
            "train_ids": pick("alpha", 0, 6) + pick("beta", 10, 15) + pick("gamma", 10, 17),
            "validation_ids": pick("alpha", 35, 37) + pick("beta", 35, 37) + pick("gamma", 35, 37),
            "target_layer": "layer_3",
            "target_layer_whitelist": ["layer_1", "layer_2", "layer_3"],
        },
        {
            "name": "signals_mix",
            "classes": ["alpha", "delta"],
            # delta[:6] and vision_gamma share real samples.
            "train_ids": pick("alpha", 10, 17) + pick("delta", 0, 6),
            "validation_ids": pick("alpha", 37, 40) + pick("delta", 35, 38),
            "target_layer": "layer_2",
            "target_layer_whitelist": ["layer_1", "layer_2", "layer_3"],
        },
    ]

    # The five required invalid input examples live in the real tasks.json.
    tasks.extend(
        [
            {
                "name": "inj_missing_sample",
                "classes": ["alpha", "beta"],
                "train_ids": pick("alpha", 20, 22) + ["missing-sample-404"],
                "validation_ids": pick("beta", 20, 22),
                "target_layer": "layer_1",
                "target_layer_whitelist": ["layer_1", "layer_2", "layer_3"],
                "_injection_id": "INJ-001",
            },
            {
                "name": "inj_duplicate_classes",
                "classes": ["alpha", "alpha"],
                "train_ids": pick("alpha", 22, 24),
                "validation_ids": pick("alpha", 24, 26),
                "target_layer": "layer_1",
                "target_layer_whitelist": ["layer_1", "layer_2", "layer_3"],
                "_injection_id": "INJ-002",
            },
            {
                "name": "inj_target_not_whitelisted",
                "classes": ["gamma", "delta"],
                "train_ids": pick("gamma", 20, 22),
                "validation_ids": pick("delta", 20, 22),
                "target_layer": "layer_2",
                "target_layer_whitelist": ["layer_1"],
                "_injection_id": "INJ-003",
            },
            {
                "name": "inj_zero_rank",
                "classes": ["alpha", "beta"],
                "train_ids": pick("alpha", 26, 29) + pick("beta", 26, 29),
                "validation_ids": pick("alpha", 29, 31) + pick("beta", 29, 31),
                "target_layer": "layer_1",
                "target_layer_whitelist": ["layer_1", "layer_2", "layer_3"],
                "adapter_rank": 0,
                "_injection_id": "INJ-004",
            },
            {
                "name": "inj_empty_validation",
                "classes": ["gamma", "delta"],
                "train_ids": pick("gamma", 22, 25) + pick("delta", 22, 25),
                "validation_ids": [],
                "target_layer": "layer_1",
                "target_layer_whitelist": ["layer_1", "layer_2", "layer_3"],
                "_injection_id": "INJ-005",
            },
        ]
    )
    tasks_document = {
        "schema_version": 1,
        "shared_feature_dim": feature_dim,
        "target_layer_whitelist": list(LAYER_NAMES),
        "tasks": tasks,
    }

    layer_1 = _encoder_layer(rng, "layer_1", "1.0.0", feature_dim, 9)
    layer_2 = _encoder_layer(rng, "layer_2", "1.0.0", 9, 7)
    layer_3 = _encoder_layer(rng, "layer_3", "1.0.0", 7, 5)
    encoder = {
        "schema_version": 1,
        "shared_feature_dim": feature_dim,
        "normalization": {
            "mean": [0.01 * (index + 1) for index in range(feature_dim)],
            "std": [0.95 + 0.03 * index for index in range(feature_dim)],
        },
        "layers": [layer_1, layer_2, layer_3],
    }

    config = {
        "activation": "tanh",
        "adapter_type": "low_rank",
        "rank": 2,
        "scale": 0.9,
        "learning_rate": 0.08,
        "epochs": 16,
        "batch_size": 4,
        "validation_metric": "macro_f1",
        "validation_threshold": 0.45,
        "max_concurrency": 3,
        "snapshot_interval": 2,
        "seed_offset": seed,
    }

    replacement_weights = rng.normal(0.0, math.sqrt(2.0 / 9), size=(9, 7)).round(12).tolist()
    replacement_bias = rng.normal(0.0, 0.05, size=7).round(12).tolist()
    replacement_alpha_vector = (
        class_centers["alpha"] + rng.normal(0.0, 0.58, size=feature_dim)
    ).round(12).tolist()
    replacement_gamma_vector = (
        class_centers["gamma"] + rng.normal(0.0, 0.58, size=feature_dim)
    ).round(12).tolist()
    audio_task = {
        "name": "audio_alpha",
        "classes": ["alpha", "gamma"],
        "train_ids": pick("alpha", 20, 26) + pick("gamma", 0, 4),
        "validation_ids": pick("alpha", 26, 29) + pick("gamma", 4, 7),
        "target_layer": "layer_1",
        "target_layer_whitelist": ["layer_1", "layer_2", "layer_3"],
    }
    edit_plan = {
        "schema_version": 1,
        "batches": [
            {
                "batch_id": "E1-mixed-invalidations",
                "operations": [
                    {
                        "op": "replace_sample_features",
                        "sample_id": "s0001",
                        "features": replacement_alpha_vector,
                        "label": "alpha",
                    },
                    {
                        "op": "replace_base_layer",
                        "layer": "layer_2",
                        "weights": replacement_weights,
                        "bias": replacement_bias,
                        "version": "2.0.0",
                    },
                    {"op": "add_task", "task": audio_task},
                    {"op": "delete_task", "name": "signals_mix"},
                    {
                        "op": "set_classes",
                        "name": "vision_alpha",
                        "classes": ["alpha", "beta", "epsilon"],
                    },
                ],
            },
            {
                "batch_id": "E2-adapter-hyperparameters",
                "operations": [
                    {
                        "op": "update_adapter_config",
                        "changes": {"rank": 3, "scale": 0.55},
                    }
                ],
            },
            {
                "batch_id": "E3-selective-task-refresh",
                "operations": [
                    {
                        "op": "set_classes",
                        "name": "signals_abg",
                        "classes": ["alpha", "beta", "gamma", "epsilon"],
                    },
                    {
                        "op": "replace_sample_features",
                        "sample_id": "s0003",
                        "features": replacement_gamma_vector,
                        "label": "gamma",
                    },
                ],
            },
        ],
    }

    injections = {
        "schema_version": 1,
        "injections": [
            {
                "id": "INJ-001",
                "kind": "missing_sample_id",
                "task": "inj_missing_sample",
                "path": "tasks.json#/tasks/4/train_ids/2",
                "business_code": "E_TASK_INVALID",
            },
            {
                "id": "INJ-002",
                "kind": "invalid_classes",
                "task": "inj_duplicate_classes",
                "path": "tasks.json#/tasks/5/classes",
                "business_code": "E_TASK_INVALID",
            },
            {
                "id": "INJ-003",
                "kind": "target_layer_not_whitelisted",
                "task": "inj_target_not_whitelisted",
                "path": "tasks.json#/tasks/6",
                "business_code": "E_TASK_INVALID",
            },
            {
                "id": "INJ-004",
                "kind": "zero_rank",
                "task": "inj_zero_rank",
                "path": "tasks.json#/tasks/7/adapter_rank",
                "business_code": "E_RANK_INVALID",
            },
            {
                "id": "INJ-005",
                "kind": "empty_validation",
                "task": "inj_empty_validation",
                "path": "tasks.json#/tasks/8/validation_ids",
                "business_code": "E_TASK_INVALID",
                "reason_code": "E_VALIDATION_EMPTY",
            },
            {
                "id": "INJ-006",
                "kind": "runtime_exception",
                "task": "vision_gamma",
                "phase": "training",
                "exception_type": "InjectedTrainingError",
                "business_code": "E_TRAINING_EXCEPTION",
                "path": "tasks.json#/tasks/1/_injection_id",
            },
        ],
    }

    _write_json(root / "tasks.json", tasks_document)
    feature_lines = "".join(
        canonical_json(features[sample_id]) + "\n"
        for sample_id in sorted(features)
    )
    atomic_write_text(root / "features.jsonl", feature_lines)
    _write_json(root / "encoder.json", encoder)
    atomic_write_text(root / "config.yaml", yaml.safe_dump(config, sort_keys=True))
    _write_json(root / "edit_plan.json", edit_plan)
    _write_json(root / "injections.json", injections)
    return {
        "root": str(root),
        "seed": seed,
        "sample_count": len(features),
        "task_count": len(tasks),
        "valid_task_count": 4,
        "injected_failure_task_count": 5,
        "runtime_failure_task_count": 1,
    }
