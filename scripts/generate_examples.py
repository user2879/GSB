"""Generate the deterministic local example workspace.

The generated data is intentionally small, CPU-only, and independent of any
network account. ``examples/features.json`` is the source of truth afterwards;
this script is provided only so the fixture can be reproduced exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SEED = 20261118


def write(name: str, value: object) -> None:
    path = EXAMPLES / name
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    feature_dim = 8

    task_shapes = [
        ("sentiment", ["negative", "positive"], [10, 10], [7, 7]),
        ("topic", ["finance", "sports", "tech"], [10, 10, 10], [7, 7, 7]),
        ("intent", ["book", "cancel", "info"], [8, 8, 8], [5, 5, 5]),
        ("mood", ["low", "high", "mid"], [6, 6, 0], [4, 4, 0]),
        ("urgency", ["critical", "normal"], [12, 12], [8, 8]),
    ]

    samples: dict[str, dict[str, object]] = {}
    tasks: list[dict[str, object]] = []
    sample_number = 1
    for task_name, classes, per_class, per_class_train in task_shapes:
        train_ids: list[str] = []
        val_ids: list[str] = []
        for class_index, class_name in enumerate(classes):
            count = per_class[class_index]
            if count == 0:
                continue
            center = rng.normal(0.0, 2.0, size=feature_dim)
            center[class_index % feature_dim] += 2.0
            for index in range(count):
                features = center + rng.normal(0.0, 0.16, size=feature_dim)
                sample_id = f"s{sample_number:03d}"
                sample_number += 1
                samples[sample_id] = {
                    "features": [float(value) for value in features],
                    "label": class_name,
                }
                if index < per_class_train[class_index]:
                    train_ids.append(sample_id)
                else:
                    val_ids.append(sample_id)
        tasks.append(
            {
                "name": task_name,
                "classes": list(classes),
                "train_ids": train_ids,
                "val_ids": val_ids,
            }
        )

    write(
        "features.json",
        {
            "seed": SEED,
            "feature_dim": feature_dim,
            "samples": samples,
        },
    )
    task_document = {
        "shared_feature_dim": feature_dim,
        "tasks": tasks[:4],
    }
    write("tasks.json", task_document)

    all_vectors = np.asarray(
        [samples[sample_id]["features"] for sample_id in sorted(samples)],
        dtype=np.float64,
    )
    mean = all_vectors.mean(axis=0)
    std = all_vectors.std(axis=0)
    std[std < 1e-6] = 1.0
    weights = rng.normal(0.0, 1.0 / np.sqrt(feature_dim), size=(6, feature_dim))
    bias = rng.normal(0.0, 0.05, size=6)
    write(
        "encoder.json",
        {
            "input_dim": feature_dim,
            "output_dim": 6,
            "weights": weights.tolist(),
            "bias": bias.tolist(),
            "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        },
    )
    write(
        "adapter_config.json",
        {
            "adapter_type": "linear_probe",
            "rank": 2,
            "learning_rate": 0.40,
            "epochs": 160,
            "batch_size": 8,
            "validation_threshold": 0.70,
            "concurrency": 4,
            "snapshot_commit_interval": 2,
            "convergence_tolerance": 1e-12,
            "lora_scale": 0.20,
            "task_overrides": {
                "intent": {
                    "adapter_type": "low_rank",
                    "rank": 2,
                    "learning_rate": 0.20,
                }
            },
        },
    )
    urgency_task = tasks[4]
    sentiment_val_replacements = sorted(tasks[0]["val_ids"])
    sentiment_train_replacements = sorted(tasks[0]["train_ids"])
    val_sample_id = sentiment_val_replacements[2]
    train_sample_id = sentiment_train_replacements[3]
    val_features = list(samples[val_sample_id]["features"])
    val_features[0] += 0.19
    val_features[3] -= 0.11
    train_features = list(samples[train_sample_id]["features"])
    train_features[1] -= 0.17
    train_features[5] += 0.09
    write(
        "edits.json",
        [
            {"op": "add_task", "task": urgency_task},
            {"op": "delete_task", "name": "topic"},
            {
                "op": "set_classes",
                "name": "mood",
                "classes": ["high", "low"],
            },
            {
                "op": "replace_sample",
                "sample_id": val_sample_id,
                "features": val_features,
                "label": "negative",
            },
            {
                "op": "replace_sample",
                "sample_id": train_sample_id,
                "features": train_features,
                "label": "negative",
            },
            {
                "op": "set_adapter_params",
                "name": "sentiment",
                "changes": {"learning_rate": 0.22},
            },
        ],
    )
    write(
        "replace_sample.json",
        {
            "sample_id": val_sample_id,
            "features": val_features,
            "label": "negative",
        },
    )
    write(
        "injections.json",
        {
            "injections": [
                {
                    "id": "INJ-001",
                    "type": "missing_sample",
                    "task": "sentiment",
                    "sample_id": "missing-999",
                },
                {
                    "id": "INJ-002",
                    "type": "invalid_classes",
                    "task": "topic",
                    "classes": ["tech", "tech"],
                },
                {"id": "INJ-003", "type": "zero_rank", "task": "intent"},
                {"id": "INJ-004", "type": "negative_lr", "task": "mood"},
            ]
        },
    )


if __name__ == "__main__":
    main()
