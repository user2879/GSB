from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

from .canonical import atomic_write_text
from .config import DEFAULT_CONFIG, dump_config_yaml


DEMO_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "data": {
        "id_field": "id",
        "label_field": "label",
        "split_field": "split",
        "numeric_columns": ["age", "income"],
        "categorical_columns": ["region", "channel"],
        "categorical_domains": {
            "region": ["east", "west", "north", "south"],
            "channel": ["web", "store", "partner"],
        },
    },
    "features": {"bins": 4},
    "training": {
        "selected_candidate": 0,
        "candidates": [
            {
                "id": "balanced",
                "initial_weight_scale": 0.05,
                "initial_bias": 0.0,
                "l2": 0.001,
            },
            {
                "id": "regularized",
                "initial_weight_scale": 0.02,
                "initial_bias": 0.0,
                "l2": 0.01,
            },
        ],
        "batch_size": 4,
        "learning_rate": 0.1,
        "epochs": 4,
    },
    "quantization": {"fractional_bits": 12, "scale_level": 4},
    "concurrency": {
        "max_workers": 4,
        "max_workers_limit": 8,
        "batch_budget": 128,
    },
    "snapshot": {"commit_interval_batches": 1},
}


def make_sample(index: int, seed: int = 202604) -> dict[str, Any]:
    rng = random.Random(seed * 100003 + index)
    age = round(rng.uniform(18.0, 72.0), 3)
    income = round(rng.uniform(20.0, 210.0), 3)
    region = ["east", "west", "north", "south"][index % 4]
    channel = ["web", "store", "partner"][(index * 2 + 1) % 3]
    if index % 13 == 0:
        age = None
    if index % 17 == 0:
        channel = None
    score = (
        (0.0 if age is None else (age - 42.0) / 18.0)
        + (income - 100.0) / 80.0
        + {"east": 0.4, "west": -0.2, "north": 0.1, "south": -0.4}[region]
        + {"web": 0.25, "store": -0.15, "partner": 0.05}.get(channel, 0.0)
        + rng.uniform(-0.6, 0.6)
    )
    probability = 1.0 / (1.0 + math.exp(-score))
    label = 1 if probability >= 0.5 else 0
    split = "validation" if index % 5 == 0 else "train"
    return {
        "id": f"S{index + 1:04d}",
        "age": age,
        "income": income,
        "region": region,
        "channel": channel,
        "label": label,
        "split": split,
    }


def demo_edit_plan(seed: int = 202604) -> dict[str, Any]:
    appended_batch_1 = [make_sample(index, seed) for index in range(24, 32)]
    appended_batch_3 = [make_sample(index, seed) for index in range(32, 36)]
    return {
        "schema_version": 1,
        "seed": seed,
        "batches": [
            {
                "batch_id": "b001-append-correct",
                "operations": [
                    {"type": "append_samples", "samples": appended_batch_1},
                    {
                        "type": "correct_sample",
                        "sample_id": "S0007",
                        "patch": {"age": 41.5, "channel": "partner"},
                    },
                ],
            },
            {
                "batch_id": "b002-feature-bins",
                "operations": [
                    {"type": "update_feature_params", "patch": {"bins": 6}}
                ],
            },
            {
                "batch_id": "b003-model-hyperparams",
                "operations": [
                    {
                        "type": "append_samples",
                        "samples": appended_batch_3,
                    },
                    {
                        "type": "update_model_hyperparams",
                        "patch": {
                            "selected_candidate": "regularized",
                            "learning_rate": 0.08,
                            "batch_size": 5,
                        },
                    },
                ],
            },
            {
                "batch_id": "b004-quantization-and-label",
                "operations": [
                    {
                        "type": "update_quantization",
                        "patch": {"fractional_bits": 10, "scale_level": 8},
                    },
                    {
                        "type": "correct_sample",
                        "sample_id": "S0012",
                        "patch": {"label": 0},
                    },
                ],
            },
        ],
    }


def generate_demo_workspace(
    workspace: str | Path,
    *,
    seed: int = 202604,
    reset: bool = False,
) -> dict[str, Path]:
    root = Path(workspace)
    if reset and root.exists():
        shutil.rmtree(root)
    samples_dir = root / "samples"
    configs_dir = root / "configs"
    plans_dir = root / "plans"
    for directory in (samples_dir, configs_dir, plans_dir, root / "snapshots", root / "reports"):
        directory.mkdir(parents=True, exist_ok=True)
    samples = [make_sample(index, seed) for index in range(24)]
    sample_lines = [
        json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for sample in samples
    ]
    sample_path = samples_dir / "samples.jsonl"
    atomic_write_text(sample_path, "\n".join(sample_lines) + "\n")
    config_path = configs_dir / "pipeline.yaml"
    atomic_write_text(config_path, dump_config_yaml(DEMO_CONFIG))
    plan_path = plans_dir / "edits.json"
    atomic_write_text(
        plan_path,
        json.dumps(demo_edit_plan(seed), ensure_ascii=False, indent=2) + "\n",
    )
    return {
        "workspace": root,
        "samples": sample_path,
        "config": config_path,
        "plan": plan_path,
    }
