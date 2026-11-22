from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DATA_SPEC: dict[str, Any] = {
    "name": "test_panel",
    "rows": 72,
    "seed": 424242,
    "split": {"train_fraction": 0.7, "validation_fraction": 0.3},
    "target": {
        "name": "approved",
        "kind": "binary",
        "latent_weights": [1.1, -0.7],
        "noise_scale": 0.8,
        "offset": 0.05,
    },
    "columns": [
        {
            "name": "income",
            "kind": "numeric",
            "latent_weights": [1.0, 0.1],
            "noise_scale": 0.5,
            "mean": 50.0,
            "scale": 12.0,
            "missing_rate": 0.04,
        },
        {
            "name": "tenure",
            "kind": "numeric",
            "latent_weights": [0.2, 1.0],
            "noise_scale": 0.6,
            "mean": 6.0,
            "scale": 2.0,
            "missing_rate": 0.05,
        },
        {
            "name": "utilization",
            "kind": "numeric",
            "latent_weights": [-0.7, 0.4],
            "noise_scale": 0.6,
            "mean": 0.35,
            "scale": 0.15,
            "missing_rate": 0.06,
        },
        {
            "name": "bureau_score",
            "kind": "numeric",
            "latent_weights": [1.0, -0.3],
            "noise_scale": 0.4,
            "mean": 680.0,
            "scale": 40.0,
            "missing_rate": 0.02,
        },
        {
            "name": "region",
            "kind": "categorical",
            "latent_weights": [0.3, 0.7],
            "noise_scale": 0.2,
            "categories": ["north", "south", "west"],
            "drop_first": True,
            "missing_rate": 0.03,
        },
        {
            "name": "channel",
            "kind": "categorical",
            "latent_weights": [0.6, -0.4],
            "noise_scale": 0.2,
            "categories": ["partner", "direct", "online"],
            "drop_first": True,
            "missing_rate": 0.02,
        },
        {
            "name": "income_copy",
            "kind": "redundant",
            "source": "income",
            "multiplier": 1.0,
            "offset": 0.0,
            "noise_scale": 0.0,
            "missing_rate": 0.0,
        },
    ],
}


BASE_CONFIG: dict[str, Any] = {
    "name": "unit_test_experiment",
    "seed": 424242,
    "exchange_version": "v2",
    "conversion_tolerance": 0.001,
    "training": {
        "epochs": 6,
        "learning_rate": 0.075,
        "batch_size": 8,
        "tolerance": 1e-12,
        "l1_zero_subgradient": 0.0,
        "snapshot_every_batches": 4,
    },
    "budget": {"global_epochs": 100, "per_model_epochs": 20},
    "runtime": {"max_concurrency": 3, "concurrency_limit": 8},
    "snapshot": {"strict_resume": False, "keep_generations": 4},
    "models": [
        {
            "name": "ridge_core",
            "scheme": "l2",
            "penalty": 0.12,
            "features": ["income", "tenure", "utilization", "bureau_score", "region"],
        },
        {
            "name": "lasso_compact",
            "scheme": "l1",
            "penalty": 0.025,
            "features": ["income", "tenure", "bureau_score", "region"],
        },
        {
            "name": "elastic_balanced",
            "scheme": "elastic_net",
            "penalty": 0.04,
            "mix": 0.55,
            "features": [
                "income",
                "tenure",
                "utilization",
                "bureau_score",
                "region",
                "channel",
                "income_copy",
            ],
        },
    ],
}


class LabFixture:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_spec = copy.deepcopy(DATA_SPEC)
        self.config = copy.deepcopy(BASE_CONFIG)
        self.data_path = self.root / "data.yaml"
        self.config_path = self.root / "experiment.yaml"
        self._write_data()
        self.config["data"] = str(self.data_path)
        self.write_config()

    def _write_data(self) -> None:
        self.data_path.write_text(yaml.safe_dump(self.data_spec, sort_keys=False), encoding="utf-8")

    def write_config(self, name: str = "experiment.yaml") -> Path:
        path = self.root / name
        path.write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")
        return path

    def variant_config(self, name: str, *, column_variant: int | None = None) -> Path:
        config = copy.deepcopy(self.config)
        data_spec = copy.deepcopy(self.data_spec)
        if column_variant is not None:
            for column in data_spec["columns"]:
                if column["name"] == "utilization":
                    column["variant"] = column_variant
            data_path = self.root / f"data-{name}.yaml"
            data_path.write_text(yaml.safe_dump(data_spec, sort_keys=False), encoding="utf-8")
            config["data"] = str(data_path)
        path = self.root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path


def model_by_name(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(model for model in report["models"] if model["name"] == name)
