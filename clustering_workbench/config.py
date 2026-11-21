"""YAML configuration loading, defaults, overrides, and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .codes import Failure
from .utils import deep_merge, hash_object, read_json


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "seed": 20261122,
    "paths": {
        "dataset": "samples/demo_dataset.yaml",
        "outputs": "outputs",
    },
    "data": {
        "include_columns": [],
        "expected_types": {},
        "target_columns": [],
        "sample_scale": 1.0,
        "sample_overrides": {},
        "max_rows": 1200,
    },
    "preprocessing": {
        "method": "standard",
        "train_fraction": 0.7,
        "drop_redundant": True,
        "epsilon": 1e-9,
        "quantile_points": 65,
    },
    "methods": {},
    "evaluation": {
        "metrics": ["silhouette", "calinski_harabasz", "density_log_likelihood"],
        "primary_metric": "silhouette",
        "density_reg_covar": 1e-6,
        "boundary_threshold": None,
    },
    "scheduler": {
        "global_iteration_budget": 240,
        "per_candidate_iteration_quota": 50,
        "cache_resident_blocks": 8,
        "block_size": 64,
        "max_workers": 4,
        "snapshot_commit_interval": 2,
        "algorithm_version": "det-v1",
    },
    "reuse": {
        "enabled": True,
        "salt": "diagnostic-workbench-v1",
    },
    "snapshot": {
        "enabled": True,
        "version": 1,
    },
    "injections": [],
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, list[Failure]]:
    config_path = Path(path).expanduser().resolve()
    raw = load_yaml(config_path)
    config = deep_merge(DEFAULT_CONFIG, raw)
    config = deep_merge(config, overrides or {})
    failures: list[Failure] = []

    scheduler = config.get("scheduler", {})
    try:
        workers = int(scheduler.get("max_workers", 1))
    except (TypeError, ValueError):
        workers = 0
    if workers <= 0:
        failures.append(
            Failure(
                code="CFG_CONCURRENCY_NONPOSITIVE",
                stage="config",
                item="scheduler.max_workers",
                message="scheduler.max_workers must be a positive integer; runtime falls back to 1.",
                details={"configured": scheduler.get("max_workers")},
            )
        )
        scheduler["max_workers"] = 1

    try:
        budget = int(scheduler.get("global_iteration_budget", 0))
    except (TypeError, ValueError):
        budget = 0
    if budget <= 0:
        failures.append(
            Failure(
                code="CFG_GLOBAL_BUDGET_NONPOSITIVE",
                stage="config",
                item="scheduler.global_iteration_budget",
                message="scheduler.global_iteration_budget must be positive.",
                details={"configured": scheduler.get("global_iteration_budget")},
            )
        )
    scheduler["global_iteration_budget"] = max(0, budget)

    try:
        quota = int(scheduler.get("per_candidate_iteration_quota", 0))
    except (TypeError, ValueError):
        quota = 0
    if quota <= 0:
        failures.append(
            Failure(
                code="CFG_PARAMETER_OUT_OF_RANGE",
                stage="config",
                item="scheduler.per_candidate_iteration_quota",
                message="per_candidate_iteration_quota must be positive; runtime falls back to 1.",
                details={"configured": scheduler.get("per_candidate_iteration_quota")},
            )
        )
        quota = 1
    scheduler["per_candidate_iteration_quota"] = quota

    block_size = max(1, int(scheduler.get("block_size", 64) or 64))
    cache_blocks = max(1, int(scheduler.get("cache_resident_blocks", 1) or 1))
    commit_interval = max(1, int(scheduler.get("snapshot_commit_interval", 1) or 1))
    scheduler["block_size"] = block_size
    scheduler["cache_resident_blocks"] = cache_blocks
    scheduler["snapshot_commit_interval"] = commit_interval
    config["scheduler"] = scheduler

    dataset_path = Path(config["paths"]["dataset"])
    if not dataset_path.is_absolute():
        dataset_path = (config_path.parent / dataset_path).resolve()
    output_path = Path(config["paths"]["outputs"])
    if not output_path.is_absolute():
        output_path = (config_path.parent / output_path).resolve()
    config["paths"]["dataset"] = str(dataset_path)
    config["paths"]["outputs"] = str(output_path)

    method = config.get("preprocessing", {}).get("method", "standard")
    if method not in {"standard", "quantile"}:
        failures.append(
            Failure(
                code="CFG_PARAMETER_OUT_OF_RANGE",
                stage="config",
                item="preprocessing.method",
                message="preprocessing.method must be 'standard' or 'quantile'.",
                details={"configured": method},
            )
        )
        config["preprocessing"]["method"] = "standard"

    return config, config_path, failures


def config_fingerprint(config: dict[str, Any]) -> str:
    """Identity of a complete run configuration.

    Runtime path values and timestamp-like fields are intentionally absent from a
    config file; callers can still pass explicit output directories separately.
    """
    identity = deepcopy(config)
    identity.pop("paths", None)
    return hash_object("cfg", identity)


def dataset_descriptor_path(config: dict[str, Any], config_path: Path) -> Path:
    path = Path(config["paths"]["dataset"])
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def injection_fingerprint(config: dict[str, Any]) -> str:
    return hash_object("inj", config.get("injections", []))


def load_latest_report(output_dir: str | Path) -> dict[str, Any] | None:
    return read_json(Path(output_dir) / "latest_report.json")
