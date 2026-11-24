from __future__ import annotations

from typing import Any

import pytest

from cluster_workbench.config import normalize_config
from cluster_workbench.data import normalize_data_description


@pytest.fixture
def base_data() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "seed": 31415,
        "rows": 54,
        "columns": [
            {
                "name": "n1",
                "type": "numeric",
                "distribution": "mixture",
                "components": [
                    {"weight": 1.0, "mean": -3.0, "std": 0.32},
                    {"weight": 1.0, "mean": 0.0, "std": 0.35},
                    {"weight": 1.0, "mean": 3.0, "std": 0.34},
                ],
                "missing_rate": 0.06,
            },
            {
                "name": "n2",
                "type": "numeric",
                "source": "n1",
                "multiplier": 1.0,
                "offset": 0.45,
                "missing_rate": 0.03,
            },
            {
                "name": "n3",
                "type": "numeric",
                "distribution": "uniform",
                "low": -1.0,
                "high": 1.0,
                "missing_rate": 0.0,
            },
            {
                "name": "c1",
                "type": "categorical",
                "categories": ["a", "b", "c"],
                "weights": [0.4, 0.35, 0.25],
                "missing_rate": 0.02,
            },
            {
                "name": "ccos",
                "type": "categorical",
                "categories": ["k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7"],
                "missing_rate": 0.04,
            },
            {
                "name": "redundant_n1",
                "type": "numeric",
                "source": "n1",
                "multiplier": 2.0,
                "offset": 0.5,
                "missing_rate": 0.02,
            },
        ],
    }


@pytest.fixture
def base_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "seed": 2718,
        "preprocessing": {
            "method": "standardize",
            "train_fraction": 0.6,
            "missing_fill": "median",
            "quantile_tie_policy": "average_rank",
        },
        "evaluation": {
            "metrics": ["silhouette", "calinski_harabasz", "density_log_likelihood"],
            "density_regularization": 1e-6,
        },
        "methods": {
            "kmeans": {"max_iter": 20, "tolerance": 1e-8},
            "dbscan": {"metric": "euclidean"},
            "gmm": {
                "max_iter": 30,
                "tolerance": 1e-4,
                "covariance_regularization": 1e-5,
                "init": "farthest",
            },
        },
        "scheduler": {
            "global_iteration_budget": 500,
            "per_candidate_iteration_quota": 50,
            "concurrency": 3,
            "interaction_latency_budget_ms": 500,
            "snapshot_commit_interval": 2,
        },
        "candidates": [],
        "injections": [],
    }


def candidate(
    identifier: str,
    method: str = "kmeans",
    *,
    params: dict[str, Any] | None = None,
    features: list[str] | None = None,
    train_fraction: float = 0.6,
) -> dict[str, Any]:
    if params is None:
        if method == "kmeans":
            params = {"n_clusters": 3}
        elif method == "dbscan":
            params = {"eps": 1.2, "min_samples": 4}
        else:
            params = {"n_components": 3, "covariance_type": "diag"}
    return {
        "id": identifier,
        "method": method,
        "params": params,
        "feature_columns": features or ["n1", "n2", "n3", "c1", "ccos", "redundant_n1"],
        "train_fraction": train_fraction,
    }


def normalized_pair(
    config: dict[str, Any], data: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_config, config_issues = normalize_config(config)
    assert not config_issues
    normalized_data, data_issues = normalize_data_description(data)
    assert not data_issues
    return normalized_config, normalized_data
