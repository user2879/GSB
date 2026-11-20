"""Ensemble scoring, validation metrics, loss trajectory, and attribution."""

from __future__ import annotations

import math
from typing import Any, Iterable

from .config import ExperimentConfig
from .data import PreparedData
from .fingerprint import fingerprint, rounded, rounded_tree
from .tree import leaf_value, path_contributions


def tree_product_fingerprint(trees: Iterable[dict[str, Any]]) -> str:
    entries = [
        {
            "tree_index": int(tree["tree_index"]),
            "status": tree["status"],
            "key": tree["key"],
            "artifact_fingerprint": tree["artifact_fingerprint"],
        }
        for tree in sorted(trees, key=lambda item: int(item["tree_index"]))
    ]
    return fingerprint({"schema": "tree-product/v1", "trees": entries})


def ensemble_cache_key(
    config: ExperimentConfig,
    data: PreparedData,
    trees: list[dict[str, Any]],
) -> str:
    product_fingerprint = tree_product_fingerprint(trees)
    return fingerprint(
        {
            "stage": "ensemble/v1",
            "tree_product_fingerprint": product_fingerprint,
            "data_fingerprint": data.data_fingerprint,
            "learning_rate": config.learning_rate,
            "ensemble_kind": config.ensemble_kind,
            "base_score": rounded(data.base_score, config.precision_decimals),
            "precision_decimals": config.precision_decimals,
        }
    )


def contributions_cache_key(ensemble: dict[str, Any], trees: list[dict[str, Any]]) -> str:
    return fingerprint(
        {
            "stage": "contributions/v1",
            "ensemble_fingerprint": ensemble["fingerprint"],
            "tree_product_fingerprint": ensemble["tree_product_fingerprint"],
            "tree_artifacts": [
                {
                    "tree_index": tree["tree_index"],
                    "artifact_fingerprint": tree["artifact_fingerprint"],
                }
                for tree in sorted(trees, key=lambda item: int(item["tree_index"]))
            ],
        }
    )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _log_loss(raw_scores: list[float], labels: list[int]) -> float:
    total = 0.0
    for raw, label in zip(raw_scores, labels):
        total += max(raw, 0.0) + math.log1p(math.exp(-abs(raw))) - float(label) * raw
    return total / float(len(raw_scores))


def _accuracy(raw_scores: list[float], labels: list[int]) -> float:
    correct = sum(1 for raw, label in zip(raw_scores, labels) if (_sigmoid(raw) >= 0.5) == bool(label))
    return correct / float(len(raw_scores))


def _metrics(raw_train: list[float], raw_validation: list[float], data: PreparedData, decimals: int) -> dict[str, Any]:
    train_labels = [data.target[index] for index in data.train_indices]
    validation_labels = [data.target[index] for index in data.validation_indices]
    return {
        "train": {
            "rows": len(raw_train),
            "accuracy": rounded(_accuracy(raw_train, train_labels), decimals),
            "log_loss": rounded(_log_loss(raw_train, train_labels), decimals),
        },
        "validation": {
            "rows": len(raw_validation),
            "accuracy": rounded(_accuracy(raw_validation, validation_labels), decimals),
            "log_loss": rounded(_log_loss(raw_validation, validation_labels), decimals),
        },
    }


def _applied_tree_weights(config: ExperimentConfig, applied_indices: list[int]) -> dict[int, float]:
    if not applied_indices:
        return {}
    if config.ensemble_kind == "random_forest":
        weight = config.learning_rate / float(len(applied_indices))
    else:
        weight = config.learning_rate
    return {index: weight for index in applied_indices}


def compute_ensemble(
    config: ExperimentConfig,
    data: PreparedData,
    binning: dict[str, Any],
    trees: list[dict[str, Any]],
    key: str | None = None,
) -> dict[str, Any]:
    decimals = config.precision_decimals
    trees_by_index = {int(tree["tree_index"]): tree for tree in trees}
    applied_indices = sorted(index for index, tree in trees_by_index.items() if tree["status"] == "ok")
    failed_indices = sorted(index for index, tree in trees_by_index.items() if tree["status"] != "ok")
    weights = _applied_tree_weights(config, applied_indices)
    feature_names = list(data.feature_names)

    leaves_by_tree: dict[int, list[float]] = {}
    for tree_index in applied_indices:
        tree = trees_by_index[tree_index]
        leaves: list[float] = []
        for row_index in range(data.row_count):
            bins_by_feature = {
                feature: int(binning["columns"][feature]["bin_indices"][row_index])
                for feature in feature_names
            }
            leaves.append(leaf_value(tree, bins_by_feature))
        leaves_by_tree[tree_index] = leaves

    raw_all = [float(data.base_score) for _ in range(data.row_count)]
    for tree_index in applied_indices:
        weight = weights[tree_index]
        leaves = leaves_by_tree[tree_index]
        for row_index in range(data.row_count):
            raw_all[row_index] += weight * leaves[row_index]

    train_indices = list(data.train_indices)
    validation_indices = list(data.validation_indices)

    # Build the trajectory from the same deterministic tree order. A failed tree keeps
    # the previous loss and is explicitly marked as not applied.
    trajectory_raw = [float(data.base_score) for _ in range(data.row_count)]
    loss_trajectory: list[dict[str, Any]] = []
    applied_count = 0
    for tree_index in sorted(trees_by_index):
        tree = trees_by_index[tree_index]
        if tree["status"] != "ok":
            raw_train = [trajectory_raw[index] for index in train_indices]
            raw_validation = [trajectory_raw[index] for index in validation_indices]
            train_labels = [data.target[index] for index in train_indices]
            validation_labels = [data.target[index] for index in validation_indices]
            loss_trajectory.append(
                {
                    "tree_index": tree_index,
                    "applied": False,
                    "applied_tree_count": applied_count,
                    "train_log_loss": rounded(_log_loss(raw_train, train_labels), decimals),
                    "validation_log_loss": rounded(_log_loss(raw_validation, validation_labels), decimals),
                }
            )
            continue
        applied_count += 1
        weight = weights[tree_index]
        leaves = leaves_by_tree[tree_index]
        for row_index in range(data.row_count):
            trajectory_raw[row_index] += weight * leaves[row_index]
        raw_train = [trajectory_raw[index] for index in train_indices]
        raw_validation = [trajectory_raw[index] for index in validation_indices]
        train_labels = [data.target[index] for index in train_indices]
        validation_labels = [data.target[index] for index in validation_indices]
        loss_trajectory.append(
            {
                "tree_index": tree_index,
                "applied": True,
                "applied_tree_count": applied_count,
                "train_log_loss": rounded(_log_loss(raw_train, train_labels), decimals),
                "validation_log_loss": rounded(_log_loss(raw_validation, validation_labels), decimals),
            }
        )

    raw_train = [raw_all[index] for index in train_indices]
    raw_validation = [raw_all[index] for index in validation_indices]
    metrics = _metrics(raw_train, raw_validation, data, decimals)
    validation_set = set(validation_indices)
    predictions: list[dict[str, Any]] = []
    for row_index in range(data.row_count):
        raw = raw_all[row_index]
        probability = _sigmoid(raw)
        predictions.append(
            {
                "row_id": data.row_ids[row_index],
                "split": "validation" if row_index in validation_set else "train",
                "label": data.target[row_index],
                "raw_score": rounded(raw, decimals),
                "probability": rounded(probability, decimals),
            }
        )

    body: dict[str, Any] = {
        "schema": "ensemble/v1",
        "key": key or ensemble_cache_key(config, data, trees),
        "tree_product_fingerprint": tree_product_fingerprint(trees),
        "parameters": {
            "learning_rate": config.learning_rate,
            "ensemble_kind": config.ensemble_kind,
            "base_score": rounded(data.base_score, decimals),
            "tree_weights": {str(index): rounded(weight, decimals) for index, weight in weights.items()},
        },
        "applied_trees": applied_indices,
        "failed_trees": failed_indices,
        "loss_trajectory": loss_trajectory,
        "metrics": metrics,
        "predictions": predictions,
    }
    body = rounded_tree(body, decimals)
    body["fingerprint"] = fingerprint({"schema": "ensemble-product/v1", "body": body})
    return body


def compute_contributions(
    config: ExperimentConfig,
    data: PreparedData,
    binning: dict[str, Any],
    trees: list[dict[str, Any]],
    ensemble: dict[str, Any],
    key: str | None = None,
) -> dict[str, Any]:
    decimals = config.precision_decimals
    feature_names = list(data.feature_names)
    attribution_features = feature_names + ["__bias__"]
    trees_by_index = {int(tree["tree_index"]): tree for tree in trees}
    applied_indices = [int(index) for index in ensemble["applied_trees"]]
    weights = {int(index): float(weight) for index, weight in ensemble["parameters"]["tree_weights"].items()}

    samples: list[dict[str, Any]] = []
    feature_totals = {feature: [] for feature in attribution_features}
    tree_totals: dict[int, dict[str, list[float]]] = {
        index: {feature: [] for feature in attribution_features} for index in applied_indices
    }
    raw_by_row_id = {
        int(prediction["row_id"]): float(prediction["raw_score"])
        for prediction in ensemble["predictions"]
    }

    for row_index in range(data.row_count):
        bins_by_feature = {
            feature: int(binning["columns"][feature]["bin_indices"][row_index])
            for feature in feature_names
        }
        by_tree: dict[str, Any] = {}
        by_feature = {feature: 0.0 for feature in attribution_features}
        for tree_index in applied_indices:
            tree = trees_by_index[tree_index]
            leaf_contribution, contributions = path_contributions(
                tree,
                bins_by_feature,
                feature_names,
                weights[tree_index],
                decimals,
            )
            contributions = {feature: rounded(contributions.get(feature, 0.0), decimals) for feature in attribution_features}
            by_tree[str(tree_index)] = {
                "leaf_contribution": leaf_contribution,
                "features": contributions,
            }
            for feature, value in contributions.items():
                by_feature[feature] = rounded(by_feature.get(feature, 0.0) + value, decimals)
                feature_totals[feature].append(value)
                tree_totals[tree_index][feature].append(value)
        samples.append(
            {
                "row_id": data.row_ids[row_index],
                "base_score": rounded(float(data.base_score), decimals),
                "raw_score": rounded(raw_by_row_id[data.row_ids[row_index]], decimals),
                "by_tree": by_tree,
                "by_feature": by_feature,
            }
        )

    feature_summary: dict[str, Any] = {}
    for feature in attribution_features:
        values = feature_totals[feature]
        feature_summary[feature] = {
            "total": rounded(math.fsum(values), decimals) if values else 0.0,
            "absolute_total": rounded(math.fsum(abs(value) for value in values), decimals) if values else 0.0,
            "by_tree": {
                str(tree_index): rounded(math.fsum(tree_totals[tree_index][feature]), decimals)
                for tree_index in applied_indices
            },
        }

    tree_summary: dict[str, Any] = {}
    for tree_index in applied_indices:
        tree_summary[str(tree_index)] = {
            "total": rounded(
                math.fsum(value for values in tree_totals[tree_index].values() for value in values),
                decimals,
            ),
            "by_feature": {
                feature: rounded(math.fsum(tree_totals[tree_index][feature]), decimals)
                for feature in attribution_features
            },
            "rows": data.row_count,
        }

    body: dict[str, Any] = {
        "schema": "contributions/v1",
        "key": key or contributions_cache_key(ensemble, trees),
        "ensemble_fingerprint": ensemble["fingerprint"],
        "tree_product_fingerprint": ensemble["tree_product_fingerprint"],
        "feature_names": attribution_features,
        "samples": samples,
        "summary": {
            "by_feature": feature_summary,
            "by_tree": tree_summary,
        },
    }
    body = rounded_tree(body, decimals)
    body["fingerprint"] = fingerprint({"schema": "contributions-product/v1", "body": body})
    return body
