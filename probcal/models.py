from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .util import (
    fingerprint,
    quantize_probability,
)


def build_model_stage(
    data_artifact: dict[str, Any],
    model_config: list[dict[str, Any]],
    fixed: dict[str, int],
    model_key: str,
    max_workers: int,
) -> dict[str, Any]:
    tasks: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for model in model_config:
        for params in model["candidates"]:
            params_hash = fingerprint(params)[:16]
            stable_key = (model["name"], 0, params_hash)
            task = {
                "model_name": model["name"],
                "model_type": model["type"],
                "params": params,
                "params_hash": params_hash,
                "fold_id": 0,
                "stable_key": stable_key,
            }
            tasks.append((stable_key, task))
    tasks.sort(key=lambda item: item[0])
    results = _run_tasks(tasks, max_workers, lambda item: _train_candidate(data_artifact, item, fixed))
    results.sort(
        key=lambda result: (
            result["model_name"],
            result["fold_id"],
            result["params_hash"],
        )
    )
    return {
        "model_key": model_key,
        "data_key": data_artifact["data_key"],
        "models": results,
        "execution": {
            "task_order_policy": "sorted by (model_name, fold_id, params_hash)",
            "prediction_contract": "validation labels are not passed to any prediction function",
        },
    }


def _run_tasks(tasks: list[tuple[Any, Any]], max_workers: int, function: Any) -> list[Any]:
    if not tasks:
        return []
    if max_workers <= 1 or len(tasks) == 1:
        return [function(task) for _, task in tasks]
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="probcal") as pool:
        futures = {pool.submit(function, task): stable_key for stable_key, task in tasks}
        outputs_by_key = {}
        for future, stable_key in futures.items():
            outputs_by_key[stable_key] = future.result()
    return [outputs_by_key[stable_key] for stable_key, _ in tasks]


def _train_candidate(
    data_artifact: dict[str, Any], task: dict[str, Any], fixed: dict[str, int]
) -> dict[str, Any]:
    train = data_artifact["train"]
    validation = data_artifact["validation"]
    model_type = task["model_type"]
    params = task["params"]
    if model_type == "logistic_regression":
        trained = _fit_logistic(train, params)
        train_probabilities = _predict_logistic(trained, train["features"], fixed)
        validation_probabilities = _predict_logistic(trained, validation["features"], fixed)
        model_payload = {
            "standardization": trained["standardization"],
            "weights": [round(float(value), 12) for value in trained["weights"]],
            "bias": round(float(trained["bias"]), 12),
        }
    elif model_type == "decision_tree":
        trained = _fit_tree(train, params)
        train_probabilities = _predict_tree_fixed(trained, train["features"], fixed)
        validation_probabilities = _predict_tree_fixed(trained, validation["features"], fixed)
        model_payload = {
            "nodes": trained["nodes"],
            "max_depth": int(params["max_depth"]),
        }
    else:
        raise ValueError(f"unsupported model type: {model_type}")

    model_id = f"{task['model_name']}-{task['params_hash']}"
    return {
        "model_name": task["model_name"],
        "model_id": model_id,
        "model_type": model_type,
        "fold_id": int(task["fold_id"]),
        "params": params,
        "params_hash": task["params_hash"],
        "stable_key": list(task["stable_key"]),
        "loss_trajectory": trained["loss_trajectory"],
        "model": model_payload,
        "raw_probabilities": {
            "train": {
                "sample_ids": list(train["sample_ids"]),
                "probabilities_fixed": train_probabilities,
            },
            "validation": {
                "sample_ids": list(validation["sample_ids"]),
                "probabilities_fixed": validation_probabilities,
            },
        },
        "prediction_audit": {
            "train_labels_fit_model": True,
            "validation_labels_read": False,
            "prediction_input": "validation features only",
        },
    }


def _fit_logistic(train: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    features = np.asarray(train["features"], dtype=np.float64)
    labels = np.asarray(train["labels"], dtype=np.float64)
    numeric_count = _numeric_feature_count(train)
    means = features[:, :numeric_count].mean(axis=0) if numeric_count else np.zeros(0)
    stds = features[:, :numeric_count].std(axis=0) if numeric_count else np.zeros(0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    standardized = features.copy()
    if numeric_count:
        standardized[:, :numeric_count] = (features[:, :numeric_count] - means) / stds
    weights = np.zeros(standardized.shape[1], dtype=np.float64)
    bias = 0.0
    learning_rate = float(params["learning_rate"])
    l2 = float(params.get("l2", 0.0))
    batch_size = int(params["batch_size"])
    epochs = int(params["epochs"])
    losses: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        for start in range(0, len(labels), batch_size):
            stop = min(len(labels), start + batch_size)
            batch_x = standardized[start:stop]
            batch_y = labels[start:stop]
            logits = batch_x @ weights + bias
            probabilities = _sigmoid_array(logits)
            residual = probabilities - batch_y
            gradient = batch_x.T @ residual / (stop - start)
            gradient += l2 * weights
            weights -= learning_rate * gradient
            bias -= learning_rate * float(residual.mean())
        full_logits = standardized @ weights + bias
        full_probabilities = _sigmoid_array(full_logits)
        loss = _log_loss(labels, full_probabilities)
        losses.append({"iteration": epoch, "loss": round(loss, 12)})
    return {
        "weights": weights,
        "bias": bias,
        "means": means,
        "stds": stds,
        "numeric_count": numeric_count,
        "loss_trajectory": losses,
        "standardization": {
            "means": [round(float(value), 12) for value in means],
            "stds": [round(float(value), 12) for value in stds],
        },
    }


def _fit_tree(train: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    features = np.asarray(train["features"], dtype=np.float64)
    labels = np.asarray(train["labels"], dtype=np.int64)
    max_depth = int(params["max_depth"])
    min_leaf = int(params["min_samples_leaf"])
    min_gain = float(params.get("min_gain", 0.0))
    nodes: dict[int, dict[str, Any]] = {
        0: _make_node(0, 0, list(range(len(labels))), labels)
    }
    frontier = [0]
    next_node_id = 1
    losses: list[dict[str, Any]] = []
    for depth in range(max_depth):
        next_frontier: list[int] = []
        for node_id in frontier:
            node = nodes[node_id]
            split = _best_split(features, labels, node["indices"], min_leaf, min_gain)
            if split is None:
                continue
            feature, threshold, left_indices, right_indices = split
            node["feature"] = feature
            node["threshold"] = round(float(threshold), 12)
            left_id = next_node_id
            right_id = next_node_id + 1
            next_node_id += 2
            nodes[left_id] = _make_node(left_id, depth + 1, left_indices, labels)
            nodes[right_id] = _make_node(right_id, depth + 1, right_indices, labels)
            node["left"] = left_id
            node["right"] = right_id
            next_frontier.extend([left_id, right_id])
        if not next_frontier:
            break
        frontier = next_frontier
        probabilities = _tree_probabilities(nodes, 0, features)
        losses.append({"iteration": depth + 1, "loss": round(_log_loss(labels, probabilities), 12)})
    serialized_nodes = []
    for node_id in sorted(nodes):
        node = nodes[node_id]
        serialized_nodes.append(
            {
                "node_id": node_id,
                "depth": node["depth"],
                "count": node["count"],
                "positive_count": node["positive_count"],
                "probability": round(float(node["probability"]), 12),
                "feature": node["feature"],
                "threshold": node["threshold"],
                "left": node["left"],
                "right": node["right"],
            }
        )
    return {
        "nodes": serialized_nodes,
        "node_index": {node["node_id"]: node for node in serialized_nodes},
        "loss_trajectory": losses,
    }


def _make_node(
    node_id: int, depth: int, indices: list[int], labels: np.ndarray
) -> dict[str, Any]:
    indices = sorted(indices)
    count = len(indices)
    positive_count = int(labels[indices].sum()) if count else 0
    probability = (positive_count + 1.0) / (count + 2.0)
    return {
        "node_id": node_id,
        "depth": depth,
        "indices": indices,
        "count": count,
        "positive_count": positive_count,
        "probability": probability,
        "feature": None,
        "threshold": None,
        "left": None,
        "right": None,
    }


def _best_split(
    features: np.ndarray,
    labels: np.ndarray,
    indices: list[int],
    min_leaf: int,
    min_gain: float,
) -> tuple[int, float, list[int], list[int]] | None:
    if len(indices) < 2 * min_leaf:
        return None
    parent = _gini(labels[indices])
    best: tuple[int, float, list[int], list[int]] | None = None
    best_gain = min_gain
    for feature_index in range(features.shape[1]):
        pairs = sorted((float(features[index, feature_index]), index) for index in indices)
        for position in range(len(pairs) - 1):
            value, _ = pairs[position]
            next_value, _ = pairs[position + 1]
            if value == next_value:
                continue
            threshold = (value + next_value) / 2.0
            left = [index for row_value, index in pairs if row_value <= threshold]
            if len(left) < min_leaf or len(indices) - len(left) < min_leaf:
                continue
            left_set = set(left)
            right = [index for index in indices if index not in left_set]
            weighted = (len(left) / len(indices)) * _gini(labels[left]) + (
                len(right) / len(indices)
            ) * _gini(labels[right])
            gain = parent - weighted
            if gain > best_gain + 1e-15:
                best_gain = gain
                best = (feature_index, threshold, sorted(left), sorted(right))
    return best


def _gini(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    positive = float(labels.sum()) / len(labels)
    return 1.0 - positive * positive - (1.0 - positive) * (1.0 - positive)


def _tree_probabilities(
    nodes: dict[int, dict[str, Any]], root: int, features: np.ndarray
) -> np.ndarray:
    outputs = []
    for row_index in range(features.shape[0]):
        node_id = root
        while True:
            node = nodes[node_id]
            feature = node["feature"]
            if feature is None:
                outputs.append(node["probability"])
                break
            if float(features[row_index, feature]) <= float(node["threshold"]):
                node_id = int(node["left"])
            else:
                node_id = int(node["right"])
    return np.asarray(outputs, dtype=np.float64)


def _predict_logistic(
    trained: dict[str, Any], features: list[list[float]], fixed: dict[str, int]
) -> list[int]:
    matrix = np.asarray(features, dtype=np.float64)
    numeric_count = trained["numeric_count"]
    if numeric_count:
        matrix[:, :numeric_count] = (
            matrix[:, :numeric_count] - trained["means"]
        ) / trained["stds"]
    probabilities = _sigmoid_array(matrix @ trained["weights"] + trained["bias"])
    return [
        quantize_probability(float(value), fixed["quantization_levels"], fixed["scale"])
        for value in probabilities
    ]


def _predict_tree_fixed(
    trained: dict[str, Any], features: list[list[float]], fixed: dict[str, int]
) -> list[int]:
    matrix = np.asarray(features, dtype=np.float64)
    probabilities = _tree_probabilities(trained["node_index"], 0, matrix)
    return [
        quantize_probability(float(value), fixed["quantization_levels"], fixed["scale"])
        for value in probabilities
    ]


def _numeric_feature_count(train: dict[str, Any]) -> int:
    return sum(1 for name in train["feature_names"] if "=" not in name)


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))
