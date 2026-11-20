"""Deterministic linear classification with dependency fingerprints."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .errors import E_MODEL_FAILED, E_UPSTREAM_STALE, WorkbenchError, diagnostic
from .util import sha256_obj


def _cell_value(row: dict[str, Any], column: str, cells: dict[str, Any]) -> Any:
    raw = row["values"].get(column)
    if raw is not None:
        return raw
    cell = cells.get(row["row_id"], {}).get(column)
    if cell is None:
        return None
    return cell.get("value")


def build_feature_dataset(
    rows: list[dict[str, Any]],
    cells: dict[str, Any],
    config: dict[str, Any],
    excluded_row_ids: set[str],
) -> tuple[list[str], list[str], list[list[float]], list[int]]:
    target = config["target_column"]
    feature_columns = [
        column
        for column in sorted(config["columns"])
        if column not in {config["id_column"], target}
    ]
    feature_names: list[str] = []
    for column in feature_columns:
        spec = config["columns"][column]
        if spec["type"] == "numeric":
            feature_names.append(column)
        else:
            for value in sorted(spec.get("domain", [])):
                feature_names.append(f"{column}={value}")

    row_ids: list[str] = []
    vectors: list[list[float]] = []
    labels: list[int] = []
    for row in sorted(rows, key=lambda item: item["row_id"]):
        if row["row_id"] in excluded_row_ids:
            continue
        label_value = _cell_value(row, target, cells)
        if label_value is None or label_value not in (0, 1, 0.0, 1.0):
            continue
        complete = True
        vector: list[float] = []
        for column in feature_columns:
            spec = config["columns"][column]
            value = _cell_value(row, column, cells)
            if value is None:
                complete = False
                break
            if spec["type"] == "numeric":
                vector.append(float(value))
            else:
                for domain_value in sorted(spec.get("domain", [])):
                    vector.append(1.0 if value == domain_value else 0.0)
        if not complete:
            continue
        row_ids.append(row["row_id"])
        vectors.append(vector)
        labels.append(int(label_value))
    return feature_names, row_ids, vectors, labels


def deterministic_split(
    row_ids: list[str], labels: list[int], seed: int, validation_fraction: float
) -> tuple[list[int], list[int]]:
    if len(row_ids) < 4:
        raise WorkbenchError(E_MODEL_FAILED, "at least four complete labelled rows are required")
    order = sorted(
        range(len(row_ids)),
        key=lambda index: (
            hashlib.sha256(f"{seed}\x1f{row_ids[index]}".encode("utf-8")).hexdigest(),
            row_ids[index],
        ),
    )
    validation_count = max(1, int(round(len(row_ids) * validation_fraction)))
    validation_count = min(validation_count, len(row_ids) - 3)
    validation = sorted(order[:validation_count])
    training = sorted(order[validation_count:])
    if len({labels[index] for index in training}) < 2:
        # Deterministic repair if a hash split happened to produce one class.
        by_label: dict[int, list[int]] = {}
        for index in training:
            by_label.setdefault(labels[index], []).append(index)
        missing_label = 0 if 0 not in by_label else 1
        candidates = [index for index in validation if labels[index] == missing_label]
        if candidates:
            moving = candidates[0]
            validation.remove(moving)
            training.append(moving)
            training.sort()
            validation.sort()
    return training, validation


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _binary_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))


def _model_input_fingerprint(
    feature_names: list[str],
    row_ids: list[str],
    vectors: list[list[float]],
    labels: list[int],
    training_indices: list[int],
    validation_indices: list[int],
    config: dict[str, Any],
    imputation_fingerprint: str,
) -> str:
    return sha256_obj(
        {
            "level": "linear_model",
            "imputation_fingerprint": imputation_fingerprint,
            "target": config["target_column"],
            "feature_names": feature_names,
            "rows": [
                {
                    "row_id": row_ids[index],
                    "features": vectors[index],
                    "label": labels[index],
                }
                for index in range(len(row_ids))
            ],
            "training_row_ids": [row_ids[index] for index in training_indices],
            "validation_row_ids": [row_ids[index] for index in validation_indices],
            "model": config["model"],
            "validation_fraction": config["validation_fraction"],
            "seed": config["seed"],
        }
    )


def train_and_evaluate(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    cells: dict[str, Any],
    excluded_row_ids: set[str],
    imputation_fingerprint: str,
    previous_model: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    feature_names, row_ids, vectors, labels = build_feature_dataset(
        rows, cells, config, excluded_row_ids
    )
    if len(row_ids) < 4:
        raise WorkbenchError(
            E_MODEL_FAILED,
            "fewer than four complete labelled rows remain after row failures",
        )
    training_indices, validation_indices = deterministic_split(
        row_ids, labels, int(config["seed"]), float(config["validation_fraction"])
    )
    fingerprint = _model_input_fingerprint(
        feature_names,
        row_ids,
        vectors,
        labels,
        training_indices,
        validation_indices,
        config,
        imputation_fingerprint,
    )
    diagnostics: list[dict[str, Any]] = []
    if previous_model and previous_model.get("status") == "ok":
        same = (
            previous_model.get("input_fingerprint") == fingerprint
            and previous_model.get("imputation_fingerprint") == imputation_fingerprint
        )
        if same:
            reused = dict(previous_model)
            reused["reused"] = True
            return reused, {"reused": True, "recomputed": False, "stale_rejected": False}, diagnostics
        diagnostics.append(
            diagnostic(
                E_UPSTREAM_STALE,
                "persisted model fingerprint does not match the current imputation data; old result rejected",
                stage="model",
                expected_imputation_fingerprint=imputation_fingerprint,
                stored_imputation_fingerprint=previous_model.get("imputation_fingerprint"),
            )
        )

    matrix = np.asarray(vectors, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.float64)
    train_x = matrix[training_indices]
    train_y = labels_array[training_indices]
    val_x = matrix[validation_indices]
    val_y = labels_array[validation_indices]

    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    train_x_scaled = (train_x - means) / stds
    val_x_scaled = (val_x - means) / stds

    rng_free_weights = np.zeros(train_x_scaled.shape[1], dtype=np.float64)
    bias = 0.0
    learning_rate = float(config["model"]["learning_rate"])
    l2 = float(config["model"]["l2"])
    batch_size = int(config["model"]["batch_size"])
    epochs = int(config["model"]["epochs"])
    losses: list[float] = []
    # Fixed order every epoch: row identifiers were sorted before deterministic split.
    training_order = list(range(len(training_indices)))
    for _epoch in range(epochs):
        for start in range(0, len(training_order), batch_size):
            batch_indices = training_order[start : start + batch_size]
            batch_x = train_x_scaled[batch_indices]
            batch_y = train_y[batch_indices]
            logits = batch_x @ rng_free_weights + bias
            probabilities = _sigmoid(logits)
            errors = probabilities - batch_y
            grad_w = (batch_x.T @ errors) / len(batch_indices) + l2 * rng_free_weights
            grad_b = float(np.mean(errors))
            rng_free_weights = rng_free_weights - learning_rate * grad_w
            bias -= learning_rate * grad_b
        full_probabilities = _sigmoid(train_x_scaled @ rng_free_weights + bias)
        epoch_loss = _binary_log_loss(full_probabilities, train_y) + 0.5 * l2 * float(
            np.sum(rng_free_weights * rng_free_weights)
        )
        losses.append(float(epoch_loss))

    validation_probabilities = _sigmoid(val_x_scaled @ rng_free_weights + bias)
    predictions = (validation_probabilities >= 0.5).astype(np.int64)
    accuracy = float(np.mean(predictions == val_y))
    log_loss = _binary_log_loss(validation_probabilities, val_y)
    model = {
        "status": "ok",
        "input_fingerprint": fingerprint,
        "imputation_fingerprint": imputation_fingerprint,
        "feature_names": feature_names,
        "training_row_ids": [row_ids[index] for index in training_indices],
        "validation_row_ids": [row_ids[index] for index in validation_indices],
        "training_row_count": len(training_indices),
        "validation_row_count": len(validation_indices),
        "excluded_row_ids": sorted(excluded_row_ids),
        "loss_trajectory": losses,
        "accuracy": accuracy,
        "log_loss": log_loss,
        "weights": [float(value) for value in rng_free_weights],
        "bias": float(bias),
        "feature_means": [float(value) for value in means],
        "feature_stds": [float(value) for value in stds],
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "l2": l2,
        "reused": False,
    }
    model["model_fingerprint"] = sha256_obj(model)
    return model, {"reused": False, "recomputed": True, "stale_rejected": bool(diagnostics)}, diagnostics


def assert_model_fresh(model: dict[str, Any] | None, imputation_fingerprint: str) -> None:
    if model is None or model.get("status") != "ok":
        raise WorkbenchError(E_UPSTREAM_STALE, "no usable trained model is available")
    if model.get("imputation_fingerprint") != imputation_fingerprint:
        raise WorkbenchError(
            E_UPSTREAM_STALE,
            "model was trained on stale imputation data and must not be used",
            {
                "stored_imputation_fingerprint": model.get("imputation_fingerprint"),
                "current_imputation_fingerprint": imputation_fingerprint,
            },
        )
