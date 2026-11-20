"""Linear-probe and low-rank task adapters with deterministic CPU training."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from .canonical import sha256_json
from .encoder import target_representation
from .errors import (
    AdapterExecutionError,
    PipelineError,
    RankInvalidError,
    TrainingExecutionError,
    ValidationExecutionError,
)
from .inputs import TaskSpec


def _seed_from_cache_key(cache_key: str, adapter_type: str, rank: int, scale: float) -> int:
    payload = f"{cache_key}\x00{adapter_type}\x00{rank}\x00{scale:.17g}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def _cross_entropy_and_gradient(
    logits: np.ndarray, targets: np.ndarray
) -> tuple[float, np.ndarray]:
    probabilities = _softmax(logits)
    count = logits.shape[0]
    correct = probabilities[np.arange(count), targets]
    loss = float(-np.mean(np.log(np.maximum(correct, 1e-300))))
    gradient = probabilities
    gradient[np.arange(count), targets] -= 1.0
    gradient /= float(count)
    return loss, gradient


def _parameter_state(
    adapter_type: str,
    rank: int,
    input_dim: int,
    class_count: int,
    cache_key: str,
    scale: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(_seed_from_cache_key(cache_key, adapter_type, rank, scale))
    if adapter_type == "linear_probe":
        left = rng.normal(0.0, math.sqrt(2.0 / max(input_dim, 1)), size=(input_dim, class_count))
        bias = np.zeros(class_count, dtype=np.float64)
        return {
            "adapter_type": adapter_type,
            "rank": rank,
            "scale": scale,
            "input_dim": input_dim,
            "class_count": class_count,
            "left": left.astype(np.float64, copy=False),
            "right": None,
            "bias": bias,
        }
    if adapter_type == "low_rank":
        left = rng.normal(0.0, math.sqrt(2.0 / max(input_dim, 1)), size=(input_dim, rank))
        right = rng.normal(0.0, math.sqrt(2.0 / max(rank * class_count, 1)), size=(rank, class_count))
        bias = np.zeros(class_count, dtype=np.float64)
        return {
            "adapter_type": adapter_type,
            "rank": rank,
            "scale": scale,
            "input_dim": input_dim,
            "class_count": class_count,
            "left": left.astype(np.float64, copy=False),
            "right": right.astype(np.float64, copy=False),
            "bias": bias,
        }
    raise AdapterExecutionError(f"unsupported adapter type: {adapter_type}")


def _logits(state: dict[str, Any], features: np.ndarray) -> np.ndarray:
    if state["adapter_type"] == "linear_probe":
        return features @ state["left"] + state["bias"]
    hidden = features @ state["left"]
    return state["scale"] * (hidden @ state["right"]) + state["bias"]


def _gradients(state: dict[str, Any], features: np.ndarray, delta: np.ndarray) -> dict[str, np.ndarray]:
    if state["adapter_type"] == "linear_probe":
        return {
            "left": features.T @ delta,
            "right": None,
            "bias": np.sum(delta, axis=0),
        }
    hidden = features @ state["left"]
    return {
        "left": state["scale"] * (features.T @ (delta @ state["right"].T)),
        "right": state["scale"] * (hidden.T @ delta),
        "bias": np.sum(delta, axis=0),
    }


def _apply_gradient_step(state: dict[str, Any], gradients: dict[str, np.ndarray], learning_rate: float) -> None:
    state["left"] = state["left"] - learning_rate * gradients["left"]
    if gradients["right"] is not None:
        state["right"] = state["right"] - learning_rate * gradients["right"]
    state["bias"] = state["bias"] - learning_rate * gradients["bias"]


def _predict(state: dict[str, Any], features: np.ndarray) -> np.ndarray:
    return np.argmax(_logits(state, features), axis=1)


def compute_metrics(labels: list[int], predictions: list[int], class_count: int) -> dict[str, float]:
    if not labels:
        raise ValidationExecutionError("validation sample set is empty")
    correct = sum(int(actual == predicted) for actual, predicted in zip(labels, predictions))
    accuracy = correct / float(len(labels))
    f1_values: list[float] = []
    for class_index in range(class_count):
        tp = sum(
            1
            for actual, predicted in zip(labels, predictions)
            if actual == class_index and predicted == class_index
        )
        fp = sum(
            1
            for actual, predicted in zip(labels, predictions)
            if actual != class_index and predicted == class_index
        )
        fn = sum(
            1
            for actual, predicted in zip(labels, predictions)
            if actual == class_index and predicted != class_index
        )
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else (2.0 * tp) / float(denominator))
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / float(class_count),
    }


def _check_runtime_injection(
    runtime: dict[tuple[str, str], dict[str, Any]], task_name: str, phase: str
) -> None:
    entry = runtime.get((task_name, phase))
    if entry is None:
        return
    exception_type = str(entry.get("exception_type", "InjectedRuntimeError"))
    message = str(entry.get("message", f"injected exception in {phase} stage"))
    if phase == "training":
        raise TrainingExecutionError(
            message,
            details={"exception_type": exception_type, "phase": "training"},
        )
    raise ValidationExecutionError(
        message,
        details={"exception_type": exception_type, "phase": "validation"},
    )


def train_and_validate(
    *,
    task: TaskSpec,
    cache_key: str,
    shared_forward: dict[str, Any],
    features: dict[str, dict[str, Any]],
    config: dict[str, Any],
    runtime: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    adapter_type = str(config["adapter_type"])
    rank = int(task.adapter_rank_override if task.adapter_rank_override is not None else config["rank"])
    scale = float(config["scale"])
    if rank <= 0:
        raise RankInvalidError(f"adapter rank must be positive before construction (got {rank})")

    try:
        classes = list(task.effective_classes)
        class_index = {label: index for index, label in enumerate(classes)}
        train_ids = sorted(task.train_ids)
        validation_ids = sorted(task.validation_ids)
        train_features = np.stack(
            [
                target_representation(shared_forward, sample_id, task.target_layer)
                for sample_id in train_ids
            ]
        )
        train_labels = np.asarray(
            [class_index[features[sample_id]["label"]] for sample_id in train_ids],
            dtype=np.int64,
        )

        _check_runtime_injection(runtime, task.name, "training")
        state = _parameter_state(
            adapter_type=adapter_type,
            rank=rank,
            input_dim=int(train_features.shape[1]),
            class_count=len(classes),
            cache_key=cache_key,
            scale=scale,
        )
        learning_rate = float(config["learning_rate"])
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        losses: list[float] = []
        for _epoch in range(epochs):
            epoch_loss = 0.0
            total_examples = 0
            for start in range(0, len(train_ids), batch_size):
                stop = min(start + batch_size, len(train_ids))
                batch_features = train_features[start:stop]
                batch_labels = train_labels[start:stop]
                logits = _logits(state, batch_features)
                loss, delta = _cross_entropy_and_gradient(logits, batch_labels)
                batch_count = stop - start
                epoch_loss += loss * batch_count
                total_examples += batch_count
                gradients = _gradients(state, batch_features, delta)
                _apply_gradient_step(state, gradients, learning_rate)
            losses.append(epoch_loss / float(total_examples))
    except PipelineError:
        raise
    except Exception as exc:
        raise TrainingExecutionError(
            str(exc),
            details={"exception_type": type(exc).__name__, "phase": "training"},
        ) from exc

    try:
        validation_features = np.stack(
            [
                target_representation(shared_forward, sample_id, task.target_layer)
                for sample_id in validation_ids
            ]
        )
        validation_labels = [
            class_index[features[sample_id]["label"]] for sample_id in validation_ids
        ]
        _check_runtime_injection(runtime, task.name, "validation")
        predictions = _predict(state, validation_features).tolist()
        metrics = compute_metrics(validation_labels, predictions, len(classes))
        metric_name = str(config["validation_metric"])
        metric_value = float(metrics[metric_name])
        threshold = float(config["validation_threshold"])
        prediction_rows = [
            {
                "sample_id": sample_id,
                "label": classes[actual_index],
                "predicted": classes[predicted_index],
            }
            for sample_id, actual_index, predicted_index in zip(
                validation_ids, validation_labels, predictions
            )
        ]
    except PipelineError:
        raise
    except Exception as exc:
        raise ValidationExecutionError(
            str(exc),
            details={"exception_type": type(exc).__name__, "phase": "validation"},
        ) from exc
    left = np.asarray(state["left"], dtype=np.float64)
    right = None if state["right"] is None else np.asarray(state["right"], dtype=np.float64)
    bias = np.asarray(state["bias"], dtype=np.float64)
    parameter_payload: dict[str, Any] = {
        "left": left.tolist(),
        "right": None if right is None else right.tolist(),
        "bias": bias.tolist(),
    }
    parameter_shapes: dict[str, list[int] | None] = {
        "left": list(left.shape),
        "right": None if right is None else list(right.shape),
        "bias": list(bias.shape),
    }
    return {
        "task_name": task.name,
        "sort_key": task.sort_key,
        "target_layer": task.target_layer,
        "classes": classes,
        "cache_key": cache_key,
        "status": "success",
        "adapter_type": adapter_type,
        "rank": rank,
        "scale": scale,
        "parameter_shapes": parameter_shapes,
        "parameter_fingerprint": sha256_json(parameter_payload),
        "parameters": parameter_payload,
        "loss_trajectory": losses,
        "validation": {
            "sample_count": len(validation_ids),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "metric": metric_name,
            "metric_value": metric_value,
            "threshold": threshold,
            "passed": metric_value >= threshold,
            "predictions": prediction_rows,
        },
    }
