"""Deterministic linear-probe and low-rank adapter training/evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .canonical import array_fingerprint, parameters_fingerprint
from .data import EffectiveAdapterConfig, FeatureSample, TaskDescription
from .errors import EvaluationError, TrainingError


@dataclass(frozen=True)
class TrainingResult:
    parameters: Mapping[str, np.ndarray]
    losses: tuple[float, ...]
    converged: bool
    epochs_run: int
    summary: Mapping[str, Any]


def initialize_parameters(
    adapter_type: str,
    rank: int,
    input_dim: int,
    class_count: int,
) -> dict[str, np.ndarray]:
    """Create deterministic zero/low-rank initialization without RNG."""

    head = np.zeros((class_count, input_dim), dtype=np.float64)
    bias = np.zeros((class_count,), dtype=np.float64)
    if adapter_type == "linear_probe":
        return {"head": head, "bias": bias}
    row = np.arange(1, rank + 1, dtype=np.float64)[:, None]
    column = np.arange(1, input_dim + 1, dtype=np.float64)[None, :]
    lora_a = 0.01 * np.sin(row * column * 0.37)
    lora_b = np.zeros((input_dim, rank), dtype=np.float64)
    return {"lora_A": lora_a, "lora_B": lora_b, "head": head, "bias": bias}


def _adapted_representation(
    representations: np.ndarray,
    parameters: Mapping[str, np.ndarray],
    config: EffectiveAdapterConfig,
) -> tuple[np.ndarray, Mapping[str, np.ndarray]]:
    if config.adapter_type == "linear_probe":
        return representations, {}
    intermediate = np.dot(representations, parameters["lora_A"].T)
    delta = np.dot(intermediate, parameters["lora_B"].T) * config.lora_scale
    return representations + delta, {"intermediate": intermediate, "delta": delta}


def _forward(
    representations: np.ndarray,
    parameters: Mapping[str, np.ndarray],
    config: EffectiveAdapterConfig,
) -> tuple[np.ndarray, Mapping[str, np.ndarray]]:
    adapted, cache = _adapted_representation(representations, parameters, config)
    logits = np.dot(adapted, parameters["head"].T) + parameters["bias"]
    extras = dict(cache)
    extras["adapted"] = adapted
    return logits, extras


def _softmax(logits: np.ndarray) -> np.ndarray:
    maximum = np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(logits - maximum)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    maximum = np.max(logits, axis=1, keepdims=True)
    log_normalizer = maximum[:, 0] + np.log(
        np.sum(np.exp(logits - maximum), axis=1)
    )
    return float(np.mean(log_normalizer - logits[np.arange(logits.shape[0]), labels]))


def _training_summary(parameters: Mapping[str, np.ndarray]) -> dict[str, Any]:
    items: dict[str, Any] = {}
    count = 0
    for name in sorted(parameters):
        array = np.asarray(parameters[name], dtype=np.float64)
        count += int(array.size)
        items[name] = {
            "shape": list(array.shape),
            "count": int(array.size),
            "l2_norm": float(np.linalg.norm(array)),
            "max_abs": float(np.max(np.abs(array))) if array.size else 0.0,
            "mean_abs": float(np.mean(np.abs(array))) if array.size else 0.0,
            "digest": array_fingerprint(array),
        }
    return {
        "parameter_count": count,
        "parameters": items,
        "digest": parameters_fingerprint(parameters),
    }


def _ordered_arrays(
    sample_ids: Sequence[str],
    samples: Mapping[str, FeatureSample],
    representations: Mapping[str, Any],
    class_to_index: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    vectors: list[list[float]] = []
    labels: list[int] = []
    for sample_id in sorted(sample_ids):
        if sample_id not in samples or sample_id not in representations:
            raise TrainingError(f"training sample {sample_id!r} lacks a shared representation")
        vector = representations[sample_id].get("vector")
        if not isinstance(vector, list) or not vector:
            raise TrainingError(f"training sample {sample_id!r} representation is malformed")
        label = samples[sample_id].label
        if label not in class_to_index:
            raise TrainingError(f"training sample {sample_id!r} label is outside task classes")
        vectors.append(vector)
        labels.append(class_to_index[label])
    return (
        np.asarray(vectors, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
    )


def train_adapter(
    task: TaskDescription,
    samples: Mapping[str, FeatureSample],
    representations: Mapping[str, Any],
    config: EffectiveAdapterConfig,
) -> TrainingResult:
    """Train in sorted sample/batch order using full deterministic SGD."""

    class_to_index = {label: index for index, label in enumerate(task.classes)}
    x_values, y_values = _ordered_arrays(
        task.train_ids, samples, representations, class_to_index
    )
    if x_values.shape[0] == 0:
        raise TrainingError("task has no training samples")
    parameters = initialize_parameters(
        config.adapter_type,
        config.rank,
        int(x_values.shape[1]),
        len(task.classes),
    )
    losses: list[float] = []
    batch_size = min(config.batch_size, x_values.shape[0])
    converged = False
    epochs_run = 0
    for epoch in range(config.epochs):
        weighted_loss = 0.0
        for start in range(0, x_values.shape[0], batch_size):
            batch_x = x_values[start : start + batch_size]
            batch_y = y_values[start : start + batch_size]
            logits, extras = _forward(batch_x, parameters, config)
            loss = _cross_entropy(logits, batch_y)
            weighted_loss += loss * int(batch_x.shape[0])
            probabilities = _softmax(logits)
            indicator = np.zeros_like(probabilities)
            indicator[np.arange(batch_x.shape[0]), batch_y] = 1.0
            grad_logits = (probabilities - indicator) / float(batch_x.shape[0])
            adapted = extras["adapted"]
            grad_head = np.dot(grad_logits.T, adapted)
            grad_bias = np.sum(grad_logits, axis=0)
            grad_lora_a = None
            grad_lora_b = None
            if config.adapter_type == "low_rank":
                grad_h = np.dot(grad_logits, parameters["head"])
                grad_delta = grad_h * config.lora_scale
                grad_lora_b = np.dot(grad_delta.T, extras["intermediate"])
                grad_intermediate = np.dot(grad_delta, parameters["lora_B"])
                grad_lora_a = np.dot(grad_intermediate.T, batch_x)
            parameters["head"] -= config.learning_rate * grad_head
            parameters["bias"] -= config.learning_rate * grad_bias
            if config.adapter_type == "low_rank":
                assert grad_lora_b is not None and grad_lora_a is not None
                parameters["lora_B"] -= config.learning_rate * grad_lora_b
                parameters["lora_A"] -= config.learning_rate * grad_lora_a
        epoch_loss = float(weighted_loss / x_values.shape[0])
        epochs_run = epoch + 1
        if losses and abs(losses[-1] - epoch_loss) <= config.convergence_tolerance:
            losses.append(epoch_loss)
            converged = True
            break
        losses.append(epoch_loss)
    summary = _training_summary(parameters)
    return TrainingResult(
        parameters=parameters,
        losses=tuple(losses),
        converged=converged,
        epochs_run=epochs_run,
        summary=summary,
    )


def _macro_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    classes: Sequence[str],
) -> tuple[float, dict[str, dict[str, float]]]:
    class_metrics: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for class_index, class_name in enumerate(classes):
        true_positive = int(np.sum((labels == class_index) & (predictions == class_index)))
        false_positive = int(np.sum((labels != class_index) & (predictions == class_index)))
        false_negative = int(np.sum((labels == class_index) & (predictions != class_index)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = (2.0 * true_positive / denominator) if denominator else 0.0
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        class_metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(labels == class_index)),
        }
        f1_values.append(float(f1))
    macro = float(math.fsum(f1_values) / len(f1_values)) if f1_values else 0.0
    return macro, class_metrics


def evaluate_adapter(
    task: TaskDescription,
    samples: Mapping[str, FeatureSample],
    representations: Mapping[str, Any],
    config: EffectiveAdapterConfig,
    parameters: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    class_to_index = {label: index for index, label in enumerate(task.classes)}
    x_values, y_values = _ordered_arrays(
        task.val_ids, samples, representations, class_to_index
    )
    if x_values.shape[0] == 0:
        raise EvaluationError("task has no validation samples")
    logits, _ = _forward(x_values, parameters, config)
    predictions = np.argmax(logits, axis=1)
    accuracy = float(np.mean(predictions == y_values))
    macro_f1, class_metrics = _macro_f1(y_values, predictions, task.classes)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "threshold": config.validation_threshold,
        "meets_threshold": bool(accuracy >= config.validation_threshold),
        "validation_count": int(x_values.shape[0]),
        "class_metrics": class_metrics,
    }
