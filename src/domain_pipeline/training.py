from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .canonical import IMPLEMENTATION_VERSION, fingerprint, stable_float
from .config import PipelineConfig
from .loader import ValidSample
from .scheduler import ScheduledTask, run_tasks


@dataclass
class TrainingRun:
    artifact: dict[str, Any]
    gradient_evaluations: int


def _training_dataset(
    samples: list[ValidSample], feature_artifact: dict[str, Any]
) -> dict[str, Any]:
    rows = {row["sample_id"]: row for row in feature_artifact["rows"]}
    entries = [
        {
            "sample_id": sample.sample_id,
            "vector": rows[sample.sample_id]["vector"],
            "label": sample.label,
        }
        for sample in sorted(samples, key=lambda item: item.sample_id)
        if sample.sample_id in rows
    ]
    return {
        "sample_ids": [entry["sample_id"] for entry in entries],
        "entries": entries,
        "feature_schema_fingerprint": feature_artifact[
            "feature_schema_fingerprint"
        ],
    }


def training_dataset_fingerprint(
    samples: list[ValidSample], feature_artifact: dict[str, Any]
) -> str:
    return fingerprint("training-dataset:v1", _training_dataset(samples, feature_artifact))


def training_cache_key(
    config: PipelineConfig,
    samples: list[ValidSample],
    feature_artifact: dict[str, Any],
) -> str:
    return fingerprint(
        "training-stage:v1",
        {
            "implementation": IMPLEMENTATION_VERSION,
            "dataset_fingerprint": training_dataset_fingerprint(
                samples, feature_artifact
            ),
            "params": config.training_params(),
        },
    )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        term = math.exp(-value)
        return 1.0 / (1.0 + term)
    term = math.exp(value)
    return term / (1.0 + term)


def _dot(weights: list[float], vector: list[float], bias: float) -> float:
    value = bias
    for index in range(len(weights)):
        value += weights[index] * vector[index]
    return value


def _log_loss(probability: float, label: int) -> float:
    if label == 1:
        return -math.log(max(probability, 1e-15))
    return -math.log(max(1.0 - probability, 1e-15))


def _initial_weights(dimension: int, candidate: dict[str, Any]) -> list[float]:
    scale = float(candidate["initial_weight_scale"])
    weights: list[float] = []
    for index in range(dimension):
        pattern = ((index * 37 + 11) % 101 - 50) / 50.0
        weights.append(stable_float(scale * pattern))
    return weights


def train_model(
    config: PipelineConfig,
    train_samples: list[ValidSample],
    feature_artifact: dict[str, Any],
) -> TrainingRun:
    dataset = _training_dataset(train_samples, feature_artifact)
    entries = dataset["entries"]
    params = config.training_params()
    candidate = params["candidate"]
    batch_size = int(params["batch_size"])
    learning_rate = float(params["learning_rate"])
    epochs = int(params["epochs"])
    l2 = float(candidate["l2"])

    dimension = (
        len(entries[0]["vector"])
        if entries
        else len(feature_artifact.get("feature_names", []))
    )
    weights = _initial_weights(dimension, candidate)
    bias = float(candidate["initial_bias"])
    losses: list[float] = []
    batch_order: list[list[str]] = []
    gradient_evaluations = 0

    for epoch in range(epochs):
        for batch_index, start in enumerate(range(0, len(entries), batch_size)):
            batch = entries[start : start + batch_size]
            if not batch:
                continue
            batch_ids = [entry["sample_id"] for entry in batch]
            batch_order.append(batch_ids)
            weight_snapshot = list(weights)
            bias_snapshot = float(bias)
            tasks: list[ScheduledTask] = []
            for entry in batch:
                vector = list(entry["vector"])
                label = int(entry["label"])

                def gradient(
                    vector: list[float] = vector,
                    label: int = label,
                    weight_snapshot: list[float] = weight_snapshot,
                    bias_snapshot: float = bias_snapshot,
                ) -> tuple[list[float], float]:
                    probability = _sigmoid(
                        _dot(weight_snapshot, vector, bias_snapshot)
                    )
                    error = probability - label
                    result = [error * value for value in vector]
                    return result, error

                tasks.append(
                    ScheduledTask(
                        key=(
                            f"train:e{epoch:04d}:b{batch_index:04d}:"
                            f"s{entry['sample_id']}:gradient"
                        ),
                        cost=0,
                        function=gradient,
                    )
                )
            scheduled = run_tasks(
                tasks,
                max_workers=int(config.concurrency["max_workers"]),
                budget=None,
                budget_name="training_gradients",
            )
            gradient_evaluations += len(batch)
            summed_weights = [0.0] * dimension
            summed_bias = 0.0
            for task in sorted(tasks, key=lambda item: item.key):
                weight_gradient, bias_gradient = scheduled.results[task.key]
                for index in range(dimension):
                    summed_weights[index] += weight_gradient[index]
                summed_bias += bias_gradient
            divisor = float(len(batch))
            for index in range(dimension):
                gradient_value = summed_weights[index] / divisor + l2 * weights[index]
                weights[index] = stable_float(weights[index] - learning_rate * gradient_value)
            bias = stable_float(bias - learning_rate * (summed_bias / divisor))

        epoch_loss = 0.0
        for entry in entries:
            probability = _sigmoid(
                _dot(weights, entry["vector"], bias)
            )
            epoch_loss += _log_loss(probability, int(entry["label"]))
        if entries:
            epoch_loss /= float(len(entries))
        losses.append(stable_float(epoch_loss))

    artifact: dict[str, Any] = {
        "artifact_type": "training",
        "cache_key": training_cache_key(config, train_samples, feature_artifact),
        "implementation": IMPLEMENTATION_VERSION,
        "dataset_fingerprint": training_dataset_fingerprint(
            train_samples, feature_artifact
        ),
        "feature_schema_fingerprint": feature_artifact[
            "feature_schema_fingerprint"
        ],
        "params": params,
        "sample_ids": dataset["sample_ids"],
        "feature_dimension": dimension,
        "batch_order": batch_order,
        "losses": losses,
        "final_weights": weights,
        "final_bias": bias,
    }
    artifact["content_fingerprint"] = fingerprint("training-artifact:v1", artifact)
    return TrainingRun(
        artifact=artifact,
        gradient_evaluations=gradient_evaluations,
    )
