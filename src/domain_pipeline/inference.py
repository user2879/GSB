from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .canonical import (
    IMPLEMENTATION_VERSION,
    fingerprint,
    round_half_up_nonnegative,
    stable_float,
)
from .config import PipelineConfig
from .loader import ValidSample
from .scheduler import ScheduledTask, run_tasks


@dataclass
class InferenceRun:
    inference_artifact: dict[str, Any]
    summary_artifact: dict[str, Any]
    row_cache: dict[str, dict[str, Any]]
    rows_total: int
    rows_reused: int
    rows_computed: int
    inference_evaluations: int


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    term = math.exp(value)
    return term / (1.0 + term)


def _dot(weights: list[float], vector: list[float], bias: float) -> float:
    value = bias
    for index in range(len(weights)):
        value += weights[index] * vector[index]
    return value


def validation_dataset(
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


def validation_dataset_fingerprint(
    samples: list[ValidSample], feature_artifact: dict[str, Any]
) -> str:
    return fingerprint(
        "validation-dataset:v1", validation_dataset(samples, feature_artifact)
    )


def inference_cache_key(
    config: PipelineConfig,
    validation_samples: list[ValidSample],
    feature_artifact: dict[str, Any],
    model_artifact: dict[str, Any],
) -> str:
    return fingerprint(
        "inference-stage:v1",
        {
            "implementation": IMPLEMENTATION_VERSION,
            "model_cache_key": model_artifact["cache_key"],
            "validation_dataset_fingerprint": validation_dataset_fingerprint(
                validation_samples, feature_artifact
            ),
            "quantization": config.quantization_params(),
        },
    )


def summary_cache_key(
    config: PipelineConfig, inference_artifact: dict[str, Any]
) -> str:
    return fingerprint(
        "summary-stage:v1",
        {
            "implementation": IMPLEMENTATION_VERSION,
            "inference_cache_key": inference_artifact["cache_key"],
            "aggregation": {
                "sort": "sample_id_ascending",
                "classification_threshold": "half_up_fixed_point",
                "quantization": config.quantization_params(),
            },
        },
    )


def _row_cache_key(
    sample_id: str,
    vector: list[float],
    label: int,
    model_cache_key: str,
    quantization: dict[str, Any],
) -> str:
    return fingerprint(
        "inference-row:v1",
        {
            "sample_id": sample_id,
            "vector": vector,
            "label": label,
            "model_cache_key": model_cache_key,
            "quantization": quantization,
        },
    )


def _predict_row(
    entry: dict[str, Any],
    model_artifact: dict[str, Any],
    quantization: dict[str, Any],
) -> dict[str, Any]:
    vector = entry["vector"]
    logit = _dot(
        model_artifact["final_weights"],
        vector,
        model_artifact["final_bias"],
    )
    probability = min(max(_sigmoid(logit), 0.0), 1.0)
    fractional_bits = int(quantization["fractional_bits"])
    scale_level = int(quantization["scale_level"])
    denominator = 1 << fractional_bits
    scaled = probability * denominator / scale_level
    quantized_integer = round_half_up_nonnegative(scaled) * scale_level
    quantized_integer = min(max(quantized_integer, 0), denominator)
    predicted_label = 1 if 2 * quantized_integer >= denominator else 0
    label = int(entry["label"])
    return {
        "sample_id": entry["sample_id"],
        "label": label,
        "logit": stable_float(logit),
        "raw_probability": stable_float(probability),
        "quantized_integer": quantized_integer,
        "quantized_probability": stable_float(
            quantized_integer / float(denominator)
        ),
        "predicted_label": predicted_label,
        "correct": predicted_label == label,
    }


def _quantized_loss_probability(quantized_integer: int, denominator: int) -> float:
    if quantized_integer <= 0:
        return 0.5 / float(denominator)
    if quantized_integer >= denominator:
        return 1.0 - 0.5 / float(denominator)
    return quantized_integer / float(denominator)


def _build_summary(
    config: PipelineConfig,
    predictions: list[dict[str, Any]],
    inference_artifact: dict[str, Any],
) -> dict[str, Any]:
    denominator = 1 << int(config.quantization["fractional_bits"])
    correct = 0
    log_loss_sum = 0.0
    integer_sum = 0
    sample_results: list[dict[str, Any]] = []
    for prediction in sorted(predictions, key=lambda item: item["sample_id"]):
        label = int(prediction["label"])
        quantized_integer = int(prediction["quantized_integer"])
        if prediction["correct"]:
            correct += 1
        probability = _quantized_loss_probability(quantized_integer, denominator)
        if label == 1:
            log_loss_sum += -math.log(probability)
        else:
            log_loss_sum += -math.log(1.0 - probability)
        integer_sum += quantized_integer
        sample_results.append(
            {
                "sample_id": prediction["sample_id"],
                "label": label,
                "quantized_integer": quantized_integer,
                "quantized_probability": prediction["quantized_probability"],
                "predicted_label": prediction["predicted_label"],
                "correct": prediction["correct"],
            }
        )
    count = len(sample_results)
    accuracy = correct / float(count) if count else 0.0
    log_loss = log_loss_sum / float(count) if count else 0.0
    summary: dict[str, Any] = {
        "artifact_type": "summary",
        "cache_key": summary_cache_key(config, inference_artifact),
        "implementation": IMPLEMENTATION_VERSION,
        "inference_cache_key": inference_artifact["cache_key"],
        "sorted_by": "sample_id",
        "sample_count": count,
        "correct_count": correct,
        "accuracy": stable_float(accuracy),
        "log_loss": stable_float(log_loss),
        "quantized_integer_sum": integer_sum,
        "quantization": config.quantization_params(),
        "sample_results": sample_results,
    }
    summary["content_fingerprint"] = fingerprint("summary-artifact:v1", summary)
    return summary


def build_summary_artifact(
    config: PipelineConfig,
    predictions: list[dict[str, Any]],
    inference_artifact: dict[str, Any],
) -> dict[str, Any]:
    return _build_summary(config, predictions, inference_artifact)


def infer_and_summarize(
    config: PipelineConfig,
    validation_samples: list[ValidSample],
    feature_artifact: dict[str, Any],
    model_artifact: dict[str, Any],
    *,
    row_cache: dict[str, dict[str, Any]],
) -> InferenceRun:
    dataset = validation_dataset(validation_samples, feature_artifact)
    quantization = config.quantization_params()
    tasks: list[ScheduledTask] = []
    row_keys: dict[str, str] = {}
    newly_cached: dict[str, dict[str, Any]] = {}
    for entry in dataset["entries"]:
        row_key = _row_cache_key(
            entry["sample_id"],
            entry["vector"],
            int(entry["label"]),
            model_artifact["cache_key"],
            quantization,
        )
        row_keys[entry["sample_id"]] = row_key
        cached = row_cache.get(row_key)
        if cached is not None:
            tasks.append(
                ScheduledTask(
                    key=f"inference:{entry['sample_id']}",
                    cost=0,
                    function=lambda cached=cached: cached,
                )
            )
        else:
            tasks.append(
                ScheduledTask(
                    key=f"inference:{entry['sample_id']}",
                    cost=1,
                    function=lambda entry=entry: _predict_row(
                        entry, model_artifact, quantization
                    ),
                )
            )

    scheduled = run_tasks(
        tasks,
        max_workers=int(config.concurrency["max_workers"]),
        budget=None,
        budget_name="inference_rows",
    )
    predictions: list[dict[str, Any]] = []
    rows_reused = 0
    rows_computed = 0
    for entry in dataset["entries"]:
        task_key = f"inference:{entry['sample_id']}"
        prediction = scheduled.results[task_key]
        predictions.append(prediction)
        row_key = row_keys[entry["sample_id"]]
        if row_key in row_cache:
            rows_reused += 1
        else:
            rows_computed += 1
            newly_cached[row_key] = prediction
    predictions.sort(key=lambda item: item["sample_id"])
    updated_cache = dict(row_cache)
    updated_cache.update(newly_cached)

    inference_artifact: dict[str, Any] = {
        "artifact_type": "inference",
        "cache_key": inference_cache_key(
            config, validation_samples, feature_artifact, model_artifact
        ),
        "implementation": IMPLEMENTATION_VERSION,
        "model_cache_key": model_artifact["cache_key"],
        "validation_dataset_fingerprint": validation_dataset_fingerprint(
            validation_samples, feature_artifact
        ),
        "quantization": quantization,
        "fixed_point_denominator": 1 << int(quantization["fractional_bits"]),
        "sample_ids": dataset["sample_ids"],
        "predictions": predictions,
    }
    inference_artifact["content_fingerprint"] = fingerprint(
        "inference-artifact:v1", inference_artifact
    )
    summary = _build_summary(config, predictions, inference_artifact)
    return InferenceRun(
        inference_artifact=inference_artifact,
        summary_artifact=summary,
        row_cache=updated_cache,
        rows_total=len(dataset["entries"]),
        rows_reused=rows_reused,
        rows_computed=rows_computed,
        inference_evaluations=len(dataset["entries"]),
    )
