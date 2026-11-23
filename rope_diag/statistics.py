from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .attention import ScoreResult
from .config import DiagnosticConfig
from .errors import ErrorCode
from .samples import AttentionSample


def _aggregate_values(
    values: Sequence[float], sample_ids: Sequence[str]
) -> dict[str, Any]:
    if not values:
        return {
            "score_count": 0,
            "sample_count": 0,
            "mean": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "score_count": int(array.size),
        "sample_count": len(set(sample_ids)),
        "mean": float(np.mean(array, dtype=np.float64)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def distance_bucket(distance: int, training_context_length: int) -> str:
    return "in_training" if distance < training_context_length else "extrapolation"


def compute_statistics(
    samples: Sequence[AttentionSample],
    score_results: Mapping[str, ScoreResult],
    config: DiagnosticConfig,
) -> dict[str, Any]:
    by_distance_values: dict[tuple[int, str], list[float]] = defaultdict(list)
    by_distance_samples: dict[tuple[int, str], list[str]] = defaultdict(list)
    in_values: list[float] = []
    in_samples: list[str] = []
    out_values: list[float] = []
    out_samples: list[str] = []

    for sample in samples:
        result = score_results.get(sample.sample_id)
        if result is None:
            continue
        distances = sample.relative_distances()
        if result.scores.shape[0] != distances.shape[0]:
            continue
        for distance_value, score in zip(distances.tolist(), result.scores.tolist()):
            distance = int(distance_value)
            bucket = distance_bucket(distance, config.training_context_length)
            key = (distance, bucket)
            by_distance_values[key].append(float(score))
            by_distance_samples[key].append(sample.sample_id)
            if bucket == "in_training":
                in_values.append(float(score))
                in_samples.append(sample.sample_id)
            else:
                out_values.append(float(score))
                out_samples.append(sample.sample_id)

    by_distance: list[dict[str, Any]] = []
    for distance, bucket in sorted(by_distance_values):
        aggregate = _aggregate_values(
            by_distance_values[(distance, bucket)], by_distance_samples[(distance, bucket)]
        )
        by_distance.append({"distance": distance, "bucket": bucket, **aggregate})

    in_training = _aggregate_values(in_values, in_samples)
    extrapolation = _aggregate_values(out_values, out_samples)
    status = "ok" if extrapolation["score_count"] > 0 else ErrorCode.NO_EXTRAPOLATION_SAMPLE
    payload: dict[str, Any] = {
        "status": status,
        "training_context_length": config.training_context_length,
        "classification_rule": "relative_distance < training_context_length => in_training",
        "in_training": {**in_training, "bucket": "in_training"},
        "extrapolation": {**extrapolation, "bucket": "extrapolation"},
        "by_distance": by_distance,
    }
    if status != "ok":
        payload["error"] = {
            "code": ErrorCode.NO_EXTRAPOLATION_SAMPLE,
            "message": "外推区间内样本数为零，已终止该方案的统计汇总",
        }
    return payload


def extrapolation_scores(
    samples: Sequence[AttentionSample],
    score_results: Mapping[str, ScoreResult],
    training_context_length: int,
) -> list[float]:
    values: list[float] = []
    for sample in samples:
        result = score_results.get(sample.sample_id)
        if result is None:
            continue
        distances = sample.relative_distances()
        for index, distance in enumerate(distances.tolist()):
            if int(distance) >= training_context_length:
                values.append(float(result.scores[index]))
    return values

