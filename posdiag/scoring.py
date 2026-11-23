from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Sequence

import numpy as np

from .encoding import EncodedSequence, encode_sequence
from .models import AttentionSample, PositionConfig


ERROR_STAGES = ("position_encoding", "attention_score", "attention_weight")
NO_ERROR_STAGE = "none"


@dataclass(frozen=True)
class AttentionResult:
    scores: tuple[float, ...]
    weights: tuple[float, ...]
    distances: tuple[int, ...]
    buckets: tuple[str, ...]


def classify_distance(distance: int, training_context_length: int) -> str:
    return "in_training" if distance < training_context_length else "extrapolation"


def stable_softmax(scores: Sequence[float]) -> tuple[float, ...]:
    if not scores:
        return ()
    values = np.asarray(scores, dtype=np.float64)
    maximum = float(np.max(values))
    exponentials = np.exp(values - maximum)
    denominator = float(np.sum(exponentials, dtype=np.float64))
    if denominator == 0.0 or not math.isfinite(denominator):
        raise ValueError("softmax normalization failed")
    return tuple(float(value) for value in (exponentials / denominator).tolist())


def attention_scores(
    query_vector: np.ndarray,
    key_vectors: Sequence[np.ndarray],
    query_position: int,
    key_positions: Sequence[int],
    training_context_length: int,
) -> AttentionResult:
    query = np.asarray(query_vector, dtype=np.float64)
    scores: list[float] = []
    distances: list[int] = []
    buckets: list[str] = []
    for key_vector, key_position in zip(key_vectors, key_positions, strict=True):
        key = np.asarray(key_vector, dtype=np.float64)
        score = float(np.dot(query, key))
        distance = abs(int(query_position) - int(key_position))
        scores.append(score)
        distances.append(distance)
        buckets.append(classify_distance(distance, training_context_length))
    weights = stable_softmax(scores)
    return AttentionResult(
        scores=tuple(scores),
        weights=weights,
        distances=tuple(distances),
        buckets=tuple(buckets),
    )


def _value_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(fmean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def sample_tier_summary(result: AttentionResult) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {"in_training": [], "extrapolation": []}
    for score, bucket in zip(result.scores, result.buckets, strict=True):
        grouped[bucket].append(score)
    return {
        "in_training": _value_summary(grouped["in_training"]),
        "extrapolation": _value_summary(grouped["extrapolation"]),
        "empty_keys": not result.scores,
        "self_attention_pairs": sum(1 for distance in result.distances if distance == 0),
    }


def attention_cache_entry(
    config: PositionConfig,
    sample: AttentionSample,
    scheme: str,
    result: AttentionResult,
) -> dict[str, Any]:
    return {
        "stage": "attention",
        "sample_id": sample.sample_id,
        "scheme": scheme,
        "training_context_length": config.training_context_length,
        "scores": [float(value) for value in result.scores],
        "weights": [float(value) for value in result.weights],
        "distances": [int(value) for value in result.distances],
        "buckets": list(result.buckets),
        "tier_summary": sample_tier_summary(result),
    }


def attention_result_from_payload(payload: dict[str, Any]) -> AttentionResult:
    return AttentionResult(
        scores=tuple(float(value) for value in payload["scores"]),
        weights=tuple(float(value) for value in payload["weights"]),
        distances=tuple(int(value) for value in payload["distances"]),
        buckets=tuple(str(value) for value in payload["buckets"]),
    )


def _relative_vector_error(actual: np.ndarray, reference: np.ndarray) -> float:
    difference = float(np.linalg.norm(actual - reference))
    denominator = max(float(np.linalg.norm(reference)), 1e-300)
    return difference / denominator


def _relative_scalar_error(actual: float, reference: float) -> float:
    return abs(float(actual) - float(reference)) / max(abs(float(reference)), 1e-12)


def _first_exceeded(values: dict[str, float], tolerance: float) -> str:
    for stage in ERROR_STAGES:
        if values[stage] > tolerance:
            return stage
    return NO_ERROR_STAGE


@dataclass(frozen=True)
class ErrorTrace:
    position_encoding: float
    attention_score: float
    attention_weight: float
    first_exceeded: str
    reference_query_position: int
    reference_key_positions: tuple[int, ...]
    external_query: bool
    external_key_count: int

    def deviations(self) -> dict[str, float]:
        return {
            "position_encoding": self.position_encoding,
            "attention_score": self.attention_score,
            "attention_weight": self.attention_weight,
        }


def trace_errors(
    config: PositionConfig,
    sample: AttentionSample,
    scheme: str,
    actual_encoded: EncodedSequence,
    actual_attention: AttentionResult,
) -> ErrorTrace:
    training_boundary = config.training_context_length - 1
    reference_query_position = min(sample.query_position, training_boundary)
    reference_key_positions = tuple(min(position, training_boundary) for position in sample.key_positions)
    reference_encoded = encode_sequence(
        config,
        sample,
        scheme,
        query_position=reference_query_position,
        key_positions=reference_key_positions,
    )
    reference_attention = attention_scores(
        reference_encoded.query_vector,
        reference_encoded.key_vectors,
        reference_query_position,
        reference_key_positions,
        config.training_context_length,
    )

    position_errors: list[float] = []
    if sample.query_position >= config.training_context_length:
        position_errors.append(_relative_vector_error(actual_encoded.query_vector, reference_encoded.query_vector))
    external_key_count = 0
    for index, key_position in enumerate(sample.key_positions):
        if key_position >= config.training_context_length:
            external_key_count += 1
            position_errors.append(
                _relative_vector_error(actual_encoded.key_vectors[index], reference_encoded.key_vectors[index])
            )
    position_error = max(position_errors, default=0.0)

    score_errors: list[float] = []
    for index, key_position in enumerate(sample.key_positions):
        if sample.query_position >= config.training_context_length or key_position >= config.training_context_length:
            score_errors.append(
                _relative_scalar_error(actual_attention.scores[index], reference_attention.scores[index])
            )
    score_error = max(score_errors, default=0.0)

    if actual_attention.weights and reference_attention.weights:
        weight_error = max(
            abs(actual - reference)
            for actual, reference in zip(actual_attention.weights, reference_attention.weights, strict=True)
        )
    else:
        weight_error = 0.0

    deviations = {
        "position_encoding": float(position_error),
        "attention_score": float(score_error),
        "attention_weight": float(weight_error),
    }
    return ErrorTrace(
        position_encoding=deviations["position_encoding"],
        attention_score=deviations["attention_score"],
        attention_weight=deviations["attention_weight"],
        first_exceeded=_first_exceeded(deviations, config.tolerance),
        reference_query_position=reference_query_position,
        reference_key_positions=reference_key_positions,
        external_query=sample.query_position >= config.training_context_length,
        external_key_count=external_key_count,
    )


def error_cache_entry(
    config: PositionConfig,
    sample: AttentionSample,
    scheme: str,
    trace: ErrorTrace,
) -> dict[str, Any]:
    return {
        "stage": "error_trace",
        "sample_id": sample.sample_id,
        "scheme": scheme,
        "tolerance": config.tolerance,
        "deviations": trace.deviations(),
        "first_exceeded": trace.first_exceeded,
        "reference_query_position": trace.reference_query_position,
        "reference_key_positions": list(trace.reference_key_positions),
        "external_query": trace.external_query,
        "external_key_count": trace.external_key_count,
        "reference_rule": "clamp-to-training-boundary-L-1",
    }


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def distribution_summary(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
            "values": [],
        }
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": float(fmean(ordered)),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
        "values": ordered,
    }


def aggregate_layer_statistics(
    attention_entries: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for entry in sorted(attention_entries, key=lambda item: (item["sample_id"], item["scheme"])):
        scheme = str(entry["scheme"])
        scheme_stats = output.setdefault(
            scheme,
            {
                "in_training": {"all_scores": [], "by_distance": {}},
                "extrapolation": {"all_scores": [], "by_distance": {}},
            },
        )
        for score, distance, bucket in zip(entry["scores"], entry["distances"], entry["buckets"], strict=True):
            bucket_stats = scheme_stats[str(bucket)]
            bucket_stats["all_scores"].append(float(score))
            by_distance = bucket_stats["by_distance"].setdefault(str(int(distance)), [])
            by_distance.append(float(score))

    rendered: dict[str, dict[str, Any]] = {}
    for scheme in sorted(output):
        rendered[scheme] = {}
        for bucket in ("in_training", "extrapolation"):
            bucket_data = output[scheme][bucket]
            by_distance = {
                distance: _value_summary(values)
                for distance, values in sorted(
                    bucket_data["by_distance"].items(), key=lambda item: int(item[0])
                )
            }
            rendered[scheme][bucket] = {
                **_value_summary(bucket_data["all_scores"]),
                "by_distance": by_distance,
            }
    return rendered


def aggregate_error_distributions(trace_entries: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for entry in sorted(trace_entries, key=lambda item: (item["sample_id"], item["scheme"])):
        scheme = str(entry["scheme"])
        scheme_group = grouped.setdefault(scheme, {stage: [] for stage in ERROR_STAGES})
        deviations = entry["deviations"]
        for stage in ERROR_STAGES:
            scheme_group[stage].append(float(deviations[stage]))
    return {
        scheme: {stage: distribution_summary(values) for stage, values in stage_map.items()}
        for scheme, stage_map in sorted(grouped.items())
    }


def boundary_reasons(config: PositionConfig, sample: AttentionSample) -> list[str]:
    reasons: list[str] = []
    if sample.empty_keys:
        reasons.append("empty_key_list")
    if sample.self_attention:
        reasons.append("self_attention")
    boundary_distance = config.training_context_length
    if any(abs(sample.query_position - key_position) == boundary_distance for key_position in sample.key_positions):
        reasons.append("relative_distance_at_training_boundary")
    training_boundary = config.training_context_length - 1
    if sample.query_position in (training_boundary, config.training_context_length):
        reasons.append("query_at_training_boundary")
    if any(position in (training_boundary, config.training_context_length) for position in sample.key_positions):
        reasons.append("key_at_training_boundary")
    return reasons
