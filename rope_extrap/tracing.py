from __future__ import annotations

from typing import Iterable

import numpy as np

from .canonical import sha256_json
from .encoding import encode_sample
from .errors import E_COMPUTE_NONFINITE, BusinessError
from .models import (
    AttentionSample,
    EncodedSample,
    KeyTrace,
    PositionEncodingConfig,
    ScoreResult,
    TraceResult,
)


TRACE_STAGES = ("position_vector", "score", "attention_weight")


def _relative_difference(actual: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), 1.0e-12)
    return float(np.linalg.norm(actual - reference) / denominator)


def _relative_scalar_difference(actual: float, reference: float) -> float:
    return abs(float(actual) - float(reference)) / max(abs(float(reference)), 1.0e-12)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - float(np.max(scores))
    exponentials = np.exp(shifted)
    denominator = float(np.sum(exponentials, dtype=np.float64))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("softmax denominator is not positive and finite")
    return exponentials / denominator


def trace_sample(
    config: PositionEncodingConfig,
    sample: AttentionSample,
    encoded: EncodedSample,
    scores: ScoreResult,
) -> TraceResult:
    reference_query_position = min(sample.query_position, config.training_context_length)
    reference_key_positions = tuple(
        min(position, config.training_context_length) for position in sample.key_positions
    )
    reference_encoded = encode_sample(
        config,
        sample,
        positions=(reference_query_position, reference_key_positions),
        kept_chunks=encoded.kept_chunks,
    )
    actual_query = np.asarray(encoded.query_vector, dtype=np.float64)
    actual_keys = np.asarray(encoded.key_vectors, dtype=np.float64)
    reference_query = np.asarray(reference_encoded.query_vector, dtype=np.float64)
    reference_keys = np.asarray(reference_encoded.key_vectors, dtype=np.float64)
    reference_scores = np.asarray(reference_keys @ reference_query, dtype=np.float64)
    if not np.all(np.isfinite(reference_scores)):
        raise BusinessError(
            E_COMPUTE_NONFINITE,
            "reference attention score is not finite",
            stage="error_trace",
            sample_id=sample.sample_id,
        )
    reference_weights = _softmax(reference_scores)
    actual_weights = np.asarray(scores.attention_weights, dtype=np.float64)

    query_position_deviation = _relative_difference(actual_query, reference_query)
    key_traces: list[KeyTrace] = []
    external_traces: list[KeyTrace] = []
    for index, entry in enumerate(scores.entries):
        key_position_deviation = _relative_difference(actual_keys[index], reference_keys[index])
        position_deviation = max(query_position_deviation, key_position_deviation)
        score_deviation = _relative_scalar_difference(
            entry.score,
            float(reference_scores[index]),
        )
        weight_deviation = _relative_scalar_difference(
            float(actual_weights[index]),
            float(reference_weights[index]),
        )
        deviations = {
            "position_vector": position_deviation,
            "score": score_deviation,
            "attention_weight": weight_deviation,
        }
        first_exceeded_stage: str | None = None
        for stage in TRACE_STAGES:
            if deviations[stage] > config.error_tolerance:
                first_exceeded_stage = stage
                break
        trace = KeyTrace(
            key_position=entry.key_position,
            relative_distance=entry.relative_distance,
            interval=entry.interval,
            reference_query_position=reference_query_position,
            reference_key_position=reference_key_positions[index],
            stage_deviations=deviations,
            first_exceeded_stage=first_exceeded_stage,
            max_deviation=max(deviations.values()),
        )
        key_traces.append(trace)
        if entry.interval == "extrapolation":
            external_traces.append(trace)

    stage_max_deviations = {
        stage: (max(trace.stage_deviations[stage] for trace in external_traces) if external_traces else 0.0)
        for stage in TRACE_STAGES
    }
    worst_trace = (
        max(external_traces, key=lambda trace: (trace.max_deviation, -trace.relative_distance, trace.key_position))
        if external_traces
        else None
    )
    violation: dict[str, object] | None = None
    if worst_trace is not None and worst_trace.first_exceeded_stage is not None:
        over_tolerance_traces = [trace for trace in external_traces if trace.first_exceeded_stage is not None]
        violation = {
            "sample_id": sample.sample_id,
            "key_position": worst_trace.key_position,
            "relative_distance": worst_trace.relative_distance,
            "first_exceeded_stage": worst_trace.first_exceeded_stage,
            "max_deviation": worst_trace.max_deviation,
            "stage_deviations": dict(worst_trace.stage_deviations),
            "over_tolerance_key_count": len(over_tolerance_traces),
        }
    fingerprint = sha256_json(
        {
            "scheme": encoded.scheme,
            "actual_position_fingerprint": encoded.position_fingerprint,
            "reference_position_fingerprint": reference_encoded.position_fingerprint,
            "reference_scores": [float(value) for value in reference_scores],
            "reference_weights": [float(value) for value in reference_weights],
            "tolerance": config.error_tolerance,
        }
    )
    return TraceResult(
        reference_query_position=reference_query_position,
        key_traces=tuple(key_traces),
        stage_max_deviations=stage_max_deviations,
        first_exceeded_stage=(worst_trace.first_exceeded_stage if worst_trace is not None else None),
        violation=violation,
        trace_fingerprint=fingerprint,
    )


def key_trace_dict(trace: KeyTrace) -> dict[str, object]:
    return {
        "key_position": trace.key_position,
        "relative_distance": trace.relative_distance,
        "interval": trace.interval,
        "reference_query_position": trace.reference_query_position,
        "reference_key_position": trace.reference_key_position,
        "stage_deviations": dict(trace.stage_deviations),
        "first_exceeded_stage": trace.first_exceeded_stage,
        "max_deviation": trace.max_deviation,
    }


def aggregate_external_trace_statistics(
    trace_results: dict[str, TraceResult],
    score_results: dict[str, ScoreResult],
) -> dict[str, object]:
    external_traces: list[KeyTrace] = []
    for sample_id in sorted(trace_results):
        external_traces.extend(
            trace for trace in trace_results[sample_id].key_traces if trace.interval == "extrapolation"
        )
    exceeding = [trace for trace in external_traces if trace.first_exceeded_stage is not None]
    stage_counts = {stage: 0 for stage in TRACE_STAGES}
    for trace in exceeding:
        if trace.first_exceeded_stage is not None:
            stage_counts[trace.first_exceeded_stage] += 1
    return {
        "extrapolation_key_count": len(external_traces),
        "over_tolerance_key_count": len(exceeding),
        "extrapolation_sample_count": sum(
            trace_results[sample_id].violation is not None for sample_id in sorted(trace_results)
        ),
        "first_exceeded_stage_counts": stage_counts,
        "mean_position_vector_deviation": _mean(
            [trace.stage_deviations["position_vector"] for trace in external_traces]
        ),
        "max_position_vector_deviation": _max(
            [trace.stage_deviations["position_vector"] for trace in external_traces]
        ),
        "mean_score_deviation": _mean([trace.stage_deviations["score"] for trace in external_traces]),
        "max_score_deviation": _max([trace.stage_deviations["score"] for trace in external_traces]),
        "mean_attention_weight_deviation": _mean(
            [trace.stage_deviations["attention_weight"] for trace in external_traces]
        ),
        "max_attention_weight_deviation": _max(
            [trace.stage_deviations["attention_weight"] for trace in external_traces]
        ),
        "mean_max_deviation": _mean([trace.max_deviation for trace in external_traces]),
        "max_max_deviation": _max([trace.max_deviation for trace in external_traces]),
    }


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return float(np.mean(materialized, dtype=np.float64))


def _max(values: Iterable[float]) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return float(np.max(materialized))
