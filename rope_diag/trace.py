from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .attention import ScoreResult, score_attention
from .config import DiagnosticConfig
from .models import EncodedSample
from .rope import (
    apply_input_precision_and_quantization,
    apply_rotation,
    apply_rotation_batch,
    generate_position_encoding,
)
from .samples import AttentionSample


STAGE_ORDER = ("position_vector", "score", "attention")


def _stage_tolerance(config: DiagnosticConfig, stage: str) -> float:
    return {
        "position_vector": config.tolerances.position_vector,
        "score": config.tolerances.score,
        "attention": config.tolerances.attention,
    }[stage]


def trace_sample_errors(
    sample: AttentionSample,
    encoded: EncodedSample,
    score_result: ScoreResult,
    config: DiagnosticConfig,
    dimension_indices: Sequence[int],
    chunk_size: int,
) -> dict[str, Any] | None:
    distances = sample.relative_distances()
    extrapolation_mask = distances >= config.training_context_length
    if not np.any(extrapolation_mask):
        return None

    reference_limit = config.training_context_length - 1
    reference_query_position = min(sample.query_position, reference_limit)
    reference_key_positions = [min(position, reference_limit) for position in sample.key_positions]

    query_prepared = apply_input_precision_and_quantization(sample.query_features, config)
    keys_prepared = np.vstack(
        [apply_input_precision_and_quantization(row, config) for row in sample.key_features]
    )
    reference_query_encoding = generate_position_encoding(
        encoded.scheme, reference_query_position, config
    )
    reference_key_encodings = tuple(
        generate_position_encoding(encoded.scheme, position, config)
        for position in reference_key_positions
    )
    reference_query_rotated = apply_rotation(query_prepared, reference_query_encoding)
    reference_keys_rotated = apply_rotation_batch(keys_prepared, reference_key_encodings)
    reference_scores = score_attention(
        reference_query_rotated, reference_keys_rotated, dimension_indices, chunk_size
    )

    query_position_error = float(
        np.linalg.norm(encoded.query_encoding.vector - reference_query_encoding.vector, ord=2)
        / np.sqrt(float(config.dimension))
    )
    key_position_errors = np.asarray(
        [
            np.linalg.norm(actual.vector - reference.vector, ord=2)
            / np.sqrt(float(config.dimension))
            for actual, reference in zip(encoded.key_encodings, reference_key_encodings)
        ],
        dtype=np.float64,
    )
    selected_key_errors = key_position_errors[extrapolation_mask]
    position_error = float(
        np.mean(np.concatenate((np.asarray([query_position_error]), selected_key_errors)))
    )
    score_error = float(
        np.mean(
            np.abs(score_result.scores[extrapolation_mask] - reference_scores.scores[extrapolation_mask])
        )
    )
    attention_error = float(
        np.mean(
            np.abs(
                score_result.weights[extrapolation_mask]
                - reference_scores.weights[extrapolation_mask]
            )
        )
    )

    raw_errors = {
        "position_vector": position_error,
        "score": score_error,
        "attention": attention_error,
    }
    stages: list[dict[str, Any]] = []
    first_exceeded: str | None = None
    for stage in STAGE_ORDER:
        tolerance = _stage_tolerance(config, stage)
        exceeded = raw_errors[stage] > tolerance
        if exceeded and first_exceeded is None:
            first_exceeded = stage
        stages.append(
            {
                "stage": stage,
                "deviation": raw_errors[stage],
                "tolerance": tolerance,
                "exceeded": exceeded,
            }
        )
    return {
        "sample_id": sample.sample_id,
        "query_position": sample.query_position,
        "reference_query_position": reference_query_position,
        "reference_key_positions": reference_key_positions,
        "extrapolation_key_count": int(np.sum(extrapolation_mask)),
        "query_position_vector_deviation": query_position_error,
        "key_position_vector_deviation_mean": float(np.mean(selected_key_errors)),
        "stages": stages,
        "first_exceeded_stage": first_exceeded,
        "score_reference": "clip_each_position_to_training_context_length_minus_one",
    }


def compute_error_trace(
    scheme: str,
    samples: Sequence[AttentionSample],
    encoded_samples: Mapping[str, EncodedSample],
    score_results: Mapping[str, ScoreResult],
    config: DiagnosticConfig,
    dimension_indices: Sequence[int],
    chunk_size: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda item: item.sample_id):
        encoded = encoded_samples.get(sample.sample_id)
        score_result = score_results.get(sample.sample_id)
        if encoded is None or score_result is None:
            continue
        record = trace_sample_errors(
            sample,
            encoded,
            score_result,
            config,
            dimension_indices,
            chunk_size,
        )
        if record is not None:
            records.append(record)

    first_exceeded_counts = {stage: 0 for stage in STAGE_ORDER}
    for record in records:
        stage = record["first_exceeded_stage"]
        if stage is not None:
            first_exceeded_counts[stage] += 1

    summary: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        deviations = [
            next(item["deviation"] for item in record["stages"] if item["stage"] == stage)
            for record in records
        ]
        summary.append(
            {
                "stage": stage,
                "mean_deviation": float(np.mean(np.asarray(deviations, dtype=np.float64)))
                if deviations
                else None,
                "max_deviation": float(np.max(np.asarray(deviations, dtype=np.float64)))
                if deviations
                else None,
                "tolerance": _stage_tolerance(config, stage),
                "first_exceeded_sample_count": first_exceeded_counts[stage],
            }
        )
    return {
        "status": "ok" if records else "no_extrapolation_records",
        "reference_rule": "clip_each_position_to_training_context_length_minus_one",
        "stage_order": list(STAGE_ORDER),
        "stages": summary,
        "samples": records,
        "overall_first_exceeded": (
            next(
                (
                    stage
                    for stage in STAGE_ORDER
                    if any(record["first_exceeded_stage"] == stage for record in records)
                ),
                None,
            )
            if records
            else None
        ),
    }

