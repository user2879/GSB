"""Three-stage extrapolation error propagation tracing."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .attention import AttentionResult
from .config import SCHEME_RANK, Tolerances
from .encoding import SampleEncoding
from .util import clean_float


STAGE_ORDER = ("position_vector", "dot_score", "normalized_attention_weight")


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values, dtype=np.float64))))


def trace_attention(
    attention: AttentionResult,
    encoding: SampleEncoding,
    tolerances: Tolerances,
    train_context_length: int,
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    reference_encoding = attention.reference_encoding
    query_position_delta = encoding.query_encoded - reference_encoding.query_encoded
    query_position_error = _max_abs(query_position_delta)
    query_l2 = float(np.linalg.norm(query_position_delta) / np.sqrt(float(encoding.dimension)))

    tolerance_values = {
        "position_vector": tolerances.position_vector,
        "dot_score": tolerances.dot_score,
        "normalized_attention_weight": tolerances.normalized_attention_weight,
    }

    for key_index, layer in enumerate(attention.layers):
        if layer != "out_of_train":
            continue
        key_position_delta = encoding.key_encoded[key_index] - reference_encoding.key_encoded[key_index]
        key_position_error = _max_abs(key_position_delta)
        key_l2 = float(np.linalg.norm(key_position_delta) / np.sqrt(float(encoding.dimension)))
        position_error = max(query_position_error, key_position_error)
        score_error = abs(
            float(attention.scores[key_index]) - float(attention.reference_scores[key_index])
        )
        weight_error = abs(
            float(attention.weights[key_index]) - float(attention.reference_weights[key_index])
        )
        deviations = {
            "position_vector": position_error,
            "dot_score": score_error,
            "normalized_attention_weight": weight_error,
        }
        stage_details: dict[str, dict[str, Any]] = {}
        exceeded_flags: list[bool] = []
        for stage in STAGE_ORDER:
            exceeded = bool(deviations[stage] > tolerance_values[stage])
            exceeded_flags.append(exceeded)
            stage_details[stage] = {
                "deviation": clean_float(deviations[stage]),
                "tolerance": clean_float(tolerance_values[stage]),
                "exceeded": exceeded,
            }
        if stage_details["position_vector"] is not None:
            stage_details["position_vector"]["query_max_abs"] = clean_float(query_position_error)
            stage_details["position_vector"]["key_max_abs"] = clean_float(key_position_error)
            stage_details["position_vector"]["query_normalized_l2"] = clean_float(query_l2)
            stage_details["position_vector"]["key_normalized_l2"] = clean_float(key_l2)

        first_exceeded = next(
            (stage for stage, exceeded in zip(STAGE_ORDER, exceeded_flags) if exceeded),
            None,
        )
        traces.append(
            {
                "scheme": attention.scheme,
                "sample_id": attention.sample_id,
                "query_position": int(attention.query_position),
                "key_index": int(key_index),
                "key_position": int(attention.key_positions[key_index]),
                "relative_distance": int(attention.relative_distances[key_index]),
                "layer": "out_of_train",
                "stage_order": list(STAGE_ORDER),
                "stage_deviations": stage_details,
                "first_exceeded_stage": first_exceeded,
                "later_stage_values_recorded": True,
                "contradictory_tolerance_judgement": len(set(exceeded_flags)) > 1,
                "reference_policy": "analytic standard-RoPE extension at the actual position",
                "deviation_precision": "float64",
            }
        )
    return traces


def summarize_traces(traces: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        grouped.setdefault(str(trace["scheme"]), []).append(trace)
    summary: list[dict[str, Any]] = []
    for scheme in sorted(grouped, key=lambda item: (SCHEME_RANK.get(item, 99), item)):
        items = grouped[scheme]
        counts = {stage: 0 for stage in STAGE_ORDER}
        counts["none"] = 0
        for item in items:
            key = item["first_exceeded_stage"] or "none"
            counts[key] = counts.get(key, 0) + 1
        summary.append(
            {
                "scheme": scheme,
                "trace_count": len(items),
                "sample_count": len({str(item["sample_id"]) for item in items}),
                "first_exceeded_stage_counts": counts,
                "contradictory_tolerance_judgement_count": sum(
                    1 for item in items if item["contradictory_tolerance_judgement"]
                ),
            }
        )
    return summary
