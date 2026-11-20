from __future__ import annotations

from typing import Any

from .models import CacheEntry, EncodedSample, KeyScore, KeyTrace, ScoreResult, TraceResult


def encoded_to_dict(encoded: EncodedSample) -> dict[str, Any]:
    return {
        "query_position": encoded.query_position,
        "query_vector": list(encoded.query_vector),
        "key_positions": list(encoded.key_positions),
        "key_vectors": [list(row) for row in encoded.key_vectors],
        "kept_chunks": list(encoded.kept_chunks),
        "kept_dimensions": list(encoded.kept_dimensions),
        "pruned_dimensions": list(encoded.pruned_dimensions),
        "effective_base": encoded.effective_base,
        "position_scale": encoded.position_scale,
        "scheme": encoded.scheme,
        "position_fingerprint": encoded.position_fingerprint,
    }


def encoded_from_dict(value: dict[str, Any]) -> EncodedSample:
    return EncodedSample(
        query_position=int(value["query_position"]),
        query_vector=tuple(float(item) for item in value["query_vector"]),
        key_positions=tuple(int(item) for item in value["key_positions"]),
        key_vectors=tuple(tuple(float(item) for item in row) for row in value["key_vectors"]),
        kept_chunks=tuple(int(item) for item in value["kept_chunks"]),
        kept_dimensions=tuple(int(item) for item in value["kept_dimensions"]),
        pruned_dimensions=tuple(int(item) for item in value["pruned_dimensions"]),
        effective_base=float(value["effective_base"]),
        position_scale=float(value["position_scale"]),
        scheme=str(value["scheme"]),
        position_fingerprint=str(value["position_fingerprint"]),
    )


def score_to_dict(result: ScoreResult) -> dict[str, Any]:
    return {
        "entries": [
            {
                "key_position": entry.key_position,
                "relative_distance": entry.relative_distance,
                "interval": entry.interval,
                "score": entry.score,
                "attention_weight": entry.attention_weight,
                "dominant_dimension": entry.dominant_dimension,
                "dominant_contribution": entry.dominant_contribution,
                "contributions": list(entry.contributions),
            }
            for entry in result.entries
        ],
        "has_extrapolation": result.has_extrapolation,
        "sample_interval": result.sample_interval,
        "dominant_dimension": result.dominant_dimension,
        "dominant_contribution_abs": result.dominant_contribution_abs,
        "raw_scores": list(result.raw_scores),
        "attention_weights": list(result.attention_weights),
        "score_fingerprint": result.score_fingerprint,
    }


def score_from_dict(value: dict[str, Any]) -> ScoreResult:
    return ScoreResult(
        entries=tuple(
            KeyScore(
                key_position=int(entry["key_position"]),
                relative_distance=int(entry["relative_distance"]),
                interval=str(entry["interval"]),
                score=float(entry["score"]),
                attention_weight=float(entry["attention_weight"]),
                dominant_dimension=int(entry["dominant_dimension"]),
                dominant_contribution=float(entry["dominant_contribution"]),
                contributions=tuple(float(item) for item in entry["contributions"]),
            )
            for entry in value["entries"]
        ),
        has_extrapolation=bool(value["has_extrapolation"]),
        sample_interval=str(value["sample_interval"]),
        dominant_dimension=int(value["dominant_dimension"]),
        dominant_contribution_abs=float(value["dominant_contribution_abs"]),
        raw_scores=tuple(float(item) for item in value["raw_scores"]),
        attention_weights=tuple(float(item) for item in value["attention_weights"]),
        score_fingerprint=str(value["score_fingerprint"]),
    )


def trace_to_dict(result: TraceResult) -> dict[str, Any]:
    return {
        "reference_query_position": result.reference_query_position,
        "key_traces": [
            {
                "key_position": trace.key_position,
                "relative_distance": trace.relative_distance,
                "interval": trace.interval,
                "reference_query_position": trace.reference_query_position,
                "reference_key_position": trace.reference_key_position,
                "stage_deviations": dict(trace.stage_deviations),
                "first_exceeded_stage": trace.first_exceeded_stage,
                "max_deviation": trace.max_deviation,
            }
            for trace in result.key_traces
        ],
        "stage_max_deviations": dict(result.stage_max_deviations),
        "first_exceeded_stage": result.first_exceeded_stage,
        "violation": result.violation,
        "trace_fingerprint": result.trace_fingerprint,
    }


def trace_from_dict(value: dict[str, Any]) -> TraceResult:
    return TraceResult(
        reference_query_position=int(value["reference_query_position"]),
        key_traces=tuple(
            KeyTrace(
                key_position=int(trace["key_position"]),
                relative_distance=int(trace["relative_distance"]),
                interval=str(trace["interval"]),
                reference_query_position=int(trace["reference_query_position"]),
                reference_key_position=int(trace["reference_key_position"]),
                stage_deviations={
                    str(stage): float(deviation)
                    for stage, deviation in trace["stage_deviations"].items()
                },
                first_exceeded_stage=trace["first_exceeded_stage"],
                max_deviation=float(trace["max_deviation"]),
            )
            for trace in value["key_traces"]
        ),
        stage_max_deviations={
            str(stage): float(deviation) for stage, deviation in value["stage_max_deviations"].items()
        },
        first_exceeded_stage=value["first_exceeded_stage"],
        violation=value["violation"],
        trace_fingerprint=str(value["trace_fingerprint"]),
    )


def entry_result_to_dict(stage: str, value: Any) -> dict[str, Any]:
    if stage == "position_encoding":
        return encoded_to_dict(value)
    if stage == "attention_score":
        return score_to_dict(value)
    if stage == "error_trace":
        return trace_to_dict(value)
    raise ValueError(f"unknown stage: {stage}")


def entry_result_from_dict(stage: str, value: dict[str, Any]) -> Any:
    if stage == "position_encoding":
        return encoded_from_dict(value)
    if stage == "attention_score":
        return score_from_dict(value)
    if stage == "error_trace":
        return trace_from_dict(value)
    raise ValueError(f"unknown stage: {stage}")


def cache_entry_to_dict(entry: CacheEntry) -> dict[str, Any]:
    return {
        "stage": entry.stage,
        "cache_key": entry.cache_key,
        "sample_id": entry.sample_id,
        "scheme": entry.scheme,
        "feature_line": entry.feature_line,
        "source_ref": entry.source_ref,
        "result": entry_result_to_dict(entry.stage, entry.result),
    }


def cache_entry_from_dict(value: dict[str, Any]) -> CacheEntry:
    stage = str(value["stage"])
    return CacheEntry(
        stage=stage,
        cache_key=str(value["cache_key"]),
        sample_id=str(value["sample_id"]),
        scheme=str(value["scheme"]),
        feature_line=int(value["feature_line"]),
        source_ref=str(value["source_ref"]),
        result=entry_result_from_dict(stage, value["result"]),
    )

