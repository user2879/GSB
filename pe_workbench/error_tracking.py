"""Float64 reference path and progressive error attribution."""

from __future__ import annotations

from typing import Any

from .config import WorkbenchConfig
from .encoding import encode_vector
from .hashing import stable_float
from .models import Sample
from .quantization import (
    float64_dot,
    quantization_profile,
    quantize_vector,
    reference_score,
    softmax_after_max_shift,
)


STAGE_ORDER = ("position_encoding", "dot_score", "attention_weight")
STAGE_TOLERANCE = {
    "position_encoding": "position_vector",
    "dot_score": "dot_score",
    "attention_weight": "attention_weight",
}


def classify_region(distance: int, train_context_length: int) -> str:
    return "extrapolation" if distance >= train_context_length else "in_training"


def _max_abs_delta(left: list[float], right: list[float]) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def build_reference(
    sample: Sample,
    config: WorkbenchConfig,
) -> dict[str, Any]:
    query = encode_vector(
        sample.query_vector,
        sample.query_position,
        scheme=config.scheme,
        base_frequency=config.base_frequency,
        dimension=config.dimension,
        scale_factor=config.scale_factor,
        precision_level="fp64",
    )
    keys = [
        encode_vector(
            key_vector,
            key_position,
            scheme=config.scheme,
            base_frequency=config.base_frequency,
            dimension=config.dimension,
            scale_factor=config.scale_factor,
            precision_level="fp64",
        )
        for key_position, key_vector in zip(sample.key_positions, sample.key_vectors)
    ]
    scores = [
        reference_score(float64_dot(query.vector, key.vector), config.dimension)
        for key in keys
    ]
    normalization = softmax_after_max_shift([stable_float(score) for score in scores])
    return {
        "query": query,
        "keys": keys,
        "scores": [stable_float(score) for score in scores],
        "normalization": normalization,
    }


def track_errors(
    sample: Sample,
    position_artifact: dict[str, Any],
    score_artifact: dict[str, Any],
    config: WorkbenchConfig,
    *,
    cache_key: str,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference = reference or build_reference(sample, config)
    quant_profile = quantization_profile(
        config.quantization_level, config.fixed_decimal_bits
    )
    quantized_query = quantize_vector(
        position_artifact["query"]["vector"],
        level=config.quantization_level,
        fixed_decimal_bits=config.fixed_decimal_bits,
    )
    key_records: list[dict[str, Any]] = []
    extrapolation_records: list[dict[str, Any]] = []

    reference_weights = reference["normalization"]["weights"]
    fixed_weights = score_artifact["normalization"]["weights"]
    for index, key_position in enumerate(sample.key_positions):
        distance = abs(sample.query_position - key_position)
        region = classify_region(distance, config.train_context_length)
        quantized_key = quantize_vector(
            position_artifact["keys"][index]["vector"],
            level=config.quantization_level,
            fixed_decimal_bits=config.fixed_decimal_bits,
        )
        query_position_delta = _max_abs_delta(
            quantized_query["dequantized"], list(reference["query"].vector)
        )
        key_position_delta = _max_abs_delta(
            quantized_key["dequantized"], list(reference["keys"][index].vector)
        )
        position_delta = max(query_position_delta, key_position_delta)
        fixed_score = float(score_artifact["keys"][index]["score"])
        reference_score_value = float(reference["scores"][index])
        dot_delta = abs(fixed_score - reference_score_value)
        fixed_weight = float(fixed_weights[index]) if fixed_weights else 0.0
        reference_weight = float(reference_weights[index]) if reference_weights else 0.0
        attention_delta = abs(fixed_weight - reference_weight)
        record = {
            "index": index,
            "key_position": key_position,
            "relative_distance": distance,
            "region": region,
            "quantized_key_clipped_count": quantized_key["clipped_count"],
            "query_position_deviation": stable_float(query_position_delta),
            "key_position_deviation": stable_float(key_position_delta),
            "position_deviation": stable_float(position_delta),
            "dot_score_deviation": stable_float(dot_delta),
            "attention_weight_deviation": stable_float(attention_delta),
            "fixed_score": stable_float(fixed_score),
            "reference_score": stable_float(reference_score_value),
            "fixed_weight": stable_float(fixed_weight),
            "reference_weight": stable_float(reference_weight),
            "fixed_max_index": score_artifact["normalization"]["max_index"],
            "reference_max_index": reference["normalization"]["max_index"],
            "fixed_denominator": score_artifact["normalization"]["denominator"],
            "reference_denominator": reference["normalization"]["denominator"],
        }
        key_records.append(record)
        if region == "extrapolation":
            extrapolation_records.append(record)

    evaluated = bool(extrapolation_records)
    stage_errors = {
        "position_encoding": stable_float(
            max((record["position_deviation"] for record in extrapolation_records), default=0.0)
        )
        if evaluated
        else None,
        "dot_score": stable_float(
            max((record["dot_score_deviation"] for record in extrapolation_records), default=0.0)
        )
        if evaluated
        else None,
        "attention_weight": stable_float(
            max(
                (record["attention_weight_deviation"] for record in extrapolation_records),
                default=0.0,
            )
        )
        if evaluated
        else None,
    }
    tolerances = config.tolerances.to_dict()
    first_exceeded_stage: str | None = None
    first_exceeded_value: float | None = None
    if evaluated:
        for stage in STAGE_ORDER:
            value = stage_errors[stage]
            assert value is not None
            if value > float(tolerances[STAGE_TOLERANCE[stage]]):
                first_exceeded_stage = stage
                first_exceeded_value = value
                break
    attention_overtolerance = bool(
        evaluated
        and stage_errors["attention_weight"] is not None
        and stage_errors["attention_weight"] > float(tolerances["attention_weight"])
    )
    return {
        "stage": "error_tracking",
        "cache_key": cache_key,
        "sample_id": sample.sample_id,
        "scheme": config.scheme,
        "scheme_display": config.scheme_display,
        "precision_level": config.precision_level,
        "quantization_level": config.quantization_level,
        "fixed_decimal_bits": config.fixed_decimal_bits,
        "quantization_profile": quant_profile,
        "train_context_length": config.train_context_length,
        "tolerances": tolerances,
        "evaluated": evaluated,
        "stage_errors": stage_errors,
        "first_exceeded_stage": first_exceeded_stage,
        "first_exceeded_value": first_exceeded_value,
        "attention_overtolerance": attention_overtolerance,
        "keys": key_records,
        "reference": {
            "precision": "fp64",
            "scores": reference["scores"],
            "weights": reference_weights,
            "max_index": reference["normalization"]["max_index"],
            "denominator": reference["normalization"]["denominator"],
        },
        "fixed": {
            "scores": [record["fixed_score"] for record in key_records],
            "weights": [record["fixed_weight"] for record in key_records],
            "max_index": score_artifact["normalization"]["max_index"],
            "denominator": score_artifact["normalization"]["denominator"],
        },
        "sort_key": f"{sample.sample_id}|error_tracking|{config.scheme}",
    }
