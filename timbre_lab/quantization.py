from __future__ import annotations

import math
from typing import Any

from .config import AnalysisConfig, digest_json


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def quantize_value(value: float, low: float, high: float, bits: int) -> dict[str, Any]:
    if not math.isfinite(value):
        value = low
    max_integer = (1 << bits) - 1
    step = (high - low) / max_integer
    clamped = min(max(value, low), high)
    integer = min(max(_round_half_up((clamped - low) / step), 0), max_integer)
    reconstructed = low + integer * step
    normalized = integer / max_integer
    return {
        "bit_width": bits,
        "levels": max_integer + 1,
        "step": step,
        "integer": integer,
        "normalized": normalized,
        "reconstructed": reconstructed,
        "residual": value - reconstructed,
        "value": value,
    }


def description_cache_key(feature_cache_key: str, config: AnalysisConfig, overrides: dict[str, int] | None) -> str:
    plan = config.quantization_plan(overrides)
    return digest_json(
        {
            "feature": feature_cache_key,
            "quantization_config": config.quantization_fingerprint(overrides),
            "bit_widths": {item["name"]: item["bit_width"] for item in plan},
        }
    )


def build_description(
    features: dict[str, Any],
    config: AnalysisConfig,
    overrides: dict[str, int] | None = None,
) -> dict[str, Any]:
    plan = config.quantization_plan(overrides)
    dimensions: list[dict[str, Any]] = []
    vector: list[float] = []
    integers: list[int] = []
    for item in plan:
        value = float(features["statistics"].get(item["metric"], 0.0))
        low, high = item["range"]
        quantized = quantize_value(value, low, high, item["bit_width"])
        dimension_result = {
            "name": item["name"],
            "metric": item["metric"],
            **quantized,
            "selection": item["source"],
            "dynamic_range": [low, high],
            "allowed_bit_widths": item["allowed_bit_widths"],
        }
        dimensions.append(dimension_result)
        vector.append(quantized["normalized"])
        integers.append(quantized["integer"])
    return {
        "segment_id": features["segment_id"],
        "file": features["file"],
        "content_fingerprint": features["content_fingerprint"],
        "cache_key": description_cache_key(features["cache_key"], config, overrides),
        "quantization_fingerprint": config.quantization_fingerprint(overrides),
        "dimensions": dimensions,
        "integers": integers,
        "vector": vector,
        "bit_widths": {item["name"]: item["bit_width"] for item in dimensions},
        "upstream": {"feature": features["cache_key"]},
    }


def reference_levels(dimension: dict[str, Any]) -> list[int]:
    allowed = [int(value) for value in dimension.get("allowed_bit_widths", [dimension["bit_width"]])]
    selected = int(dimension["bit_width"])
    before = [value for value in allowed if value < selected]
    after = [value for value in allowed if value > selected]
    reference = after[0] if after else (before[-1] if before else selected)
    return [selected, reference]


def dual_level_reference(
    features: dict[str, Any],
    description: dict[str, Any],
    config: AnalysisConfig,
    overrides: dict[str, int] | None,
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for dimension, source_dimension in zip(description["dimensions"], config.dimensions):
        levels = reference_levels(dimension)
        value = float(features["statistics"].get(source_dimension.metric, 0.0))
        level_values = []
        for bits in levels:
            quantized = quantize_value(value, *source_dimension.dynamic_range, bits)
            level_values.append(
                {
                    "bit_width": bits,
                    "step": quantized["step"],
                    "integer": quantized["integer"],
                    "normalized": quantized["normalized"],
                    "residual": quantized["residual"],
                }
            )
        references.append({"dimension": source_dimension.name, "levels": level_values})
    return references


def similarity(
    left: dict[str, Any],
    right: dict[str, Any],
    config: AnalysisConfig,
) -> tuple[float, list[dict[str, Any]]]:
    weights = config.similarity_weights
    denominator = sum(weights.values())
    differences: list[dict[str, Any]] = []
    weighted_sum = 0.0
    for dimension in config.dimensions:
        left_dimension = next(item for item in left["dimensions"] if item["name"] == dimension.name)
        right_dimension = next(item for item in right["dimensions"] if item["name"] == dimension.name)
        difference = abs(float(left_dimension["normalized"]) - float(right_dimension["normalized"]))
        contribution = weights[dimension.name] * difference
        weighted_sum += contribution
        differences.append(
            {
                "name": dimension.name,
                "weight": weights[dimension.name],
                "left_normalized": left_dimension["normalized"],
                "right_normalized": right_dimension["normalized"],
                "absolute_difference": difference,
                "weighted_difference": contribution,
            }
        )
    score = 1.0 - weighted_sum / denominator if denominator else 0.0
    return max(0.0, min(1.0, score)), differences
