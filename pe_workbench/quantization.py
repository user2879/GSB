"""Deterministic fixed-point quantization, integer dot products, and softmax."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from typing import Any, Iterable

import numpy as np

from .hashing import stable_float

SCORE_DECIMAL_PLACES = 12
SCORE_SCALE = 10**SCORE_DECIMAL_PLACES

QUANTIZATION_PROFILES: dict[str, dict[str, Any]] = {
    "fine": {"bit_offset": 0, "clip": 8.0},
    "balanced": {"bit_offset": -2, "clip": 4.0},
    "coarse": {"bit_offset": -4, "clip": 2.0},
}


def quantization_profile(level: str, fixed_decimal_bits: int) -> dict[str, Any]:
    if level not in QUANTIZATION_PROFILES:
        raise ValueError(f"unsupported quantization level {level}")
    profile = QUANTIZATION_PROFILES[level]
    bits = max(2, int(fixed_decimal_bits) + int(profile["bit_offset"]))
    return {
        "level": level,
        "effective_fractional_bits": bits,
        "scale": 1 << bits,
        "clip": float(profile["clip"]),
        "max_integer": int(float(profile["clip"]) * (1 << bits)),
        "rounding": "nearest_even",
    }


def quantize_vector(
    values: Iterable[float],
    *,
    level: str,
    fixed_decimal_bits: int,
) -> dict[str, Any]:
    profile = quantization_profile(level, fixed_decimal_bits)
    scale = int(profile["scale"])
    max_integer = int(profile["max_integer"])
    integers: list[int] = []
    clipped_count = 0
    source = [float(value) for value in values]
    for value in source:
        if not math.isfinite(value):
            raise ValueError("cannot quantize a non-finite value")
        scaled = value * scale
        if scaled > max_integer:
            scaled = float(max_integer)
            clipped_count += 1
        elif scaled < -max_integer:
            scaled = float(-max_integer)
            clipped_count += 1
        integers.append(int(np.rint(scaled)))
    dequantized = [value / scale for value in integers]
    return {
        "integers": integers,
        "dequantized": dequantized,
        "profile": profile,
        "clipped_count": clipped_count,
    }


def integer_dot_with_accumulation(
    left: list[int],
    right: list[int],
    block_size: int,
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("quantized vectors must have equal length")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    accumulation: list[dict[str, int]] = []
    dimension_steps: list[dict[str, int]] = []
    cumulative = 0
    for start in range(0, len(left), block_size):
        end = min(start + block_size, len(left))
        block_sum = 0
        for dimension in range(start, end):
            product = int(left[dimension]) * int(right[dimension])
            block_sum += product
            cumulative += product
            dimension_steps.append(
                {
                    "dimension": dimension,
                    "product": product,
                    "cumulative_sum": cumulative,
                }
            )
        accumulation.append(
            {
                "dim_start": start,
                "dim_end": end - 1,
                "block_sum": block_sum,
                "cumulative_sum": cumulative,
            }
        )
    return {
        "raw_dot": cumulative,
        "accumulation": accumulation,
        "dimension_steps": dimension_steps,
    }


def fixed_score_units(raw_dot: int, bits: int, dimension: int) -> int:
    """Round ``raw_dot / (2^(2b) * sqrt(d))`` to 12 decimal places exactly."""
    with localcontext() as context:
        context.prec = 60
        exact = Decimal(int(raw_dot)) / Decimal(1 << (2 * int(bits)))
        exact /= Decimal(int(dimension)).sqrt()
        return int((exact * SCORE_SCALE).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def fixed_scores_from_units(score_units: Iterable[int]) -> list[float]:
    return [stable_float(units / SCORE_SCALE) for units in score_units]


def reference_score(raw_dot: float, dimension: int) -> float:
    return float(raw_dot) / math.sqrt(int(dimension))


def float64_dot(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values):
        raise ValueError("vectors must have equal length")
    total = 0.0
    for dimension in range(len(left_values)):
        total += float(left_values[dimension]) * float(right_values[dimension])
    return total


def softmax_after_max_shift(
    scores: list[float],
    *,
    digits: int = SCORE_DECIMAL_PLACES,
) -> dict[str, Any]:
    if not scores:
        return {
            "max_index": None,
            "max_score": None,
            "weights": [],
            "denominator": None,
            "shifted_logits": [],
        }
    max_index = max(range(len(scores)), key=lambda index: (scores[index], -index))
    max_score = scores[max_index]
    shifted = [stable_float(score - max_score, digits + 2) for score in scores]
    exponentials = [math.exp(value) for value in shifted]
    denominator = math.fsum(exponentials)
    weights = [stable_float(value / denominator, digits) for value in exponentials]
    return {
        "max_index": max_index,
        "max_score": stable_float(max_score),
        "weights": weights,
        "denominator": stable_float(denominator),
        "shifted_logits": shifted,
    }


def score_fixed_path(
    encoded_query: dict[str, Any],
    encoded_keys: list[dict[str, Any]],
    query_position: int,
    key_positions: tuple[int, ...],
    *,
    dimension: int,
    dimension_block_size: int,
    quantization_level: str,
    fixed_decimal_bits: int,
) -> dict[str, Any]:
    query_quantized = quantize_vector(
        encoded_query["vector"],
        level=quantization_level,
        fixed_decimal_bits=fixed_decimal_bits,
    )
    key_records: list[dict[str, Any]] = []
    score_units: list[int] = []
    profile = query_quantized["profile"]
    bits = int(profile["effective_fractional_bits"])
    for index, (key_position, encoded_key) in enumerate(zip(key_positions, encoded_keys)):
        key_quantized = quantize_vector(
            encoded_key["vector"],
            level=quantization_level,
            fixed_decimal_bits=fixed_decimal_bits,
        )
        dot = integer_dot_with_accumulation(
            query_quantized["integers"],
            key_quantized["integers"],
            dimension_block_size,
        )
        units = fixed_score_units(int(dot["raw_dot"]), bits, dimension)
        score_units.append(units)
        key_records.append(
            {
                "index": index,
                "key_position": key_position,
                "relative_distance": abs(int(query_position) - int(key_position)),
                "quantized_key_clipped_count": key_quantized["clipped_count"],
                "raw_dot": dot["raw_dot"],
                "accumulation": dot["accumulation"],
                "dimension_steps": dot["dimension_steps"],
                "score_units": units,
                "score": stable_float(units / SCORE_SCALE),
            }
        )
    softmax = softmax_after_max_shift(fixed_scores_from_units(score_units))
    for index, key_record in enumerate(key_records):
        key_record["weight_units"] = int(
            round(softmax["weights"][index] * SCORE_SCALE)
        ) if softmax["weights"] else 0
        key_record["weight"] = softmax["weights"][index] if softmax["weights"] else None
    return {
        "quantization": {
            **profile,
            "query_clipped_count": query_quantized["clipped_count"],
            "query_integers": query_quantized["integers"],
            "query_dequantized": [stable_float(value) for value in query_quantized["dequantized"]],
            "dimension_block_size": dimension_block_size,
        },
        "keys": key_records,
        "normalization": softmax,
        "attention_scale": stable_float(1.0 / math.sqrt(dimension)),
    }
