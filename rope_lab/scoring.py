"""Ordered fixed-point dot products, stable normalization and attribution."""
from __future__ import annotations

import math
from typing import Any

from .fixedpoint import dequantize, wrap_signed


def dot_dimension_integer_term(q_query_value: int, q_key_value: int) -> int:
    return int(q_query_value) * int(q_key_value)


def fixed_dot_product(
    q_query: list[int],
    q_key: list[int],
    fraction_bits: int,
    accumulator_bits: int,
    *,
    simulate_overflow: bool,
) -> dict[str, Any]:
    """Dot product with dimension order fixed to ascending index."""
    if len(q_query) != len(q_key):
        raise ValueError("fixed dot product dimension mismatch")

    accumulator = 0
    terms: list[int] = []
    overflow_count = 0
    overflow_steps: list[int] = []

    for dimension_index in range(len(q_query)):
        term = dot_dimension_integer_term(q_query[dimension_index], q_key[dimension_index])
        terms.append(term)
        accumulator += term
        if simulate_overflow:
            wrapped, overflowed = wrap_signed(accumulator, accumulator_bits)
            if overflowed:
                overflow_count += 1
                overflow_steps.append(dimension_index)
            accumulator = wrapped

    return {
        "q_terms": terms,
        "q_score": accumulator,
        "score": dequantize(accumulator, fraction_bits * 2),
        "overflow_count": overflow_count,
        "overflow_steps": overflow_steps,
        "simulate_overflow": simulate_overflow,
    }


def float_dot_product(query: list[float], key: list[float]) -> float:
    """Explicit ascending-dimension float64 dot product."""
    if len(query) != len(key):
        raise ValueError("float dot product dimension mismatch")
    total = 0.0
    for dimension_index in range(len(query)):
        total += float(query[dimension_index]) * float(key[dimension_index])
    return total


def stable_softmax(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"row_max": 0.0, "exponents": [], "sum": 0.0, "probabilities": []}

    row_max = float(scores[0])
    for index in range(1, len(scores)):
        if float(scores[index]) > row_max:
            row_max = float(scores[index])

    exponents: list[float] = []
    for score in scores:
        value = math.exp(float(score) - row_max)
        if not math.isfinite(value):
            value = 0.0
        exponents.append(value)

    denominator = 0.0
    for value in exponents:
        denominator += value

    if denominator <= 0.0 or not math.isfinite(denominator):
        probabilities = [1.0 / len(scores) for _ in scores]
    else:
        probabilities = [value / denominator for value in exponents]

    return {
        "row_max": row_max,
        "exponents": exponents,
        "sum": denominator,
        "probabilities": probabilities,
    }


def extrapolation_bin(overrun: int, bins: list[int]) -> str:
    if overrun <= 0:
        return "inside"
    previous = 0
    for edge in bins:
        if overrun <= edge:
            return f"extrap_{previous + 1}_to_{edge}"
        previous = edge
    return f"extrap_gt_{previous}"


def classify_key(
    query_position: int,
    key_position: int,
    train_length: int,
    target_length: int,
    bins: list[int],
) -> dict[str, Any]:
    relative_distance = abs(int(key_position) - int(query_position))
    overrun = relative_distance - int(train_length)
    category = "inside" if overrun <= 0 else "extrapolated"
    return {
        "relative_distance": relative_distance,
        "train_length": int(train_length),
        "overrun": overrun,
        "category": category,
        "bin": extrapolation_bin(overrun, bins),
        "beyond_target": relative_distance > int(target_length),
    }


def element_error(truth: float, candidate: float) -> dict[str, Any]:
    absolute_error = abs(float(candidate) - float(truth))
    zero_branch = float(truth) == 0.0
    if zero_branch:
        relative_error = absolute_error
    else:
        relative_error = absolute_error / abs(float(truth))
    return {
        "abs_error": absolute_error,
        "rel_error": relative_error,
        "zero_truth_branch": zero_branch,
    }


def aggregate_element_errors(elements: list[dict[str, Any]], abs_tolerance: float, rel_tolerance: float) -> dict[str, Any]:
    if not elements:
        elements = [{"truth": 0.0, "candidate": 0.0, "abs_error": 0.0, "rel_error": 0.0, "zero_truth_branch": True}]

    worst_index = 0
    for index in range(1, len(elements)):
        if elements[index]["abs_error"] > elements[worst_index]["abs_error"]:
            worst_index = index

    worst = elements[worst_index]
    abs_exceeded = worst["abs_error"] > abs_tolerance
    rel_exceeded = (not worst["zero_truth_branch"]) and worst["rel_error"] > rel_tolerance
    return {
        "abs_error": worst["abs_error"],
        "rel_error": worst["rel_error"],
        "zero_truth_branch": bool(worst["zero_truth_branch"]),
        "worst": {k: v for k, v in worst.items() if k not in {"abs_error", "rel_error", "zero_truth_branch"}},
        "element_count": len(elements),
        "zero_branch_count": sum(1 for item in elements if item["zero_truth_branch"]),
        "abs_tolerance": abs_tolerance,
        "rel_tolerance": rel_tolerance,
        "abs_exceeded": abs_exceeded,
        "rel_exceeded": rel_exceeded,
        "tolerance_exceeded": abs_exceeded or rel_exceeded,
        "first_exceeded": False,
    }


def ranking_pair_flips(
    subset_indexes: list[int],
    truth_scores: list[float],
    candidate_scores: list[float],
) -> list[dict[str, Any]]:
    flips: list[dict[str, Any]] = []
    for left_pos in range(len(subset_indexes)):
        left = subset_indexes[left_pos]
        for right_pos in range(left_pos + 1, len(subset_indexes)):
            right = subset_indexes[right_pos]
            truth_delta = truth_scores[left] - truth_scores[right]
            candidate_delta = candidate_scores[left] - candidate_scores[right]
            if truth_delta == 0.0 or candidate_delta == 0.0:
                continue
            if (truth_delta > 0) != (candidate_delta > 0):
                flips.append(
                    {
                        "left_key_index": left,
                        "right_key_index": right,
                        "truth_delta": truth_delta,
                        "candidate_delta": candidate_delta,
                    }
                )
    return flips
