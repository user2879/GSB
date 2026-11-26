from __future__ import annotations

from typing import Any

import numpy as np

from .hashing import fingerprint
from .quant import relative_or_absolute_error, round_shift_half_even
from .schemes import make_operator

LAYER_INSIDE = "inside_train"
LAYER_EXTRAPOLATION = "outside_extrapolation"
STAGE_ORDER = ("position", "score", "weight")


def layer_name(query_position: int, key_position: int, train_length: int) -> str:
    distance = abs(int(query_position) - int(key_position))
    return LAYER_INSIDE if distance < int(train_length) else LAYER_EXTRAPOLATION


def layer_indices(sample: dict[str, Any]) -> dict[str, list[int]]:
    result = {LAYER_INSIDE: [], LAYER_EXTRAPOLATION: []}
    for index, position in enumerate(sample["key_positions"]):
        layer = layer_name(sample["query_position"], position, sample["train_length"])
        result[layer].append(index)
    return result


def fixed_scores(position_artifact: dict[str, Any]) -> dict[str, Any]:
    frac_bits = int(position_artifact["frac_bits"])
    scale = 1 << frac_bits
    query_int = position_artifact["query"]["encoded_int"]
    rows = []
    score_ints: list[int] = []

    for key in position_artifact["keys"]:
        key_int = key["encoded_int"]
        products: list[int] = []
        intermediates: list[int] = []
        accumulator = 0
        for dimension_index in range(len(query_int)):
            product = int(query_int[dimension_index]) * int(key_int[dimension_index])
            products.append(product)
            # Explicit ascending dimension order; never use ndarray sum/reduction.
            accumulator += product
            intermediates.append(accumulator)
        score_int = round_shift_half_even(accumulator, frac_bits)
        score_ints.append(score_int)
        rows.append({
            "key_index": key["key_index"],
            "position": key["position"],
            "products": products,
            "accumulator_intermediate": intermediates,
            "accumulator_int": accumulator,
            "score_int": score_int,
            "score_float": score_int / scale,
            "rounding_residual": score_int / scale - accumulator / (scale * scale),
        })

    max_score_int = max(score_ints) if score_ints else 0
    shifted_ints = [value - max_score_int for value in score_ints]
    exp_values: list[float] = []
    denominator = np.float64(0.0)
    for shifted in shifted_ints:
        value = float(np.exp(np.float64(shifted) / np.float64(scale), dtype=np.float64))
        exp_values.append(value)
        denominator += np.float64(value)
    weights = [float(np.float64(value) / denominator) for value in exp_values]

    score_floats = [row["score_float"] for row in rows]
    max_score_float = max(score_floats) if score_floats else 0.0
    float_shifted = [float(np.float64(value) - np.float64(max_score_float))
                     for value in score_floats]
    for index, row in enumerate(rows):
        row["rank_score"] = index
        row["shifted_int"] = shifted_ints[index]
        row["exp_value"] = exp_values[index]
        row["weight"] = weights[index]
        row["float_shifted_score"] = float_shifted[index]

    return {
        "scale": scale,
        "max_score_int": max_score_int,
        "max_score_float": max_score_float,
        "denominator": float(denominator),
        "weights": weights,
        "rows": rows,
    }


def reference_scores(float_encoding: dict[str, Any]) -> dict[str, Any]:
    query = [np.float64(x) for x in float_encoding["query"]["encoded_vector"]]
    rows = []
    scores: list[float] = []
    dimension = len(query)

    for key in float_encoding["keys"]:
        key_vector = [np.float64(x) for x in key["encoded_vector"]]
        accumulator = np.float64(0.0)
        intermediates: list[float] = []
        for dimension_index in range(dimension):
            accumulator += query[dimension_index] * key_vector[dimension_index]
            intermediates.append(float(accumulator))
        score = float(accumulator)
        scores.append(score)
        rows.append({
            "key_index": key["key_index"],
            "position": key["position"],
            "score": score,
            "accumulator_intermediate": intermediates,
        })

    max_score = float(max(np.float64(score) for score in scores)) if scores else 0.0
    exp_values: list[float] = []
    denominator = np.float64(0.0)
    for score in scores:
        shifted = np.float64(score) - np.float64(max_score)
        value = float(np.exp(shifted, dtype=np.float64))
        exp_values.append(value)
        denominator += np.float64(value)
    weights = [float(np.float64(value) / denominator) for value in exp_values]
    for index, row in enumerate(rows):
        row["shifted_score"] = float(np.float64(scores[index]) - np.float64(max_score))
        row["exp_value"] = exp_values[index]
        row["weight"] = weights[index]

    return {
        "max_score": max_score,
        "denominator": float(denominator),
        "weights": weights,
        "rows": rows,
    }


def compute_encoding_and_scores(
    sample: dict[str, Any],
    scheme: dict[str, Any],
    frac_bits: int | None = None,
    quant_tier: str | None = None,
) -> dict[str, Any]:
    frac_bits = int(sample.get("_frac_bits", 16)) if frac_bits is None else int(frac_bits)
    quant_tier = str(sample.get("_quant_tier", "Q32")) if quant_tier is None else str(quant_tier)
    operator = make_operator(scheme)
    float_encoding = operator.float_encoding(sample)
    position = operator.fixed_encoding(sample, frac_bits, quant_tier)
    score = fixed_scores(position)
    reference = reference_scores(float_encoding)
    digest = fingerprint("position-digest:v1", {
        "query_pe_int": position["query"]["pe_int"],
        "query_encoded_int": position["query"]["encoded_int"],
        "keys": [
            {
                "position": key["position"],
                "pe_int": key["pe_int"],
                "encoded_int": key["encoded_int"],
            }
            for key in position["keys"]
        ],
    })
    return {
        "position": position,
        "score": score,
        "reference": reference,
        "float_encoding": float_encoding,
        "position_digest": digest,
    }
