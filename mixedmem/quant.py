"""Explicit fixed-point arithmetic and deterministic float32 kernels.

Numerical policy (recorded separately in attribution output):

* quantised operands use round-half-away-from-zero;
* quantised operands saturate to signed int24;
* dot-product accumulators saturate to signed int64;
* every reduction follows ascending input-index order (no BLAS reduction);
* dequantisation is ``accumulator / scale**2`` followed by float32 casting;
* lease ordering never participates in a numerical reduction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


Q_MIN = -(2**23)
Q_MAX = 2**23 - 1
ACC_MIN = -(2**63 - 1)
ACC_MAX = 2**63 - 1
F32_EPS = float(np.finfo(np.float32).eps)


def quantize_scalar(value: float, scale: int) -> int:
    if scale <= 0:
        raise ValueError("scale must be positive")
    scaled = float(value) * scale
    if scaled >= 0:
        rounded = math.floor(scaled + 0.5)
    else:
        rounded = math.ceil(scaled - 0.5)
    return min(Q_MAX, max(Q_MIN, int(rounded)))


def quantize_values(values: np.ndarray, scale: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32, order="C")
    result = np.empty(source.shape, dtype=np.int64, order="C")
    flat_source = source.reshape(-1)
    flat_result = result.reshape(-1)
    for index, value in enumerate(flat_source):
        flat_result[index] = quantize_scalar(float(value), scale)
    return result


def _saturating_add(left: int, right: int) -> int:
    total = left + right
    if total > ACC_MAX:
        return ACC_MAX
    if total < ACC_MIN:
        return ACC_MIN
    return total


def _saturating_mul(left: int, right: int) -> int:
    total = left * right
    if total > ACC_MAX:
        return ACC_MAX
    if total < ACC_MIN:
        return ACC_MIN
    return total


@dataclass(frozen=True)
class FixedRowResult:
    values: np.ndarray
    saturation_count: int
    quantized_weight_max_abs: int
    quantized_input_max_abs: int


def fixed_forward_rows(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    scale: int,
) -> FixedRowResult:
    """Run a fixed-point affine transform for each sample in ascending order."""

    x = np.asarray(inputs, dtype=np.float32, order="C")
    w = np.asarray(weights, dtype=np.float32, order="C")
    b = np.asarray(bias, dtype=np.float32, order="C")
    if x.ndim != 2 or w.ndim != 2 or b.ndim != 1:
        raise ValueError("expected inputs/weights matrices and a bias vector")
    if x.shape[1] != w.shape[1] or w.shape[0] != b.shape[0]:
        raise ValueError("incompatible fixed-point shapes")
    quantized_weights = quantize_values(w, scale)
    quantized_bias = quantize_values(b, scale)
    output = np.empty((x.shape[0], w.shape[0]), dtype=np.float32, order="C")
    saturation_count = 0
    denominator = scale * scale
    for sample_index in range(x.shape[0]):
        quantized_input = quantize_values(x[sample_index], scale)
        for output_index in range(w.shape[0]):
            accumulator = _saturating_mul(int(quantized_bias[output_index]), scale)
            for input_index in range(x.shape[1]):
                product = _saturating_mul(
                    int(quantized_input[input_index]),
                    int(quantized_weights[output_index, input_index]),
                )
                previous = accumulator
                accumulator = _saturating_add(accumulator, product)
                if accumulator in (ACC_MIN, ACC_MAX) and previous != accumulator:
                    saturation_count += 1
            dequantized = float(accumulator) / float(denominator)
            output[sample_index, output_index] = np.float32(dequantized)
    return FixedRowResult(
        values=output,
        saturation_count=saturation_count,
        quantized_weight_max_abs=int(np.max(np.abs(quantized_weights))) if quantized_weights.size else 0,
        quantized_input_max_abs=int(np.max(np.abs(quantize_values(x, scale)))) if x.size else 0,
    )


def float_forward_rows(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """NumPy-backed float32 affine transform with a fixed scalar reduction."""

    x = np.asarray(inputs, dtype=np.float32, order="C")
    w = np.asarray(weights, dtype=np.float32, order="C")
    b = np.asarray(bias, dtype=np.float32, order="C")
    if x.ndim != 2 or w.ndim != 2 or b.ndim != 1:
        raise ValueError("expected inputs/weights matrices and a bias vector")
    if x.shape[1] != w.shape[1] or w.shape[0] != b.shape[0]:
        raise ValueError("incompatible float32 shapes")
    output = np.empty((x.shape[0], w.shape[0]), dtype=np.float32, order="C")
    for sample_index in range(x.shape[0]):
        for output_index in range(w.shape[0]):
            accumulator = np.float32(0.0)
            for input_index in range(x.shape[1]):
                product = np.float32(
                    w[output_index, input_index] * x[sample_index, input_index]
                )
                accumulator = np.float32(accumulator + product)
            output[sample_index, output_index] = np.float32(
                accumulator + b[output_index]
            )
    return output


def error_bound_components(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    scale: int,
) -> dict[str, float]:
    """Derive a conservative layer bound split into three named components.

    Integer accumulation is exact but the float32 reference has a reduction
    rounding term.  The third term is structurally zero because no numerical
    reduction is ordered by lease allocation.
    """

    x = np.asarray(inputs, dtype=np.float32, order="C")
    w = np.asarray(weights, dtype=np.float32, order="C")
    b = np.asarray(bias, dtype=np.float32, order="C")
    if x.size == 0 or w.size == 0:
        return {
            "quantization_truncation": 0.0,
            "accumulation_order": 0.0,
            "lease_reordering": 0.0,
            "combined": 0.0,
        }
    weight_error = 0.5 / float(scale)
    activation_error = 0.5 / float(scale)
    max_truncation = 0.0
    max_reduction = 0.0
    term_count = x.shape[1]
    for sample_index in range(x.shape[0]):
        sample = x[sample_index].astype(np.float64)
        for output_index in range(w.shape[0]):
            row = w[output_index].astype(np.float64)
            truncation = float(
                np.sum(row * activation_error + sample * weight_error)
                + term_count * activation_error * weight_error
                + 0.5 / float(scale)
            )
            products = np.abs(row * sample)
            reduction = float(
                (term_count + 2) * F32_EPS * (float(np.sum(products)) + abs(float(b[output_index])) + 1.0)
            )
            max_truncation = max(max_truncation, truncation)
            max_reduction = max(max_reduction, reduction)
    lease_reordering = 0.0
    return {
        "quantization_truncation": max_truncation,
        "accumulation_order": max_reduction,
        "lease_reordering": lease_reordering,
        "combined": max_truncation + max_reduction + lease_reordering,
    }


def error_statistics(reference: np.ndarray, fixed: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float32, order="C")
    actual = np.asarray(fixed, dtype=np.float32, order="C")
    if ref.shape != actual.shape:
        raise ValueError("reference and fixed shapes differ")
    if ref.size == 0:
        return {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "sum_abs_error": 0.0,
            "max_rel_error": 0.0,
            "element_count": 0,
        }
    delta = np.abs(ref.astype(np.float64) - actual.astype(np.float64))
    denominator = np.maximum(np.abs(ref.astype(np.float64)), 1.0e-12)
    return {
        "max_abs_error": float(np.max(delta)),
        "mean_abs_error": float(np.mean(delta)),
        "sum_abs_error": float(np.sum(delta)),
        "max_rel_error": float(np.max(delta / denominator)),
        "element_count": int(ref.size),
    }


def stable_array_digest(array: np.ndarray) -> str:
    import hashlib

    contiguous = np.asarray(array, dtype="<f4", order="C")
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()
