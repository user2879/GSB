"""Float32 and integer fixed-point forward paths with explicit numerics."""

from __future__ import annotations

import numpy as np

from .util import stable_seed


INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def quantize(values: np.ndarray, scale: int) -> np.ndarray:
    """Round to nearest, ties to even, then saturate to signed int32."""

    if isinstance(scale, bool) or not isinstance(scale, int) or not 0 <= scale <= 6:
        raise ValueError("scale must be an integer in [0, 6]")
    array = np.asarray(values, dtype=np.float64)
    finite = np.nan_to_num(
        array,
        nan=0.0,
        posinf=float(INT32_MAX),
        neginf=float(INT32_MIN),
    )
    scaled = finite * float(1 << scale)
    rounded = np.rint(scaled)
    clipped = np.clip(rounded, float(INT32_MIN), float(INT32_MAX))
    return clipped.astype(np.int32)


def dequantize(integer_values: np.ndarray, scale: int) -> np.ndarray:
    array = np.asarray(integer_values, dtype=np.int64)
    return (array.astype(np.float64) / float(1 << scale)).astype(np.float32)


def saturating_add_int64(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.int64)
    b = np.asarray(right, dtype=np.int64)
    result = np.add(a, b, dtype=np.int64)
    overflow_pos = (b > 0) & (result < a)
    overflow_neg = (b < 0) & (result > a)
    result = np.where(overflow_pos, np.int64(INT64_MAX), result)
    result = np.where(overflow_neg, np.int64(INT64_MIN), result)
    return result


def round_shift_saturating_int32(values: np.ndarray, scale: int) -> np.ndarray:
    """Signed nearest-even division by 2**scale, then int32 saturation."""

    divisor = 1 << scale
    integer = np.asarray(values, dtype=np.int64)
    quotient = np.floor_divide(integer, divisor)
    remainder = integer - quotient * divisor
    twice = remainder * 2
    increment = (twice > divisor) | ((twice == divisor) & ((quotient & 1) != 0))
    quotient = quotient + increment.astype(np.int64)
    quotient = np.clip(quotient, INT32_MIN, INT32_MAX)
    return quotient.astype(np.int32)


def float32_layer_forward(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    x = np.asarray(inputs, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    b = np.asarray(bias, dtype=np.float32)
    preactivation = np.matmul(x, w, dtype=np.float32) + b
    return np.tanh(preactivation).astype(np.float32)


def _integer_accumulate_standard(q_inputs: np.ndarray, q_weights: np.ndarray) -> np.ndarray:
    result = np.matmul(q_inputs.astype(np.int64), q_weights.astype(np.int64))
    return np.clip(result, INT64_MIN, INT64_MAX).astype(np.int64)


def _integer_accumulate_sequential(q_inputs: np.ndarray, q_weights: np.ndarray) -> np.ndarray:
    qx = np.asarray(q_inputs, dtype=np.int64)
    qw = np.asarray(q_weights, dtype=np.int64)
    accumulator = np.zeros(qw.shape[1], dtype=np.int64)
    for index in range(qw.shape[0]):
        product = qx[index] * qw[index]
        accumulator = saturating_add_int64(accumulator, product)
    return accumulator


def fixed_point_layer_forward(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    scale: int,
    *,
    accumulation_order: str = "standard",
) -> tuple[np.ndarray, dict[str, object]]:
    """Quantized weight/activation/accumulator path.

    q = round_nearest_ties_even(x * 2**scale), saturated to int32.
    Products are int64. Accumulation and bias addition saturate to int64.
    The output accumulator is divided by 2**scale with ties-to-even and
    saturated to int32 before tanh and a final int32 activation quantization.
    """

    if scale < 0 or scale > 6:
        raise ValueError("scale must be in [0, 6]")
    x = np.asarray(inputs, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    b = np.asarray(bias, dtype=np.float32)
    q_inputs = quantize(x, scale)
    q_weights = quantize(w, scale)
    q_bias = quantize(b, scale).astype(np.int64) * np.int64(1 << scale)

    if accumulation_order == "sequential":
        raw_accumulator = _integer_accumulate_sequential(q_inputs, q_weights)
    elif accumulation_order == "standard":
        raw_accumulator = _integer_accumulate_standard(q_inputs, q_weights)
    else:
        raise ValueError("accumulation_order must be standard or sequential")
    accumulator = saturating_add_int64(raw_accumulator, q_bias)
    preactivation_q = round_shift_saturating_int32(accumulator, scale)
    preactivation = dequantize(preactivation_q, scale)
    activated = np.tanh(preactivation).astype(np.float32)
    output_q = quantize(activated, scale)
    output = dequantize(output_q, scale)
    return output, {
        "q_inputs": q_inputs,
        "q_weights": q_weights,
        "q_bias": q_bias.astype(np.int32),
        "accumulator": accumulator,
        "preactivation_q": preactivation_q,
        "output_q": output_q,
    }


def accumulation_order_delta(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    scale: int,
) -> float:
    """Difference between vectorized and sequential integer accumulation."""

    _, standard = fixed_point_layer_forward(
        inputs, weights, bias, scale, accumulation_order="standard"
    )
    _, sequential = fixed_point_layer_forward(
        inputs, weights, bias, scale, accumulation_order="sequential"
    )
    delta = np.abs(
        np.asarray(standard["accumulator"], dtype=np.float64)
        - np.asarray(sequential["accumulator"], dtype=np.float64)
    )
    return float(delta.max(initial=0.0) / float(1 << (2 * scale)))


def deterministic_layer_parameters(
    seed: int,
    layer_id: str,
    input_dim: int,
    output_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_seed(seed, layer_id, "weights", input_dim, output_dim))
    weights = np.round(rng.standard_normal((input_dim, output_dim)) * 0.28, 8).astype(np.float32)
    bias = np.round(rng.standard_normal(output_dim) * 0.035, 8).astype(np.float32)
    return weights, bias


def adapt_input(vector: np.ndarray, input_dim: int) -> np.ndarray:
    source = np.asarray(vector, dtype=np.float32).reshape(-1)
    if source.size >= input_dim:
        return source[:input_dim].astype(np.float32, copy=True)
    result = np.zeros(input_dim, dtype=np.float32)
    if source.size:
        result[: source.size] = source
    return result
