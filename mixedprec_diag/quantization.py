from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


@dataclass(frozen=True)
class QuantizationResult:
    values: np.ndarray
    saturations: int

    @property
    def as_list(self) -> list[int]:
        return [int(value) for value in self.values.tolist()]


def scale_parts(power: int) -> tuple[int, int]:
    """Return (numerator, denominator) for 10**power as an exact integer ratio."""

    if power >= 0:
        return 10**power, 1
    return 1, 10 ** (-power)


def saturate(value: int, lower: int = INT32_MIN, upper: int = INT32_MAX) -> int:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def round_half_away_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), denominator)
    if remainder * 2 >= denominator:
        quotient += 1
    return sign * quotient


def quantize_array(
    values: np.ndarray | Iterable[float],
    power: int,
    *,
    lower: int = INT32_MIN,
    upper: int = INT32_MAX,
) -> QuantizationResult:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("cannot quantize non-finite values")
    numerator, denominator = scale_parts(power)
    scaled = array * float(numerator) / float(denominator)
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    lower_float = float(lower)
    upper_float = float(upper)
    if upper == INT64_MAX:
        upper_float = float(np.nextafter(2.0**63, 0.0))
    clipped = np.clip(rounded, lower_float, upper_float)
    saturations = int(np.count_nonzero(rounded != clipped))
    return QuantizationResult(clipped.astype(np.int64), saturations)


def dequantize_value(value: int, power: int) -> float:
    numerator, denominator = scale_parts(power)
    return float(value) * float(denominator) / float(numerator)


def saturating_dot(
    input_q: np.ndarray,
    weight_q: np.ndarray,
    bias_q: np.ndarray,
    power: int,
) -> tuple[np.ndarray, int]:
    """Integer dot product followed by rescaling from S^2 to S and saturation.

    Python integers are used for accumulation so an intermediate cannot silently
    wrap. The public result is explicitly saturated to signed int32 activation
    storage after the S^2 -> S rounding step.
    """

    inputs = [int(value) for value in input_q.tolist()]
    weights = weight_q.tolist()
    biases = [int(value) for value in bias_q.tolist()]
    numerator, denominator = scale_parts(power)
    outputs: list[int] = []
    saturations = 0
    for output_index, bias in enumerate(biases):
        accumulator = 0
        for input_index, input_value in enumerate(inputs):
            accumulator += input_value * int(weights[input_index][output_index])
        scaled = round_half_away_div(accumulator * denominator, numerator)
        combined = scaled + bias
        clipped = saturate(combined, INT32_MIN, INT32_MAX)
        if clipped != combined:
            saturations += 1
        outputs.append(clipped)
    return np.asarray(outputs, dtype=np.int64), saturations


def saturating_matvec(
    input_q: np.ndarray,
    weight_q: np.ndarray,
    bias_q: np.ndarray,
    power: int,
) -> tuple[np.ndarray, int, int]:
    output, saturations = saturating_dot(input_q, weight_q, bias_q, power)
    activation = np.maximum(output, 0)
    return activation, saturations, int(np.count_nonzero(activation != output))


def fixed_linear_trace(
    input_q: np.ndarray,
    weight_q: np.ndarray,
    bias_q: np.ndarray,
    power: int,
    relu: bool,
) -> dict[str, object]:
    preactivation, saturations = saturating_dot(input_q, weight_q, bias_q, power)
    if relu:
        output = np.maximum(preactivation, 0)
    else:
        output = preactivation
    return {
        "preactivation": preactivation,
        "output": output,
        "saturation_count": saturations,
        "negative_clipped_count": int(np.count_nonzero(preactivation != output)),
        "rounding_direction": "nearest; exact ties away from zero",
        "overflow_policy": "saturate to signed int32 activation storage",
    }
