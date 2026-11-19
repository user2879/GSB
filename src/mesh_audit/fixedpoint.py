from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from typing import Iterable

import numpy as np


INT64_LIMIT = (1 << 63) - 1


def round_ratio(numerator: int, denominator: int) -> int:
    """Round an exact rational to the nearest integer, ties away from zero."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def quantize_scalar(value: float, scale: int) -> int:
    if not np.isfinite(value):
        raise ValueError("cannot quantize a non-finite coordinate")
    product = Decimal(str(value)) * Decimal(scale)
    quantized = int(product.to_integral_value(rounding=ROUND_HALF_UP))
    if abs(quantized) > INT64_LIMIT:
        raise OverflowError("fixed-point coordinate exceeds int64 range")
    return quantized


def quantize_rows(rows: Iterable[Iterable[float]], scale: int) -> np.ndarray:
    values = [[quantize_scalar(float(value), scale) for value in row] for row in rows]
    return np.asarray(values, dtype=np.int64)


def fixed_as_float(fixed: np.ndarray, scale: int) -> np.ndarray:
    return fixed.astype(np.float64) / float(scale)


def fraction_from_decimal(value: float) -> Fraction:
    return Fraction(str(value))


def laplacian_fixed_row(
    old: tuple[int, int, int],
    neighbor_sum: tuple[int, int, int],
    degree: int,
    lambda_fraction: Fraction,
) -> tuple[int, int, int]:
    """Exact fixed-point uniform Laplacian: old + lambda*(mean-old)."""

    num = lambda_fraction.numerator
    den = lambda_fraction.denominator
    denominator = den * degree
    result: list[int] = []
    for axis in range(3):
        numerator = old[axis] * denominator + num * (
            neighbor_sum[axis] - old[axis] * degree
        )
        result.append(round_ratio(numerator, denominator))
    return (result[0], result[1], result[2])


def laplacian_float_row(
    old: tuple[float, float, float],
    neighbor_sum: tuple[float, float, float],
    degree: int,
    lam: float,
) -> tuple[float, float, float]:
    mean = tuple(value / float(degree) for value in neighbor_sum)
    return tuple(old[axis] + lam * (mean[axis] - old[axis]) for axis in range(3))


def mean_fixed_row(fixed_rows: np.ndarray) -> tuple[int, int, int]:
    count = int(fixed_rows.shape[0])
    sums = [sum(int(value) for value in fixed_rows[:, axis]) for axis in range(3)]
    result = tuple(round_ratio(value, count) for value in sums)
    return result  # type: ignore[return-value]


def mean_float_row(float_rows: np.ndarray) -> tuple[float, float, float]:
    mean = np.mean(float_rows, axis=0, dtype=np.float64)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def per_vertex_errors(
    fixed: np.ndarray,
    reference: np.ndarray,
    scale: int,
    tolerance: float,
) -> list[dict[str, object]]:
    fixed_world = fixed_as_float(fixed, scale)
    delta = fixed_world - reference
    max_abs = np.max(np.abs(delta), axis=1)
    records: list[dict[str, object]] = []
    for vertex in range(int(fixed.shape[0])):
        records.append(
            {
                "vertex": vertex,
                "fixed": [int(value) for value in fixed[vertex]],
                "reference": [float(value) for value in reference[vertex]],
                "delta": [float(value) for value in delta[vertex]],
                "max_abs_error": float(max_abs[vertex]),
                "exceeds_tolerance": bool(max_abs[vertex] > tolerance),
            }
        )
    return records


def deviation_field(fixed: np.ndarray, reference: np.ndarray, scale: int) -> np.ndarray:
    return np.max(np.abs(fixed_as_float(fixed, scale) - reference), axis=1)
