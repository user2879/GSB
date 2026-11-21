"""Deterministic activation kernels and explicit rounding boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ExecutionConfig, PrecisionSpec, QuantizationSpec


ACTIVATION_IMPL_VERSION = "act-v1"
PERMUTE_IMPL_VERSION = "perm-v1"
FUSION_IMPL_VERSION = "fused-v1"

APPROXIMATIONS: dict[str, str] = {
    "exp": f"numpy.exp/{np.__version__} (platform IEEE-754 binary libm)",
    "sqrt": f"numpy.sqrt/{np.__version__} (IEEE-754 square root)",
    "tanh": f"numpy.tanh/{np.__version__} (platform libm)",
    "erf": "abramowitz-stegun-7.1.26 polynomial (fixed coefficient/order)",
}


@dataclass(frozen=True)
class BoundaryResult:
    values: np.ndarray
    rounding_order: tuple[str, ...]
    round_count_delta: int


def _as_precision(values: np.ndarray, precision: PrecisionSpec) -> np.ndarray:
    return np.asarray(values, dtype=precision.dtype).copy(order="C")


def _preserve_nonfinite(original: np.ndarray, rounded: np.ndarray) -> np.ndarray:
    mask = ~np.isfinite(original)
    if np.any(mask):
        rounded = np.array(rounded, copy=True)
        rounded[mask] = original[mask]
    return rounded


def fixed_quantize(values: np.ndarray, quantization: QuantizationSpec, dtype: np.dtype) -> np.ndarray:
    if quantization.scale is None:
        return np.asarray(values, dtype=dtype).copy(order="C")
    source = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = np.rint(source * quantization.scale) / quantization.scale
    rounded = _preserve_nonfinite(source, rounded)
    return np.asarray(rounded, dtype=dtype).copy(order="C")


def round_decimal(values: np.ndarray, decimal_places: int, dtype: np.dtype) -> np.ndarray:
    scale = float(10**decimal_places)
    source = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = np.rint(source * scale) / scale
    rounded = _preserve_nonfinite(source, rounded)
    return np.asarray(rounded, dtype=dtype).copy(order="C")


def unfused_boundary(values: np.ndarray, config: ExecutionConfig) -> BoundaryResult:
    precision = config.precision
    if config.quantization.scale is None:
        rounded = round_decimal(values, precision.decimal_places, precision.dtype)
        return BoundaryResult(rounded, ("decimal",), 1)
    quantized = fixed_quantize(values, config.quantization, precision.dtype)
    rounded = round_decimal(quantized, precision.decimal_places, precision.dtype)
    return BoundaryResult(rounded, ("fixed", "decimal"), 2)


def fused_boundary(values: np.ndarray, config: ExecutionConfig) -> BoundaryResult:
    precision = config.precision
    rounded = round_decimal(values, precision.decimal_places, precision.dtype)
    if config.quantization.scale is None:
        return BoundaryResult(rounded, ("decimal",), 1)
    quantized = fixed_quantize(rounded, config.quantization, precision.dtype)
    return BoundaryResult(quantized, ("decimal", "fixed"), 2)


def _erf_approx(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Abramowitz-Stegun 7.1.26, evaluated in a fixed operation order."""

    x = np.asarray(values, dtype=dtype)
    a1 = dtype.type(0.254829592)
    a2 = dtype.type(-0.284496736)
    a3 = dtype.type(1.421413741)
    a4 = dtype.type(-1.453152027)
    a5 = dtype.type(1.061405429)
    p = dtype.type(0.3275911)
    sign = np.sign(x)
    absolute = np.abs(x)
    t = dtype.type(1.0) / (dtype.type(1.0) + p * absolute)
    polynomial = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    result = dtype.type(1.0) - polynomial * np.exp(-(absolute * absolute))
    return np.asarray(sign * result, dtype=dtype)


def activate(values: np.ndarray, op: str, params: dict[str, Any], config: ExecutionConfig) -> np.ndarray:
    dtype = config.precision.dtype
    x = _as_precision(values, config.precision)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if op == "relu":
            slope = dtype.type(params.get("negative_slope", params.get("alpha", 0.0)))
            if float(slope) == 0.0:
                return np.asarray(np.maximum(x, dtype.type(0.0)), dtype=dtype)
            return np.asarray(np.where(x >= dtype.type(0.0), x, slope * x), dtype=dtype)
        if op == "sigmoid":
            return np.asarray(dtype.type(1.0) / (dtype.type(1.0) + np.exp(-x)), dtype=dtype)
        if op == "tanh":
            return np.asarray(np.tanh(x), dtype=dtype)
        if op == "gelu":
            approximation = params.get("approximation", params.get("approximate", "erf"))
            if approximation == "tanh":
                coefficient = np.sqrt(dtype.type(2.0) / dtype.type(np.pi))
                inner = coefficient * (x + dtype.type(0.044715) * x * x * x)
                result = dtype.type(0.5) * x * (dtype.type(1.0) + np.tanh(inner))
            else:
                sqrt_two = np.sqrt(dtype.type(2.0))
                erf_arg = x / sqrt_two
                result = dtype.type(0.5) * x * (dtype.type(1.0) + _erf_approx(erf_arg, dtype))
            result = np.asarray(result, dtype=dtype)
            result = np.where(np.isposinf(x), dtype.type(np.inf), result)
            result = np.where(np.isneginf(x), dtype.type(0.0), result)
            return np.asarray(result, dtype=dtype)
    raise ValueError(f"unsupported activation: {op}")


def activation_approximation_metadata(op: str, params: dict[str, Any]) -> dict[str, Any]:
    used = ["erf", "sqrt"] if op == "gelu" and params.get("approximation", "erf") != "tanh" else []
    if op == "gelu" and params.get("approximation", "erf") == "tanh":
        used = ["tanh", "sqrt"]
    if op == "sigmoid":
        used = ["exp"]
    if op == "tanh":
        used = ["tanh"]
    return {
        "implementations": {name: APPROXIMATIONS[name] for name in dict.fromkeys(used)},
        "used": list(dict.fromkeys(used)),
        "exp": APPROXIMATIONS["exp"],
        "sqrt": APPROXIMATIONS["sqrt"],
        "erf": APPROXIMATIONS["erf"],
        "tanh": APPROXIMATIONS["tanh"],
        "all_available": APPROXIMATIONS,
        "fixed_operation_order": True,
    }


def permute_values(values: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(np.asarray(values), axes=perm))


def make_element_metadata(
    shape: tuple[int, ...],
    order: list[str],
    round_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    order_array = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        order_array[index] = list(order)
    round_array = np.full(shape, round_count, dtype=np.int64)
    return order_array, round_array


def permute_element_metadata(
    order_array: np.ndarray,
    round_array: np.ndarray,
    perm: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.ascontiguousarray(np.transpose(order_array, axes=perm)),
        np.ascontiguousarray(np.transpose(round_array, axes=perm)),
    )


def append_element_order(
    order_array: np.ndarray,
    labels: list[str],
    round_delta: int,
    round_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.empty(order_array.shape, dtype=object)
    for index in np.ndindex(order_array.shape):
        result[index] = list(order_array[index]) + labels
    return result, np.asarray(round_array, dtype=np.int64) + round_delta
