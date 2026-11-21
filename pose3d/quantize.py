"""Fixed-point quantization helpers with explicit round-half-even semantics."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def scale_for_bits(fractional_bits: int) -> int:
    if fractional_bits < 0:
        raise ValueError("fractional_bits must be non-negative")
    return 1 << fractional_bits


def quantize_values(values: Iterable[float] | np.ndarray, fractional_bits: int) -> dict[str, Any]:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    scale = scale_for_bits(fractional_bits)
    scaled = np.rint(array * float(scale))
    if not np.all(np.isfinite(scaled)):
        raise ValueError("quantization produced non-finite integers")
    ints = np.asarray(scaled, dtype=np.int64)
    restored = ints.astype(np.float64) / float(scale)
    return {
        "fractional_bits": int(fractional_bits),
        "scale": int(scale),
        "rounding": "round_half_even",
        "int": ints.tolist(),
        "quantized": restored.tolist(),
    }


def quantize_point(point: Any, fractional_bits: int) -> dict[str, Any]:
    return quantize_values(np.asarray(point, dtype=np.float64), fractional_bits)


def quantize_residuals(residuals: list[dict[str, Any]], fractional_bits: int) -> dict[str, Any]:
    values = [float(item["residual_px"]) for item in residuals]
    quantized = quantize_values(values, fractional_bits)
    records = []
    for item, integer, restored in zip(
        residuals, quantized["int"], quantized["quantized"], strict=True
    ):
        records.append(
            {
                "view_index": int(item["view_index"]),
                "residual_px": float(item["residual_px"]),
                "residual_int": int(integer),
                "residual_quantized_px": float(restored),
                "quantization_error_px": float(restored - item["residual_px"]),
            }
        )
    return {
        "fractional_bits": quantized["fractional_bits"],
        "scale": quantized["scale"],
        "rounding": quantized["rounding"],
        "int": quantized["int"],
        "quantized": quantized["quantized"],
        "records": records,
    }
