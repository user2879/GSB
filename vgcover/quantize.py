"""Coverage-to-u8 quantization with explicit rounding and reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import CoverageError, ErrorCode
from .hashing import hash_float32, hash_uint8


QUANTIZATION_POLICY = {
    "version": "vgcover.coverage_u8.round_half_up.v1",
    "input": "float32 coverage in [0, 1]",
    "output": "uint8 grayscale in [0, 255]",
    "step": 1.0 / 255.0,
    "rounding": "nearest_half_up",
    "formula": "q = floor(c * 255 + 0.5)",
    "reconstruction": "c_hat = q / 255",
    "max_abs_error_bound": 0.5 / 255.0,
    "over_tolerance_test": "abs_error > max_abs_error_bound + 1e-7",
}


@dataclass
class QuantizationResult:
    gray: np.ndarray
    reconstructed_coverage: np.ndarray
    quant_error: np.ndarray
    reconstruction_error: np.ndarray
    policy: dict[str, Any]
    max_abs_error: float
    max_reconstruction_abs_error: float
    over_tolerance_pixels: int
    total_pixels: int
    distribution: dict[str, Any]
    gray_hash: str
    quant_error_hash: str
    reconstruction_error_hash: str


def _validate_coverage(coverage: np.ndarray) -> np.ndarray:
    array = np.asarray(coverage, dtype=np.float32)
    if array.ndim != 2:
        raise CoverageError(
            ErrorCode.COVERAGE_RANGE,
            "coverage map must be a two-dimensional array",
            details={"shape": list(array.shape)},
        )
    if not np.all(np.isfinite(array)):
        raise CoverageError(
            ErrorCode.COVERAGE_RANGE,
            "coverage map contains non-finite values",
        )
    minimum = float(np.min(array)) if array.size else 0.0
    maximum = float(np.max(array)) if array.size else 0.0
    if minimum < 0.0 or maximum > 1.0:
        raise CoverageError(
            ErrorCode.COVERAGE_RANGE,
            "coverage values must lie in [0, 1]",
            details={"minimum": minimum, "maximum": maximum},
        )
    return array


def _error_distribution(
    absolute_error: np.ndarray,
    *,
    max_abs_error: float,
    over_tolerance_pixels: int,
) -> dict[str, Any]:
    if absolute_error.size:
        finite = absolute_error.reshape(-1).astype(np.float64)
        mean_abs = float(np.mean(finite))
        rms = float(np.sqrt(np.mean(finite * finite)))
        p95 = float(np.percentile(finite, 95.0))
        nonzero_pixels = int(np.count_nonzero(finite > 0.0))
    else:
        finite = np.empty(0, dtype=np.float64)
        mean_abs = 0.0
        rms = 0.0
        p95 = 0.0
        nonzero_pixels = 0

    bound = float(QUANTIZATION_POLICY["max_abs_error_bound"])
    if finite.size:
        histogram_counts, histogram_edges = np.histogram(
            finite,
            bins=16,
            range=(0.0, max(bound, float(np.max(finite))) + 1e-12),
        )
        counts = [int(value) for value in histogram_counts.tolist()]
        edges = [float(value) for value in histogram_edges.tolist()]
    else:
        counts = [0] * 16
        edges = [float(bound * index / 16.0) for index in range(17)]

    return {
        "total_pixels": int(absolute_error.size),
        "nonzero_error_pixels": nonzero_pixels,
        "mean_abs_error": mean_abs,
        "rms_error": rms,
        "p95_abs_error": p95,
        "max_abs_error": float(max_abs_error),
        "max_abs_error_bound": bound,
        "over_tolerance_pixels": int(over_tolerance_pixels),
        "abs_error_histogram": {
            "bin_count": 16,
            "edges": edges,
            "counts": counts,
        },
    }


def quantize_coverage(coverage: np.ndarray) -> QuantizationResult:
    """Quantize float coverage to u8 using nearest-half-up.

    The input is never silently clipped: values outside [0, 1] raise
    ``E_COVERAGE_RANGE``. This makes the public error ledger honest.
    """

    source = _validate_coverage(coverage)
    scaled = source.astype(np.float64) * 255.0
    quantized = np.floor(scaled + 0.5).astype(np.int64)
    gray = np.clip(quantized, 0, 255).astype(np.uint8)
    reconstructed = gray.astype(np.float32) / np.float32(255.0)
    quant_error = (reconstructed - source).astype(np.float32)
    reconstruction_error = quant_error.copy()
    absolute_error = np.abs(quant_error)
    absolute_reconstruction_error = np.abs(reconstruction_error)
    max_abs_error = float(np.max(absolute_error)) if absolute_error.size else 0.0
    max_reconstruction_abs_error = (
        float(np.max(absolute_reconstruction_error)) if absolute_reconstruction_error.size else 0.0
    )
    bound = float(QUANTIZATION_POLICY["max_abs_error_bound"])
    over_tolerance = int(np.count_nonzero(absolute_error > bound + 1e-7))
    distribution = _error_distribution(
        absolute_error,
        max_abs_error=max_abs_error,
        over_tolerance_pixels=over_tolerance,
    )
    distribution.update(
        {
            "reconstruction_max_abs_error": max_reconstruction_abs_error,
            "reconstruction_mean_abs_error": distribution["mean_abs_error"],
            "reconstruction_rms_error": distribution["rms_error"],
        }
    )
    return QuantizationResult(
        gray=gray,
        reconstructed_coverage=reconstructed.astype(np.float32),
        quant_error=quant_error,
        reconstruction_error=reconstruction_error,
        policy=dict(QUANTIZATION_POLICY),
        max_abs_error=max_abs_error,
        max_reconstruction_abs_error=max_reconstruction_abs_error,
        over_tolerance_pixels=over_tolerance,
        total_pixels=int(source.size),
        distribution=distribution,
        gray_hash=hash_uint8(gray),
        quant_error_hash=hash_float32(quant_error),
        reconstruction_error_hash=hash_float32(reconstruction_error),
    )


def reconstruct_coverage(gray: np.ndarray) -> np.ndarray:
    array = np.asarray(gray)
    if array.ndim != 2 or array.dtype != np.uint8:
        raise CoverageError(
            ErrorCode.COVERAGE_RANGE,
            "gray map must be a two-dimensional uint8 array",
            details={"dtype": str(array.dtype), "shape": list(array.shape)},
        )
    return (array.astype(np.float32) / np.float32(255.0)).astype(np.float32)
