"""Deterministic fixed-point quantization.

Floating feature values are converted to signed fixed-point integers using a
configurable quantization level (scaling rule) and fractional bit width. The
quantization residual is preserved alongside the integer so that the original
float can be reconstructed and the rounding policy is auditable.

No NumPy global state or floating-point environment is modified anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import ConfigError

ALLOWED_QUANT_LEVELS = ("Q0.8", "Q1.15", "Q2.30", "scale10")
ALLOWED_FRAC_BITS = (8, 15, 30)
ALLOWED_ROUNDING = ("HALF_UP", "TRUNC", "HALF_EVEN")


def quant_scale(quant_level: str, frac_bits: int) -> float:
    """Return the multiplicative scale used to map a float to an integer."""
    if quant_level not in ALLOWED_QUANT_LEVELS:
        raise ConfigError(f"unsupported quant_level {quant_level!r}")
    if frac_bits not in ALLOWED_FRAC_BITS:
        raise ConfigError(f"unsupported frac_bits {frac_bits!r}")
    if quant_level == "Q0.8":
        if frac_bits != 8:
            raise ConfigError("Q0.8 requires frac_bits=8")
        return float(1 << 8)
    if quant_level == "Q1.15":
        if frac_bits != 15:
            raise ConfigError("Q1.15 requires frac_bits=15")
        return float(1 << 15)
    if quant_level == "Q2.30":
        if frac_bits != 30:
            raise ConfigError("Q2.30 requires frac_bits=30")
        return float(1 << 30)
    # "scale10" keeps a decimal-oriented scale but uses the requested frac bits
    # only as an enum-compatible precision declaration.
    return 10.0 ** min(9, frac_bits // 4 + 6)


def _round_array(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "TRUNC":
        return np.trunc(values)
    if mode == "HALF_EVEN":
        # np.rint implements banker's rounding and is vectorized/deterministic.
        return np.rint(values)
    # HALF_UP: deterministic for every finite input, including negative values.
    return np.floor(values + 0.5)


@dataclass(frozen=True)
class FixedPointQuantizer:
    """Stateless fixed-point quantizer derived from configuration."""

    quant_level: str
    frac_bits: int
    rounding: str = "HALF_UP"

    def __post_init__(self):
        if self.rounding not in ALLOWED_ROUNDING:
            raise ConfigError(f"unsupported rounding {self.rounding!r}")
        # Validate enum compatibility eagerly.
        quant_scale(self.quant_level, self.frac_bits)

    @property
    def scale(self) -> float:
        return quant_scale(self.quant_level, self.frac_bits)

    def quantize(self, values) -> tuple[np.ndarray, np.ndarray]:
        """Return (integer vector, residual vector).

        residual = original_float - integer / scale, rounded to 12 decimals so
        identical inputs always produce byte-identical JSON output.
        """
        arr = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ConfigError("non-finite value cannot be quantized")
        scaled = arr * self.scale
        integers = _round_array(scaled, self.rounding).astype(np.int64)
        residual = np.round(arr - integers.astype(np.float64) / self.scale, 12)
        return integers, residual.astype(np.float64)

    def dequantize(self, integers) -> np.ndarray:
        arr = np.asarray(integers, dtype=np.int64)
        return arr.astype(np.float64) / self.scale

    def to_meta(self) -> dict:
        return {
            "quant_level": self.quant_level,
            "frac_bits": self.frac_bits,
            "rounding": self.rounding,
            "scale": self.scale,
        }


def quantize_probability(probability: float, quantizer: FixedPointQuantizer) -> int:
    """Quantize a [0, 1] probability to a fixed-point integer score."""
    probability = float(probability)
    if not math.isfinite(probability):
        raise ConfigError("non-finite probability")
    probability = min(1.0, max(0.0, probability))
    integers, _ = quantizer.quantize(np.asarray([probability], dtype=np.float64))
    return int(integers[0])


def threshold_integer(threshold, quantizer: FixedPointQuantizer) -> int:
    """Convert a probability threshold to the integer used for exact compares."""
    integers, _ = quantizer.quantize(
        np.asarray([float(threshold)], dtype=np.float64)
    )
    return int(integers[0])
