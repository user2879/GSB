from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import CODES, StageFault
from .hashing import array_fingerprint


ALLOWED_FRAC_BITS = (4, 8, 12)
ALLOWED_QUANT_MODES = ("round_half_up", "floor", "round_half_even")


@dataclass(frozen=True)
class QuantSpec:
    frac_bits: int = 8
    mode: str = "round_half_up"

    def __post_init__(self) -> None:
        if self.frac_bits not in ALLOWED_FRAC_BITS:
            raise StageFault(
                CODES["QUANT_INVALID"],
                f"frac_bits must be one of {ALLOWED_FRAC_BITS}",
                field_name="fixed_frac_bits",
                details={"value": self.frac_bits},
            )
        if self.mode not in ALLOWED_QUANT_MODES:
            raise StageFault(
                CODES["QUANT_INVALID"],
                f"quant_mode must be one of {ALLOWED_QUANT_MODES}",
                field_name="quant_mode",
                details={"value": self.mode},
            )

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits

    def fingerprint(self) -> dict[str, Any]:
        return {"frac_bits": self.frac_bits, "mode": self.mode, "scale": self.scale}


def validate_quant_spec(frac_bits: Any, mode: Any) -> QuantSpec:
    try:
        return QuantSpec(int(frac_bits), str(mode))
    except StageFault:
        raise
    except (TypeError, ValueError) as exc:
        raise StageFault(
            CODES["QUANT_INVALID"], f"invalid quantization specification: {exc}"
        ) from exc


def quantize_rational(
    numerator: np.ndarray | int,
    denominator: int,
    spec: QuantSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Quantize numerator/denominator * scale using integer arithmetic.

    The returned residual numerator is relative to denominator*scale and is
    signed.  No floating-point values participate in the arithmetic.
    """
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    num = np.asarray(numerator, dtype=np.int64)
    scale = spec.scale
    scaled_num = num * scale
    den = np.int64(denominator)

    if spec.mode == "floor":
        quotient = _floor_divide(scaled_num, den)
    elif spec.mode == "round_half_up":
        quotient = _round_half_up(scaled_num, den)
    else:
        quotient = _round_half_even(scaled_num, den)

    quotient = quotient.astype(np.int64)
    residual_num = scaled_num - quotient * den
    trace = {
        "quant_step": f"1/{scale}",
        "scale": scale,
        "mode": spec.mode,
        "denominator": int(denominator),
        "integer_dtype": "int64",
        "integer_shape": list(quotient.shape),
        "integer_fingerprint": None,
        "residual_dtype": "int64",
        "residual_fingerprint": None,
        "residual_abs_max": int(np.max(np.abs(residual_num))) if residual_num.size else 0,
    }
    trace["integer_fingerprint"] = array_fingerprint(quotient)
    trace["residual_fingerprint"] = array_fingerprint(residual_num)
    return quotient, trace


def _floor_divide(num: np.ndarray, den: np.int64 | int) -> np.ndarray:
    quotient = np.floor_divide(num, den)
    remainder = num - quotient * den
    return quotient - (remainder < 0).astype(np.int64)


def _round_half_up(num: np.ndarray, den: np.int64) -> np.ndarray:
    """Round nearest with ties away from zero."""
    sign = np.sign(num)
    absolute = np.abs(num)
    rounded_abs = (absolute * 2 + den) // (den * 2)
    return (rounded_abs * sign).astype(np.int64)


def _round_half_even(num: np.ndarray, den: np.int64) -> np.ndarray:
    quotient = np.floor_divide(num, den)
    remainder = num - quotient * den
    twice_remainder = remainder * 2
    round_up = twice_remainder > den
    half = twice_remainder == den
    quotient = quotient + round_up.astype(np.int64)
    quotient = quotient + (half & ((quotient % 2) != 0)).astype(np.int64)
    return quotient


def rescale_fixed(
    values: np.ndarray,
    from_frac_bits: int,
    to_spec: QuantSpec,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Rescale Q(from_frac_bits) integers to Q(to_spec.frac_bits)."""
    values = np.asarray(values, dtype=np.int64)
    if from_frac_bits == to_spec.frac_bits:
        return values, None
    if from_frac_bits < to_spec.frac_bits:
        shift = to_spec.frac_bits - from_frac_bits
        result = values << shift
        return result, {
            "quant_step": f"1/{to_spec.scale}",
            "mode": "left_shift",
            "shift": shift,
            "integer_fingerprint": None,
        }
    shift = from_frac_bits - to_spec.frac_bits
    denominator = 1 << shift
    quotient = _divide_by_mode(values.astype(np.int64), denominator, to_spec.mode)
    residual = values - quotient * denominator
    trace = {
        "quant_step": f"1/{to_spec.scale}",
        "mode": to_spec.mode,
        "right_shift": shift,
        "integer_fingerprint": array_fingerprint(quotient),
        "residual_fingerprint": array_fingerprint(residual),
        "residual_abs_max": int(np.max(np.abs(residual))) if residual.size else 0,
    }
    return quotient.astype(np.int64), trace


def _divide_by_mode(num: np.ndarray, denominator: int, mode: str) -> np.ndarray:
    den = np.int64(denominator)
    if mode == "floor":
        return _floor_divide(num, den)
    if mode == "round_half_even":
        return _round_half_even(num, den)
    return _round_half_up(num, den)


def round_integer_ratio(
    numerator: np.ndarray | int, denominator: int, spec: QuantSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Round numerator/denominator to integers without floating point."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    num = np.asarray(numerator, dtype=np.int64)
    quotient = _divide_by_mode(num, denominator, spec.mode).astype(np.int64)
    residual = num - quotient * np.int64(denominator)
    return quotient, residual


def assert_finite_numeric(array: Any, stage: str) -> np.ndarray:
    converted = np.asarray(array)
    if not np.issubdtype(converted.dtype, np.number):
        raise StageFault(
            CODES["DIM_MISMATCH"], f"{stage}: expected numeric latent, got {converted.dtype}"
        )
    if not np.all(np.isfinite(converted)):
        bad = int(np.size(converted) - np.isfinite(converted).sum())
        raise StageFault(
            CODES["NONFINITE_LATENT"],
            f"{stage}: {bad} non-finite latent entries",
            details={"stage": stage, "nonfinite_count": bad},
        )
    return converted
