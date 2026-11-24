"""Integer fixed-point arithmetic with explicit rounding modes."""

from __future__ import annotations

import numpy as np


ROUNDING_MODES = frozenset({"half_up", "half_even", "floor", "ceil", "truncate"})


def _abs_round_div(numerator: np.ndarray, denominator: int, mode: str) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.int64)
    sign = np.where(numerator < 0, -1, 1).astype(np.int64)
    absolute = np.abs(numerator)
    quotient, remainder = np.divmod(absolute, np.int64(denominator))
    if mode == "truncate":
        rounded = quotient
    elif mode == "floor":
        rounded = quotient + ((remainder != 0) & (numerator < 0)).astype(np.int64)
    elif mode == "ceil":
        rounded = quotient + ((remainder != 0) & (numerator > 0)).astype(np.int64)
    elif mode == "half_up":
        rounded = quotient + (2 * remainder >= denominator).astype(np.int64)
    elif mode == "half_even":
        above_half = 2 * remainder > denominator
        exactly_half = 2 * remainder == denominator
        rounded = quotient + (above_half | (exactly_half & (quotient % 2 == 1))).astype(np.int64)
    else:
        raise ValueError(f"unsupported rounding mode: {mode}")
    return sign * rounded


def round_div(numerator: np.ndarray | int, denominator: int, mode: str) -> np.ndarray:
    """Round numerator/denominator to an integer using a fixed mode.

    denominator must be positive. The function is exact for int64 inputs.
    """

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return _abs_round_div(np.asarray(numerator, dtype=np.int64), denominator, mode)


def _round_div_checked(numerator: np.ndarray | int, denominator: int, mode: str) -> np.ndarray:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return _abs_round_div(np.asarray(numerator, dtype=np.int64), denominator, mode)


def clamp_fixed(values: np.ndarray, bit_width: int) -> np.ndarray:
    low = -(1 << (bit_width - 1))
    high = (1 << (bit_width - 1)) - 1
    return np.clip(np.asarray(values, dtype=np.int64), low, high)


def to_fixed(
    values: np.ndarray,
    fraction_bits: int,
    bit_width: int,
    mode: str = "half_even",
) -> np.ndarray:
    scale = float(1 << fraction_bits)
    scaled = np.asarray(values, dtype=np.float64) * scale
    if mode == "truncate":
        result = np.trunc(scaled)
    elif mode == "floor":
        result = np.floor(scaled)
    elif mode == "ceil":
        result = np.ceil(scaled)
    elif mode == "half_up":
        result = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    elif mode == "half_even":
        result = np.rint(scaled)
    else:
        raise ValueError(f"unsupported rounding mode: {mode}")
    return clamp_fixed(result.astype(np.int64), bit_width)


def quantize_fixed(raw: np.ndarray, step: int, mode: str) -> np.ndarray:
    if step <= 0:
        raise ValueError("quantization step must be positive")
    return _round_div_checked(raw, step, mode) * step


def quantization_residual(raw: np.ndarray, quantized: np.ndarray) -> np.ndarray:
    return np.asarray(raw, dtype=np.int64) - np.asarray(quantized, dtype=np.int64)


def quantization_rounding(config_rounding: dict[str, str]) -> str:
    """Use the diffuse channel's rounding mode for scalar quantization."""

    return config_rounding.get("diffuse", "half_even")


CHANNELS = ("diffuse", "roughness", "metallic", "normal_perturbation")
CHANNEL_LIMITS = {
    "diffuse": (0, 1),
    "roughness": (0, 1),
    "metallic": (0, 1),
    "normal_perturbation": (-1, 1),
}


def extract_channel(field: np.ndarray, scale: int, channel: str, mode: str = "half_even") -> np.ndarray:
    """Map one quantized layer field to a signed fixed-point channel."""

    field = np.asarray(field, dtype=np.int64)
    if channel == "diffuse":
        result = np.clip(field, 0, scale)
    elif channel == "roughness":
        result = scale - np.clip(field, 0, scale)
    elif channel == "metallic":
        result = _round_div_checked(field * field, scale, mode)
        result = np.clip(result, 0, scale)
    elif channel == "normal_perturbation":
        result = 2 * field - scale
        result = np.clip(result, -scale, scale)
    else:
        raise ValueError(f"unknown channel: {channel}")
    return result.astype(np.int64)


def fixed_to_unit(values: np.ndarray, scale: int) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) / float(scale)


def fixed_channel_to_uint8(values: np.ndarray, scale: int, channel: str) -> np.ndarray:
    unit = fixed_to_unit(values, scale)
    if channel == "normal_perturbation":
        unit = (unit + 1.0) * 0.5
    unit = np.clip(unit, 0.0, 1.0)
    return np.rint(unit * 255.0).astype(np.uint8)
