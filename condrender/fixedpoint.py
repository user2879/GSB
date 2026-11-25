from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np

from .errors import Codes, Failure, ValidationError


PROFILE_FRAC_BITS = {
    "Q8_FAST": 8,
    "Q12_BALANCED": 12,
    "Q16_DIAG": 16,
}
ALLOWED_FRAC_BITS = (8, 12, 16)
ALLOWED_ROUNDING = ("HALF_AWAY_FROM_ZERO", "HALF_EVEN", "TRUNCATE")
BASIS = 10000


def q_scale(frac_bits: int) -> int:
    return 1 << int(frac_bits)


def parse_decimal(value: Any, field: str = "value") -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValidationError(Failure(
            Codes.COMMAND_INVALID, f"{field} is not a decimal number: {value!r}"
        )) from exc


def decimal_to_q(value: Any, frac_bits: int, field: str = "value") -> int:
    decimal = parse_decimal(value, field)
    scaled = decimal * q_scale(frac_bits)
    rounded = int(scaled.to_integral_value(rounding="ROUND_HALF_EVEN"))
    return rounded


def validate_quant_config(config: dict[str, Any]) -> tuple[int, str]:
    fixed = config.get("fixed_point", {})
    profile = fixed.get("quant_profile")
    frac_bits = fixed.get("frac_bits")
    if profile not in PROFILE_FRAC_BITS:
        raise ValidationError(Failure(
            Codes.QUANT_ENUM, f"quant_profile must be one of {sorted(PROFILE_FRAC_BITS)}",
            details={"actual": profile},
        ))
    if frac_bits not in ALLOWED_FRAC_BITS:
        raise ValidationError(Failure(
            Codes.FRAC_BITS_ENUM, f"frac_bits must be one of {ALLOWED_FRAC_BITS}",
            details={"actual": frac_bits},
        ))
    if PROFILE_FRAC_BITS[profile] != frac_bits:
        raise ValidationError(Failure(
            Codes.QUANT_ENUM, "quant_profile and frac_bits do not describe the same gear",
            details={"profile": profile, "frac_bits": frac_bits},
        ))
    rounding = fixed.get("rounding_mode", "HALF_AWAY_FROM_ZERO")
    if rounding not in ALLOWED_ROUNDING:
        raise ValidationError(Failure(
            Codes.ROUNDING_ENUM, f"rounding_mode must be one of {ALLOWED_ROUNDING}"
        ))
    return int(frac_bits), str(rounding)


def round_div(num: int, den: int, mode: str = "HALF_AWAY_FROM_ZERO") -> tuple[int, str, int]:
    """Round num/den and return (integer, direction relative to exact value, residual)."""
    if den <= 0:
        raise ValueError("den must be positive")
    if mode == "TRUNCATE":
        rounded = int(num / den)
    elif mode == "HALF_EVEN":
        sign = -1 if num < 0 else 1
        quotient, remainder = divmod(abs(num), den)
        twice = remainder * 2
        if twice > den or (twice == den and quotient % 2 == 1):
            quotient += 1
        rounded = sign * quotient
    else:
        sign = -1 if num < 0 else 1
        quotient, remainder = divmod(abs(num), den)
        if remainder * 2 >= den:
            quotient += 1
        rounded = sign * quotient
    difference_num = rounded * den - num
    if difference_num > 0:
        direction = "UP"
    elif difference_num < 0:
        direction = "DOWN"
    else:
        direction = "EXACT"
    residual = num - rounded * den
    return rounded, direction, residual


def fixed_mul(a_q: int, b_q: int, frac_bits: int,
              mode: str = "HALF_AWAY_FROM_ZERO") -> tuple[int, str, int]:
    """Multiply two Q-fraction integers and return Q result plus rounding trace."""
    return round_div(int(a_q) * int(b_q), q_scale(frac_bits), mode)


def basis_weight(value: Any, field: str) -> int:
    decimal = parse_decimal(value, field)
    return int((decimal * BASIS).to_integral_value(rounding="ROUND_HALF_EVEN"))


def round_div_array(num: np.ndarray | list[int], den: int,
                     mode: str = "HALF_AWAY_FROM_ZERO"):
    values = np.asarray(num, dtype=np.int64)
    if den <= 0:
        raise ValueError("den must be positive")
    if mode == "TRUNCATE":
        rounded = (values // den).astype(np.int64)
    elif mode == "HALF_EVEN":
        sign = np.sign(values)
        quotient, remainder = np.divmod(np.abs(values), den)
        up = (2 * remainder > den) | ((2 * remainder == den) & (quotient % 2 == 1))
        quotient = quotient + up.astype(np.int64)
        rounded = sign * quotient
    else:
        sign = np.sign(values)
        quotient, remainder = np.divmod(np.abs(values), den)
        quotient = quotient + (2 * remainder >= den).astype(np.int64)
        rounded = sign * quotient
    difference_num = rounded * den - values
    direction = np.where(difference_num > 0, 1,
                         np.where(difference_num < 0, -1, 0)).astype(np.int8)
    residual = values - rounded * den
    return rounded.astype(np.int64), direction, residual.astype(np.int64)
