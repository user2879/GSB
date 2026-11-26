"""定点整数域工具：量化档位枚举、定点换算与可记录方向的舍入。"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import PipelineValidationError, E_CONFIG_INVALID

ALLOWED_FRAC_BITS = (2, 4, 8, 16)
QUANT_LEVELS = {"coarse": 2, "standard": 4, "fine": 8, "ultra": 16}


def validate_quant(frac_bits: int, quant_level: str) -> None:
    if frac_bits not in ALLOWED_FRAC_BITS:
        raise PipelineValidationError(
            E_CONFIG_INVALID, "<config>", "frac_bits",
            f"frac_bits {frac_bits} 不在允许枚举 {ALLOWED_FRAC_BITS} 内")
    if quant_level not in QUANT_LEVELS:
        raise PipelineValidationError(
            E_CONFIG_INVALID, "<config>", "quant_level",
            f"quant_level {quant_level!r} 不在允许枚举 {sorted(QUANT_LEVELS)} 内")
    if QUANT_LEVELS[quant_level] != frac_bits:
        raise PipelineValidationError(
            E_CONFIG_INVALID, "<config>", "quant_level",
            f"quant_level {quant_level!r} 对应位宽 {QUANT_LEVELS[quant_level]} "
            f"与 frac_bits={frac_bits} 不一致")


@dataclass(frozen=True)
class FixedPoint:
    frac_bits: int

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits

    def from_int(self, value: int) -> int:
        return int(value) << self.frac_bits

    def from_real(self, value: float) -> int:
        """半向上取整的确定性实数转定点，负数按绝对值处理后还原符号。"""
        sign = -1 if value < 0 else 1
        scaled = abs(value) * self.scale
        return sign * int(scaled + 0.5)

    def mul(self, left: int, right: int) -> int:
        product = left * right
        if product >= 0:
            return product >> self.frac_bits
        return -((-product) >> self.frac_bits)

    def div(self, numerator: int, denominator: int) -> int:
        """两个原始整数相除，返回 q 定点值（numerator/denominator * scale）。"""
        if denominator == 0:
            raise ZeroDivisionError("fixed point division by zero")
        scaled = numerator * self.scale
        sign = -1 if (scaled < 0) != (denominator < 0) else 1
        return sign * (abs(scaled) // abs(denominator))

    def round_to_int(self, value: int) -> tuple[int, str]:
        """定点值转整数像素，半向上取整；返回 (整数, 舍入方向 exact/up/down)。"""
        sign = -1 if value < 0 else 1
        magnitude = abs(value)
        half = 1 << (self.frac_bits - 1)
        fraction = magnitude & (self.scale - 1)
        base = magnitude >> self.frac_bits
        if fraction == 0:
            return (base if sign >= 0 else -base), "exact"
        if fraction >= half:
            rounded = base + 1
            return (rounded if sign >= 0 else -rounded), "up"
        return (base if sign >= 0 else -base), "down"

    def mean(self, total: int, count: int) -> int:
        if count <= 0:
            raise ZeroDivisionError("mean over empty set")
        return self.div(total, count)


def int_cos_sin(degrees: int, frac_bits: int) -> tuple[int, int]:
    """整数度数(仅支持小角度枚举)的定点 cos/sin，使用本地常量表，无浮点。"""
    table = {
        -5: (9962, -872),
        -3: (9986, -523),
        0: (10000, 0),
        3: (9986, 523),
        5: (9962, 872),
    }
    if degrees not in table:
        raise ValueError(f"rotation degrees {degrees} not in deterministic table")
    cos_q4, sin_q4 = table[degrees]
    shift = frac_bits - 4
    if shift >= 0:
        return cos_q4 << shift, sin_q4 << shift
    return cos_q4 >> (-shift), sin_q4 >> (-shift)
