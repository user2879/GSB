"""定点整数运算原语。

校准全程以定点整数按固定运算顺序推进，float64 仅作为参考值保留，
不参与任何后续运算，保证同一输入与配置下逐位一致。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ARTIFACT_VERSION = "1.0.0"
ROUNDING_MODE = "ROUND_HALF_EVEN"

ALLOWED_FRAC_BITS = (8, 12, 16)
ALLOWED_QUANT_GEARS = (64, 256, 1024)


def half_even_div(num: int, den: int) -> int:
    """带符号的半偶舍入整数除法（banker's rounding）。"""
    if den <= 0:
        raise ValueError("denominator must be positive")
    q, r = divmod(abs(num), den)
    if r * 2 > den or (r * 2 == den and (q & 1) == 1):
        q += 1
    return q if num >= 0 else -q


@dataclass(frozen=True)
class FixedPoint:
    frac_bits: int
    quant_gear: int

    def __post_init__(self) -> None:
        if self.frac_bits not in ALLOWED_FRAC_BITS:
            raise ValueError(f"frac_bits must be one of {ALLOWED_FRAC_BITS}, got {self.frac_bits}")
        if self.quant_gear not in ALLOWED_QUANT_GEARS:
            raise ValueError(f"quant_gear must be one of {ALLOWED_QUANT_GEARS}, got {self.quant_gear}")

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits

    @property
    def headroom(self) -> int:
        # 量化档位决定饱和余量：64->1, 256->2, 1024->4（以 1.0 为单位）。
        return {64: 1, 256: 2, 1024: 4}[self.quant_gear]

    @property
    def limit(self) -> int:
        return self.headroom * self.scale

    def quantize(self, value: float) -> tuple[int, float]:
        """float64 -> 定点整数，返回 (整数, 量化残差 float64)。"""
        if not np.isfinite(value):
            raise ValueError("non-finite value cannot be quantized")
        scaled = value * self.scale
        candidate = int(np.round(scaled))  # numpy 默认半偶舍入
        clamped = max(-self.limit, min(self.limit, candidate))
        residual = float(value - clamped / self.scale)
        return clamped, float(residual)

    def to_float(self, value: int) -> float:
        return value / self.scale

    def mul_scaled(self, a: int, b_q: int) -> int:
        """定点乘法 a * (b_q / scale)，半偶舍入回定点。"""
        return half_even_div(a * b_q, self.scale)

    def add(self, a: int, b: int) -> int:
        return a + b

    def clamp(self, value: int) -> int:
        return max(-self.limit, min(self.limit, value))

    def metadata(self) -> dict:
        return {
            "artifact_version": ARTIFACT_VERSION,
            "rounding_mode": ROUNDING_MODE,
            "frac_bits": self.frac_bits,
            "quant_gear": self.quant_gear,
            "scale": self.scale,
            "headroom": self.headroom,
            "limit": self.limit,
        }
