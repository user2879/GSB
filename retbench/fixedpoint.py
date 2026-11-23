"""定点量化与定点整数运算。

允许枚举：
- 小数位宽 FRAC_BITS: 8 / 12 / 16；
- 量化档位 TIER: coarse(粗) / standard(标) / fine(细)。

量化语义（确定性）：
    step = tier_scale * 2 ** (-frac_bits)
    q    = clip(rint(x / step), INT32_MIN, INT32_MAX)
    residual = x - q * step

档位变化会同时改变 step 与整数结果，从而真实改变下游整数排序。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALLOWED_FRAC_BITS = (8, 12, 16)
ALLOWED_TIERS = ("coarse", "standard", "fine")
TIER_SCALE = {"coarse": 4.0, "standard": 1.0, "fine": 0.25}
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class QuantSpec:
    frac_bits: int
    tier: str

    @property
    def step(self) -> float:
        return TIER_SCALE[self.tier] * 2.0 ** (-self.frac_bits)

    @property
    def scale(self) -> int:
        """1/step 的整数近似，用于定点乘加归一。"""

        return int(round(1.0 / self.step))

    def key(self) -> str:
        return f"q{self.frac_bits}-{self.tier}"


def validate(frac_bits: int, tier: str) -> None:
    if frac_bits not in ALLOWED_FRAC_BITS:
        raise ValueError(f"非法定点小数位宽 {frac_bits!r}，允许 {ALLOWED_FRAC_BITS}")
    if tier not in ALLOWED_TIERS:
        raise ValueError(f"非法量化档位 {tier!r}，允许 {ALLOWED_TIERS}")


def quantize_array(values: np.ndarray, spec: QuantSpec) -> dict:
    """把浮点数组量化为定点整数，返回步长/整数/残差等可审计记录。"""

    validate(spec.frac_bits, spec.tier)
    x = np.asarray(values, dtype=np.float64)
    step = spec.step
    scaled = np.rint(x / step)
    clipped = int(np.count_nonzero((scaled < INT32_MIN) | (scaled > INT32_MAX)))
    q = np.clip(scaled, INT32_MIN, INT32_MAX).astype(np.int64)
    residual = x - q.astype(np.float64) * step
    return {
        "frac_bits": spec.frac_bits,
        "tier": spec.tier,
        "step": step,
        "q": q,
        "residual": residual,
        "clipped": clipped,
    }


def quantize_scalar(value: float, spec: QuantSpec) -> int:
    return int(quantize_array(np.array([value], dtype=np.float64), spec)["q"][0])


def dequantize(q: np.ndarray, spec: QuantSpec) -> np.ndarray:
    return q.astype(np.float64) * spec.step


def fixed_dot(q_a: np.ndarray, q_b: np.ndarray, spec: QuantSpec) -> int:
    """两个同档位定点向量的定点内积：先逐元素整数乘，再按 scale 归一。

    全部运算保持 int64，顺序由调用方保证（维度升序）。
    """

    prod = (np.asarray(q_a, dtype=np.int64) * np.asarray(q_b, dtype=np.int64)).sum()
    scale = spec.scale
    if prod >= 0:
        return int(prod // scale)
    return -int((-prod) // scale)


def fixed_weight(value: float, spec: QuantSpec) -> int:
    """把权重系数转为同档位定点整数。"""

    return quantize_scalar(float(value), spec)
