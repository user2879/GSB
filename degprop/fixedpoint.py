"""整数定点基础设施与量化误差界推导。

缩放因子 ``F = 10**k``（k 取 0..3）。所有比例型量（阈值、可用比例、
容量分摊份额）均量化为分母 F 的定点整数；计数型容量以 ``count * F``
统一编码，保证容量分摊与阈值比较全部为整数运算。

截断方向
    量化与除法一律向负无穷截断（floor）。本组件输入均为非负数，
    等价于向下截断，量化结果不大于真实值，残差记录为中间量。

饱和规则
    定点整数以有符号 64 位为声明域，``FP_MAX = 2**63 - 1``。
    加/减/乘结果越界即饱和到 FP_MAX（或 0），并在返回的中间量里
    置 ``saturated=True``；计数容量与配额的原始乘积由领域层在饱和
    发生前单独守卫，触发 ``E_CAPACITY``。

误差界（绝对误差，单位为真实比例）
    * 单值量化误差 ``q = 1/F``
    * 路径聚合：以 (值, 绝对误差) 区间逐算子传播，输出全部中间量
    * 方案判定：两候选方案聚合误差界之和 ``J = B_a + B_b``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np

FP_MAX = 2**63 - 1
FP_MIN = 0


@dataclass(frozen=True)
class Scaling:
    """定点缩放参数。"""

    exponent: int

    def __post_init__(self) -> None:
        if not isinstance(self.exponent, int) or not 0 <= self.exponent <= 3:
            raise ValueError("scale exponent 必须是 0..3 的整数")
        object.__setattr__(self, "factor", 10**self.exponent)

    factor: int = 0

    @property
    def quantum(self) -> Fraction:
        """单值量化绝对误差上界 q = 1/F。"""
        return Fraction(1, self.factor)

    def encode_ratio(self, value: float | Fraction | int) -> int:
        """把 [0,1] 比例量化为定点整数（floor 截断）。"""
        scaled = int(Fraction(value).numerator * self.factor / Fraction(value).denominator)
        return _saturate(scaled)

    def encode_count(self, count: int) -> int:
        """整数容量计数统一编码为 count*F。"""
        if not isinstance(count, int):
            raise TypeError("容量计数必须为整数")
        return _saturate(count * self.factor)

    def decode_count(self, encoded: int) -> int:
        return encoded // self.factor

    def ratio(self, numerator: int, denominator: int) -> tuple[int, int]:
        """成功率定点值与余数：返回 (floor(num*F/den), 余数)。"""
        if denominator == 0:
            return 0, 0
        scaled_num = numerator * self.factor
        return _saturate(scaled_num // denominator), scaled_num % denominator

    def encode_ratios(self, values: Iterable[float]) -> np.ndarray:
        """NumPy 批量量化，dtype=int64，越界饱和。"""
        arr = np.floor(np.asarray(list(values), dtype=np.float64) * self.factor)
        return np.clip(arr, FP_MIN, FP_MAX).astype(np.int64)


def _saturate(value: int) -> int:
    if value > FP_MAX:
        return FP_MAX
    if value < FP_MIN:
        return FP_MIN
    return value


@dataclass
class Interval:
    """带绝对误差界的定点量（value 为定点整数，err 以定点单位表示）。"""

    value: int
    err: int
    intermediates: list[dict] = field(default_factory=list)

    def as_fraction(self, factor: int) -> tuple[Fraction, Fraction]:
        return Fraction(self.value, factor), Fraction(self.err, factor)


def quantized_interval(raw: Fraction, scaling: Scaling, name: str) -> Interval:
    """单值量化：输出定点值、截断残差与误差上界 q。"""
    q = scaling.factor  # 定点单位下 q=1（真实单位 1/F）
    value = scaling.encode_ratio(raw)
    remainder = raw * scaling.factor - value
    return Interval(
        value=value,
        err=q,
        intermediates=[
            {
                "op": "quantize",
                "name": name,
                "raw": str(raw),
                "fixed": value,
                "truncation": "floor",
                "remainder_fixed": float(remainder),
                "abs_error_bound_fixed": q,
                "abs_error_bound_real": str(scaling.quantum),
            }
        ],
    )


def interval_add(left: Interval, right: Interval, factor: int, name: str) -> Interval:
    """定点加法区间传播：值相加，误差界相加，带饱和标记。"""
    raw_sum = left.value + right.value
    saturated = raw_sum > FP_MAX or raw_sum < FP_MIN
    value = _saturate(raw_sum)
    err = _saturate(left.err + right.err)
    inter = list(left.intermediates) + list(right.intermediates)
    inter.append(
        {
            "op": "add",
            "name": name,
            "lhs": left.value,
            "rhs": right.value,
            "result": value,
            "err_fixed": err,
            "saturated": saturated,
        }
    )
    return Interval(value=value, err=err, intermediates=inter)


def interval_mul(left: Interval, right: Interval, factor: int, name: str) -> Interval:
    """定点乘法：(v1*v2)/F，误差按一阶区间公式传播。

    真实区间 [v-e, v+e]（定点单位）乘积上界展开：
        err = v1*e2 + v2*e1 + e1*e2  （再除以 F 转回定点单位）
    """
    v1, e1, v2, e2 = left.value, left.err, right.value, right.err
    raw = v1 * v2 // factor
    err_num = v1 * e2 + v2 * e1 + e1 * e2
    err = _saturate(err_num // factor + (1 if err_num % factor else 0))  # 误差上界向上取整
    saturated = v1 * v2 > FP_MAX * factor
    value = _saturate(raw)
    inter = list(left.intermediates) + list(right.intermediates)
    inter.append(
        {
            "op": "mul",
            "name": name,
            "lhs": v1,
            "rhs": v2,
            "result": value,
            "err_numerator": err_num,
            "err_fixed": err,
            "saturated": saturated,
        }
    )
    return Interval(value=value, err=err, intermediates=inter)


def path_aggregate(bounds: Sequence[Interval], factor: int, name: str) -> Interval:
    """一条传播路径上的比例聚合误差上界（连乘 + 链上加法合并）。

    路径因子先连乘，再按到达序累加，全部为整数运算，保留中间量。
    """
    if not bounds:
        return Interval(0, 0, [{"op": "empty_path", "name": name}])
    acc = bounds[0]
    for idx, nxt in enumerate(bounds[1:], start=1):
        acc = interval_mul(acc, nxt, factor, f"{name}.mul{idx}")
    acc.intermediates.append(
        {
            "op": "path_aggregate",
            "name": name,
            "length": len(bounds),
            "path_value_fixed": acc.value,
            "path_error_bound_fixed": acc.err,
            "path_error_bound_real": str(Fraction(acc.err, factor)),
        }
    )
    return acc


def judgment_bound(
    scheme_a: Interval, scheme_b: Interval, factor: int, name: str
) -> dict:
    """两降级方案比较的合成判定误差界 J = B_a + B_b。"""
    bound = _saturate(scheme_a.err + scheme_b.err)
    diff = abs(scheme_a.value - scheme_b.value)
    indistinguishable = diff <= bound
    return {
        "name": name,
        "value_a_fixed": scheme_a.value,
        "value_b_fixed": scheme_b.value,
        "abs_diff_fixed": diff,
        "error_a_fixed": scheme_a.err,
        "error_b_fixed": scheme_b.err,
        "judgment_bound_fixed": bound,
        "judgment_bound_real": str(Fraction(bound, factor)),
        "indistinguishable": indistinguishable,
        "intermediates": list(scheme_a.intermediates) + list(scheme_b.intermediates),
    }


def numpy_window_rates(success: np.ndarray, total: np.ndarray, factor: int) -> np.ndarray:
    """NumPy 向量化窗口成功率定点值（floor，total=0 记 0）。"""
    total_safe = np.where(total == 0, 1, total)
    rates = np.floor(success.astype(np.int64) * factor // total_safe.astype(np.int64))
    return np.where(total == 0, 0, rates).astype(np.int64)
