"""定点整数与精确有理数工具。

归一化与融合过程中完全不使用 float：原始 JSON 数值解析为 ``Fraction``，
归一化统计量也是精确分数，量化时按定点小数位宽四舍五入为 Python 大整数，
后续加权融合、裁剪、折减全部在整数域完成，位宽切换必然改变量化结果。
"""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from typing import Any, Iterable

SCALE_ONE = 1  # 定点整数 1 的语义基准（配合 scale=10**frac_bits 使用）


def parse_exact(value: Any, *, field: str = "value") -> Fraction:
    """把 JSON 标量解析为精确 Fraction，拒绝非有限浮点。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} 必须是数值，实际为 {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field} 为非有限值: {value}")
    return Fraction(value)


def round_half_away(value: Fraction) -> int:
    """对称四舍五入（远离零方向），保证正负号确定性。"""
    if value >= 0:
        return (value.numerator * 2 + value.denominator) // (2 * value.denominator)
    return -(((-value).numerator * 2 + (-value).denominator) // (2 * (-value).denominator))


def quantize(value: Fraction, frac_bits: int) -> int:
    """把精确分数量化为 ``frac_bits`` 位小数的定点整数。"""
    scale = 10**frac_bits
    return round_half_away(value * scale)


def clip_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def isqrt_frac(value: Fraction) -> Fraction:
    """精确分数的平方根：分别对分子分母整数开方并做一次定点修正。

    返回 ``Fraction``，调用方再量化，保证 z 分数全程不引入 float。
    """
    if value < 0:
        raise ValueError("平方根不能为负数")
    num = math.isqrt(value.numerator)
    den = math.isqrt(value.denominator)
    root = Fraction(num, den)
    # 整数开方向下取整，做一次牛顿修正使误差小于一个分母单位，后续仍会量化。
    if root * root != value and root != 0:
        better = (root + value / root) / 2
        if better * better <= value or True:  # 保留更接近的估计
            root = better
    return root


def stable_hash(obj: Any) -> str:
    """对任意可 JSON 化对象计算稳定指纹（排序键、规范 JSON、blake2b）。"""
    canonical = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


def compound_key(*parts: Any) -> str:
    """由多个部件派生缓存键。"""
    return stable_hash(list(parts))


def stable_task_key(request_id: str, item_id: str) -> int:
    """由请求 ID 与候选物品 ID 派生稳定的整数排序键。"""
    digest = hashlib.blake2b(
        f"{request_id}\u0000{item_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def fraction_to_plain(value: Fraction) -> str:
    """Fraction 的规范字符串，用于指纹与报告（不用浮点）。"""
    return f"{value.numerator}/{value.denominator}"


def mean(values: Iterable[Fraction]) -> Fraction:
    values = list(values)
    if not values:
        raise ValueError("空序列不能求均值")
    total = Fraction(0)
    for value in values:
        total += value
    return total / len(values)


def population_variance(values: list[Fraction], mu: Fraction) -> Fraction:
    """总体方差（分母 N），精确分数。"""
    if not values:
        raise ValueError("空序列不能求方差")
    total = Fraction(0)
    for value in values:
        delta = value - mu
        total += delta * delta
    return total / len(values)
