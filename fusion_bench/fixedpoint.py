"""定点整数运算：全程 int64，舍入方式固定为 half-even（银行家舍入）。

约定：定点值 = 真实值 * (1 << frac_bits)，以 int64 存储。
所有除法与乘法还原均使用确定性 half-even 舍入，禁止任何运行时随机状态。
"""

from fractions import Fraction

import numpy as np

ROUNDING = "half_even"


def scale_of(frac_bits):
    return 1 << frac_bits


def _round_fraction_half_even(fr):
    """Fraction -> int，half-even 舍入。"""
    q = fr.numerator // fr.denominator  # 向下取整
    r = fr - q
    twice = 2 * r
    if twice > 1:
        return q + 1
    if twice < 1:
        return q
    return q if q % 2 == 0 else q + 1


def float_to_fixed(value, frac_bits):
    """浮点（配置值）精确转定点，half-even。"""
    return _round_fraction_half_even(Fraction(value) * scale_of(frac_bits))


def div_round(numer, denom):
    """int64 数组/标量除以正整数 denom，half-even 舍入，返回 int64。"""
    a = np.asarray(numer, dtype=np.int64)
    d = np.int64(denom)
    if d <= 0:
        raise ValueError("denom must be positive")
    sign = np.where(a < 0, np.int64(-1), np.int64(1))
    mag = np.abs(a)
    q = mag // d
    r = mag - q * d
    up = (2 * r > d) | ((2 * r == d) & ((q & 1) == 1))
    out = sign * (q + up.astype(np.int64))
    if out.ndim == 0:
        return int(out)
    return out


def mul_fixed(a, b, frac_bits):
    """定点乘法：(a * b) >> frac_bits，half-even。"""
    return div_round(np.asarray(a, dtype=np.int64) * np.asarray(b, dtype=np.int64),
                     scale_of(frac_bits))


def quantize(values, frac_bits, quant_levels, quant_range_fixed):
    """按量化档位对齐定点值。

    quantum = quant_range / quant_levels（定点单位），
    值对齐到最近的 quantum 整数倍（half-even），并裁剪到 [-range, range]。
    """
    v = np.asarray(values, dtype=np.int64)
    quantum = max(1, int(div_round(quant_range_fixed, quant_levels)))
    aligned = div_round(v, quantum) * np.int64(quantum)
    clipped = np.clip(aligned, -quant_range_fixed, quant_range_fixed)
    if clipped.ndim == 0:
        return int(clipped)
    return clipped.astype(np.int64)


def exp_fixed(x_fixed, frac_bits, terms=24):
    """定点 exp，仅支持 x <= 0（融合前已做减最大值稳定化）。

    先将 x 折半至 [-0.5, 0] 再用 Taylor 级数（避免大负值直接展开的
    灾难性抵消），再逐次平方还原。全部整数定点运算，确定性结果。
    """
    scale = scale_of(frac_bits)
    x = max(int(x_fixed), -16 * scale)  # 下溢保护：exp(-16) 已低于量化粒度
    halvings = 0
    while x < -(scale // 2):
        x = div_round(x, 2)
        halvings += 1
    result = scale  # 1.0
    term = scale
    for n in range(1, terms):
        term = div_round(term * x, scale)
        term = div_round(term, n)
        if term == 0:
            break
        result += term
    for _ in range(halvings):
        result = mul_fixed(result, result, frac_bits)
    return int(result)


def mean_fixed(values):
    """int64 序列均值的定点值（half-even）。"""
    arr = np.asarray(values, dtype=np.int64)
    return div_round(int(arr.sum(dtype=np.int64)), int(arr.size))
