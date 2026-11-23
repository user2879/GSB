"""定点量化与纯整数域定点运算（确定性，多次运行逐元素一致）。"""

from __future__ import annotations

import math

import numpy as np

_LN2 = 0.6931471805599453


def quantize(arr: np.ndarray, frac_bits: int, level_bits: int) -> np.ndarray:
    """按量化档位把浮点张量转为定点整数（int64 存储，值域受档位限制）。"""
    scale = 1 << frac_bits
    q = np.rint(np.asarray(arr, dtype=np.float64) * scale).astype(np.int64)
    lo = -(1 << (level_bits - 1))
    hi = (1 << (level_bits - 1)) - 1
    return np.clip(q, lo, hi)


def dequantize_scalar(value: int, frac_bits: int) -> float:
    return value / float(1 << frac_bits)


def dequantize(arr, frac_bits: int) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64) / float(1 << frac_bits)


def saturate(value: int, bits: int):
    """把整数饱和到声明位宽的有符号范围，返回 (值, 是否溢出)。"""
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    if value > hi:
        return hi, True
    if value < lo:
        return lo, True
    return value, False


def fp_exp(x_fp: int, frac_bits: int) -> int:
    """定点 exp：输入 x_fp = x * 2^frac_bits（要求 x <= 0），
    返回 exp(x) * 2^frac_bits 的整数近似。

    纯 Python 整数运算（任意精度），结果完全确定。
    范围约化：exp(x) = 2^k * exp(r)，r ∈ [0, ln2)，Taylor 展开求和。
    """
    if x_fp >= 0:
        return 1 << frac_bits
    ln2_fp = int(round(_LN2 * (1 << frac_bits)))
    k = x_fp // ln2_fp  # 向负无穷取整，k <= 0
    r_fp = x_fp - k * ln2_fp  # 0 <= r_fp < ln2_fp
    # exp(r)，r ∈ [0, ln2)
    total = 1 << frac_bits
    term = 1 << frac_bits
    n = 1
    while term:
        term = (term * r_fp) // (n << frac_bits)
        total += term
        n += 1
        if n > 200:  # 防御性上限，正常远早于此收敛到 0
            break
    # 乘上 2^k（k <= 0，右移）
    shift = -k
    if shift >= 63:
        return 0
    return total >> shift


def inv_sqrt_fp(head_dim: int, frac_bits: int) -> int:
    return int(round((1.0 / math.sqrt(head_dim)) * (1 << frac_bits)))
