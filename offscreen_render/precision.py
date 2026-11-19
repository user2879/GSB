"""单精度中间量模拟（业务要求 4）。

渲染主结果始终使用 float64 逐段累乘；单精度路径在每一步把中间透射率
量化到规格声明的有效尾数位（IEEE-754 舍入到最近、偶数保留），
从而可复核地观察有限有效位带来的误差产生与传播。
"""
from __future__ import annotations

import math

import numpy as np


def quantize_mantissa(value: float, mantissa_bits: int) -> float:
    """把 float64 数值舍入到仅保留 ``mantissa_bits`` 位尾数的浮点格式。

    mantissa_bits=23 等价于 IEEE-754 binary32（单精度）。
    对 0、非有限值与次常规量做保守处理。
    """
    if value is None:
        return value
    v = float(value)
    if v == 0.0 or not math.isfinite(v):
        return v
    sign = -1.0 if v < 0.0 else 1.0
    magnitude = abs(v)
    # 二进制指数 floor(log2(m))
    exponent = math.floor(math.log2(magnitude))
    scale = 2.0 ** (exponent - mantissa_bits)
    # 舍入到最近、 ties to even
    scaled = magnitude / scale
    lower = math.floor(scaled)
    frac = scaled - lower
    if frac > 0.5 or (frac == 0.5 and (lower % 2 == 1)):
        q = lower + 1
    else:
        q = lower
    result = sign * q * scale
    # 处理舍入后指数进位（例如量化到 2 的幂边界）
    if not math.isfinite(result):
        return v
    return result


def quantize_array(values: np.ndarray, mantissa_bits: int) -> np.ndarray:
    """向量化版本，逐元素调用尾数量化。"""
    flat = values.reshape(-1)
    out = np.empty_like(flat, dtype=np.float64)
    for i in range(flat.size):
        out[i] = quantize_mantissa(float(flat[i]), mantissa_bits)
    return out.reshape(values.shape)
