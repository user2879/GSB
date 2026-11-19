"""16 位半浮点（IEEE 754 binary16）线性光编解码。

数值口径声明
------------
* 解码（half -> float32）使用 IEEE 754 **就近舍入、偶数优先**
  （round-to-nearest-even, RNE）。binary16 的全部有限值都能被 binary32
  精确表示，因此 half -> float32 实际上是精确无损转换。
* 编码（float32 -> half）同样使用 RNE：24 位有效数（含隐含 1）右移后按
  guard/round/sticky 三位做偶数优先舍入，subnormal 半浮点同口径处理。
* NaN / Inf 处理策略（工作台合成数据不含这些值，这里仅定义防御口径）：
    - 解码时 +/-Inf 按 **饱和值 65504.0（带符号）** 替换；
    - 解码时任意 NaN（signaling/quiet）替换为 **0.0**；
    - 编码时 +/-Inf 与超界值饱和到 +/-65504；NaN 编码为 0。
  因此经过本模块落盘的帧数据保证不含 NaN/Inf。
* 本模块同时提供纯 Python 标量参考实现与 numpy (>=1.26, 默认 RNE)
  向量化路径；两条路径在测试中逐值比对，结果完全一致。
"""

from __future__ import annotations

import numpy as np

HALF_MAX = 65504.0
_HALF_MAX_EXP = 31  # half 指数域保留给 Inf/NaN 的值


def decode_half_scalar(bits: int) -> float:
    """单个 binary16 位模式解码为 Python float（float32 可精确表示）。"""
    bits &= 0xFFFF
    sign = -1.0 if (bits & 0x8000) else 1.0
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x03FF

    if exponent == 0:
        if mantissa == 0:
            return sign * 0.0
        # subnormal: (-1)^s * 2^-24 * m
        return sign * float(mantissa) * (2.0 ** -24)
    if exponent == _HALF_MAX_EXP:
        # Inf / NaN 防御口径
        if mantissa == 0:
            return sign * HALF_MAX
        return 0.0
    value = float((1 << 10) + mantissa) * (2.0 ** (exponent - 25))
    return sign * value


def encode_half_scalar(value: float) -> int:
    """单个 Python float 编码为 binary16 位模式（RNE + 饱和 + NaN 归零）。"""
    if value != value:  # NaN
        return 0
    if value in (float("inf"), float("-inf")):
        return 0xFBFF if value < 0 else 0x7BFF
    if value >= HALF_MAX:
        return 0x7BFF
    if value <= -HALF_MAX:
        return 0xFBFF

    sign_bit = 0
    if value < 0.0 or (value == 0.0 and np.signbit(value)):
        sign_bit = 0x8000
        value = -value
    if value == 0.0:
        return sign_bit

    bits = int(np.float32(value).view(np.uint32))
    f_exp = ((bits >> 23) & 0xFF) - 127
    f_mant = bits & 0x7FFFFF
    sig = f_mant | 0x800000  # 24 位有效数，隐含 1 在 bit23

    if f_exp < -25:
        return sign_bit

    if f_exp < -14:
        # subnormal：右移 r 位，使隐含 1 对齐到 half 的 bit (exp+24)
        r = -f_exp - 1  # e=-15 -> 14, ..., e=-25 -> 24
        kept = sig >> r
        guard = (sig >> (r - 1)) & 1
        sticky = 1 if (sig & ((1 << (r - 1)) - 1)) else 0
        lsb = kept & 1
        if guard and (lsb or sticky):
            kept += 1  # 进位到 0x400 即最小正常值 (exp 字段 bit10)
        return sign_bit | (kept & 0x7FF)

    # normal：保留高 11 位（含隐含 1），舍弃低 13 位
    new_exp = f_exp + 15
    if new_exp >= _HALF_MAX_EXP:
        return sign_bit | 0x7BFF
    kept = sig >> 13
    guard = (sig >> 12) & 1
    round_bit = (sig >> 11) & 1
    sticky = 1 if (sig & ((1 << 11) - 1)) else 0
    lsb = kept & 1
    if guard and (lsb or round_bit or sticky):
        kept += 1
        if kept == 0x800:  # 尾数进位溢出，指数 +1
            kept = 0x400
            new_exp += 1
    half_mant = kept & 0x3FF
    return sign_bit | (new_exp << 10) | half_mant


def half_bytes_to_float32(raw: np.ndarray) -> np.ndarray:
    """binary16 位模式数组 (..., uint16) -> float32 数组，RNE 精确转换。"""
    if raw.dtype != np.uint16:
        raise TypeError("half decode expects uint16 bit patterns")
    out = raw.view(np.float16).astype(np.float32)
    # NaN/Inf 防御口径（先处理 NaN，避免对 NaN 取 sign 触发告警）
    out = np.where(np.isnan(out), np.float32(0.0), out)
    with np.errstate(invalid="ignore"):
        out = np.where(np.isinf(out), np.sign(out) * np.float32(HALF_MAX), out)
    return out.astype(np.float32)


def float32_to_half_bytes(values: np.ndarray) -> np.ndarray:
    """float32 数组 -> binary16 位模式数组 (uint16)，RNE + 饱和 + NaN 归零。"""
    with np.errstate(over="ignore"):
        f16 = np.float16(values)  # numpy 默认 RNE
    u16 = f16.view(np.uint16).copy()
    # numpy 对超界值产生 +/-Inf(0x7C00/0xFC00)，按饱和口径替换
    u16 = np.where(u16 == 0x7C00, np.uint16(0x7BFF), u16)
    u16 = np.where(u16 == 0xFC00, np.uint16(0xFBFF), u16)
    # NaN 位模式 -> 0
    nan_mask = ((u16 & 0x7C00) == 0x7C00) & ((u16 & 0x03FF) != 0)
    u16 = np.where(nan_mask, np.uint16(0), u16)
    return u16.astype(np.uint16)
