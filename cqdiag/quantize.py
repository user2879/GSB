"""逐列定点重缩放与饱和截断内核 (业务要求 3).

定点表示: 浮点值 x 先转为定点整数 fx = round(x * 2^frac_bits);
缩放因子同样定点化 scale_fp = round(scale * 2^frac_bits) (>=1),
量化整数 q = round_half_away(fx / scale_fp) + zero_point.
frac_bits 与量化位宽真实改变中间整数与截断集合.
"""

from __future__ import annotations

import numpy as np


def scale_to_fixed(scale: float, frac_bits: int) -> int:
    """缩放因子定点化, 保证至少为 1 避免除零."""
    return max(1, int(np.round(scale * (1 << frac_bits))))


def _round_half_away_div(fx: np.ndarray, denom: int) -> np.ndarray:
    """对整数数组做 round_half_away_from_zero(fx / denom) 的精确整数运算."""
    fx = fx.astype(np.int64)
    sign = np.where(fx < 0, -1, 1)
    mag = np.abs(fx)
    if denom % 2 == 0:
        q = (mag + denom // 2) // denom
    else:
        q = (2 * mag + denom) // (2 * denom)
    return sign * q


def quantize_column(
    col_values: np.ndarray,
    col_missing: np.ndarray,
    scale: float,
    zero_point: int,
    qmin: int,
    qmax: int,
    frac_bits: int,
    saturation: str,
) -> dict:
    """返回逐元素定点整数、截断方向与截断前定点值."""
    n = col_values.size
    scale_fp = scale_to_fixed(scale, frac_bits)
    fx = np.round(col_values * (1 << frac_bits)).astype(np.int64)
    pre_clip = _round_half_away_div(fx, scale_fp) + zero_point
    clipped = np.clip(pre_clip, qmin, qmax)
    clip_dir = np.sign(pre_clip - clipped).astype(np.int64)  # -1 下溢, 0 未截断, +1 上溢
    q_out: list[int | None] = []
    pre_out: list[int | None] = []
    dir_out: list[int | None] = []
    for i in range(n):
        if col_missing[i]:
            q_out.append(None)
            pre_out.append(None)
            dir_out.append(None)
        else:
            q_out.append(int(clipped[i]))
            pre_out.append(int(pre_clip[i]))
            dir_out.append(int(clip_dir[i]))
    clipped_count = int(np.count_nonzero(clip_dir[~col_missing]))
    return {
        "q": q_out,
        "pre_clip": pre_out,
        "clip_dir": dir_out,
        "clipped_count": clipped_count,
        "scale_fp": int(scale_fp),
        "frac_bits": frac_bits,
        "saturation": saturation,
        "qmin": qmin,
        "qmax": qmax,
    }
