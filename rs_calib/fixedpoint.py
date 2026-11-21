"""定点量化与饱和处理（确定性，纯 NumPy/标准库）。"""
from __future__ import annotations

import math

import numpy as np

from .errors import E_FIXEDPOINT_OVERFLOW, StageError


def to_fixed_point(values: np.ndarray, total_bits: int, frac_bits: int,
                   *, tile_id: str, band: int, stage: str) -> np.ndarray:
    """浮点物理量 -> 有符号定点整数。

    量化档位为 2**frac_bits，四舍六入五成双（NumPy 默认 bankers rounding），
    保证逐位确定。超出有符号位宽范围时抛 E_FIXEDPOINT_OVERFLOW。
    """
    if frac_bits < 0 or frac_bits >= total_bits:
        raise StageError(
            E_FIXEDPOINT_OVERFLOW,
            f"定点小数位宽 frac_bits={frac_bits} 与总位宽 {total_bits} 不合法",
            tile_id=tile_id, band=band, stage=stage,
        )
    scale = 1 << frac_bits
    quantized = np.rint(np.asarray(values, dtype=np.float64) * scale)
    int_min = -(1 << (total_bits - 1))
    int_max = (1 << (total_bits - 1)) - 1
    if quantized.size and (np.min(quantized) < int_min or np.max(quantized) > int_max):
        raise StageError(
            E_FIXEDPOINT_OVERFLOW,
            f"定点值超出 {total_bits} 位有符号范围 [{int_min},{int_max}]",
            tile_id=tile_id, band=band, stage=stage,
        )
    return quantized.astype(np.int64)


def fixed_point_scale(frac_bits: int) -> int:
    return 1 << frac_bits


def finite_float(value, code, message, *, tile_id=None, band=None, stage):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise StageError(code, message, tile_id=tile_id, band=band, stage=stage)
    return float(value)
