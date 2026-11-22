"""Integer fixed-point water-level ratios built with NumPy.

The ratio of used/capacity is scaled by ``scale_factor`` (a documented power of
ten) and truncated toward zero using integer floor division.  No floating point
is involved, so the same inputs always produce the same integers.

Quantization: for capacity C>0 and used U (0<=U<=C)
    ratio_q = floor(U * S / C)
    exact   = U * S / C
    0 <= exact - ratio_q < 1  (in scaled integer units), i.e. the absolute
    error bound is 1/S in ratio units, achieved in the worst case.
"""
from __future__ import annotations

import numpy as np


def quantize_ratio(used: int, capacity: int, scale_factor: int) -> int:
    if capacity <= 0:
        return 0
    used_arr = np.int64(used)
    cap_arr = np.int64(capacity)
    scale_arr = np.int64(scale_factor)
    return int(np.floor_divide(used_arr * scale_arr, cap_arr))


def quantization_error_bound(scale_factor: int) -> str:
    return f"abs(error) < 1/{scale_factor} ratio units (floor truncation)"
