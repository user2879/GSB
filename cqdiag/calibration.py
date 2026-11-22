"""逐列分位标定与缩放因子/零点推导 (业务要求 2).

固定规则 (写入产物元数据):
- 插值规则: linear —— 排序后位置 pos = p*(n-1), 在 floor/ceil 两点间线性插值;
- 并列值决断: 不特判, 由排序+线性插值天然确定 (稳定排序, 相同值插值结果相同);
- 缺失元素不参与标定, 记录有效元素数 valid_count 与引用的元素下标集合.
"""

from __future__ import annotations

import numpy as np

INTERPOLATION_RULE = "linear"
TIE_RULE = "stable_sort_linear_interp"


def quantile_permille(sorted_vals: np.ndarray, permille: int) -> float:
    """升序数组上按千分比取分位值, 线性插值."""
    n = sorted_vals.shape[0]
    if n == 0:
        raise ValueError("空列无法标定")
    if n == 1:
        return float(sorted_vals[0])
    pos = (permille / 1000.0) * (n - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def quant_range(bits: int) -> tuple[int, int]:
    """有符号量化位宽的整数区间 [qmin, qmax]."""
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def calibrate_column(
    col_values: np.ndarray,
    col_missing: np.ndarray,
    lo_permille: int,
    hi_permille: int,
) -> dict:
    """分位标定阶段产物 (不依赖位宽): 分位值 + 有效元素数 + 引用元素集合."""
    valid_idx = np.nonzero(~col_missing)[0]
    valid = col_values[valid_idx]
    if valid.size == 0:
        raise ValueError("列内无有效元素")
    sorted_vals = np.sort(valid, kind="stable")
    return {
        "q_lo": quantile_permille(sorted_vals, lo_permille),
        "q_hi": quantile_permille(sorted_vals, hi_permille),
        "col_min": float(sorted_vals[0]),
        "col_max": float(sorted_vals[-1]),
        "valid_count": int(valid.size),
        "total_count": int(col_values.size),
        "element_refs": [int(i) for i in valid_idx],
        "meta": {"interpolation": INTERPOLATION_RULE, "tie_rule": TIE_RULE},
    }


def derive_scale_zero_point(calib: dict, scheme: str, bits: int) -> dict:
    """由分位标定结果推导缩放因子与零点 (依赖方案与量化位宽)."""
    qmin, qmax = quant_range(bits)
    q_lo = calib["q_lo"]
    q_hi = calib["q_hi"]
    if scheme == "symmetric":
        max_abs = max(abs(q_lo), abs(q_hi))
        scale = max_abs / qmax if max_abs > 0 else 1.0
        zero_point = 0
    else:
        span = q_hi - q_lo
        scale = span / (qmax - qmin) if span > 0 else 1.0
        zp_f = qmin - q_lo / scale
        zero_point = int(np.round(zp_f))
        zero_point = int(min(max(zero_point, qmin), qmax))
    return {
        "scale": float(scale),
        "zero_point": int(zero_point),
        "qmin": qmin,
        "qmax": qmax,
        "scheme": scheme,
        "bits": bits,
    }
