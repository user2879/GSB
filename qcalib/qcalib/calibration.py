"""逐列组分位标定与缩放因子、零点推导（纯函数、确定性）。"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from .config import GroupConfig
from .fingerprint import cache_key, content_fingerprint, fp_float, fp_floats

# 固定的插值与并列值决断规则（写入产物元数据）
INTERPOLATION_RULE = "linear"
TIE_BREAK_RULE = "lower-index-on-equal"


def quantile_permille(sorted_values: List[float], permille: int) -> float:
    """对已升序、非缺失数组按千分位 (0..1000) 取分位值。

    插值：线性插值；并列值决断：取下标更小者（lower-index-on-equal）。
    permille=0 -> 最小值，permille=1000 -> 最大值。
    """
    n = len(sorted_values)
    if n == 0:
        raise ValueError("空数组无法计算分位值")
    if n == 1:
        return float(sorted_values[0])
    if permille <= 0:
        return float(sorted_values[0])
    if permille >= 1000:
        return float(sorted_values[-1])
    position = (permille / 1000.0) * (n - 1)
    lower = int(position)  # 向零取整；并列时天然保留更低下标
    upper = lower + 1
    if upper >= n:
        return float(sorted_values[-1])
    weight = position - lower
    low_val = sorted_values[lower]
    high_val = sorted_values[upper]
    if low_val == high_val:  # 并列值决断：低下标
        return float(low_val)
    return float(low_val + (high_val - low_val) * weight)


@dataclass
class ColumnCalib:
    group_id: str
    column: str
    n_used: int
    q_lo_value: float
    q_hi_value: float
    scale: float
    zero_point: int
    qbits: int
    fbits: int
    scheme: str
    rounding: str
    saturation: str
    interpolation_rule: str
    tie_break_rule: str
    rows_fingerprint: str
    params_fingerprint: str
    cache_key: str
    stage: str = "calibration"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroupCalib:
    group_id: str
    columns: List[ColumnCalib]
    cache_keys: List[str] = field(default_factory=list)


def _clip_bounds(qbits: int, fbits: int, scheme: str) -> tuple[int, int]:
    """定点重缩放的裁剪（饱和）整数区间。

    LSB 尺度 span/2^qbits 固定；小数位宽 fbits 占用低位后，
    真实值可表示的整数峰值为 2^(qbits-fbits)（对称）或 [0, 2^(qbits-fbits))。
    """
    head = qbits - fbits
    if head <= 0:
        # 退化：至少保留 1 个整数档
        head = 1
    if scheme == "symmetric":
        bound = (1 << head) - 1
        return -bound, bound
    return 0, (1 << head) - 1


def _derive_symmetric(
    q_lo: float, q_hi: float, qbits: int, fbits: int
) -> tuple[float, int]:
    _, hi = _clip_bounds(qbits, fbits, "symmetric")
    amplitude = max(abs(q_lo), abs(q_hi))
    if amplitude == 0.0:
        amplitude = 1.0  # 退化保护，避免零缩放
    # 尺度 = 幅值 / 有效对称整数峰值；fbits 改变峰值从而改变尺度
    scale = amplitude / float(hi)
    return scale, 0


def _derive_asymmetric(
    q_lo: float, q_hi: float, qbits: int, fbits: int
) -> tuple[float, int]:
    _, hi = _clip_bounds(qbits, fbits, "asymmetric")
    span = q_hi - q_lo
    if span <= 0.0:
        span = 1.0  # 退化保护
    scale = span / float(hi)  # 尺度随 fbits 决定的档数变化
    lo, clip_hi = _clip_bounds(qbits, fbits, "asymmetric")
    zero_point = int(round(-q_lo / scale))
    zero_point = min(max(zero_point, lo), clip_hi)  # 裁剪到允许区间
    return scale, zero_point


def calibrate_column(
    cfg: GroupConfig,
    column: str,
    vector: List[Optional[float]],
    row_ids: List[str],
    used_row_ids: Optional[List[str]] = None,
) -> ColumnCalib:
    present = [v for v in vector if v is not None]
    if not present:
        raise ValueError(f"列 {column} 全部缺失，无法标定")
    sorted_values = sorted(present)
    q_lo = quantile_permille(sorted_values, cfg.q_lo_permille)
    q_hi = quantile_permille(sorted_values, cfg.q_hi_permille)

    if cfg.scheme == "symmetric":
        scale, zero_point = _derive_symmetric(q_lo, q_hi, cfg.qbits, cfg.fbits)
    else:
        scale, zero_point = _derive_asymmetric(q_lo, q_hi, cfg.qbits, cfg.fbits)

    if used_row_ids is None:
        used_row_ids = [rid for rid, v in zip(row_ids, vector) if v is not None]
    rows_fp = content_fingerprint(
        {
            "column": column,
            "rows": used_row_ids,
            "values_fp": fp_floats(sorted_values),
        }
    )
    key = cache_key(
        "calibration",
        [rows_fp],
        params=cfg.params_fingerprint,
        qbits=cfg.qbits,
        fbits=cfg.fbits,
        exchange=0,
    )
    return ColumnCalib(
        group_id=cfg.group_id,
        column=column,
        n_used=len(sorted_values),
        q_lo_value=q_lo,
        q_hi_value=q_hi,
        scale=scale,
        zero_point=zero_point,
        qbits=cfg.qbits,
        fbits=cfg.fbits,
        scheme=cfg.scheme,
        rounding=cfg.rounding,
        saturation=cfg.saturation,
        interpolation_rule=INTERPOLATION_RULE,
        tie_break_rule=TIE_BREAK_RULE,
        rows_fingerprint=rows_fp,
        params_fingerprint=cfg.params_fingerprint,
        cache_key=key,
    )
