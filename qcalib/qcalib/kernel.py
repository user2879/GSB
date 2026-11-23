"""逐列组定点重缩放与饱和截断内核。

为保证任意平台逐字段一致，定点中间量全部使用 Python 整数运算
（不依赖 NumPy 浮点累加的平台差异）。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from .calibration import ColumnCalib
from .config import GroupConfig
from .calibration import _clip_bounds
from .errors import E_ACC_OVERFLOW, FailureCollector
from .fingerprint import cache_key, content_fingerprint


def _round_int(value: float, mode: str) -> int:
    if mode == "truncate":
        return int(value)  # 向零
    if mode == "floor":
        import math

        return math.floor(value)
    if mode == "half_even":
        import decimal

        return int(decimal.Decimal(repr(value)).quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_EVEN))
    # half_up：四舍五入（远离零的半数）
    import math

    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


@dataclass
class ElementResult:
    row_id: str
    original: Optional[float]
    raw_int: Optional[int]      # 截断前定点值（未裁剪）
    quantized: Optional[int]    # 最终定点整数
    dequantized: Optional[float]
    direction: str              # "none" / "low" / "high" / "missing" / "reject"
    saturated: bool
    missing: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ColumnRescale:
    group_id: str
    column: str
    scale: float
    zero_point: int
    qbits: int
    fbits: int
    saturation: str
    elements: List[ElementResult]
    n_saturated: int
    n_missing: int
    accumulator: int
    accumulator_overflow: bool
    overflow_positions: List[Dict[str, Any]] = field(default_factory=list)
    cache_key: str = ""
    stage: str = "rescale"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


def rescale_column(
    cfg: GroupConfig,
    calib: ColumnCalib,
    vector: List[Optional[float]],
    row_ids: List[str],
    accumulator_bits: int = 32,
) -> tuple[ColumnRescale, FailureCollector]:
    collector = FailureCollector(source=f"rescale:{cfg.group_id}:{calib.column}")
    # 标准补码定点：q = round(x / scale) + zp，反量化 x = (q - zp) * scale。
    # 裁剪区间随 fbits 收缩；scale(LSB) 在标定阶段确定。
    lo, hi = _clip_bounds(cfg.qbits, cfg.fbits, cfg.scheme)
    scale = calib.scale
    if scale <= 0:
        scale = 1e-12

    elements: List[ElementResult] = []
    n_saturated = 0
    n_missing = 0
    # Python 整数累加器，再按位宽判越界
    accumulator = 0
    acc_lo = -(1 << (accumulator_bits - 1))
    acc_hi = (1 << (accumulator_bits - 1)) - 1
    overflow_positions: List[Dict[str, Any]] = []

    for position, (row_id, original) in enumerate(zip(row_ids, vector)):
        if original is None:
            n_missing += 1
            elements.append(
                ElementResult(row_id, None, None, None, None, "missing", False, True)
            )
            continue
        raw = _round_int(original / scale, cfg.rounding) + calib.zero_point
        if raw < lo:
            direction = "low"
        elif raw > hi:
            direction = "high"
        else:
            direction = "none"

        if direction != "none":
            n_saturated += 1
            if cfg.saturation == "clip":
                quantized = min(max(raw, lo), hi)
            else:  # reject：该元素不参与累加，记录后跳过
                elements.append(
                    ElementResult(row_id, original, raw, None, None, "reject", True, False)
                )
                continue
        else:
            quantized = raw

        accumulator += quantized
        if not (acc_lo <= accumulator <= acc_hi):
            collector.add(
                E_ACC_OVERFLOW,
                f"累加越界: group={cfg.group_id} column={calib.column} pos={position}",
                location=row_id,
                group_id=cfg.group_id,
                column=calib.column,
                position=position,
                accumulator=accumulator,
                accumulator_bits=accumulator_bits,
            )
            overflow_positions.append(
                {"row_id": row_id, "position": position, "accumulator": accumulator}
            )
        dequant = (quantized - calib.zero_point) * scale
        elements.append(
            ElementResult(row_id, original, raw, quantized, dequant, direction, direction != "none", False)
        )

    upstream_fp = content_fingerprint(
        {
            "rows": row_ids,
            "calib_key": calib.cache_key,
            "fbits": cfg.fbits,
            "qbits": cfg.qbits,
            "saturation": cfg.saturation,
            "rounding": cfg.rounding,
        }
    )
    key = cache_key(
        "rescale",
        [upstream_fp, calib.cache_key],
        fbits=cfg.fbits,
        qbits=cfg.qbits,
        exchange=0,
    )
    result = ColumnRescale(
        group_id=cfg.group_id,
        column=calib.column,
        scale=calib.scale,
        zero_point=calib.zero_point,
        qbits=cfg.qbits,
        fbits=cfg.fbits,
        saturation=cfg.saturation,
        elements=elements,
        n_saturated=n_saturated,
        n_missing=n_missing,
        accumulator=accumulator,
        accumulator_overflow=bool(overflow_positions),
        overflow_positions=overflow_positions,
        cache_key=key,
    )
    return result, collector
