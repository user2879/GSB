"""饱和截断与反量化归约。

截断阶段以量化位宽允许区间为界，饱和策略真实改变整数值；
恰等于边界的值保留在区间内并标注为边界样本。
反量化把整数乘以激活/权重缩放因子之积，再量化到 frac_bits
定点网格（三种确定性舍入），最后裁剪到输出区间。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .kernel import PositionResult, _twos_complement_wrap
from .models import OpDesc


@dataclass
class TruncationResult:
    position: int
    truncated: int
    boundary_sample: bool
    clamped: bool
    strategy: str

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "truncated": self.truncated,
            "boundary_sample": self.boundary_sample,
            "clamped": self.clamped,
            "strategy": self.strategy,
        }


@dataclass
class DequantResult:
    position: int
    scaled: float
    grid_index: int
    value: float
    clipped: bool
    rounding: str
    frac_bits: int
    scale: float
    step: float
    scale_source: str

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "scaled": self.scaled,
            "grid_index": self.grid_index,
            "value": self.value,
            "clipped": self.clipped,
            "rounding": self.rounding,
            "frac_bits": self.frac_bits,
            "scale": self.scale,
            "step": self.step,
            "scale_source": self.scale_source,
        }


def truncate(desc: OpDesc, position: PositionResult, *, strategy: str | None = None,
             qbits: int | None = None) -> TruncationResult:
    strategy = strategy or desc.saturation
    qbits = qbits or desc.qbits
    lo = -(1 << (qbits - 1))
    hi = (1 << (qbits - 1)) - 1
    value = position.buffered

    # 恰等于边界的值按固定规则归入区间；经饱和截断后落在边界上的值
    # 同样标注为边界样本（含输入恰为边界与越界饱和两种情形）。
    boundary_sample = value == lo or value == hi
    if value < lo or value > hi:
        clamped = True
        if strategy == "saturate":
            truncated = hi if value > hi else lo
        else:
            truncated = _twos_complement_wrap(value, qbits)
    else:
        clamped = False
        truncated = value
    if truncated == lo or truncated == hi:
        boundary_sample = True
    return TruncationResult(
        position=position.position,
        truncated=truncated,
        boundary_sample=boundary_sample,
        clamped=clamped,
        strategy=strategy,
    )


def _round_grid_index(index_float: float, rounding: str) -> int:
    if rounding == "trunc_zero":
        # 向零舍入
        return int(math.copysign(math.floor(abs(index_float)), index_float))
    if rounding == "floor":
        return math.floor(index_float)
    if rounding == "round_nearest_even":
        # 就近舍入、半数取偶（banker's rounding）
        return int(round(index_float))
    raise ValueError(f"未知舍入模式: {rounding}")


def dequantize(
    desc: OpDesc,
    trunc: TruncationResult,
    out_channel: int,
    *,
    rounding: str | None = None,
    frac_bits: int | None = None,
    scale_override: float | None = None,
) -> DequantResult:
    rounding = rounding or desc.rounding
    frac_bits = frac_bits or desc.frac_bits
    if scale_override is None:
        scale = desc.effective_scale(out_channel)
        source = f"act_scale({desc.scale_sources.get('act_scale')}) * " \
                 f"weight_scales[{out_channel}]({desc.scale_sources.get('weight_scales')})"
    else:
        scale = float(scale_override)
        source = "injection_override"
    step = math.ldexp(1.0, -frac_bits)

    scaled = trunc.truncated * scale
    index_float = scaled / step
    grid_index = _round_grid_index(index_float, rounding)

    low_index = math.ceil(desc.out_low / step)
    high_index = math.floor(desc.out_high / step)
    clipped = grid_index < low_index or grid_index > high_index
    clipped_index = min(max(grid_index, low_index), high_index)
    value = clipped_index * step
    return DequantResult(
        position=trunc.position,
        scaled=scaled,
        grid_index=grid_index,
        value=value,
        clipped=clipped,
        rounding=rounding,
        frac_bits=frac_bits,
        scale=scale,
        step=step,
        scale_source=source,
    )
