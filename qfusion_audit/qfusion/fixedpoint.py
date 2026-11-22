"""定点量化原语。

定点约定：整数 ``q`` 表示实值 ``q * 2^-f``，其中 ``f`` 为小数位宽。
量化档位 ``g``（gear，正整数）在最小步长 ``2^-f`` 之上再放大，
真实量化步长 ``step = g * 2^-f``：

    q = round_half_up(x / step)
    residual = x - q * step          # 带符号残差，|residual| <= step/2
    direction = up / down / exact     由残差符号决定

档位越大步长越粗、整数结果与残差必然随之改变（要求 2）。
所有运算使用纯 Python ``int``，保证跨平台、跨次运行逐位确定。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict


STAGE_NORM = "normalize"
STAGE_FUSION = "channel_fusion"
STAGE_FEATURE = "feature_overlay"
STAGE_CLIP = "clip"
STAGE_ORDER = (STAGE_NORM, STAGE_FUSION, STAGE_FEATURE, STAGE_CLIP)


def round_half_up(num: float, den: float) -> int:
    """对 num/den 做远离零方向的半值舍入（half-up），den > 0。"""
    z = num / den
    if z < 0:
        return -round_half_up(-num, den)
    r = math.floor(z)
    if z - r >= 0.5:
        return r + 1
    return r


def quantization_step(frac_bits: int, gear: int) -> float:
    return gear * (2.0 ** (-frac_bits))


def residual_direction(residual: float) -> str:
    if residual > 0.0:
        return "up"
    if residual < 0.0:
        return "down"
    return "exact"


@dataclass
class Quantized:
    frac_bits: int
    gear: int
    step: float
    q: int
    residual: float
    direction: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "frac_bits": self.frac_bits,
            "gear": self.gear,
            "step": self.step,
            "q": self.q,
            "residual": self.residual,
            "direction": self.direction,
        }


def quantize(value: float, frac_bits: int, gear: int) -> Quantized:
    step = quantization_step(frac_bits, gear)
    q = round_half_up(value, step)
    residual = value - q * step
    # 消除浮点末尾抖动，使 exact 在可判定时稳定。
    if abs(residual) < step * 1e-12:
        residual = 0.0
    return Quantized(frac_bits, gear, step, q, residual,
                     residual_direction(residual))


def shift_right_half_up(value: int, bits: int) -> int:
    """整数域右移 bits 位（half-up，对负数同样远离零）。bits >= 0。"""
    if bits <= 0:
        return value
    half = 1 << (bits - 1)
    if value >= 0:
        return (value + half) >> bits
    return -((-value + half) >> bits)


def clip_int(value: int, low: int, high: int) -> int:
    if value < low:
        return low
    if value > high:
        return high
    return value


def stable_key(request_id: str, item_id: str, stage: str) -> str:
    """任务稳定排序键：由请求 ID、物品 ID、阶段名派生。"""
    return f"{request_id}\u0000{item_id}\u0000{stage}"
