"""速度与力度曲线的分段线性插值（确定性，基于 NumPy）。"""
from __future__ import annotations

import numpy as np

from .models import ExpressionParams


def tempo_at_ticks(params: ExpressionParams, ticks_per_beat: int, total_ticks: int) -> np.ndarray:
    """返回长度为 total_ticks 的数组，元素为每个 tick 的演奏速度 bpm。

    控制点先按拍位换算为 tick 坐标，控制点之间分段线性插值；
    首控制点之前取首段速度，末控制点之后取末段速度。
    """
    points = sorted(params.tempo_curve, key=lambda p: p.beat)
    ctrl_tick = np.array([p.beat * ticks_per_beat for p in points], dtype=np.float64)
    ctrl_bpm = np.array([p.bpm for p in points], dtype=np.float64)
    tick_index = np.arange(total_ticks, dtype=np.float64)
    return np.interp(tick_index, ctrl_tick, ctrl_bpm)  # 末端自动恒定


def velocity_map(params: ExpressionParams, notated_velocity: int) -> int:
    """力度映射曲线：分段线性后取整，钳制在 0..127。"""
    points = sorted(params.velocity_curve, key=lambda p: p.x)
    xs = np.array([p.x for p in points], dtype=np.float64)
    ys = np.array([p.y for p in points], dtype=np.float64)
    value = float(np.interp(float(notated_velocity), xs, ys))
    return int(max(0, min(127, np.rint(value))))


def pedal_track(params: ExpressionParams, ticks_per_beat: int, total_ticks: int) -> np.ndarray:
    """返回 0/1 整数数组，表示每个 tick 踏板是否踩下。"""
    track = np.zeros(total_ticks, dtype=np.int8)
    for segment in params.pedal_segments:
        start = int(round(segment.start_beat * ticks_per_beat))
        end = int(round(segment.end_beat * ticks_per_beat))
        lo = max(0, start)
        hi = min(total_ticks, end)
        if lo < hi:
            track[lo:hi] = 1
    return track
