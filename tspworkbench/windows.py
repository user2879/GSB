"""回测窗口切分、输入校验与确定性缺失插补。

插补只使用当前窗口内“已见”观测，规则固定且不引入随机性；
每个窗口显式记录被插补的位置，供缓存指纹与诊断报告引用。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import WorkbenchError, E_TIMESTAMP_IRREGULAR, E_NON_FINITE, E_WINDOW_TOO_LONG
from .util import fingerprint


@dataclass
class Window:
    series_id: str
    window_index: int
    timestamps: list[int]
    raw_values: list[float | None]          # None 表示原始缺失
    values: list[float]                     # 插补后的等长序列
    missing_positions: list[int]
    imputed_positions: list[int]
    train_values: list[float]               # 前 window_length 个点
    actual: list[float]                     # 后 horizon 个点（真值）
    fingerprint: str

    def content(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "window_index": self.window_index,
            "timestamps": self.timestamps,
            "raw_values": self.raw_values,
            "missing_positions": self.missing_positions,
        }


def _impute(values: list[float | None], strategy: str) -> tuple[list[float], list[int]]:
    """按固定规则填补缺失，返回 (填补后序列, 被插补位置)。仅用窗口内已见值。"""
    n = len(values)
    out: list[float | None] = list(values)
    positions = [i for i, v in enumerate(out) if v is None]
    if not positions:
        return [float(v) for v in out], []

    if strategy == "forward_fill":
        last: float | None = None
        for i in range(n):
            if out[i] is None:
                out[i] = last if last is not None else 0.0
            else:
                last = out[i]
    elif strategy == "mean_fill":
        seen = [float(v) for v in out if v is not None]
        fill = float(np.mean(seen)) if seen else 0.0
        out = [fill if v is None else float(v) for v in out]
    elif strategy == "zero_fill":
        out = [0.0 if v is None else float(v) for v in out]
    else:  # 配置层已校验，这里防御
        raise WorkbenchError("E_MISSING_ENUM", f"未知缺失策略 {strategy}")
    return [float(v) for v in out], positions


def build_windows(rows: list[dict], cfg) -> list[Window]:
    """对单条序列做等间隔/非有限校验并切出回测窗口。"""
    series_id = rows[0]["series_id"]
    timestamps = [int(r["timestamp_ms"]) for r in rows]
    if len(timestamps) >= 2:
        deltas = {timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)}
        if any(d <= 0 for d in deltas) or len(deltas) != 1:
            raise WorkbenchError(E_TIMESTAMP_IRREGULAR,
                                 f"序列 {series_id} 时间戳非等间隔",
                                 context={"series_id": series_id, "deltas": sorted(deltas)})

    raw: list[float | None] = []
    missing_positions: list[int] = []
    for i, r in enumerate(rows):
        if r.get("missing") or r.get("value") is None:
            raw.append(None)
            missing_positions.append(i)
            continue
        value = float(r["value"])
        if not math.isfinite(value):
            raise WorkbenchError(E_NON_FINITE,
                                 f"序列 {series_id} 位置 {i} 观测值非有限",
                                 context={"series_id": series_id, "position": i, "value": r.get("value")})
        raw.append(value)

    total_needed = cfg.window_length + cfg.horizon
    if len(rows) < total_needed:
        raise WorkbenchError(E_WINDOW_TOO_LONG,
                             f"序列 {series_id} 可用长度 {len(rows)} 小于窗口+预测长度 {total_needed}",
                             context={"series_id": series_id, "available": len(rows),
                                      "required": total_needed})

    windows: list[Window] = []
    window_index = 0
    start = 0
    while start + total_needed <= len(rows):
        seg_ts = timestamps[start:start + total_needed]
        seg_raw = raw[start:start + total_needed]
        values, imputed = _impute(seg_raw, cfg.missing_strategy)
        win = Window(
            series_id=series_id,
            window_index=window_index,
            timestamps=seg_ts,
            raw_values=seg_raw,
            values=values,
            missing_positions=[p for p in missing_positions if start <= p < start + total_needed],
            imputed_positions=imputed,
            train_values=values[:cfg.window_length],
            actual=values[cfg.window_length:cfg.window_length + cfg.horizon],
            fingerprint="",
        )
        win.fingerprint = fingerprint(win.content())
        windows.append(win)
        window_index += 1
        start += cfg.window_step
    return windows
