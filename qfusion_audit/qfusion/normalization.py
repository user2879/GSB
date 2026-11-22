"""通道内归一化（要求 2 的浮点前半段）。

- minmax: (x - min) / (max - min)，分母为 0 时全部置 0；
- zscore: (x - mean) / std，分母为 0 时全部置 0。

零分母按固定规则置零并产出 zero_denominator 事件，不抛异常。
结果全部落在 [-1, 1] 或 [0, 1]，便于后续定点量化。
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .config import ChannelSpec


def normalize_channel(records, spec: ChannelSpec
                      ) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    events: List[Dict[str, object]] = []
    scores = [r.raw_score for r in records]
    out: Dict[str, float] = {}

    if spec.normalization == "minmax":
        lo, hi = min(scores), max(scores)
        denom = hi - lo
        if denom == 0.0:
            events.append({"channel": spec.name,
                           "rule": "minmax_zero_range",
                           "action": "set_zero"})
            for r in records:
                out[r.item_id] = 0.0
            return out, events
        for r in records:
            out[r.item_id] = (r.raw_score - lo) / denom
        return out, events

    # zscore
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(var)
    if std == 0.0:
        events.append({"channel": spec.name,
                       "rule": "zscore_zero_std",
                       "action": "set_zero"})
        for r in records:
            out[r.item_id] = 0.0
        return out, events
    for r in records:
        z = (r.raw_score - mean) / std
        # 裁剪到 [-1, 1]，保证后续定点输入有界且确定。
        out[r.item_id] = max(-1.0, min(1.0, z))
    return out, events
