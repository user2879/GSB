"""按整数毫秒时间窗聚合每流丢包率（NumPy 实现）。

定点约定（constants.RATE_SCALE = 1_000_000，单位 BPS，1 BPS = 1e-6）：
- 比率 rate_bps = floor(scale * lost / expected)，舍入方向恒为向下取整；
- 单窗量化误差上界 eps = scale / expected（BPS），真实比率落在
  [rate_bps, rate_bps + eps)；
- 两流比较的合成误差界为两侧 eps 之和；差值落在界内时按稳定决胜规则
  （rate_bps 降序、flow 字典序升序）排序并标记 error_bound_decisive。
窗口内期望样本数 < min_samples 时标注 insufficient_samples。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .constants import RATE_ROUNDING, RATE_SCALE, RATE_UNIT
from .gaps import FlowGapConclusion
from .model import ObservationModel

ROUND_DOWN = "floor"


@dataclass
class WindowStat:
    flow: str
    window_start_ms: int
    window_end_ms: int
    expected: int
    received: int
    lost: int
    rate_bps: int
    error_bound_bps: int
    rounding: str
    insufficient_samples: bool
    error_bound_decisive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow": self.flow,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "expected": self.expected,
            "received": self.received,
            "lost": self.lost,
            "rate_bps": self.rate_bps,
            "rate_unit": RATE_UNIT,
            "error_bound_bps": self.error_bound_bps,
            "rounding": self.rounding,
            "insufficient_samples": self.insufficient_samples,
            "error_bound_decisive": self.error_bound_decisive,
        }


@dataclass
class FlowStats:
    flow: str
    windows: List[WindowStat] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"flow": self.flow,
                "windows": [w.to_dict() for w in self.windows]}


def _loss_set(conclusion: FlowGapConclusion) -> set:
    lost: set = set()
    for gap in conclusion.gaps:
        if gap.cause == "true_loss":
            lost.update(range(gap.start_seq, gap.end_seq + 1))
    return lost


def aggregate_flow(flow: str, model: ObservationModel,
                   conclusion: FlowGapConclusion, window_ms: int,
                   min_samples: int) -> FlowStats:
    """按整数毫秒网格聚合单流丢包率。

    每个窗口的期望序号集合 = 窗内首达帧序号并集 + 落入该窗时间范围的
    真实丢包序号；接收集合只含首达帧。重复/重传/迟到帧不增加接收计数。
    比率以 floor 定点表示，误差上界为 scale/expected（向上取整）。
    """
    if window_ms <= 0:
        raise ValueError("window_ms 必须为正整数")
    obs = model.flows[flow]
    frames = sorted(obs.frames, key=lambda f: (f.direction_key, f.timestamp_us,
                                               f.seq, f.offset))
    stats = FlowStats(flow=flow)
    if not frames:
        return stats

    timestamps_ms = np.array([f.timestamp_us // 1000 for f in frames],
                             dtype=np.int64)
    base = int(timestamps_ms.min())
    origin = base - (base % window_ms)

    def win_of(frame) -> int:
        return ((frame.timestamp_us // 1000) - origin) // window_ms

    # 仅保留首达帧
    first_frames: Dict[Tuple[str, int], Any] = {}
    for frame in frames:
        first_frames.setdefault((frame.direction_key, frame.seq), frame)
    ordered_first = sorted(first_frames.values(),
                           key=lambda f: (f.direction_key, f.timestamp_us,
                                          f.seq, f.offset))

    win_received: Dict[int, set] = {}
    win_direction_span: Dict[int, Dict[str, Tuple[int, int]]] = {}
    for frame in ordered_first:
        win = win_of(frame)
        win_received.setdefault(win, set()).add(frame.seq)
        span = win_direction_span.setdefault(win, {})
        prev = span.get(frame.direction_key)
        span[frame.direction_key] = ((frame.seq, frame.seq) if prev is None
                                     else (min(prev[0], frame.seq),
                                           max(prev[1], frame.seq)))

    # 真实丢包序号归入观测覆盖方向的最后一个窗口（闭窗仍未到达）
    true_loss_seqs = sorted(_loss_set(conclusion))
    last_win = max(win_received) if win_received else 0
    extra_loss_by_win: Dict[int, set] = {}
    if true_loss_seqs:
        extra_loss_by_win[last_win] = set(true_loss_seqs)

    for win in sorted(set(win_received) | set(extra_loss_by_win)):
        start_ms = origin + win * window_ms
        end_ms = start_ms + window_ms - 1
        expected_space: set = set()
        for lo, hi in win_direction_span.get(win, {}).values():
            expected_space.update(range(lo, hi + 1))
        expected_space |= extra_loss_by_win.get(win, set())
        received_seqs = win_received.get(win, set()) & expected_space
        expected = len(expected_space)
        received = len(received_seqs)
        lost = expected - received

        if expected > 0:
            rate_bps = (RATE_SCALE * lost) // expected
            error_bound = (RATE_SCALE + expected - 1) // expected
        else:
            rate_bps = 0
            error_bound = 0
        stats.windows.append(WindowStat(
            flow=flow,
            window_start_ms=start_ms,
            window_end_ms=end_ms,
            expected=expected,
            received=received,
            lost=lost,
            rate_bps=rate_bps,
            error_bound_bps=error_bound,
            rounding=ROUND_DOWN,
            insufficient_samples=expected < min_samples,
        ))
    return stats


def _stable_sort_key(stat: WindowStat):
    return (-stat.rate_bps, stat.flow)


def rank_window(stats_by_flow: Dict[str, FlowStats], window_start_ms: int
                ) -> List[WindowStat]:
    """同一时间窗跨流排序，应用误差界稳定决胜规则。"""
    window_stats = [next((w for w in fs.windows
                          if w.window_start_ms == window_start_ms), None)
                    for fs in stats_by_flow.values()]
    rows = [w for w in window_stats if w is not None]
    rows.sort(key=_stable_sort_key)

    # 标记“受误差界影响”的相邻判定
    for idx, row in enumerate(rows):
        if idx == 0:
            continue
        prev = rows[idx - 1]
        diff = abs(prev.rate_bps - row.rate_bps)
        combined_bound = prev.error_bound_bps + row.error_bound_bps
        if diff <= combined_bound:
            row.error_bound_decisive = True
            prev.error_bound_decisive = True
    return rows


def aggregate_session(model: ObservationModel,
                      conclusions: Dict[str, FlowGapConclusion], *,
                      window_ms: int = 100,
                      min_samples: int = 4) -> Dict[str, Any]:
    flows: Dict[str, FlowStats] = {}
    for flow in sorted(model.flows):
        flows[flow] = aggregate_flow(flow, model, conclusions[flow],
                                     window_ms, min_samples)

    all_starts = sorted({w.window_start_ms for fs in flows.values()
                         for w in fs.windows})
    ranking = []
    for start in all_starts:
        ranking.append({
            "window_start_ms": start,
            "order": [w.to_dict() for w in rank_window(flows, start)],
        })

    return {
        "window_ms": window_ms,
        "min_samples": min_samples,
        "rate_scale": RATE_SCALE,
        "rate_unit": RATE_UNIT,
        "rounding": ROUND_DOWN,
        "declared_rounding_note": RATE_ROUNDING,
        "flows": {k: v.to_dict() for k, v in sorted(flows.items())},
        "ranking": ranking,
    }
