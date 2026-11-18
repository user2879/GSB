"""序号缺口分析与成因归类。

设计要点：
1. 顺序无关性——入口先做确定性全序排序
   ``(direction, seq, timestamp_us, frame_offset, flags)``，
   同一份抓包样本无论以何种遍历顺序输入，结论完全一致。
2. 三类成因：
   - ``true_loss``：缺口在观测窗口结束前始终未被任何帧填充；
   - ``retransmit_fill``：缺口被带 RETX 标志的重复序号帧填充；
   - ``ooo_late``：缺口被晚到的、无 RETX 标志的重复序号帧填充，
     且乱序深度与事件清单声明（v3）相容。
3. 填充语义互相矛盾（同时存在 RETX 与非 RETX 填充、或乱序深度与
   声明不符）时不强行归类，标记为 ``undetermined`` 待定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    CAUSE_OOO,
    CAUSE_PENDING,
    CAUSE_RETRANSMIT,
    CAUSE_TRUE_LOSS,
    FLAG_RETX,
)
from .eventlog import LossEvent
from .model import ObservationModel
from .wire import Frame


@dataclass
class GapInterval:
    direction: str
    start_seq: int
    end_seq: int
    cause: str
    fill_event_ids: List[int] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.end_seq - self.start_seq + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "start_seq": self.start_seq,
            "end_seq": self.end_seq,
            "width": self.width,
            "cause": self.cause,
            "fill_event_ids": list(self.fill_event_ids),
            "evidence": dict(self.evidence),
        }


@dataclass
class FlowGapConclusion:
    flow: str
    gaps: List[GapInterval] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    coverage: Dict[str, List[int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow": self.flow,
            "coverage": {k: list(v) for k, v in self.coverage.items()},
            "counts": dict(self.counts),
            "gaps": [g.to_dict() for g in self.gaps],
        }


def _canonical_frames(frames: List[Frame]) -> List[Frame]:
    """确定性全序：顺序无关性的唯一保证点。

    按“到达顺序”排序（时间戳、文件偏移、标志位），序号不参与排序键：
    乱序/重传的判定依赖观测到达次序，若按序号排序会把迟到帧提前，
    导致缺口永远不会被打开。文件偏移作为同时间戳下的稳定决胜键。
    """
    return sorted(frames, key=lambda f: (
        f.direction_key, f.timestamp_us, f.offset, f.flags, f.seq))


def _classify_fill(fill_frame: Frame, events: List[LossEvent],
                   max_seen: int) -> str:
    depth = max_seen - fill_frame.seq
    retransmit = bool(fill_frame.flags & FLAG_RETX)
    related = [e for e in events
               if e.direction in (fill_frame.direction_key, None)]

    if retransmit:
        denied = any(e.retransmitted is False for e in related)
        return CAUSE_PENDING if denied else CAUSE_RETRANSMIT

    declared_depths = [e.ooo_depth for e in related
                       if e.ooo_depth is not None]
    if declared_depths:
        # 声明深度描述该流观测到的最大乱序深度；单个迟到帧深度不超过
        # 声明深度即相容，不强行要求逐帧相等。
        if depth <= max(declared_depths):
            return CAUSE_OOO
        return CAUSE_PENDING
    return CAUSE_OOO


def analyze_direction(frames: List[Frame],
                      events: List[LossEvent]
                      ) -> Tuple[List[GapInterval], Tuple[int, int]]:
    if not frames:
        return [], (0, 0)

    ordered = _canonical_frames(frames)
    direction = ordered[0].direction_key
    min_seq = ordered[0].seq
    max_seen = min_seq - 1
    seen_seqs: set = set()

    # 逐缺失序号跟踪：seq -> 状态记录；输出阶段再按相邻同因合并为区间。
    missing: Dict[int, Dict[str, Any]] = {}

    for frame in ordered:
        seq = frame.seq
        if seq > max_seen + 1:
            for absent in range(max_seen + 1, seq):
                missing.setdefault(absent, {
                    "fills": [], "causes": set(), "depths": []})
        max_seen = max(max_seen, seq)

        is_duplicate = seq in seen_seqs
        if not is_duplicate:
            seen_seqs.add(seq)

        state = missing.get(seq)
        if state is None:
            continue  # 非缺口序号的普通首达帧

        filled = (
            is_duplicate
            or bool(frame.flags & FLAG_RETX)
            or max_seen > seq
        )
        if not filled:
            continue

        cause = _classify_fill(frame, events, max_seen)
        state["fills"].append(frame)
        state["causes"].add(cause)
        state["depths"].append(max_seen - seq)

    gaps: List[GapInterval] = []
    for seq in sorted(missing):
        state = missing[seq]
        if not state["fills"]:
            cause = CAUSE_TRUE_LOSS
            evidence: Dict[str, Any] = {
                "filled": False, "window_closed_without_fill": True}
            event_ids: List[int] = []
        else:
            causes = state["causes"]
            cause = next(iter(causes)) if len(causes) == 1 else CAUSE_PENDING
            evidence = {
                "filled": True,
                "fill_offsets": [f.offset for f in state["fills"]],
                "fill_seqs": [f.seq for f in state["fills"]],
                "retransmit_flags": [bool(f.flags & FLAG_RETX)
                                     for f in state["fills"]],
                "ooo_depths": list(state["depths"]),
            }
            event_ids = [e.event_id for e in events
                         if e.direction in (direction, None)]
            if cause == CAUSE_PENDING:
                evidence["reason"] = "conflicting_fill_semantics"

        # 与前一缺口相邻且成因相同则合并区间
        if gaps and gaps[-1].end_seq == seq - 1 and gaps[-1].cause == cause:
            target = gaps[-1]
            target.end_seq = seq
            target.evidence.setdefault("members", []).append(seq)
            target.fill_event_ids = sorted(
                set(target.fill_event_ids) | set(event_ids))
            if "fill_offsets" in evidence:
                target.evidence.setdefault("fill_offsets", []).extend(
                    evidence["fill_offsets"])
                target.evidence.setdefault("fill_seqs", []).extend(
                    evidence["fill_seqs"])
                target.evidence.setdefault("retransmit_flags", []).extend(
                    evidence["retransmit_flags"])
                target.evidence.setdefault("ooo_depths", []).extend(
                    evidence["ooo_depths"])
        else:
            base_evidence = dict(evidence)
            base_evidence["members"] = [seq]
            gaps.append(GapInterval(
                direction=direction, start_seq=seq, end_seq=seq,
                cause=cause, fill_event_ids=list(event_ids),
                evidence=base_evidence))

    coverage = (min(f.seq for f in ordered), max(f.seq for f in ordered))
    return gaps, coverage


def analyze_model(model: ObservationModel) -> Dict[str, FlowGapConclusion]:
    """对统一观测模型逐流做缺口分析，返回确定性结论集合。"""
    conclusions: Dict[str, FlowGapConclusion] = {}
    for flow in sorted(model.flows):
        obs = model.flows[flow]
        per_direction: Dict[str, List[Frame]] = {}
        for frame in obs.frames:
            per_direction.setdefault(frame.direction_key, []).append(frame)

        conclusion = FlowGapConclusion(flow=flow)
        for direction in sorted(per_direction):
            gaps, coverage = analyze_direction(per_direction[direction],
                                               obs.events)
            conclusion.gaps.extend(gaps)
            conclusion.coverage[direction] = [coverage[0], coverage[1]]

        counts = {CAUSE_TRUE_LOSS: 0, CAUSE_RETRANSMIT: 0,
                  CAUSE_OOO: 0, CAUSE_PENDING: 0}
        for gap in conclusion.gaps:
            counts[gap.cause] = counts.get(gap.cause, 0) + gap.width
        conclusion.counts = counts
        conclusions[flow] = conclusion
    return conclusions


def total_true_loss(conclusion: FlowGapConclusion) -> int:
    return conclusion.counts.get(CAUSE_TRUE_LOSS, 0)
