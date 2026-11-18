"""两类来源到统一观测模型的合并与显式仲裁。

仲裁规则（版本化，见 constants.ARBITRATION_RULE_VERSION）：
- ``wire_preferred``（默认）：同一流两侧都出现时，时间窗与计数以线格式
  观测为准；事件清单原值整体留存在 overrides 中作为对照。
- ``event_preferred``：以事件清单为准，线格式派生值留存在 overrides 中。

序号覆盖区间冲突：当两侧都给出同一方向的覆盖区间且不完全相同（交集为空或
边界不一致）时以 E_MERGE 拒绝，并在上下文给出冲突区间。v1 事件不携带覆盖
信息，因此不参与覆盖冲突检查。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    ARBITRATION_EVENT_PREFERRED,
    ARBITRATION_RULE_VERSION,
    ARBITRATION_WIRE_PREFERRED,
    MODEL_VERSION,
    SOURCE_EVENT,
    SOURCE_WIRE,
)
from .errors import MergeConflictError
from .eventlog import NormalizedEventLog
from .model import FlowObservation, ObservationModel, OverrideRecord
from .wire import DecodedCapture


@dataclass
class MergeResult:
    model: ObservationModel
    arbitration_total: int = 0
    conflicts: List[Dict[str, Any]] = field(default_factory=list)


def _intervals_overlap(a: List[int], b: List[int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _check_coverage_conflicts(flow: str,
                              wire_coverage: Dict[str, Tuple[int, int]],
                              event_coverage: Dict[str, Dict[str, List[int]]]
                              ) -> None:
    for direction, event_interval in event_coverage.items():
        wire_interval = wire_coverage.get(direction)
        if wire_interval is None:
            # 方向无法对齐时退化为流级比较
            all_wire = list(wire_coverage.values())
            if not all_wire:
                continue
            if not any(_intervals_overlap(list(w), event_interval)
                       for w in all_wire):
                raise MergeConflictError(
                    f"流 {flow} 两类来源序号覆盖区间不相交",
                    {"flow": flow, "direction": direction,
                     "wire_intervals": [list(w) for w in all_wire],
                     "event_interval": event_interval,
                     "reason": "disjoint_coverage"})
            continue
        wire_list = list(wire_interval)
        if wire_list != list(event_interval):
            raise MergeConflictError(
                f"流 {flow} 方向 {direction} 序号覆盖区间不一致",
                {"flow": flow, "direction": direction,
                 "wire_interval": wire_list,
                 "event_interval": list(event_interval),
                 "reason": "coverage_boundary_mismatch"})


def merge_sources(capture: Optional[DecodedCapture],
                  event_logs: Optional[List[NormalizedEventLog]],
                  *,
                  rule: str = ARBITRATION_WIRE_PREFERRED,
                  wire_name: str = "capture.bin",
                  event_names: Optional[List[str]] = None
                  ) -> MergeResult:
    """把线格式抓包与事件清单归一合并为统一观测模型。"""
    if rule not in (ARBITRATION_WIRE_PREFERRED, ARBITRATION_EVENT_PREFERRED):
        raise ValueError(f"未知仲裁规则 {rule}")

    model = ObservationModel(model_version=MODEL_VERSION,
                             arbitration_rule=rule)
    result = MergeResult(model=model)
    event_names = event_names or []

    wire_coverage_all: Dict[str, Dict[str, Tuple[int, int]]] = {}
    if capture is not None:
        wire_coverage_all = capture.coverage()
        model.wire_inputs.append(wire_name)
        for frame in capture.frames:
            obs = model.flows.setdefault(frame.flow_key,
                                         FlowObservation(flow=frame.flow_key))
            obs.frames.append(frame)
            if SOURCE_WIRE not in obs.sources:
                obs.sources.append(SOURCE_WIRE)
        for flow, per_dir in wire_coverage_all.items():
            obs = model.flows.setdefault(flow, FlowObservation(flow=flow))
            obs.coverage = {d: [lo, hi] for d, (lo, hi) in per_dir.items()}

    event_names = event_names or []
    for log_idx, log in enumerate(event_logs or []):
        name = event_names[log_idx] if log_idx < len(event_names) else \
            f"events-{log_idx}.json"
        model.event_inputs.append(name)
        model.event_versions.append(log.version)
        for event in log.events:
            obs = model.flows.setdefault(event.flow,
                                         FlowObservation(flow=event.flow))
            if SOURCE_EVENT not in obs.sources:
                obs.sources.append(SOURCE_EVENT)
            obs.events.append(event)
            if obs.link_id is None and event.link_id:
                obs.link_id = event.link_id

        # 覆盖区间冲突检查（仅携带 coverage 的 v2/v3 清单）
        for flow, per_dir in log.coverage.items():
            if flow in wire_coverage_all:
                _check_coverage_conflicts(flow, wire_coverage_all[flow],
                                          per_dir)
            else:
                obs = model.flows.setdefault(flow, FlowObservation(flow=flow))
                for direction, interval in per_dir.items():
                    prev = obs.coverage.get(direction)
                    if prev is not None and prev != list(interval):
                        raise MergeConflictError(
                            f"流 {flow} 多份事件清单覆盖区间不一致",
                            {"flow": flow, "direction": direction,
                             "existing": prev, "incoming": list(interval),
                             "reason": "event_coverage_mismatch"})
                    obs.coverage[direction] = list(interval)

    # 仲裁：两侧都出现的流逐事件留存被覆盖原值
    for flow, obs in model.flows.items():
        both = SOURCE_WIRE in obs.sources and SOURCE_EVENT in obs.sources
        if not both:
            continue
        wire_window = _wire_window_ms(obs.frames)
        wire_loss = _wire_loss_count(obs.frames)
        for event in obs.events:
            event_window = (list(event.window_ms)
                            if event.window_ms else None)
            if rule == ARBITRATION_WIRE_PREFERRED:
                record = OverrideRecord(
                    event_id=event.event_id,
                    direction=event.direction,
                    rule=rule,
                    winner=SOURCE_WIRE,
                    loser=SOURCE_EVENT,
                    original_window_ms=event_window,
                    original_loss_count=event.loss_count,
                    adopted_window_ms=wire_window,
                    adopted_loss_count=wire_loss,
                )
            else:
                record = OverrideRecord(
                    event_id=event.event_id,
                    direction=event.direction,
                    rule=rule,
                    winner=SOURCE_EVENT,
                    loser=SOURCE_WIRE,
                    original_window_ms=wire_window,
                    original_loss_count=wire_loss,
                    adopted_window_ms=event_window,
                    adopted_loss_count=event.loss_count,
                )
            obs.overrides.append(record)
            obs.arbitration_count += 1
            result.arbitration_total += 1

    return result


def _wire_window_ms(frames: List[Any]) -> Optional[List[int]]:
    if not frames:
        return None
    start_us = min(f.timestamp_us for f in frames)
    end_us = max(f.timestamp_us for f in frames)
    return [start_us // 1000, end_us // 1000]


def _wire_loss_count(frames: List[Any]) -> int:
    """线格式侧的丢包计数：按方向统计缺失序号个数（重复帧不重复计）。"""
    seen: Dict[str, set] = {}
    minmax: Dict[str, Tuple[int, int]] = {}
    for frame in frames:
        bucket = seen.setdefault(frame.direction_key, set())
        bucket.add(frame.seq)
        lo, hi = minmax.get(frame.direction_key, (frame.seq, frame.seq))
        minmax[frame.direction_key] = (min(lo, frame.seq),
                                       max(hi, frame.seq))
    total = 0
    for direction, seqs in seen.items():
        lo, hi = minmax[direction]
        total += (hi - lo + 1) - len(seqs)
    return total
