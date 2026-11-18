"""统一内部观测模型。

两类来源（线格式抓包、多版本事件清单）归一后都进入该模型，
诊断、统计、快照与账本均只依赖该模型而不直接接触原始格式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .eventlog import LossEvent
from .wire import Frame


@dataclass
class OverrideRecord:
    """一次仲裁中被覆盖一侧的原始值留存。"""

    event_id: Optional[int]
    direction: Optional[str]
    rule: str
    winner: str
    loser: str
    original_window_ms: Optional[List[int]]
    original_loss_count: int
    adopted_window_ms: Optional[List[int]]
    adopted_loss_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "direction": self.direction,
            "rule": self.rule,
            "winner": self.winner,
            "loser": self.loser,
            "original_window_ms": self.original_window_ms,
            "original_loss_count": self.original_loss_count,
            "adopted_window_ms": self.adopted_window_ms,
            "adopted_loss_count": self.adopted_loss_count,
        }


@dataclass
class FlowObservation:
    """单条流的归一观测。"""

    flow: str
    frames: List[Frame] = field(default_factory=list)
    events: List[LossEvent] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    arbitration_count: int = 0
    overrides: List[OverrideRecord] = field(default_factory=list)
    coverage: Dict[str, List[int]] = field(default_factory=dict)
    link_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow": self.flow,
            "sources": sorted(set(self.sources)),
            "frame_count": len(self.frames),
            "event_count": len(self.events),
            "arbitration_count": self.arbitration_count,
            "overrides": [o.to_dict() for o in self.overrides],
            "coverage": {k: list(v) for k, v in self.coverage.items()},
            "link_id": self.link_id,
            "events": [e.to_dict() for e in self.events],
        }


@dataclass
class ObservationModel:
    """整个排查会话的统一观测模型（版本化）。"""

    model_version: str
    arbitration_rule: str
    flows: Dict[str, FlowObservation] = field(default_factory=dict)
    event_versions: List[str] = field(default_factory=list)
    wire_inputs: List[str] = field(default_factory=list)
    event_inputs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "arbitration_rule": self.arbitration_rule,
            "event_versions": sorted(set(self.event_versions)),
            "wire_inputs": list(self.wire_inputs),
            "event_inputs": list(self.event_inputs),
            "flows": {k: v.to_dict() for k, v in sorted(self.flows.items())},
        }
