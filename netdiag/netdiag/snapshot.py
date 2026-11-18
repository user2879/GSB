"""排查会话快照：版本识别与整体替换恢复。

快照持久化观测模型版本、事件清单版本集合、仲裁规则版本、链路拓扑、
缺口结论与丢包率统计。重新打开时：
- 快照版本与当前会话版本一致 -> 原样恢复；
- 任一版本不一致 -> 整体替换当前观测模型与结论（绝不增量合并），
  并在恢复报告中记录 replaced=True 与版本差异。
恢复后可以继续追加新的抓包样本（见 Session.append_capture）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import ARBITRATION_RULE_VERSION, MODEL_VERSION
from .gaps import FlowGapConclusion, GapInterval
from .topology import Topology

SNAPSHOT_FORMAT = "netdiag-session-snapshot"


@dataclass
class RestoreReport:
    replaced: bool
    snapshot_model_version: str
    current_model_version: str
    snapshot_arbitration_version: str
    current_arbitration_version: str
    event_versions: List[str]
    differences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replaced": self.replaced,
            "snapshot_model_version": self.snapshot_model_version,
            "current_model_version": self.current_model_version,
            "snapshot_arbitration_version": self.snapshot_arbitration_version,
            "current_arbitration_version": self.current_arbitration_version,
            "event_versions": list(self.event_versions),
            "differences": list(self.differences),
        }


def build_snapshot_payload(session: Any) -> Dict[str, Any]:
    return {
        "format": SNAPSHOT_FORMAT,
        "model_version": MODEL_VERSION,
        "arbitration_rule_version": ARBITRATION_RULE_VERSION,
        "event_versions": sorted(set(session.model.event_versions)),
        "topology": session.topology.to_dict(),
        "observation_model": session.model.to_dict(),
        "gap_conclusions": {
            flow: conclusion.to_dict()
            for flow, conclusion in sorted(session.conclusions.items())},
        "rate_statistics": session.statistics,
        "session_params": session.params,
        "input_summary": session.input_summary,
    }


def save_snapshot(session: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_snapshot_payload(session),
                               indent=2, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8")
    return path


def load_snapshot_raw(path: str | Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != SNAPSHOT_FORMAT:
        raise ValueError(f"非法快照格式: {data.get('format')!r}")
    return data


def compare_versions(data: Dict[str, Any]) -> RestoreReport:
    snapshot_model = data.get("model_version", "?")
    snapshot_arb = data.get("arbitration_rule_version", "?")
    differences: List[str] = []
    if snapshot_model != MODEL_VERSION:
        differences.append(
            f"model_version: snapshot={snapshot_model} current={MODEL_VERSION}")
    if snapshot_arb != ARBITRATION_RULE_VERSION:
        differences.append(
            f"arbitration: snapshot={snapshot_arb} "
            f"current={ARBITRATION_RULE_VERSION}")
    return RestoreReport(
        replaced=bool(differences),
        snapshot_model_version=snapshot_model,
        current_model_version=MODEL_VERSION,
        snapshot_arbitration_version=snapshot_arb,
        current_arbitration_version=ARBITRATION_RULE_VERSION,
        event_versions=list(data.get("event_versions", [])),
        differences=differences,
    )


def restore_conclusions(data: Dict[str, Any]
                        ) -> Dict[str, FlowGapConclusion]:
    """把快照中的结论字典还原为内存对象（整体替换用）。"""
    from .constants import (CAUSE_OOO, CAUSE_PENDING, CAUSE_RETRANSMIT,
                            CAUSE_TRUE_LOSS)
    result: Dict[str, FlowGapConclusion] = {}
    for flow, payload in data.get("gap_conclusions", {}).items():
        conclusion = FlowGapConclusion(flow=flow)
        conclusion.coverage = {k: list(v) for k, v in
                               payload.get("coverage", {}).items()}
        conclusion.counts = dict(payload.get("counts", {}))
        for item in payload.get("gaps", []):
            conclusion.gaps.append(GapInterval(
                direction=item["direction"], start_seq=item["start_seq"],
                end_seq=item["end_seq"], cause=item["cause"],
                fill_event_ids=list(item.get("fill_event_ids", [])),
                evidence=dict(item.get("evidence", {}))))
        result[flow] = conclusion
    return result
