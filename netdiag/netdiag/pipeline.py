"""端到端诊断流水线与会话状态管理。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import gaps as gap_mod
from . import merge as merge_mod
from . import snapshot as snapshot_mod
from . import statistics as stats_mod
from . import wire as wire_mod
from .constants import (ARBITRATION_WIRE_PREFERRED, DEFAULT_MIN_SAMPLES,
                        DEFAULT_WINDOW_MS)
from .eventlog import load_event_logs
from .ledger import (LedgerEntry, append_entry, build_input_summary,
                     read_entries, verify_entry)
from .logging_utils import StructuredLogger
from .model import ObservationModel
from .topology import Topology
from .wire import DecodedCapture, decode_file


@dataclass
class Session:
    model: ObservationModel
    topology: Topology
    conclusions: Dict[str, gap_mod.FlowGapConclusion]
    statistics: Dict[str, Any]
    params: Dict[str, int]
    input_summary: Dict[str, Any]
    captures: List[DecodedCapture] = field(default_factory=list)
    logger: Optional[StructuredLogger] = None

    def append_capture(self, path: str | Path) -> None:
        """恢复后追加新抓包样本：重新解码并整体重算模型与结论。"""
        capture = decode_file(path)
        self.captures.append(capture)
        combined = DecodedCapture()
        for item in self.captures:
            combined.frames.extend(item.frames)
        for frame in combined.frames:
            combined.flows.setdefault(frame.flow_key, []).append(frame)

        merged = merge_mod.merge_sources(
            combined, self._event_logs, rule=self.model.arbitration_rule,
            wire_name="appended-capture.bin")
        self.model = merged.model
        self.topology = Topology.default_for_flows(sorted(self.model.flows))
        self.conclusions = gap_mod.analyze_model(self.model)
        self.statistics = stats_mod.aggregate_session(
            self.model, self.conclusions,
            window_ms=self.params["window_ms"],
            min_samples=self.params["min_samples"])
        if self.logger:
            self.logger.info("capture_appended", path=str(path),
                             frames=len(capture.frames),
                             flows=len(self.model.flows))

    _event_logs: List[Any] = field(default_factory=list)


def run_diagnosis(*, wire_paths: List[str], event_paths: List[str],
                  window_ms: int = DEFAULT_WINDOW_MS,
                  min_samples: int = DEFAULT_MIN_SAMPLES,
                  arbitration_rule: str = ARBITRATION_WIRE_PREFERRED,
                  logger: Optional[StructuredLogger] = None,
                  topology: Optional[Topology] = None) -> Session:
    logger = logger or StructuredLogger(stream=None)
    logger.info("diagnosis_start", wire_inputs=wire_paths,
                event_inputs=event_paths, window_ms=window_ms,
                min_samples=min_samples, arbitration=arbitration_rule)

    captures = [decode_file(p) for p in wire_paths]
    combined = DecodedCapture()
    for capture in captures:
        combined.frames.extend(capture.frames)
    for frame in combined.frames:
        combined.flows.setdefault(frame.flow_key, []).append(frame)

    event_logs = load_event_logs(event_paths) if event_paths else []
    merged = merge_mod.merge_sources(
        combined, event_logs, rule=arbitration_rule,
        wire_name=";".join(wire_paths),
        event_names=[str(p) for p in event_paths])
    model = merged.model

    topo = topology or Topology.default_for_flows(sorted(model.flows))
    conclusions = gap_mod.analyze_model(model)
    statistics = stats_mod.aggregate_session(
        model, conclusions, window_ms=window_ms,
        min_samples=min_samples)

    input_summary = build_input_summary(wire_paths, event_paths)
    params = {"window_ms": window_ms, "min_samples": min_samples}
    session = Session(model=model, topology=topo, conclusions=conclusions,
                      statistics=statistics, params=params,
                      input_summary=input_summary, captures=captures,
                      logger=logger)
    session._event_logs = event_logs
    logger.info("diagnosis_complete", flows=len(model.flows),
                frames=len(combined.frames),
                arbitrations=merged.arbitration_total,
                event_versions=model.event_versions)
    return session


def persist_session(session: Session, snapshot_path: str | Path,
                    ledger_path: str | Path,
                    entry_id: Optional[str] = None) -> Dict[str, str]:
    path = snapshot_mod.save_snapshot(session, snapshot_path)
    entry = LedgerEntry(
        entry_id=entry_id or f"entry-{time.strftime('%Y%m%d%H%M%S')}",
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        input_summary=session.input_summary,
        version_bundle={
            "model_version": session.model.model_version,
            "event_versions": sorted(set(session.model.event_versions)),
            "arbitration_rule": session.model.arbitration_rule,
        },
        gap_conclusions={flow: c.to_dict()
                         for flow, c in session.conclusions.items()},
        rate_statistics=session.statistics,
        session_params=session.params,
        arbitration_rule=session.model.arbitration_rule,
    )
    append_entry(ledger_path, entry)
    if session.logger:
        session.logger.info("session_persisted", snapshot=str(path),
                            ledger=str(ledger_path), entry_id=entry.entry_id)
    return {"snapshot": str(path), "ledger": str(ledger_path),
            "entry_id": entry.entry_id}


def restore_session(snapshot_path: str | Path,
                    logger: Optional[StructuredLogger] = None
                    ) -> tuple[Session, snapshot_mod.RestoreReport]:
    """整体替换恢复：版本不一致时也以快照内容替换当前会话状态。"""
    logger = logger or StructuredLogger(stream=None)
    data = snapshot_mod.load_snapshot_raw(snapshot_path)
    report = snapshot_mod.compare_versions(data)
    logger.info("snapshot_restore", **report.to_dict())

    # 从快照重建最小可运行 Session（原始帧不在快照内，追加样本时重算）。
    # 快照只持久化仲裁规则版本，规则名从观测模型负载中恢复。
    stored_model = data.get("observation_model", {})
    model = ObservationModel(
        model_version=data["model_version"],
        arbitration_rule=stored_model.get(
            "arbitration_rule", ARBITRATION_WIRE_PREFERRED),
        event_versions=list(data.get("event_versions", [])))
    session = Session(
        model=model,
        topology=Topology.from_dict(data["topology"]),
        conclusions=snapshot_mod.restore_conclusions(data),
        statistics=data["rate_statistics"],
        params=data["session_params"],
        input_summary=data.get("input_summary", {}),
        logger=logger,
    )
    return session, report


def verify_ledger(ledger_path: str | Path, *, workspace: str | Path,
                  logger: Optional[StructuredLogger] = None) -> List[Dict[str, Any]]:
    logger = logger or StructuredLogger(stream=None)
    entries = read_entries(ledger_path)
    results = []
    for record in entries:
        results.append(verify_entry(record, workspace=workspace,
                                    logger=logger))
    return results
