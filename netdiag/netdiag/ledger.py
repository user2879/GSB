"""诊断账本与历史结论重放校验。

账本为追加式 JSON Lines，每行持久化一次排查的输入摘要、版本组合、
缺口结论与统计结论。``verify`` 按账本记录重新解码、合并与归因，
与存档逐项比较；首个不一致定位到流标识、缺口区间与字段，
无法复现以 E_REPLAY 报明依据。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import ReplayMismatchError
from .logging_utils import StructuredLogger

LEDGER_FORMAT = "netdiag-ledger"


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class LedgerEntry:
    entry_id: str
    created: str
    input_summary: Dict[str, Any]
    version_bundle: Dict[str, Any]
    gap_conclusions: Dict[str, Any]
    rate_statistics: Dict[str, Any]
    session_params: Dict[str, Any]
    arbitration_rule: str

    def to_line(self) -> str:
        return json.dumps({
            "format": LEDGER_FORMAT,
            "entry_id": self.entry_id,
            "created": self.created,
            "input_summary": self.input_summary,
            "version_bundle": self.version_bundle,
            "gap_conclusions": self.gap_conclusions,
            "rate_statistics": self.rate_statistics,
            "session_params": self.session_params,
            "arbitration_rule": self.arbitration_rule,
        }, sort_keys=True, separators=(",", ":"))


def append_entry(path: str | Path, entry: LedgerEntry) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(entry.to_line() + "\n")


def read_entries(path: str | Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for lineno, line in enumerate(Path(path).read_text(
            encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("format") != LEDGER_FORMAT:
            raise ReplayMismatchError(
                f"账本第 {lineno} 行格式标识非法",
                {"reason": "bad_ledger_format", "line": lineno})
        entries.append(record)
    return entries


def build_input_summary(wire_paths: List[str],
                        event_paths: List[str]) -> Dict[str, Any]:
    return {
        "wire": [{"path": str(p), "sha256": file_digest(p)}
                 for p in wire_paths],
        "events": [{"path": str(p), "sha256": file_digest(p)}
                   for p in event_paths],
    }


def _first_gap_mismatch(expected: Dict[str, Any],
                        actual: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for flow in sorted(expected):
        exp_flow = expected[flow]
        act_flow = actual.get(flow)
        if act_flow is None:
            return {"flow": flow, "interval": None, "field": "flow",
                    "expected": exp_flow, "actual": None,
                    "reason": "flow_missing_on_replay"}
        exp_gaps = exp_flow.get("gaps", [])
        act_gaps = act_flow.get("gaps", [])
        if len(exp_gaps) != len(act_gaps):
            return {"flow": flow, "field": "gaps.length",
                    "expected": len(exp_gaps), "actual": len(act_gaps),
                    "reason": "gap_count_mismatch"}
        for exp_gap, act_gap in zip(exp_gaps, act_gaps):
            for field_name in ("direction", "start_seq", "end_seq", "cause"):
                if exp_gap.get(field_name) != act_gap.get(field_name):
                    return {
                        "flow": flow,
                        "interval": [exp_gap.get("start_seq"),
                                     exp_gap.get("end_seq")],
                        "field": field_name,
                        "expected": exp_gap.get(field_name),
                        "actual": act_gap.get(field_name),
                        "reason": "gap_field_mismatch"}
        for field_name in ("counts", "coverage"):
            if exp_flow.get(field_name) != act_flow.get(field_name):
                return {"flow": flow, "field": field_name,
                        "interval": None,
                        "expected": exp_flow.get(field_name),
                        "actual": act_flow.get(field_name),
                        "reason": "flow_field_mismatch"}
    return None


def verify_entry(record: Dict[str, Any], *, workspace: str | Path,
                 logger: Optional[StructuredLogger] = None) -> Dict[str, Any]:
    """按账本记录重新解码、合并、归因并逐项比较。"""
    from .pipeline import run_diagnosis

    workspace = Path(workspace)

    def resolve(item_path: str) -> Path:
        candidate = Path(item_path)
        if candidate.is_absolute():
            return candidate
        under_workspace = workspace / candidate
        return under_workspace if under_workspace.exists() else candidate

    wire_paths = [resolve(item["path"])
                  for item in record["input_summary"]["wire"]]
    event_paths = [resolve(item["path"])
                   for item in record["input_summary"]["events"]]

    missing = [str(p) for p in wire_paths + event_paths if not p.exists()]
    if missing:
        raise ReplayMismatchError(
            "账本输入无法复现：样本文件缺失",
            {"reason": "input_not_reproducible", "missing": missing})

    actual_digests = {str(p): file_digest(p)
                      for p in wire_paths + event_paths}
    for item in (record["input_summary"]["wire"] +
                 record["input_summary"]["events"]):
        stored_path = str(resolve(item["path"]))
        if actual_digests.get(stored_path) != item["sha256"]:
            raise ReplayMismatchError(
                "账本输入无法复现：摘要哈希不一致",
                {"reason": "digest_mismatch", "path": stored_path,
                 "expected_sha256": item["sha256"],
                 "actual_sha256": actual_digests.get(stored_path)})

    params = record["session_params"]
    session = run_diagnosis(
        wire_paths=[str(p) for p in wire_paths],
        event_paths=[str(p) for p in event_paths],
        window_ms=params["window_ms"],
        min_samples=params["min_samples"],
        arbitration_rule=record["arbitration_rule"],
        logger=logger,
    )

    actual_conclusions = {
        flow: conclusion.to_dict()
        for flow, conclusion in session.conclusions.items()}
    mismatch = _first_gap_mismatch(record["gap_conclusions"],
                                   actual_conclusions)
    if mismatch is not None:
        raise ReplayMismatchError(
            f"账本重放首个不一致位于流 {mismatch['flow']}", mismatch)

    stored_stats = record["rate_statistics"].get("flows", {})
    actual_stats = session.statistics.get("flows", {})
    if stored_stats != actual_stats:
        for flow in sorted(stored_stats):
            if stored_stats.get(flow) != actual_stats.get(flow):
                raise ReplayMismatchError(
                    f"账本重放统计不一致位于流 {flow}",
                    {"reason": "statistics_mismatch", "flow": flow,
                     "expected": stored_stats.get(flow),
                     "actual": actual_stats.get(flow)})
        raise ReplayMismatchError("账本重放统计集合不一致",
                                  {"reason": "statistics_mismatch"})

    result = {"entry_id": record["entry_id"], "verified": True,
              "flows": len(actual_conclusions)}
    if logger:
        logger.info("ledger_verify_ok", **result)
    return result
