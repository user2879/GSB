from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def public_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in report.items() if not str(key).startswith("_")}


def write_report_atomic(path: str | Path, report: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = public_report(report)
    fd, temp_name = tempfile.mkstemp(prefix=".report-", suffix=".json", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def empty_summary() -> Dict[str, Any]:
    return {
        "sessions": [],
        "combined_stable_prefix_text": "",
    }


def build_summary(
    report: Optional[Mapping[str, Any]],
    session_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    if not report:
        return empty_summary()
    wanted = None if session_ids is None else set(session_ids)
    sessions: List[Dict[str, Any]] = []
    prefix_texts: List[str] = []
    for item in report.get("sessions", []):
        if wanted is not None and item.get("session_id") not in wanted:
            continue
        stable = item.get("stable_prefix") or {}
        tail = item.get("variable_tail") or {}
        instruction = item.get("final_instruction") or {}
        slots = item.get("slots") or {}
        summary = {
            "session_id": item.get("session_id"),
            "status": item.get("status"),
            "stable_prefix_text": stable.get("text", ""),
            "stable_prefix_length": stable.get("length", 0),
            "variable_tail": {
                "start_frame": tail.get("start_frame"),
                "end_frame": tail.get("end_frame"),
                "frame_interval": tail.get("frame_interval", []),
                "length": tail.get("length", 0),
                "text": tail.get("text", ""),
            },
            "filled_slots": slots.get("filled", []),
            "affected_intervals": item.get("affected_intervals", []),
            "final_instruction_text": instruction.get("text", ""),
            "final_tokens": instruction.get("tokens", []),
            "strategy": item.get("strategy"),
            "lookahead_frames": item.get("lookahead_frames"),
            "frame_count": item.get("frame_count", 0),
            "failures": item.get("failures", []),
        }
        sessions.append(summary)
        if summary["stable_prefix_text"]:
            prefix_texts.append(summary["stable_prefix_text"])
    sessions.sort(key=lambda value: value.get("session_id") or "")
    return {
        "sessions": sessions,
        "combined_stable_prefix_text": " | ".join(prefix_texts),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return json_safe(value.as_dict())
    return value
