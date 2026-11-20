from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .canonical import nearest_rank_percentile, read_jsonl
from .errors import ErrorCode, WorkbenchError, fail


def error_distribution(values: Iterable[float]) -> dict[str, Any]:
    ordered = [float(value) for value in values]
    if not ordered:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
        }
    return {
        "count": len(ordered),
        "min": min(ordered),
        "max": max(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": nearest_rank_percentile(ordered, 50),
        "p90": nearest_rank_percentile(ordered, 90),
        "p99": nearest_rank_percentile(ordered, 99),
    }


def _decision_map(frame_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {decision["block_id"]: decision for decision in frame_report.get("decisions", [])}


def find_first_divergence(
    left_path: str | Path,
    right_path: str | Path,
    raise_error: bool = False,
) -> dict[str, Any]:
    left_rows = read_jsonl(left_path)
    right_rows = read_jsonl(right_path)
    left_by_frame = {int(row["frame"]): row for row in left_rows}
    right_by_frame = {int(row["frame"]): row for row in right_rows}
    all_frames = sorted(set(left_by_frame) | set(right_by_frame))

    for frame in all_frames:
        if frame not in left_by_frame or frame not in right_by_frame:
            context = {
                "frame": frame,
                "reason": "frame_present_in_only_one_report",
                "left_present": frame in left_by_frame,
                "right_present": frame in right_by_frame,
                "left_input": left_by_frame.get(frame, {}).get("camera"),
                "right_input": right_by_frame.get(frame, {}).get("camera"),
            }
            if raise_error:
                fail(ErrorCode.E_TRACK_DIVERGE, "first frame-set divergence", **context)
            return {"diverged": True, "code": ErrorCode.E_TRACK_DIVERGE.value, **context}

        left_report = left_by_frame[frame]
        right_report = right_by_frame[frame]
        left_decisions = _decision_map(left_report)
        right_decisions = _decision_map(right_report)
        all_blocks = sorted(set(left_decisions) | set(right_decisions))
        for block_id in all_blocks:
            left_decision = left_decisions.get(block_id)
            right_decision = right_decisions.get(block_id)
            left_status = left_decision.get("status") if left_decision else "MISSING"
            right_status = right_decision.get("status") if right_decision else "MISSING"
            if left_status != right_status:
                context = {
                    "frame": int(frame),
                    "block_id": block_id,
                    "left_status": left_status,
                    "right_status": right_status,
                    "left_errors": left_decision.get("errors") if left_decision else None,
                    "right_errors": right_decision.get("errors") if right_decision else None,
                    "left_projection": left_decision.get("projection") if left_decision else None,
                    "right_projection": right_decision.get("projection") if right_decision else None,
                    "left_hiz_lookup": left_decision.get("hiz_lookup") if left_decision else None,
                    "right_hiz_lookup": right_decision.get("hiz_lookup") if right_decision else None,
                    "left_input_parameters": left_decision.get("input") if left_decision else None,
                    "right_input_parameters": right_decision.get("input") if right_decision else None,
                    "left_parameter_stamp": left_decision.get("parameter_stamp") if left_decision else None,
                    "right_parameter_stamp": right_decision.get("parameter_stamp") if right_decision else None,
                    "left_camera": left_report.get("camera"),
                    "right_camera": right_report.get("camera"),
                    "left_frame_parameter_stamp": left_report.get("parameter_stamp"),
                    "right_frame_parameter_stamp": right_report.get("parameter_stamp"),
                }
                if raise_error:
                    raise WorkbenchError(
                        ErrorCode.E_TRACK_DIVERGE,
                        "first per-block culling divergence",
                        context,
                    )
                return {"diverged": True, "code": ErrorCode.E_TRACK_DIVERGE.value, **context}

    return {
        "diverged": False,
        "frames_compared": len(all_frames),
        "blocks_compared_per_frame": {
            str(frame): len(_decision_map(left_by_frame[frame]))
            for frame in all_frames
        },
    }


def compare_or_raise(left_path: str | Path, right_path: str | Path) -> dict[str, Any]:
    return find_first_divergence(left_path, right_path, raise_error=True)
