"""Truth-referenced repair accuracy and smoothness evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    EVALUATION_TOLERANCES,
    FPS,
    SKELETON,
    POLICY_TEXT,
)
from .io import atomic_write_json, clean_float, sha256_json
from .math3d import quat_angle_deg, relative_rotvec
from .model import (
    DetectionResult,
    EvaluationDetail,
    EvaluationResult,
    Motion,
    RepairResult,
)


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def repaired_frame_intervals(detection: DetectionResult) -> list[tuple[int, int]]:
    frame_bad = detection.joint_bad.any(axis=1) | detection.root_bad
    indices = np.flatnonzero(frame_bad)
    if indices.size == 0:
        return []
    intervals: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = start
    for value in indices[1:]:
        current = int(value)
        if current != previous + 1:
            intervals.append((start, previous))
            start = current
        previous = current
    intervals.append((start, previous))
    return intervals


def _joint_acceleration_deg_s2(
    quats: np.ndarray, frame: int, joint: int, valid_frame: np.ndarray | None = None
) -> float | None:
    if frame <= 0 or frame >= quats.shape[0] - 1:
        return None
    if valid_frame is not None and not (valid_frame[frame - 1] and valid_frame[frame] and valid_frame[frame + 1]):
        return None
    first = relative_rotvec(quats[frame - 1, joint], quats[frame, joint])
    second = relative_rotvec(quats[frame, joint], quats[frame + 1, joint])
    return float(np.degrees(np.linalg.norm(second - first)) * (FPS**2))


def _detail_metrics(
    record: Motion,
    truth: Motion,
    detection: DetectionResult,
    repair: RepairResult,
    start: int,
    end: int,
) -> tuple[list[EvaluationDetail], list[dict[str, Any]], dict[str, Any]]:
    details: list[EvaluationDetail] = []
    acceleration: list[dict[str, Any]] = []
    angle_errors: list[float] = []
    raw_angle_errors: list[float] = []
    root_errors: list[float] = []
    raw_root_errors: list[float] = []
    affected_joints = sorted(
        {
            int(joint)
            for frame in range(start, end + 1)
            for joint in np.flatnonzero(detection.joint_bad[frame, :])
        }
    )

    for frame in range(start, end + 1):
        for joint in np.flatnonzero(detection.joint_bad[frame, :]):
            joint_index = int(joint)
            error = quat_angle_deg(repair.quats[frame, joint_index], truth.quats[frame, joint_index])
            raw_error = quat_angle_deg(record.quats[frame, joint_index], truth.quats[frame, joint_index])
            trusted = error <= EVALUATION_TOLERANCES["joint_angle_error_deg"]
            details.append(
                EvaluationDetail(
                    frame=frame,
                    joint=SKELETON[joint_index],
                    defect_type=str(detection.joint_type[frame, joint_index]),
                    joint_angle_error_deg=clean_float(error),
                    raw_joint_angle_error_deg=clean_float(raw_error),
                    root_error_m=None,
                    raw_root_error_m=None,
                    trusted=bool(trusted),
                    tolerance={
                        "joint_angle_error_deg": EVALUATION_TOLERANCES["joint_angle_error_deg"]
                    },
                )
            )
            angle_errors.append(error)
            raw_angle_errors.append(raw_error)
        if detection.root_bad[frame]:
            error = float(np.linalg.norm(repair.roots[frame] - truth.roots[frame]))
            raw_error = float(np.linalg.norm(record.roots[frame] - truth.roots[frame]))
            trusted = error <= EVALUATION_TOLERANCES["root_error_m"]
            details.append(
                EvaluationDetail(
                    frame=frame,
                    joint="Root",
                    defect_type=str(detection.root_type[frame]),
                    joint_angle_error_deg=None,
                    raw_joint_angle_error_deg=None,
                    root_error_m=clean_float(error),
                    raw_root_error_m=clean_float(raw_error),
                    trusted=bool(trusted),
                    tolerance={"root_error_m": EVALUATION_TOLERANCES["root_error_m"]},
                )
            )
            root_errors.append(error)
            raw_root_errors.append(raw_error)

    raw_valid = np.linalg.norm(record.quats, axis=2) > 1e-12
    per_joint_peaks: dict[str, dict[str, float | None]] = {}
    for joint in affected_joints:
        raw_values: list[float] = []
        repaired_values: list[float] = []
        truth_values: list[float] = []
        for frame in range(max(1, start - 1), min(record.frame_count - 1, end + 2)):
            raw_value = _joint_acceleration_deg_s2(record.quats, frame, joint, raw_valid[:, joint])
            repaired_value = _joint_acceleration_deg_s2(repair.quats, frame, joint)
            truth_value = _joint_acceleration_deg_s2(truth.quats, frame, joint)
            values = {
                "frame": frame,
                "joint": SKELETON[joint],
                "raw_deg_s2": None if raw_value is None else clean_float(raw_value),
                "repaired_deg_s2": None if repaired_value is None else clean_float(repaired_value),
                "truth_deg_s2": None if truth_value is None else clean_float(truth_value),
            }
            acceleration.append(values)
            if raw_value is not None and math.isfinite(raw_value):
                raw_values.append(raw_value)
            if repaired_value is not None and math.isfinite(repaired_value):
                repaired_values.append(repaired_value)
            if truth_value is not None and math.isfinite(truth_value):
                truth_values.append(truth_value)
        per_joint_peaks[SKELETON[joint]] = {
            "raw_peak_deg_s2": clean_float(max(raw_values)) if raw_values else None,
            "repaired_peak_deg_s2": clean_float(max(repaired_values)) if repaired_values else None,
            "truth_peak_deg_s2": clean_float(max(truth_values)) if truth_values else None,
        }

    raw_peak = max(
        (
            item["raw_peak_deg_s2"]
            for item in per_joint_peaks.values()
            if item["raw_peak_deg_s2"] is not None
        ),
        default=None,
    )
    repaired_peak = max(
        (
            item["repaired_peak_deg_s2"]
            for item in per_joint_peaks.values()
            if item["repaired_peak_deg_s2"] is not None
        ),
        default=None,
    )
    truth_peak = max(
        (
            item["truth_peak_deg_s2"]
            for item in per_joint_peaks.values()
            if item["truth_peak_deg_s2"] is not None
        ),
        default=None,
    )
    improvement = None
    if raw_peak is not None and raw_peak > 0.0 and repaired_peak is not None:
        improvement = clean_float((raw_peak - repaired_peak) / raw_peak)

    detail_bytes = [detail.to_dict() for detail in details]
    interval_metrics: dict[str, Any] = {
        "frames": [start, end],
        "frame_count": end - start + 1,
        "detail_count": len(details),
        "joints": [SKELETON[joint] for joint in affected_joints],
        "max_joint_angle_error_deg": clean_float(max(angle_errors)) if angle_errors else 0.0,
        "rms_joint_angle_error_deg": clean_float(_rms(angle_errors)),
        "max_raw_joint_angle_error_deg": clean_float(max(raw_angle_errors)) if raw_angle_errors else 0.0,
        "max_root_error_m": clean_float(max(root_errors)) if root_errors else 0.0,
        "rms_root_error_m": clean_float(_rms(root_errors)),
        "max_raw_root_error_m": clean_float(max(raw_root_errors)) if raw_root_errors else 0.0,
        "raw_acceleration_peak_deg_s2": raw_peak,
        "repaired_acceleration_peak_deg_s2": repaired_peak,
        "truth_acceleration_peak_deg_s2": truth_peak,
        "smoothness_improvement_ratio": improvement,
        "trusted": all(detail.trusted for detail in details),
        "per_joint_acceleration_peaks": per_joint_peaks,
        "metric_sha256": sha256_json(
            {"details": detail_bytes, "acceleration": acceleration, "peaks": per_joint_peaks}
        ),
    }
    return details, acceleration, interval_metrics


def evaluate_full(
    record: Motion,
    truth: Motion,
    detection: DetectionResult,
    repair: RepairResult,
) -> EvaluationResult:
    details: list[EvaluationDetail] = []
    intervals: dict[str, dict[str, Any]] = {}
    acceleration_details: dict[str, list[dict[str, Any]]] = {}
    for start, end in repaired_frame_intervals(detection):
        key = f"{start}-{end}"
        interval_details, acceleration, metrics = _detail_metrics(
            record, truth, detection, repair, start, end
        )
        details.extend(interval_details)
        intervals[key] = metrics
        acceleration_details[key] = acceleration
    details.sort(key=lambda detail: (detail.frame, detail.joint))
    summary = _summary(intervals, details)
    evaluation_hash = _evaluation_hash(details, intervals, acceleration_details, summary)
    return EvaluationResult(
        details=tuple(details),
        interval_metrics=intervals,
        acceleration_details=acceleration_details,
        summary=summary,
        evaluation_hash=evaluation_hash,
    )


def _summary(intervals: dict[str, dict[str, Any]], details: list[EvaluationDetail]) -> dict[str, Any]:
    all_angle_errors = [
        float(detail.joint_angle_error_deg)
        for detail in details
        if detail.joint_angle_error_deg is not None
    ]
    all_root_errors = [
        float(detail.root_error_m)
        for detail in details
        if detail.root_error_m is not None
    ]
    raw_peaks = [
        metrics["raw_acceleration_peak_deg_s2"]
        for metrics in intervals.values()
        if metrics["raw_acceleration_peak_deg_s2"] is not None
    ]
    repaired_peaks = [
        metrics["repaired_acceleration_peak_deg_s2"]
        for metrics in intervals.values()
        if metrics["repaired_acceleration_peak_deg_s2"] is not None
    ]
    truth_peaks = [
        metrics["truth_acceleration_peak_deg_s2"]
        for metrics in intervals.values()
        if metrics["truth_acceleration_peak_deg_s2"] is not None
    ]
    raw_peak = max(raw_peaks) if raw_peaks else None
    repaired_peak = max(repaired_peaks) if repaired_peaks else None
    truth_peak = max(truth_peaks) if truth_peaks else None
    improvement = None
    if raw_peak is not None and raw_peak > 0.0 and repaired_peak is not None:
        improvement = clean_float((raw_peak - repaired_peak) / raw_peak)
    trusted = all(detail.trusted for detail in details)
    return {
        "tolerances": {
            "joint_angle_error_deg": EVALUATION_TOLERANCES["joint_angle_error_deg"],
            "root_error_m": EVALUATION_TOLERANCES["root_error_m"],
        },
        "measurement_policy": {
            key: POLICY_TEXT[key] for key in ("endpoint_rule", "sign_rule", "root_rule", "acceleration_rule", "error_rule", "trust_rule")
        },
        "trusted": bool(trusted),
        "publication_status": "TRUSTED" if trusted else "WITHHELD_UNTRUSTED",
        "interval_count": len(intervals),
        "detail_count": len(details),
        "untrusted_detail_count": sum(1 for detail in details if not detail.trusted),
        "repaired_joint_coordinate_count": len(all_angle_errors),
        "repaired_root_coordinate_count": len(all_root_errors),
        "max_joint_angle_error_deg": clean_float(max(all_angle_errors)) if all_angle_errors else 0.0,
        "rms_joint_angle_error_deg": clean_float(_rms(all_angle_errors)),
        "max_root_error_m": clean_float(max(all_root_errors)) if all_root_errors else 0.0,
        "rms_root_error_m": clean_float(_rms(all_root_errors)),
        "raw_acceleration_peak_deg_s2": raw_peak,
        "repaired_acceleration_peak_deg_s2": repaired_peak,
        "truth_acceleration_peak_deg_s2": truth_peak,
        "smoothness_improvement_ratio": improvement,
    }


def _evaluation_hash(
    details: list[EvaluationDetail],
    intervals: dict[str, dict[str, Any]],
    acceleration_details: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
) -> str:
    return sha256_json(
        {
            "details": [detail.to_dict() for detail in details],
            "intervals": intervals,
            "acceleration": acceleration_details,
            "summary": summary,
        }
    )


def evaluate_incremental(
    record: Motion,
    truth: Motion,
    detection: DetectionResult,
    repair: RepairResult,
    base: EvaluationResult,
    dirty_frames: set[int],
) -> tuple[EvaluationResult, tuple[str, ...]]:
    new_ranges = repaired_frame_intervals(detection)
    affected_keys: set[str] = set()
    for start, end in new_ranges:
        key = f"{start}-{end}"
        # Acceleration for an interval reads one neighbouring frame on each side.
        if any(frame in dirty_frames for frame in range(max(0, start - 1), min(record.frame_count, end + 2))):
            affected_keys.add(key)
    for key in base.interval_metrics:
        start_text, end_text = key.split("-", 1)
        start, end = int(start_text), int(end_text)
        if any(frame in dirty_frames for frame in range(max(0, start - 1), min(record.frame_count, end + 2))):
            affected_keys.add(key)

    intervals: dict[str, dict[str, Any]] = {}
    acceleration_details: dict[str, list[dict[str, Any]]] = {}
    details: list[EvaluationDetail] = []
    for start, end in new_ranges:
        key = f"{start}-{end}"
        if key in affected_keys or key not in base.interval_metrics:
            interval_details, acceleration, metrics = _detail_metrics(
                record, truth, detection, repair, start, end
            )
            intervals[key] = metrics
            acceleration_details[key] = acceleration
        else:
            interval_details = []
            intervals[key] = base.interval_metrics[key]
            acceleration_details[key] = base.acceleration_details[key]
        details.extend(interval_details)
    # Details outside affected intervals are copied. Details inside affected
    # intervals were regenerated above.
    for detail in base.details:
        key = f"{detail.frame}-{detail.frame}"
        contained = any(
            start <= detail.frame <= end and f"{start}-{end}" not in affected_keys
            for start, end in new_ranges
        )
        if contained:
            details.append(detail)
    details.sort(key=lambda detail: (detail.frame, detail.joint))

    summary = _summary(intervals, details)
    evaluation_hash = _evaluation_hash(details, intervals, acceleration_details, summary)
    result = EvaluationResult(
        details=tuple(details),
        interval_metrics=intervals,
        acceleration_details=acceleration_details,
        summary=summary,
        evaluation_hash=evaluation_hash,
    )
    return result, tuple(sorted(affected_keys))


def evaluation_payload(result: EvaluationResult) -> dict[str, Any]:
    return {
        "schema": "mocap-evaluation/v1",
        "summary": result.summary,
        "policy": result.summary["measurement_policy"],
        "intervals": result.interval_metrics,
        "details": [detail.to_dict() for detail in result.details],
        "acceleration_details": result.acceleration_details,
        "evaluation_sha256": result.evaluation_hash,
    }


def write_evaluation(path: Path, result: EvaluationResult) -> str:
    return atomic_write_json(path, evaluation_payload(result))
