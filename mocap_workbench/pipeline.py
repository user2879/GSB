"""Snapshot construction, dependency closure and whole-array consistency checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import EVALUATION_TOLERANCES, FPS, SKELETON
from .detection import detect
from .errors import BusinessError, ErrorCode
from .evaluation import evaluate_full, evaluate_incremental, repaired_frame_intervals
from .io import clean_float, sha256_json
from .model import (
    ClosureReport,
    ConsistencyReport,
    DefectEntry,
    DetectionResult,
    EvaluationResult,
    Motion,
    RepairResult,
    Snapshot,
    Thresholds,
    VersionStamp,
)
from .repair import (
    _anchor_pair,
    build_anchors,
    combined_joint_anchors,
    combined_root_anchors,
    repair_dirty,
    repair_full,
    validate_keyframes,
    validate_suppressed_auto,
)


def keyframe_hash(
    keyframes: frozenset[int], suppressed_auto: frozenset[int] = frozenset()
) -> str:
    return sha256_json(
        {
            "user_keyframes": sorted(int(frame) for frame in keyframes),
            "suppressed_auto": sorted(int(frame) for frame in suppressed_auto),
        }
    )


def param_version(
    thresholds: Thresholds,
    keyframes: frozenset[int],
    suppressed_auto: frozenset[int] = frozenset(),
) -> str:
    return sha256_json(
        {
            "thresholds": thresholds.to_dict(),
            "keyframes": sorted(keyframes),
            "suppressed_auto": sorted(suppressed_auto),
        }
    )[:16]


def _version(
    sequence: int,
    record: Motion,
    thresholds: Thresholds,
    keyframes: frozenset[int],
    suppressed_auto: frozenset[int],
    detection: DetectionResult,
    repair: RepairResult,
    evaluation: EvaluationResult,
) -> VersionStamp:
    threshold_hash = thresholds.hash()
    key_hash = keyframe_hash(keyframes, suppressed_auto)
    parameter_version = param_version(thresholds, keyframes, suppressed_auto)
    content_hash = sha256_json(
        {
            "record_hash": record.content_sha256,
            "threshold_hash": threshold_hash,
            "keyframe_hash": key_hash,
            "detection_hash": detection.detection_hash,
            "repair_hash": repair.repair_hash,
            "evaluation_hash": evaluation.evaluation_hash,
            "param_version": parameter_version,
        }
    )
    return VersionStamp(
        sequence=sequence,
        param_version=parameter_version,
        record_hash=record.content_sha256,
        threshold_hash=threshold_hash,
        keyframe_hash=key_hash,
        detection_hash=detection.detection_hash,
        repair_hash=repair.repair_hash,
        evaluation_hash=evaluation.evaluation_hash,
        content_hash=content_hash,
    )


def build_full_snapshot(
    record: Motion,
    truth: Motion,
    truth_entries: tuple[DefectEntry, ...],
    thresholds: Thresholds,
    keyframes: frozenset[int] | set[int],
    sequence: int,
    committed: bool = False,
    suppressed_auto: frozenset[int] | set[int] = frozenset(),
) -> Snapshot:
    keyframe_set = frozenset(int(frame) for frame in keyframes)
    suppressed_set = validate_suppressed_auto(suppressed_auto, record)
    detection = detect(record, truth_entries, thresholds)
    keyframe_set = validate_keyframes(keyframe_set, record, detection)
    anchors, repair = repair_full(record, detection, keyframe_set, suppressed_set)
    evaluation = evaluate_full(record, truth, detection, repair)
    version = _version(
        sequence,
        record,
        thresholds,
        keyframe_set,
        suppressed_set,
        detection,
        repair,
        evaluation,
    )
    return Snapshot(
        record=record,
        truth=truth,
        thresholds=thresholds,
        keyframes=keyframe_set,
        evidence=detection_evidence_ref(record),
        detection=detection,
        anchors=anchors,
        repair=repair,
        evaluation=evaluation,
        version=version,
        suppressed_auto=suppressed_set,
        committed=committed,
    )


def detection_evidence_ref(record: Motion):
    from .detection import compute_evidence

    return compute_evidence(record)


def build_with_evidence(
    record: Motion,
    truth: Motion,
    truth_entries: tuple[DefectEntry, ...],
    evidence,
    thresholds: Thresholds,
    keyframes: frozenset[int] | set[int],
    sequence: int,
    committed: bool = False,
    suppressed_auto: frozenset[int] | set[int] = frozenset(),
) -> Snapshot:
    keyframe_set = frozenset(int(frame) for frame in keyframes)
    suppressed_set = validate_suppressed_auto(suppressed_auto, record)
    detection = detect(record, truth_entries, thresholds, evidence=evidence)
    keyframe_set = validate_keyframes(keyframe_set, record, detection)
    anchors, repair = repair_full(record, detection, keyframe_set, suppressed_set)
    evaluation = evaluate_full(record, truth, detection, repair)
    version = _version(
        sequence,
        record,
        thresholds,
        keyframe_set,
        suppressed_set,
        detection,
        repair,
        evaluation,
    )
    return Snapshot(
        record=record,
        truth=truth,
        thresholds=thresholds,
        keyframes=keyframe_set,
        evidence=evidence,
        detection=detection,
        anchors=anchors,
        repair=repair,
        evaluation=evaluation,
        version=version,
        suppressed_auto=suppressed_set,
        committed=committed,
    )


def _group_frames(frames: set[int]) -> list[tuple[int, int]]:
    if not frames:
        return []
    ordered = sorted(frames)
    result: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame != previous + 1:
            result.append((start, previous))
            start = frame
        previous = frame
    result.append((start, previous))
    return result


def _dirty_coordinates(
    old: Snapshot,
    new_detection: DetectionResult,
    new_anchors,
) -> tuple[list[tuple[int, int]], list[int]]:
    dirty_joints: set[tuple[int, int]] = set()
    old_detection = old.detection
    for joint in range(old.record.joint_count):
        old_combined = combined_joint_anchors(old.anchors, joint)
        new_combined = combined_joint_anchors(new_anchors, joint)
        candidates = set(np.flatnonzero(old_detection.joint_bad[:, joint] | new_detection.joint_bad[:, joint]).tolist())
        for frame in sorted(candidates):
            old_bad = bool(old_detection.joint_bad[frame, joint])
            new_bad = bool(new_detection.joint_bad[frame, joint])
            old_type = str(old_detection.joint_type[frame, joint]) if old_bad else ""
            new_type = str(new_detection.joint_type[frame, joint]) if new_bad else ""
            old_pair = _anchor_pair(frame, old_combined)
            new_pair = _anchor_pair(frame, new_combined)
            if old_bad != new_bad or old_type != new_type or old_pair != new_pair:
                dirty_joints.add((frame, joint))

    dirty_roots: set[int] = set()
    old_root_combined = combined_root_anchors(old.anchors)
    new_root_combined = combined_root_anchors(new_anchors)
    root_candidates = set(np.flatnonzero(old_detection.root_bad | new_detection.root_bad).tolist())
    for frame in sorted(root_candidates):
        old_bad = bool(old_detection.root_bad[frame])
        new_bad = bool(new_detection.root_bad[frame])
        old_pair = _anchor_pair(frame, old_root_combined)
        new_pair = _anchor_pair(frame, new_root_combined)
        if old_bad != new_bad or old_pair != new_pair:
            dirty_roots.add(frame)
    return sorted(dirty_joints), sorted(dirty_roots)


def _closure_graph(
    old: Snapshot,
    new_detection: DetectionResult,
    new_keyframes: frozenset[int],
    new_suppressed_auto: frozenset[int],
    changed_keys: tuple[str, ...],
    dirty_joints: list[tuple[int, int]],
    dirty_roots: list[int],
    evaluation_intervals: tuple[str, ...],
) -> dict[str, Any]:
    affected_frames = {frame for frame, _ in dirty_joints} | set(dirty_roots)
    return {
        "nodes": [
            {"id": "detection", "kind": "waste_detection", "hash": new_detection.detection_hash},
            {
                "id": "keyframes",
                "kind": "keyframe_set",
                "hash": keyframe_hash(new_keyframes, new_suppressed_auto),
                "keys": sorted(new_keyframes),
                "suppressed_auto": sorted(new_suppressed_auto),
            },
            {"id": "intervals", "kind": "keyframe_intervals"},
            {
                "id": "repair",
                "kind": "pose_repair",
                "dirty_joint_count": len(dirty_joints),
                "dirty_root_count": len(dirty_roots),
            },
            {
                "id": "evaluation",
                "kind": "truth_evaluation",
                "recomputed_intervals": list(evaluation_intervals),
            },
        ],
        "edges": [
            {
                "from": "thresholds",
                "to": "detection",
                "trigger_keys": [key for key in changed_keys if key.startswith("threshold:")],
            },
            {
                "from": "detection",
                "to": "intervals",
                "trigger_keys": ["defect_interval_membership"],
            },
            {
                "from": "keyframes",
                "to": "intervals",
                "trigger_keys": [
                    key
                    for key in changed_keys
                    if key.startswith("keyframe:") or key.startswith("auto:")
                ],
            },
            {"from": "intervals", "to": "repair", "trigger_keys": ["anchor_pair_changed"]},
            {"from": "repair", "to": "evaluation", "trigger_keys": ["pose_hash_changed"]},
        ],
        "affected_interval_ranges": [
            {"start": start, "end": end} for start, end in _group_frames(affected_frames)
        ],
        "outside_closure_pose_hash_preserved": True,
        "outside_closure_metric_hash_preserved": True,
    }


def apply_incremental(
    old: Snapshot,
    truth_entries: tuple[DefectEntry, ...],
    thresholds: Thresholds | None = None,
    keyframes: frozenset[int] | set[int] | None = None,
    sequence: int | None = None,
    committed: bool = False,
    suppressed_auto: frozenset[int] | set[int] | None = None,
    frame_timings_ms: dict[int, float] | None = None,
) -> Snapshot:
    new_thresholds = thresholds if thresholds is not None else old.thresholds
    new_keyframes = frozenset(int(f) for f in (keyframes if keyframes is not None else old.keyframes))
    new_suppressed_auto = validate_suppressed_auto(
        suppressed_auto if suppressed_auto is not None else old.suppressed_auto,
        old.record,
    )
    new_sequence = old.version.sequence + 1 if sequence is None else sequence

    changed_keys: list[str] = []
    if new_thresholds != old.thresholds:
        old_values = old.thresholds.to_dict()
        for key, value in new_thresholds.to_dict().items():
            if old_values[key] != value:
                changed_keys.append(f"threshold:{key}:{old_values[key]}->{value}")
    if new_keyframes != old.keyframes:
        for frame in sorted(new_keyframes - old.keyframes):
            changed_keys.append(f"keyframe:+{frame}")
        for frame in sorted(old.keyframes - new_keyframes):
            changed_keys.append(f"keyframe:-{frame}")
    if new_suppressed_auto != old.suppressed_auto:
        for frame in sorted(new_suppressed_auto - old.suppressed_auto):
            changed_keys.append(f"auto:-{frame}")
        for frame in sorted(old.suppressed_auto - new_suppressed_auto):
            changed_keys.append(f"auto:+{frame}")

    detection = detect(old.record, truth_entries, new_thresholds, evidence=old.evidence)
    new_keyframes = validate_keyframes(new_keyframes, old.record, detection)
    new_anchors = build_anchors(old.record, detection, new_keyframes, new_suppressed_auto)
    dirty_joints, dirty_roots = _dirty_coordinates(old, detection, new_anchors)
    dirty_frames = {frame for frame, _ in dirty_joints} | set(dirty_roots)

    if dirty_joints or dirty_roots:
        anchors, repair, actual_joints, actual_roots = repair_dirty(
            old.record,
            detection,
            new_keyframes,
            old.repair,
            dirty_joints,
            dirty_roots,
            new_suppressed_auto,
            frame_timings_ms,
        )
    else:
        anchors = new_anchors
        repair = old.repair
        actual_joints = ()
        actual_roots = ()

    if tuple(dirty_joints) != tuple(actual_joints) or tuple(dirty_roots) != tuple(actual_roots):
        raise BusinessError(
            ErrorCode.E_CLOSURE_MISMATCH,
            "actual repair set differs from the declared dependency closure",
            {
                "declared_joint_coords": [
                    {"frame": frame, "joint": SKELETON[joint]} for frame, joint in dirty_joints
                ],
                "actual_joint_coords": [
                    {"frame": frame, "joint": SKELETON[joint]} for frame, joint in actual_joints
                ],
                "declared_root_frames": dirty_roots,
                "actual_root_frames": actual_roots,
            },
        )

    if dirty_frames:
        evaluation, evaluation_intervals = evaluate_incremental(
            old.record, old.truth, detection, repair, old.evaluation, dirty_frames
        )
    else:
        evaluation = old.evaluation
        evaluation_intervals = ()

    graph = _closure_graph(
        old,
        detection,
        new_keyframes,
        new_suppressed_auto,
        tuple(changed_keys),
        dirty_joints,
        dirty_roots,
        evaluation_intervals,
    )
    closure = ClosureReport(
        trigger="threshold" if any(key.startswith("threshold:") for key in changed_keys) else "keyframe",
        changed_keys=tuple(changed_keys),
        dirty_joint_coords=tuple(dirty_joints),
        dirty_root_frames=tuple(dirty_roots),
        evaluation_intervals=evaluation_intervals,
        graph=graph,
        actual_recomputed_joint_coords=tuple(actual_joints),
        actual_recomputed_root_frames=tuple(actual_roots),
        exact=True,
    )
    version = _version(
        new_sequence,
        old.record,
        new_thresholds,
        new_keyframes,
        new_suppressed_auto,
        detection,
        repair,
        evaluation,
    )
    return Snapshot(
        record=old.record,
        truth=old.truth,
        thresholds=new_thresholds,
        keyframes=new_keyframes,
        evidence=old.evidence,
        detection=detection,
        anchors=anchors,
        repair=repair,
        evaluation=evaluation,
        version=version,
        suppressed_auto=new_suppressed_auto,
        closure=closure,
        committed=committed,
    )


def assert_closure_exact(snapshot: Snapshot) -> None:
    closure = snapshot.closure
    if closure is None:
        return
    if set(closure.dirty_joint_coords) != set(closure.actual_recomputed_joint_coords):
        raise BusinessError(
            ErrorCode.E_CLOSURE_MISMATCH,
            "closure joint coordinate set is not exact",
            closure.to_dict(),
        )
    if set(closure.dirty_root_frames) != set(closure.actual_recomputed_root_frames):
        raise BusinessError(
            ErrorCode.E_CLOSURE_MISMATCH,
            "closure root frame set is not exact",
            closure.to_dict(),
        )


def check_consistency(incremental: Snapshot, full: Snapshot) -> ConsistencyReport:
    if incremental.frame_count != full.frame_count:
        raise BusinessError(
            ErrorCode.E_FRAME_MISSING,
            "incremental and full snapshots have different frame counts",
            {"incremental": incremental.frame_count, "full": full.frame_count},
        )
    quat_component_tolerance = EVALUATION_TOLERANCES["quaternion_component_abs"]
    root_tolerance = EVALUATION_TOLERANCES["root_component_abs_m"]
    angular_tolerance = 1e-5
    quat_component_delta = np.abs(incremental.repair.quats - full.repair.quats)
    component_max = np.max(quat_component_delta, axis=2)
    incremental_q = incremental.repair.quats / np.maximum(
        1e-15, np.linalg.norm(incremental.repair.quats, axis=2, keepdims=True)
    )
    full_q = full.repair.quats / np.maximum(
        1e-15, np.linalg.norm(full.repair.quats, axis=2, keepdims=True)
    )
    dot = np.clip(np.abs(np.einsum("fji,fji->fj", incremental_q, full_q)), -1.0, 1.0)
    angular = np.degrees(2.0 * np.arccos(dot))
    joint_fail = (component_max > quat_component_tolerance) | (angular > angular_tolerance)

    root_component_delta = np.abs(incremental.repair.roots - full.repair.roots)
    root_component_max = np.max(root_component_delta, axis=1)
    root_fail = root_component_max > root_tolerance

    max_quat_component = float(np.max(component_max)) if component_max.size else 0.0
    max_angular = float(np.max(angular)) if angular.size else 0.0
    max_root_component = float(np.max(root_component_max)) if root_component_max.size else 0.0
    mismatch_count = int(np.count_nonzero(joint_fail)) + int(np.count_nonzero(root_fail))
    checked_joint_coords = incremental.frame_count * incremental.record.joint_count
    first_mismatch: dict[str, Any] | None = None
    failing_frames = np.flatnonzero(joint_fail.any(axis=1) | root_fail)
    if failing_frames.size:
        frame = int(failing_frames[0])
        coordinates = [(joint_name, joint) for joint, joint_name in enumerate(SKELETON)]
        coordinates.append(("Root", -1))
        for joint_name, joint in sorted(coordinates, key=lambda item: item[0]):
            if joint == -1:
                if not root_fail[frame]:
                    continue
                left, right = _anchor_pair(frame, combined_root_anchors(incremental.anchors))
                first_mismatch = {
                    "frame": frame,
                    "joint": "Root",
                    "stage": "repair.root_trajectory",
                    "keyframe_interval_endpoints": [left, right],
                    "param_version": incremental.version.param_version,
                    "version": incremental.version.label,
                    "max_component_delta_m": clean_float(float(root_component_max[frame])),
                }
                break
            if not joint_fail[frame, joint]:
                continue
            left, right = _anchor_pair(frame, combined_joint_anchors(incremental.anchors, joint))
            first_mismatch = {
                "frame": frame,
                "joint": joint_name,
                "stage": "repair.joint_rotation",
                "keyframe_interval_endpoints": [left, right],
                "param_version": incremental.version.param_version,
                "version": incremental.version.label,
                "max_quaternion_component_delta": clean_float(float(component_max[frame, joint])),
                "angular_delta_deg": clean_float(float(angular[frame, joint])),
            }
            break

    consistent = mismatch_count == 0
    return ConsistencyReport(
        checked_frames=incremental.frame_count,
        checked_joint_coords=checked_joint_coords,
        tolerance={
            "quaternion_component_abs": quat_component_tolerance,
            "joint_angular_deg": angular_tolerance,
            "root_component_abs_m": root_tolerance,
        },
        max_quaternion_component_abs=clean_float(max_quat_component),
        max_joint_angular_deg=clean_float(max_angular),
        max_root_component_abs_m=clean_float(max_root_component),
        mismatch_count=mismatch_count,
        first_mismatch=first_mismatch,
        incremental_hash=incremental.repair.repair_hash,
        full_hash=full.repair.repair_hash,
        consistent=consistent,
    )


def rebuild_full_for_consistency(
    snapshot: Snapshot,
    truth_entries: tuple[DefectEntry, ...],
) -> Snapshot:
    return build_with_evidence(
        snapshot.record,
        snapshot.truth,
        truth_entries,
        snapshot.evidence,
        snapshot.thresholds,
        snapshot.keyframes,
        snapshot.version.sequence,
        committed=snapshot.committed,
        suppressed_auto=snapshot.suppressed_auto,
    )
