"""Sparse-keyframe SLERP/root repair and dirty-coordinate updates."""

from __future__ import annotations

import bisect
import hashlib
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import POLICY_TEXT, SKELETON
from .errors import BusinessError, ErrorCode
from .io import atomic_write_json, clean_float, clean_vector, sha256_json
from .math3d import normalize_quat, slerp
from .model import AnchorModel, DetectionResult, Motion, RepairResult


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = start
    for item in indices[1:]:
        current = int(item)
        if current != previous + 1:
            runs.append((start, previous))
            start = current
        previous = current
    runs.append((start, previous))
    return runs


def validate_keyframes(
    keyframes: Iterable[int], record: Motion, detection: DetectionResult
) -> frozenset[int]:
    values = list(keyframes)
    for frame in values:
        if not isinstance(frame, (int, np.integer)) or isinstance(frame, bool):
            raise BusinessError(
                ErrorCode.E_KEYFRAME_INVALID,
                "keyframe must be an integer frame index",
                {"frame": frame},
            )
        if frame < 0 or frame >= record.frame_count:
            raise BusinessError(
                ErrorCode.E_KEYFRAME_INVALID,
                "keyframe lies outside the record",
                {"frame": int(frame), "frame_count": record.frame_count},
            )
        if detection.joint_bad[int(frame), :].any() or detection.root_bad[int(frame)]:
            bad_joints = [SKELETON[j] for j in np.flatnonzero(detection.joint_bad[int(frame), :])]
            if detection.root_bad[int(frame)]:
                bad_joints.append("Root")
            raise BusinessError(
                ErrorCode.E_KEYFRAME_INVALID,
                "a keyframe cannot anchor a defective frame",
                {"frame": int(frame), "defective_coordinates": bad_joints},
            )
    if len(values) != len(set(int(v) for v in values)):
        raise BusinessError(
            ErrorCode.E_KEYFRAME_INVALID,
            "duplicate keyframe frame index",
            {"keyframes": sorted(int(v) for v in values)},
        )
    return frozenset(int(frame) for frame in values)


def add_keyframe(
    current: frozenset[int], frame: int, record: Motion, detection: DetectionResult
) -> frozenset[int]:
    if frame in current:
        raise BusinessError(
            ErrorCode.E_KEYFRAME_INVALID,
            "keyframe already exists",
            {"frame": frame},
        )
    return validate_keyframes(set(current) | {frame}, record, detection)


def remove_keyframe(current: frozenset[int], frame: int) -> frozenset[int]:
    if frame not in current:
        raise BusinessError(
            ErrorCode.E_KEYFRAME_INVALID,
            "keyframe does not exist",
            {"frame": frame},
        )
    return frozenset(value for value in current if value != frame)


def validate_suppressed_auto(
    suppressed_auto: Iterable[int], record: Motion
) -> frozenset[int]:
    values: set[int] = set()
    for frame in suppressed_auto:
        if not isinstance(frame, (int, np.integer)) or isinstance(frame, bool):
            raise BusinessError(
                ErrorCode.E_KEYFRAME_INVALID,
                "suppressed automatic anchor must be an integer frame",
                {"frame": frame},
            )
        if frame < 0 or frame >= record.frame_count:
            raise BusinessError(
                ErrorCode.E_KEYFRAME_INVALID,
                "suppressed automatic anchor lies outside the record",
                {"frame": int(frame)},
            )
        values.add(int(frame))
    return frozenset(values)


def build_anchors(
    record: Motion,
    detection: DetectionResult,
    keyframes: frozenset[int],
    suppressed_auto: frozenset[int] = frozenset(),
) -> AnchorModel:
    frames = record.frame_count
    suppressed_auto = validate_suppressed_auto(suppressed_auto, record)
    joint_auto: list[tuple[int, ...]] = []
    joint_user: list[tuple[int, ...]] = []
    for joint in range(record.joint_count):
        auto: set[int] = set()
        hard_boundaries: set[int] = set()
        valid = ~detection.joint_bad[:, joint]
        if valid.any():
            hard_boundaries.add(int(np.flatnonzero(valid)[0]))
            hard_boundaries.add(int(np.flatnonzero(valid)[-1]))
            auto.update(hard_boundaries)
        for start, end in _contiguous_runs(detection.joint_bad[:, joint]):
            if start - 1 >= 0 and not detection.joint_bad[start - 1, joint]:
                auto.add(start - 1)
            if end + 1 < frames and not detection.joint_bad[end + 1, joint]:
                auto.add(end + 1)
        joint_auto.append(
            tuple(sorted({frame for frame in auto if frame not in suppressed_auto} | hard_boundaries))
        )
        joint_user.append(tuple(sorted(frame for frame in keyframes if not detection.joint_bad[frame, joint])))

    root_auto: set[int] = set()
    root_hard_boundaries: set[int] = set()
    valid_root = ~detection.root_bad
    if valid_root.any():
        root_hard_boundaries.add(int(np.flatnonzero(valid_root)[0]))
        root_hard_boundaries.add(int(np.flatnonzero(valid_root)[-1]))
        root_auto.update(root_hard_boundaries)
    for start, end in _contiguous_runs(detection.root_bad):
        if start - 1 >= 0 and not detection.root_bad[start - 1]:
            root_auto.add(start - 1)
        if end + 1 < frames and not detection.root_bad[end + 1]:
            root_auto.add(end + 1)
    root_user = tuple(sorted(keyframes))
    return AnchorModel(
        joint_auto=tuple(joint_auto),
        joint_user=tuple(joint_user),
        root_auto=tuple(
            sorted({frame for frame in root_auto if frame not in suppressed_auto} | root_hard_boundaries)
        ),
        root_user=root_user,
    )


def combined_joint_anchors(anchors: AnchorModel, joint: int) -> tuple[int, ...]:
    return tuple(sorted(set(anchors.joint_auto[joint]) | set(anchors.joint_user[joint])))


def combined_root_anchors(anchors: AnchorModel) -> tuple[int, ...]:
    return tuple(sorted(set(anchors.root_auto) | set(anchors.root_user)))


def _anchor_pair(frame: int, combined: tuple[int, ...]) -> tuple[int, int]:
    position = bisect.bisect_left(combined, frame)
    if position < len(combined) and combined[position] == frame:
        return frame, frame
    left = combined[position - 1] if position > 0 else None
    right = combined[position] if position < len(combined) else None
    if left is None and right is None:
        return frame, frame
    if left is None:
        return int(right), int(right)
    if right is None:
        return int(left), int(left)
    return int(left), int(right)


def repaired_joint_value(record: Motion, frame: int, joint: int, anchors: AnchorModel) -> np.ndarray:
    combined = combined_joint_anchors(anchors, joint)
    left, right = _anchor_pair(frame, combined)
    if left == right:
        return normalize_quat(record.quats[left, joint])
    alpha = (frame - left) / float(right - left)
    return slerp(record.quats[left, joint], record.quats[right, joint], alpha)


def repaired_root_value(record: Motion, frame: int, anchors: AnchorModel) -> np.ndarray:
    combined = combined_root_anchors(anchors)
    left, right = _anchor_pair(frame, combined)
    if left == right:
        return record.roots[left].copy()
    alpha = (frame - left) / float(right - left)
    return record.roots[left] * (1.0 - alpha) + record.roots[right] * alpha


def quantize_quat(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    return np.array(clean_vector(q), dtype=np.float64)


def quantize_root(root: np.ndarray) -> np.ndarray:
    return np.array(clean_vector(root), dtype=np.float64)


def frame_pose_hash(quats: np.ndarray, roots: np.ndarray, frame: int) -> str:
    quat = np.rint(quats[frame] * 1_000_000_000.0).astype("<i8").tobytes()
    root = np.rint(roots[frame] * 1_000_000_000.0).astype("<i8").tobytes()
    return hashlib.sha256(quat + root).hexdigest()


def repair_hash(frame_hashes: tuple[str, ...], policy: dict[str, str]) -> str:
    return sha256_json({"frame_hashes": list(frame_hashes), "policy": policy})


def repair_full(
    record: Motion,
    detection: DetectionResult,
    keyframes: frozenset[int],
    suppressed_auto: frozenset[int] = frozenset(),
) -> tuple[AnchorModel, RepairResult]:
    keyframes = validate_keyframes(keyframes, record, detection)
    anchors = build_anchors(record, detection, keyframes, suppressed_auto)
    quats = record.quats.copy()
    roots = record.roots.copy()
    changed_joints: list[tuple[int, int]] = []
    for frame in range(record.frame_count):
        for joint in np.flatnonzero(detection.joint_bad[frame, :]):
            joint_index = int(joint)
            quats[frame, joint_index, :] = quantize_quat(
                repaired_joint_value(record, frame, joint_index, anchors)
            )
            changed_joints.append((frame, joint_index))
    changed_root: list[int] = []
    for frame in np.flatnonzero(detection.root_bad):
        frame_index = int(frame)
        roots[frame_index, :] = quantize_root(repaired_root_value(record, frame_index, anchors))
        changed_root.append(frame_index)

    # Clean frames are copied from the raw record and never touched. Even a
    # user keyframe on a clean frame remains an anchor whose value is raw truth.
    frame_hashes = tuple(frame_pose_hash(quats, roots, frame) for frame in range(record.frame_count))
    policy = dict(POLICY_TEXT)
    result = RepairResult(
        quats=quats,
        roots=roots,
        frame_hashes=frame_hashes,
        changed_joint_frames=tuple(changed_joints),
        changed_root_frames=tuple(changed_root),
        repair_hash=repair_hash(frame_hashes, policy),
        policy=policy,
    )
    return anchors, result


def repair_dirty(
    record: Motion,
    detection: DetectionResult,
    keyframes: frozenset[int],
    base_repair: RepairResult,
    dirty_joints: Iterable[tuple[int, int]],
    dirty_roots: Iterable[int],
    suppressed_auto: frozenset[int] = frozenset(),
    timings_ms: dict[int, float] | None = None,
) -> tuple[AnchorModel, RepairResult, tuple[tuple[int, int], ...], tuple[int, ...]]:
    """Recompute only declared dirty coordinates; clean/unchanged bytes are copied."""

    keyframes = validate_keyframes(keyframes, record, detection)
    anchors = build_anchors(record, detection, keyframes, suppressed_auto)
    quats = base_repair.quats.copy()
    roots = base_repair.roots.copy()
    frame_hashes = list(base_repair.frame_hashes)
    actual_joint: list[tuple[int, int]] = []
    actual_root: list[int] = []
    touched_frames: set[int] = set()

    for frame, joint in sorted(set((int(f), int(j)) for f, j in dirty_joints)):
        started = time.perf_counter()
        record.check_frame(frame)
        if joint < 0 or joint >= record.joint_count:
            raise BusinessError(
                ErrorCode.E_KEYFRAME_INVALID,
                "joint index is invalid",
                {"frame": frame, "joint": joint},
            )
        if detection.joint_bad[frame, joint]:
            value = quantize_quat(repaired_joint_value(record, frame, joint, anchors))
        else:
            value = record.quats[frame, joint].copy()
        quats[frame, joint, :] = value
        actual_joint.append((frame, joint))
        touched_frames.add(frame)
        if timings_ms is not None:
            timings_ms[frame] = timings_ms.get(frame, 0.0) + (time.perf_counter() - started) * 1000.0

    for frame in sorted(set(int(frame) for frame in dirty_roots)):
        started = time.perf_counter()
        record.check_frame(frame)
        if detection.root_bad[frame]:
            value = quantize_root(repaired_root_value(record, frame, anchors))
        else:
            value = record.roots[frame].copy()
        roots[frame, :] = value
        actual_root.append(frame)
        touched_frames.add(frame)
        if timings_ms is not None:
            timings_ms[frame] = timings_ms.get(frame, 0.0) + (time.perf_counter() - started) * 1000.0

    for frame in sorted(touched_frames):
        frame_hashes[frame] = frame_pose_hash(quats, roots, frame)

    policy = dict(POLICY_TEXT)
    # These lists describe the *current* repaired set, not historical touches.
    # When a threshold edit makes a coordinate clean again, it must disappear.
    changed_joint_frames = tuple(
        (frame, joint)
        for frame in range(record.frame_count)
        for joint in np.flatnonzero(detection.joint_bad[frame, :])
    )
    changed_root_frames = tuple(int(frame) for frame in np.flatnonzero(detection.root_bad))
    result = RepairResult(
        quats=quats,
        roots=roots,
        frame_hashes=tuple(frame_hashes),
        changed_joint_frames=changed_joint_frames,
        changed_root_frames=changed_root_frames,
        repair_hash=repair_hash(tuple(frame_hashes), policy),
        policy=policy,
    )
    return anchors, result, tuple(sorted(actual_joint)), tuple(sorted(actual_root))


def repair_payload(
    snapshot_meta: dict[str, object],
    record: Motion,
    repair: RepairResult,
) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    for frame in range(record.frame_count):
        frames.append(
            {
                "frame": frame,
                "root": [clean_float(value) for value in repair.roots[frame]],
                "joints": {
                    joint: [clean_float(value) for value in repair.quats[frame, joint_index]]
                    for joint_index, joint in enumerate(SKELETON)
                },
                "frame_sha256": repair.frame_hashes[frame],
            }
        )
    return {
        "schema": "mocap-repair/v1",
        "snapshot": snapshot_meta,
        "policy": repair.policy,
        "repair_sha256": repair.repair_hash,
        "frames": frames,
    }


def write_repair(path: Path, snapshot_meta: dict[str, object], record: Motion, repair: RepairResult) -> str:
    return atomic_write_json(path, repair_payload(snapshot_meta, record, repair))
