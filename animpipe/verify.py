"""Frame-by-frame reconciliation between incremental and serial paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .animation import Animation
from .errors import E_CLOSURE_MISMATCH
from .hashutil import frame_hash
from .quat import angular_distance


@dataclass
class FrameDiff:
    frame: int
    time: float
    joints: list[int]
    joint_names: list[str]
    max_angle_rad: float
    max_translation: float
    per_joint_angle_rad: list[float]
    parameter_version: str


def compare_animations(
    expected: Animation,
    actual: Animation,
    *,
    angle_tolerance: float = 1e-9,
    translation_tolerance: float = 1e-9,
) -> dict[str, Any]:
    if expected.joint_names != actual.joint_names:
        raise ValueError("animations must share joint names")
    if expected.frame_count != actual.frame_count:
        return {
            "equal": False,
            "first_diverging_frame": 0,
            "reason": "frame_count",
            "expected_frames": expected.frame_count,
            "actual_frames": actual.frame_count,
        }

    first_diff: FrameDiff | None = None
    per_frame_max: list[float] = []
    for frame in range(expected.frame_count):
        angles = [
            angular_distance(
                expected.rotations[frame, joint],
                actual.rotations[frame, joint],
            )
            for joint in range(expected.joint_count)
        ]
        translation = float(
            np.max(
                np.abs(
                    expected.root_translations[frame] - actual.root_translations[frame]
                )
            )
        )
        max_angle = max(angles)
        per_frame_max.append(max_angle)
        if (
            max_angle > angle_tolerance
            or translation > translation_tolerance
        ) and first_diff is None:
            involved = [
                joint
                for joint, angle in enumerate(angles)
                if angle > angle_tolerance
            ]
            first_diff = FrameDiff(
                frame=frame,
                time=float(expected.times[frame]),
                joints=involved,
                joint_names=[expected.joint_names[j] for j in involved],
                max_angle_rad=max_angle,
                max_translation=translation,
                per_joint_angle_rad=angles,
                parameter_version=actual.parameter_version,
            )

    expected_hashes = [
        frame_hash(
            frame,
            expected.rotations[frame],
            expected.root_translations[frame],
            {"parameter_version": expected.parameter_version},
        )
        for frame in range(expected.frame_count)
    ]
    actual_hashes = [
        frame_hash(
            frame,
            actual.rotations[frame],
            actual.root_translations[frame],
            {"parameter_version": actual.parameter_version},
        )
        for frame in range(actual.frame_count)
    ]
    return {
        "equal": first_diff is None,
        "first_diverging_frame": None if first_diff is None else first_diff.frame,
        "frame_diff": None
        if first_diff is None
        else {
            "frame": first_diff.frame,
            "time": first_diff.time,
            "joints": first_diff.joints,
            "joint_names": first_diff.joint_names,
            "max_angle_rad": first_diff.max_angle_rad,
            "max_translation": first_diff.max_translation,
            "per_joint_angle_rad": first_diff.per_joint_angle_rad,
            "parameter_version": first_diff.parameter_version,
        },
        "max_angle_rad": max(per_frame_max, default=0.0),
        "expected_content_hash": expected.content_hash(),
        "actual_content_hash": actual.content_hash(),
        "frame_hash_equal": expected_hashes == actual_hashes,
        "angle_tolerance": angle_tolerance,
        "translation_tolerance": translation_tolerance,
    }


def assert_closure_hash_stability(
    before: list[str], after: list[str], closure_frames: set[int]
) -> None:
    """Frames outside the recomputed closure must keep identical hashes."""

    if len(before) != len(after):
        raise E_CLOSURE_MISMATCH(
            "frame count changed outside closure",
            {"before": len(before), "after": len(after)},
        )
    drifted = [
        frame
        for frame, (old, new) in enumerate(zip(before, after))
        if old != new and frame not in closure_frames
    ]
    if drifted:
        raise E_CLOSURE_MISMATCH(
            "content hash changed for frames outside the closure",
            {"drifted_frames": drifted, "closure_frames": sorted(closure_frames)},
        )
