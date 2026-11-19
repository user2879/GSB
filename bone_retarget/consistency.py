"""Tolerance-aware pose comparison with first-divergence localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import ErrorCode, PipelineError
from .math3d import quat_angle_between
from .models import Animation


@dataclass(frozen=True)
class Tolerances:
    rotation_rad: float = 1e-6
    translation: float = 1e-6
    time: float = 1e-9
    retarget_residual_rad: float = 1e-6

    def to_dict(self) -> dict[str, float]:
        return {
            "rotation_rad": self.rotation_rad,
            "translation": self.translation,
            "time": self.time,
            "retarget_residual_rad": self.retarget_residual_rad,
        }


@dataclass
class ConsistencyReport:
    left_name: str
    right_name: str
    equal: bool
    joint_count: int
    frame_count: int
    max_rotation_error_rad: float
    max_translation_error: float
    max_time_error: float
    first_divergent_frame: int | None
    first_divergent_joint: str | None
    first_divergence_kind: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left_name,
            "right": self.right_name,
            "equal": self.equal,
            "joint_count": self.joint_count,
            "frame_count": self.frame_count,
            "max_rotation_error_rad": self.max_rotation_error_rad,
            "max_translation_error": self.max_translation_error,
            "max_time_error": self.max_time_error,
            "first_divergent_frame": self.first_divergent_frame,
            "first_divergent_joint": self.first_divergent_joint,
            "first_divergence_kind": self.first_divergence_kind,
        }


def compare_animations(
    left: Animation,
    right: Animation,
    *,
    tolerances: Tolerances | None = None,
    left_name: str = "direct",
    right_name: str = "roundtrip",
    raise_on_divergence: bool = False,
) -> ConsistencyReport:
    tol = tolerances or Tolerances()
    if left.joint_names != right.joint_names:
        common_joint_count = min(left.joint_count, right.joint_count)
        first_joint = (
            left.joint_names[common_joint_count]
            if left.joint_count > right.joint_count
            else right.joint_names[common_joint_count]
        )
        for index in range(common_joint_count):
            if left.joint_names[index] != right.joint_names[index]:
                first_joint = left.joint_names[index]
                break
        error = PipelineError(
            ErrorCode.E_RETARGET_DIVERGE,
            "animations have different joint name layouts",
            joint=first_joint,
            frame=0,
            field="joints",
            details={
                "left_joint_count": left.joint_count,
                "right_joint_count": right.joint_count,
            },
        )
        if raise_on_divergence:
            raise error
        return ConsistencyReport(
            left_name=left_name,
            right_name=right_name,
            equal=False,
            joint_count=min(left.joint_count, right.joint_count),
            frame_count=min(left.frame_count, right.frame_count),
            max_rotation_error_rad=float("inf"),
            max_translation_error=float("inf"),
            max_time_error=float("inf"),
            first_divergent_frame=0,
            first_divergent_joint=first_joint,
            first_divergence_kind="joint_layout",
        )

    common_frames = min(left.frame_count, right.frame_count)
    time_errors = np.abs(left.times[:common_frames] - right.times[:common_frames])
    rotation_errors = quat_angle_between(
        left.rotations[:common_frames], right.rotations[:common_frames]
    )
    translation_errors = np.linalg.norm(
        left.root_translation[:common_frames] - right.root_translation[:common_frames], axis=1
    )
    max_time_error = float(time_errors.max()) if time_errors.size else 0.0
    max_rotation_error = float(rotation_errors.max()) if rotation_errors.size else 0.0
    max_translation_error = float(translation_errors.max()) if translation_errors.size else 0.0

    equal = left.frame_count == right.frame_count
    equal = equal and max_time_error <= tol.time
    equal = equal and max_rotation_error <= tol.rotation_rad
    equal = equal and max_translation_error <= tol.translation

    first_frame: int | None = None
    first_joint: str | None = None
    first_kind: str | None = None
    candidates: list[tuple[int, int, str, str | None]] = []
    if left.frame_count != right.frame_count:
        candidates.append((common_frames, 4, "frame_count", left.joint_names[0]))
    if common_frames:
        for frame_index in np.nonzero(time_errors > tol.time)[0]:
            candidates.append((int(frame_index), 2, "timestamp", left.joint_names[0]))
        for frame_index, joint_index in np.argwhere(rotation_errors > tol.rotation_rad):
            candidates.append(
                (
                    int(frame_index),
                    0,
                    "rotation",
                    left.joint_names[int(joint_index)],
                )
            )
        for frame_index in np.nonzero(translation_errors > tol.translation)[0]:
            candidates.append((int(frame_index), 3, "root_translation", left.joint_names[0]))
    if candidates:
        frame_index, _priority, kind, joint_name = min(candidates, key=lambda item: (item[0], item[1]))
        first_frame = frame_index
        first_joint = joint_name
        first_kind = kind

    report = ConsistencyReport(
        left_name=left_name,
        right_name=right_name,
        equal=bool(equal),
        joint_count=left.joint_count,
        frame_count=min(left.frame_count, right.frame_count),
        max_rotation_error_rad=max_rotation_error,
        max_translation_error=max_translation_error,
        max_time_error=max_time_error,
        first_divergent_frame=first_frame,
        first_divergent_joint=first_joint,
        first_divergence_kind=first_kind,
    )
    if not report.equal and raise_on_divergence:
        frame = report.first_divergent_frame
        joint = report.first_divergent_joint
        details = {
            "kind": report.first_divergence_kind,
            "max_rotation_error_rad": report.max_rotation_error_rad,
            "rotation_tolerance_rad": tol.rotation_rad,
            "max_translation_error": report.max_translation_error,
            "translation_tolerance": tol.translation,
        }
        raise PipelineError(
            ErrorCode.E_RETARGET_DIVERGE,
            "animations diverge beyond declared tolerance",
            joint=joint,
            frame=frame,
            field=report.first_divergence_kind,
            details=details,
        )
    return report
