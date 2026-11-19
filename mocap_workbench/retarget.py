"""Invertible direction-space retargeting with rest-pose conjugation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import BusinessError, ErrorCode
from .skeleton import Skeleton, rest_height, rotation_between_batch


@dataclass
class RetargetMapping:
    rest_conjugations: np.ndarray
    root_scale: float


@dataclass
class RetargetResult:
    pose: np.ndarray
    inverse_pose: np.ndarray
    roundtrip_residual: np.ndarray
    target_bone_length_error: np.ndarray
    mapping: RetargetMapping
    flagged_joints_by_frame: list[list[str]]
    joint_names: tuple[str, ...]
    joint_rotations: np.ndarray

    def report(self, tolerance: float) -> dict[str, Any]:
        per_joint_max = np.max(self.roundtrip_residual, axis=0)
        joint_statistics = [
            {
                "joint": self.joint_names[index],
                "max_roundtrip_residual_m": float(per_joint_max[index]),
                "frames_over_tolerance": int(np.sum(self.roundtrip_residual[:, index] > tolerance)),
            }
            for index in range(len(self.joint_names))
        ]
        flagged = [item for item in joint_statistics if item["frames_over_tolerance"] > 0]
        return {
            "roundtrip_tolerance_m": float(tolerance),
            "global_pose_deviation_max_m": float(np.max(self.roundtrip_residual)),
            "global_pose_deviation_mean_m": float(np.mean(self.roundtrip_residual)),
            "max_target_bone_length_error_m": float(np.max(self.target_bone_length_error)),
            "root_scale": float(self.mapping.root_scale),
            "flagged_joint_count": int(len(flagged)),
            "flagged_joints": flagged,
            "joint_maxima": joint_statistics,
            "frames_with_flagged_joints": int(
                sum(1 for flags in self.flagged_joints_by_frame if flags)
            ),
        }


def build_mapping(source: Skeleton, target: Skeleton) -> RetargetMapping:
    if source.joints != target.joints or not np.array_equal(source.parents, target.parents):
        raise BusinessError(ErrorCode.E_INPUT, "retargeting requires identical topology")
    conjugations = np.tile(np.eye(3), (source.joint_count, 1, 1))
    for joint in range(1, source.joint_count):
        conjugations[joint] = rotation_between_batch(
            source.rest_local[joint][None, :], target.rest_local[joint][None, :]
        )[0]
    scale = rest_height(target.rest_world) / max(rest_height(source.rest_world), 1e-12)
    return RetargetMapping(rest_conjugations=conjugations, root_scale=float(scale))


def retarget_pose(
    source_pose: np.ndarray,
    source: Skeleton,
    target: Skeleton,
    tolerance: float = 1e-8,
) -> RetargetResult:
    source_pose = np.asarray(source_pose, dtype=np.float64)
    if source_pose.ndim != 3 or source_pose.shape[1:] != (source.joint_count, 3):
        raise BusinessError(ErrorCode.E_INPUT, "source pose shape does not match source skeleton")
    mapping = build_mapping(source, target)
    frames = source_pose.shape[0]
    target_pose = np.zeros_like(source_pose)
    target_pose[:, 0] = target.rest_world[0] + mapping.root_scale * (
        source_pose[:, 0] - source.rest_world[0]
    )
    joint_rotations = np.tile(np.eye(3), (frames, source.joint_count, 1, 1))
    for joint in range(1, source.joint_count):
        parent = int(source.parents[joint])
        source_offset = source_pose[:, joint] - source_pose[:, parent]
        source_length = np.linalg.norm(source_offset, axis=-1)
        source_unit = source_offset / np.maximum(source_length[:, None], 1e-12)
        target_unit = np.einsum("ij,fj->fi", mapping.rest_conjugations[joint], source_unit)
        target_pose[:, joint] = target_pose[:, parent] + target.bone_lengths[joint] * target_unit
        source_rest_unit = source.rest_local[joint] / source.bone_lengths[joint]
        joint_rotations[:, joint] = rotation_between_batch(
            np.broadcast_to(source_rest_unit, target_unit.shape), target_unit
        )

    inverse_pose = inverse_retarget_pose(target_pose, source, target, mapping)
    roundtrip = np.linalg.norm(inverse_pose - source_pose, axis=-1)
    bone_error = np.zeros_like(roundtrip)
    for joint in range(1, target.joint_count):
        parent = int(target.parents[joint])
        length = np.linalg.norm(target_pose[:, joint] - target_pose[:, parent], axis=-1)
        bone_error[:, joint] = np.abs(length - target.bone_lengths[joint])
    flagged_by_frame: list[list[str]] = []
    for frame in range(frames):
        flagged_by_frame.append(
            [target.joints[index] for index in np.flatnonzero(roundtrip[frame] > tolerance)]
        )
    return RetargetResult(
        pose=target_pose,
        inverse_pose=inverse_pose,
        roundtrip_residual=roundtrip,
        target_bone_length_error=bone_error,
        mapping=mapping,
        flagged_joints_by_frame=flagged_by_frame,
        joint_names=source.joints,
        joint_rotations=joint_rotations,
    )


def inverse_retarget_pose(
    target_pose: np.ndarray,
    source: Skeleton,
    target: Skeleton,
    mapping: RetargetMapping | None = None,
) -> np.ndarray:
    target_pose = np.asarray(target_pose, dtype=np.float64)
    if target_pose.ndim != 3 or target_pose.shape[1:] != (target.joint_count, 3):
        raise BusinessError(ErrorCode.E_INPUT, "target pose shape does not match target skeleton")
    if mapping is None:
        mapping = build_mapping(source, target)
    source_pose = np.zeros_like(target_pose)
    source_pose[:, 0] = source.rest_world[0] + (
        target_pose[:, 0] - target.rest_world[0]
    ) / mapping.root_scale
    for joint in range(1, source.joint_count):
        parent = int(source.parents[joint])
        target_offset = target_pose[:, joint] - target_pose[:, parent]
        target_length = np.linalg.norm(target_offset, axis=-1)
        target_unit = target_offset / np.maximum(target_length[:, None], 1e-12)
        source_unit = np.einsum("ji,fj->fi", mapping.rest_conjugations[joint], target_unit)
        source_pose[:, joint] = source_pose[:, parent] + source.bone_lengths[joint] * source_unit
    return source_pose


def assert_retarget_tolerance(result: RetargetResult, tolerance: float) -> None:
    if np.any(result.roundtrip_residual > tolerance):
        frame, joint = np.unravel_index(np.argmax(result.roundtrip_residual), result.roundtrip_residual.shape)
        raise BusinessError(
            ErrorCode.E_RETARGET_TOL,
            "retarget roundtrip exceeded the declared tolerance",
            tolerance_m=float(tolerance),
            max_residual_m=float(result.roundtrip_residual[frame, joint]),
            frame=int(frame),
            joint=result.joint_names[int(joint)],
        )
