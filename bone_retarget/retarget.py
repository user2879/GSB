"""Rest-pose-aware motion retargeting and residual reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .consistency import Tolerances
from .errors import ErrorCode, PipelineError
from .math3d import (
    normalize_quaternion,
    quat_angle_between,
    quat_conjugate,
    quat_multiply,
    rigid_matrix,
    rotate_vector,
)
from .models import Animation, Skeleton


@dataclass
class RetargetResult:
    animation: Animation
    report: dict[str, Any]


def _validate_inputs(animation: Animation, source: Skeleton, target: Skeleton) -> dict[str, int]:
    source_mapping: dict[str, int] = {}
    for index, name in enumerate(source.joint_names):
        source_mapping[name] = index
        if name not in animation.joint_names:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "source animation is missing a source skeleton joint",
                joint=name,
                field="animation.joints",
            )
    extra_animation_joints = [name for name in animation.joint_names if name not in source_mapping]
    if extra_animation_joints:
        raise PipelineError(
            ErrorCode.E_INDEX_OVERFLOW,
            "source animation contains joints absent from the source skeleton",
            joint=extra_animation_joints[0],
            field="animation.joints",
        )
    for name in source.joint_names:
        if name not in target.joint_names:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "target skeleton is missing a joint required for retargeting",
                joint=name,
                field="target.joints",
            )
    return source_mapping


def _world_rotations_from_local(
    skeleton: Skeleton, local_rotations: np.ndarray
) -> np.ndarray:
    world = np.empty((skeleton.joint_count, 4), dtype=np.float64)
    for index, _joint in enumerate(skeleton.joints):
        parent = skeleton.parent_index(index)
        if parent is None:
            world[index] = local_rotations[index]
        else:
            world[index] = quat_multiply(world[parent], local_rotations[index])
        world[index] = normalize_quaternion(world[index])
    return world


def _global_matrices_for_frame(
    skeleton: Skeleton,
    local_rotations: np.ndarray,
    root_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_rotations = _world_rotations_from_local(skeleton, local_rotations)
    translations = np.empty((skeleton.joint_count, 3), dtype=np.float64)
    matrices = np.empty((skeleton.joint_count, 4, 4), dtype=np.float64)
    for index, joint in enumerate(skeleton.joints):
        parent = skeleton.parent_index(index)
        local_translation = joint.rest_translation.copy()
        if parent is None:
            local_translation = local_translation + root_translation
            translations[index] = local_translation
        else:
            translations[index] = translations[parent] + rotate_vector(
                world_rotations[parent], local_translation
            )
        matrices[index] = rigid_matrix(translations[index], world_rotations[index])
    return matrices, translations


def retarget_motion(
    animation: Animation,
    source: Skeleton,
    target: Skeleton,
    *,
    tolerances: Tolerances | None = None,
) -> RetargetResult:
    tol = tolerances or Tolerances()
    source_mapping = _validate_inputs(animation, source, target)
    frame_count = animation.frame_count
    target_joint_count = target.joint_count
    target_local = np.empty((frame_count, target_joint_count, 4), dtype=np.float64)
    source_rest_world = source.world_rest_rotations()
    target_rest_world = target.world_rest_rotations()
    source_root = source.root_name
    target_root = target.root_name
    target_root_index = target.joint_index(target_root)
    source_root_length = source.joints[source.joint_index(source_root)].length
    target_root_length = target.joints[target.joint_index(target_root)].length
    root_translation_scale = target_root_length / source_root_length

    source_deltas_by_frame: list[np.ndarray] = []
    for frame_index in range(frame_count):
        source_world = np.empty((source.joint_count, 4), dtype=np.float64)
        deltas = np.empty((source.joint_count, 4), dtype=np.float64)
        for joint_index, joint in enumerate(source.joints):
            parent = source.parent_index(joint_index)
            local = animation.rotations[frame_index, source_mapping[joint.name]]
            source_world[joint_index] = (
                local if parent is None else quat_multiply(source_world[parent], local)
            )
            source_world[joint_index] = normalize_quaternion(source_world[joint_index])
            deltas[joint_index] = normalize_quaternion(
                quat_multiply(source_world[joint_index], quat_conjugate(source_rest_world[joint_index]))
            )
        source_deltas_by_frame.append(deltas)

        target_world = np.empty((target_joint_count, 4), dtype=np.float64)
        for target_index, joint in enumerate(target.joints):
            target_parent = target.parent_index(target_index)
            if joint.name in source_mapping:
                source_index = source_mapping[joint.name]
                desired_world = quat_multiply(deltas[source_index], target_rest_world[target_index])
                if target_parent is None:
                    local = desired_world
                else:
                    local = quat_multiply(quat_conjugate(target_world[target_parent]), desired_world)
            else:
                if target_parent is None:
                    local = joint.rest_rotation
                    desired_world = local
                else:
                    local = joint.rest_rotation
                    desired_world = quat_multiply(target_world[target_parent], local)
            target_local[frame_index, target_index] = normalize_quaternion(local)
            target_world[target_index] = normalize_quaternion(desired_world)

    target_root_translation = animation.root_translation * root_translation_scale
    result_animation = Animation(
        joint_names=target.joint_names,
        times=animation.times.copy(),
        rotations=target_local,
        root_translation=target_root_translation,
        interpolation=animation.interpolation,
        fps=animation.fps,
    )

    samples: list[dict[str, Any]] = []
    joint_max_residual = np.zeros(target_joint_count, dtype=np.float64)
    joint_max_deviation = np.zeros(target_joint_count, dtype=np.float64)
    joint_first_flag: list[int | None] = [None] * target_joint_count
    frame_max_residual = np.zeros(frame_count, dtype=np.float64)
    frame_max_deviation = np.zeros(frame_count, dtype=np.float64)
    frame_flagged: list[list[str]] = [[] for _ in range(frame_count)]
    max_residual = 0.0
    max_deviation = 0.0
    max_residual_location: tuple[int, str] | None = None
    max_deviation_location: tuple[int, str] | None = None

    for frame_index in range(frame_count):
        local = result_animation.rotations[frame_index]
        result_world = _world_rotations_from_local(target, local)
        matrices, _translations = _global_matrices_for_frame(
            target, local, result_animation.root_translation[frame_index]
        )
        for target_index, joint in enumerate(target.joints):
            if joint.name in source_mapping:
                source_index = source_mapping[joint.name]
                desired_world = quat_multiply(
                    source_deltas_by_frame[frame_index][source_index],
                    target_rest_world[target_index],
                )
                residual = float(quat_angle_between(result_world[target_index], desired_world))
                target_delta = quat_multiply(
                    result_world[target_index], quat_conjugate(target_rest_world[target_index])
                )
                global_deviation = float(
                    quat_angle_between(
                        target_delta, source_deltas_by_frame[frame_index][source_index]
                    )
                )
            else:
                residual = 0.0
                global_deviation = 0.0
            flagged = max(residual, global_deviation) > tol.retarget_residual_rad
            joint_max_residual[target_index] = max(joint_max_residual[target_index], residual)
            joint_max_deviation[target_index] = max(
                joint_max_deviation[target_index], global_deviation
            )
            frame_max_residual[frame_index] = max(frame_max_residual[frame_index], residual)
            frame_max_deviation[frame_index] = max(
                frame_max_deviation[frame_index], global_deviation
            )
            if flagged:
                if joint_first_flag[target_index] is None:
                    joint_first_flag[target_index] = frame_index
                if joint.name not in frame_flagged[frame_index]:
                    frame_flagged[frame_index].append(joint.name)
            if residual > max_residual:
                max_residual = residual
                max_residual_location = (frame_index, joint.name)
            if global_deviation > max_deviation:
                max_deviation = global_deviation
                max_deviation_location = (frame_index, joint.name)
            local_translation = target.joints[target_index].rest_translation.copy()
            if target_index == target_root_index:
                local_translation = local_translation + result_animation.root_translation[frame_index]
            local_matrix = rigid_matrix(local_translation, local[target_index])
            samples.append(
                {
                    "frame": frame_index,
                    "time": float(result_animation.times[frame_index]),
                    "joint": joint.name,
                    "residual_rad": residual,
                    "global_deviation_rad": global_deviation,
                    "flagged": bool(flagged),
                    "target_local_rotation_wxyz": [float(value) for value in local[target_index]],
                    "target_global_rotation_wxyz": [
                        float(value) for value in result_world[target_index]
                    ],
                    "target_local_matrix": [
                        [float(value) for value in row] for row in local_matrix
                    ],
                    "target_global_matrix": [
                        [float(value) for value in row] for row in matrices[target_index]
                    ],
                }
            )

    flagged_joints = [
        target.joints[index].name
        for index in range(target_joint_count)
        if joint_first_flag[index] is not None
    ]
    report: dict[str, Any] = {
        "tolerance": tol.to_dict(),
        "retarget_residual_tolerance_rad": tol.retarget_residual_rad,
        "frame_count": frame_count,
        "joint_count": target_joint_count,
        "root_translation_scale": float(root_translation_scale),
        "source_root_length": float(source_root_length),
        "target_root_length": float(target_root_length),
        "max_residual_rad": float(max_residual),
        "max_global_deviation_rad": float(max_deviation),
        "max_residual_frame": max_residual_location[0] if max_residual_location else None,
        "max_residual_joint": max_residual_location[1] if max_residual_location else None,
        "max_deviation_frame": max_deviation_location[0] if max_deviation_location else None,
        "max_deviation_joint": max_deviation_location[1] if max_deviation_location else None,
        "flagged_joints": flagged_joints,
        "joints": [
            {
                "joint": target.joints[index].name,
                "max_residual_rad": float(joint_max_residual[index]),
                "max_global_deviation_rad": float(joint_max_deviation[index]),
                "flagged": joint_first_flag[index] is not None,
                "first_flagged_frame": joint_first_flag[index],
            }
            for index in range(target_joint_count)
        ],
        "frames": [
            {
                "frame": frame_index,
                "time": float(result_animation.times[frame_index]),
                "max_residual_rad": float(frame_max_residual[frame_index]),
                "max_global_deviation_rad": float(frame_max_deviation[frame_index]),
                "flagged_joints": frame_flagged[frame_index],
            }
            for frame_index in range(frame_count)
        ],
        "samples": samples,
    }
    return RetargetResult(animation=result_animation, report=report)


def target_transforms_document(result: RetargetResult) -> dict[str, Any]:
    report = result.report
    frames: list[dict[str, Any]] = []
    for frame_index in range(result.animation.frame_count):
        frames.append(
            {
                "frame": frame_index,
                "time": float(result.animation.times[frame_index]),
                "joints": [
                    sample for sample in report["samples"] if sample["frame"] == frame_index
                ],
            }
        )
    return {
        "schema_version": 1,
        "joint_map": {
            name: index for index, name in enumerate(result.animation.joint_names)
        },
        "frames": frames,
    }
