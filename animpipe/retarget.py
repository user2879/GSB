"""Per-frame retargeting from source skeleton to target skeleton.

Mapping rule (local space, per joint j)::

    Rt_local[j] = Rt_rest_local[j] * (Rs_rest_local[j])^-1 * Rs_local[j]

so the joint's motion *relative to its own rest pose* is preserved while
the target rest pose is respected.  Root translation is scaled by the
mean child-bone length ratio; rotation semantics are untouched.

Two independent global accumulation paths are computed: one directly from
the mapped local rotations and one from the global rest-pose delta
``Tg * Sg^-1``.  Their geodesic gap is the global pose deviation; if it
exceeds the declared budget the stage raises ``E_RETARGET_DIVERGE``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .animation import Animation, check_time_monotonic
from .errors import E_RETARGET_DIVERGE
from .hashutil import hash_dict
from .quat import (
    align_sequence,
    angular_distance,
    conjugate,
    multiply,
    normalize,
)
from .skeleton import Skeleton, rest_pose_delta, root_bone_scale

DEFAULT_DIVERGE_TOLERANCE = 1e-6


@dataclass
class RetargetResult:
    animation: Animation
    per_joint_residual: np.ndarray  # (J,) mean angular distance, radians
    frame_joint_residual: np.ndarray  # (F, J) angular distance, radians
    global_deviation: dict[str, float]
    parameter_version: str
    rest_delta: np.ndarray  # (J, 4) global rest-pose delta
    scale: float


def parameter_version(source: Skeleton, target: Skeleton, scale: float) -> str:
    payload = {
        "source": source.parameter_version(),
        "target": target.parameter_version(),
        "scale": float(scale),
    }
    return hash_dict(payload)[:16]


def retarget_frame_block(
    source: Animation,
    source_skel: Skeleton,
    target_skel: Skeleton,
    rest_delta: np.ndarray,
    scale: float,
    frame_range: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Pure function: retarget the half-open frame range ``[start, end)``."""

    start, end = frame_range
    frames = end - start
    joints = source.joint_count
    rotations = np.zeros((frames, joints, 4), dtype=np.float64)
    roots = np.zeros((frames, 3), dtype=np.float64)
    residual = np.zeros((frames, joints), dtype=np.float64)
    deviation = np.zeros((frames, joints), dtype=np.float64)

    src_rest_local = source_skel.rest_rotations
    tgt_rest_local = target_skel.rest_rotations

    for local_frame, frame in enumerate(range(start, end)):
        source_global = np.zeros((joints, 4), dtype=np.float64)
        target_global_expected = np.zeros((joints, 4), dtype=np.float64)
        target_global_actual = np.zeros((joints, 4), dtype=np.float64)
        for joint, parent in enumerate(source_skel.parents):
            src_local = normalize(source.rotations[frame, joint])
            if parent < 0:
                source_global[joint] = src_local
            else:
                source_global[joint] = multiply(source_global[parent], src_local)

            # Preserve motion relative to the joint's own local rest pose:
            # Rt_local = Rt_rest_local * Rs_rest_local^-1 * Rs_local.
            relative_motion = multiply(
                conjugate(src_rest_local[joint]), src_local
            )
            mapped_local = normalize(
                multiply(tgt_rest_local[joint], relative_motion)
            )
            rotations[local_frame, joint] = mapped_local
            residual[local_frame, joint] = angular_distance(
                relative_motion,
                multiply(conjugate(tgt_rest_local[joint]), mapped_local),
            )
            if parent < 0:
                target_global_actual[joint] = mapped_local
                # Hierarchical equivalent of the global rest-pose delta
                # Tg * Sg^-1 applied to the source global rotation.
                target_global_expected[joint] = multiply(
                    rest_delta[joint], source_global[joint]
                )
            else:
                target_global_actual[joint] = multiply(
                    target_global_actual[parent], mapped_local
                )
                target_global_expected[joint] = multiply(
                    target_global_expected[parent],
                    multiply(tgt_rest_local[joint], relative_motion),
                )
            deviation[local_frame, joint] = angular_distance(
                target_global_expected[joint], target_global_actual[joint]
            )

        roots[local_frame] = source.root_translations[frame] * scale

    return {
        "rotations": rotations,
        "root_translations": roots,
        "residual": residual,
        "deviation": deviation,
    }


def retarget(
    source: Animation,
    source_skel: Skeleton,
    target_skel: Skeleton,
    *,
    divergence_tolerance: float = DEFAULT_DIVERGE_TOLERANCE,
    parameter_tag: str | None = None,
) -> RetargetResult:
    if source.joint_count != source_skel.joint_count:
        raise ValueError("animation joint count disagrees with source skeleton")
    check_time_monotonic(source.times)
    scale = root_bone_scale(source_skel, target_skel)
    rest_delta = rest_pose_delta(source_skel, target_skel)
    tag = parameter_tag or parameter_version(source_skel, target_skel, scale)

    block = retarget_frame_block(
        source, source_skel, target_skel, rest_delta, scale, (0, source.frame_count)
    )
    max_deviation = float(np.max(block["deviation"]))
    if max_deviation > divergence_tolerance:
        flat_index = int(np.argmax(block["deviation"]))
        frame, joint = np.unravel_index(flat_index, block["deviation"].shape)
        raise E_RETARGET_DIVERGE(
            "retarget global accumulation paths diverged beyond tolerance",
            {
                "frame": int(frame),
                "joint": int(joint),
                "joint_name": target_skel.names[joint],
                "deviation_rad": max_deviation,
                "tolerance_rad": divergence_tolerance,
                "parameter_version": tag,
            },
        )

    aligned_rotations = align_sequence(block["rotations"])
    animation = Animation(
        joint_names=target_skel.names,
        times=source.times.copy(),
        rotations=aligned_rotations,
        root_translations=block["root_translations"],
        interp=source.interp,
        quat_order="wxyz",
        parameter_version=tag,
    )
    per_joint = np.mean(block["residual"], axis=0)
    report = {
        "max_deviation_rad": max_deviation,
        "mean_deviation_rad": float(np.mean(block["deviation"])),
        "max_residual_rad": float(np.max(block["residual"])),
        "root_scale": scale,
        "tolerance_rad": divergence_tolerance,
    }
    return RetargetResult(
        animation=animation,
        per_joint_residual=per_joint,
        frame_joint_residual=block["residual"],
        global_deviation=report,
        parameter_version=tag,
        rest_delta=rest_delta,
        scale=scale,
    )
