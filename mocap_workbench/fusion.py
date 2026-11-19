"""Least-squares multi-camera fusion and hierarchical missing-joint completion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import BusinessError, ErrorCode, ensure
from .skeleton import Skeleton, rotation_between_batch, safe_unit


@dataclass
class FusionResult:
    pose: np.ndarray
    least_squares_residual: np.ndarray
    completed: np.ndarray
    observed_camera_count: np.ndarray
    bone_length_error: np.ndarray

    def report(self) -> dict[str, Any]:
        observed = self.observed_camera_count > 0
        finite = np.isfinite(self.least_squares_residual)
        return {
            "frames": int(self.pose.shape[0]),
            "joints": int(self.pose.shape[1]),
            "observed_joint_fraction": float(np.mean(observed)) if observed.size else 0.0,
            "completed_joint_fraction": float(np.mean(self.completed)) if self.completed.size else 0.0,
            "max_least_squares_residual_m": float(np.max(self.least_squares_residual[finite]))
            if np.any(finite)
            else 0.0,
            "mean_least_squares_residual_m": float(np.mean(self.least_squares_residual[finite]))
            if np.any(finite)
            else 0.0,
            "max_bone_length_error_m": float(np.max(self.bone_length_error))
            if self.bone_length_error.size
            else 0.0,
            "residual_units": "weighted RMS distance in metres; NaN marks hierarchy-completed joints",
        }


def _frame_parent_orientation(
    pose: np.ndarray,
    resolved: np.ndarray,
    skeleton: Skeleton,
    parent: int,
) -> np.ndarray:
    children = skeleton.children(parent)
    best_rotation: np.ndarray | None = None
    best_length = -1.0
    for child in children:
        if not resolved[child]:
            continue
        rest = skeleton.rest_local[child]
        actual = pose[child] - pose[parent]
        rest_length = float(np.linalg.norm(rest))
        if rest_length > best_length:
            best_rotation = rotation_between_batch(rest[None, :], actual[None, :])[0]
            best_length = rest_length
    if best_rotation is not None:
        return best_rotation
    # No resolved child gives a directional reference; retain the rest orientation.
    return np.eye(3)


def _temporal_estimates(pose: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fill unobserved joints from nearest visible neighbours for continuity.

    These estimates are only used as a direction prior by ``_complete_frame``.
    The hierarchy still determines the parent anchor and the declared bone
    length is restored before the result is accepted.
    """
    estimates = np.array(pose, copy=True)
    frames, joints, _ = estimates.shape
    for joint in range(joints):
        visible_frames = np.flatnonzero(observed[:, joint])
        if visible_frames.size == 0:
            continue
        for frame in np.flatnonzero(~observed[:, joint]):
            left = visible_frames[visible_frames < frame]
            right = visible_frames[visible_frames > frame]
            if left.size and right.size:
                before, after = int(left[-1]), int(right[0])
                ratio = (frame - before) / float(after - before)
                estimates[frame, joint] = (
                    (1.0 - ratio) * pose[before, joint] + ratio * pose[after, joint]
                )
            elif left.size:
                estimates[frame, joint] = pose[int(left[-1]), joint]
            else:
                estimates[frame, joint] = pose[int(right[0]), joint]
    return estimates


def _complete_frame(
    pose: np.ndarray,
    observed: np.ndarray,
    skeleton: Skeleton,
    temporal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.array(pose, copy=True)
    resolved = np.asarray(observed, dtype=bool).copy()
    completed = np.zeros((skeleton.joint_count,), dtype=bool)
    if not np.any(resolved):
        raise BusinessError(ErrorCode.E_OBS_MISSING, "no visible joint is available in a frame")

    if not resolved[0]:
        if temporal is not None and np.all(np.isfinite(temporal[0])):
            result[0] = temporal[0]
        else:
            known = np.flatnonzero(resolved)
            # Translation-only anchoring is deliberately used before directions are known.
            offsets = skeleton.rest_world[known] - skeleton.rest_world[0]
            candidates = result[known] - offsets
            result[0] = np.mean(candidates, axis=0)
        resolved[0] = True
        completed[0] = True

    for joint in range(1, skeleton.joint_count):
        if resolved[joint]:
            continue
        parent = int(skeleton.parents[joint])
        if not resolved[parent]:
            continue
        prior = None
        if temporal is not None:
            offset = temporal[joint] - temporal[parent]
            length = float(np.linalg.norm(offset))
            if np.isfinite(length) and length > 1e-12:
                prior = offset / length * skeleton.bone_lengths[joint]
        if prior is None:
            orientation = _frame_parent_orientation(result, resolved, skeleton, parent)
            prior = orientation @ skeleton.rest_local[joint]
        result[joint] = result[parent] + prior
        resolved[joint] = True
        completed[joint] = True

    if not np.all(resolved):
        missing = [skeleton.joints[i] for i in np.flatnonzero(~resolved)]
        raise BusinessError(
            ErrorCode.E_OBS_MISSING,
            "hierarchy completion could not reach every joint",
            missing_joints=missing,
        )

    for joint in range(1, skeleton.joint_count):
        if not completed[joint]:
            continue
        parent = int(skeleton.parents[joint])
        offset = result[joint] - result[parent]
        length = np.linalg.norm(offset)
        if length > 1e-12:
            result[joint] = result[parent] + (offset / length) * skeleton.bone_lengths[joint]
    return result, completed


def fuse_observations(
    skeleton: Skeleton,
    observations: np.ndarray,
    visibility: np.ndarray,
    sigma: np.ndarray,
    visibility_override: np.ndarray | None = None,
    sigma_override: np.ndarray | None = None,
) -> FusionResult:
    obs = np.asarray(observations, dtype=np.float64)
    vis = np.asarray(visibility, dtype=bool)
    scale = np.asarray(sigma, dtype=np.float64)
    if visibility_override is not None:
        vis = np.asarray(visibility_override, dtype=bool)
    if sigma_override is not None:
        scale = np.asarray(sigma_override, dtype=np.float64)
    if obs.ndim != 4 or obs.shape[-1] != 3:
        raise BusinessError(ErrorCode.E_INPUT, "observations must have shape (frames, joints, cameras, 3)")
    if vis.shape != obs.shape[:3] or scale.shape != obs.shape[:3]:
        raise BusinessError(ErrorCode.E_INPUT, "visibility and sigma shapes must match observations")
    if obs.shape[1] != skeleton.joint_count:
        raise BusinessError(ErrorCode.E_INPUT, "observation joint count does not match skeleton")
    if not np.all(np.isfinite(scale[vis])):
        raise BusinessError(ErrorCode.E_INPUT, "visible observations must have finite noise scales")
    if not np.all(np.isfinite(obs[vis])):
        raise BusinessError(ErrorCode.E_INPUT, "visible observations must be finite")
    if np.any(scale[vis] <= 0.0):
        raise BusinessError(ErrorCode.E_INPUT, "visible observations must have positive noise scales")

    frames, joints, cameras, _ = obs.shape
    usable = vis & np.isfinite(obs).all(axis=-1) & np.isfinite(scale) & (scale > 0.0)
    if not np.any(usable):
        raise BusinessError(ErrorCode.E_OBS_MISSING, "no visible observation exists")

    weights = np.zeros((frames, joints, cameras), dtype=np.float64)
    weights[usable] = 1.0 / np.square(scale[usable])
    weight_sum = np.sum(weights, axis=2)
    observed = weight_sum > 0.0
    pose = np.full((frames, joints, 3), np.nan, dtype=np.float64)
    safe_weight = np.where(observed, weight_sum, 1.0)
    pose[observed] = np.sum(weights[:, :, :, None] * np.where(usable[:, :, :, None], obs, 0.0), axis=2)[observed] / safe_weight[observed, None]

    residual = np.full((frames, joints), np.nan, dtype=np.float64)
    if np.any(observed):
        difference = np.where(usable[:, :, :, None], obs - pose[:, :, None, :], 0.0)
        weighted_sq = np.sum(weights * np.sum(difference * difference, axis=-1), axis=2)
        residual[observed] = np.sqrt(weighted_sq[observed] / weight_sum[observed])

    completed = np.zeros((frames, joints), dtype=bool)
    temporal_estimates = _temporal_estimates(pose, observed)
    for frame in range(frames):
        if not np.all(observed[frame]):
            frame_pose, frame_completed = _complete_frame(
                pose[frame], observed[frame], skeleton, temporal_estimates[frame]
            )
            pose[frame] = frame_pose
            completed[frame] = frame_completed

    bone_error = np.zeros((frames, joints), dtype=np.float64)
    for joint in range(1, joints):
        parent = int(skeleton.parents[joint])
        length = np.linalg.norm(pose[:, joint] - pose[:, parent], axis=-1)
        bone_error[:, joint] = np.abs(length - skeleton.bone_lengths[joint])

    # Restore declared source bone lengths after per-joint LS fusion. This keeps
    # retargeting invertible while retaining the LS estimates as the data term.
    from .constraints import project_bone_lengths

    project_bone_lengths(pose, skeleton)
    for joint in range(1, joints):
        parent = int(skeleton.parents[joint])
        length = np.linalg.norm(pose[:, joint] - pose[:, parent], axis=-1)
        bone_error[:, joint] = np.abs(length - skeleton.bone_lengths[joint])

    return FusionResult(
        pose=pose,
        least_squares_residual=residual,
        completed=completed,
        observed_camera_count=np.sum(vis, axis=2),
        bone_length_error=bone_error,
    )


def visible_camera_sets(visibility: np.ndarray) -> list[list[list[int]]]:
    result: list[list[list[int]]] = []
    for frame in np.asarray(visibility, dtype=bool):
        result.append([np.flatnonzero(row).astype(int).tolist() for row in frame])
    return result
