"""Cross-strategy pose and constraint-residual consistency checks."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


def _distance_to_boundary(frame: int, boundaries: list[int]) -> int | None:
    if not boundaries:
        return None
    return int(min(abs(frame - candidate) for candidate in boundaries))


def compare_pose_results(
    results: dict[str, dict[str, Any]],
    position_tolerance_m: float = 0.05,
    residual_tolerance: float = 0.02,
    overlap_frames: int = 48,
) -> dict[str, Any]:
    pair_reports: list[dict[str, Any]] = []
    all_consistent = True
    for left_name, right_name in combinations(sorted(results), 2):
        left = results[left_name]
        right = results[right_name]
        left_pose = np.asarray(left["pose"])
        right_pose = np.asarray(right["pose"])
        if left_pose.shape != right_pose.shape:
            pair_reports.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "consistent": False,
                    "reason": "pose shapes differ",
                }
            )
            all_consistent = False
            continue
        joint_distance = np.linalg.norm(left_pose - right_pose, axis=-1)
        frame_max = np.max(joint_distance, axis=1)
        max_frame = int(np.argmax(frame_max))
        first_divergent = np.flatnonzero(frame_max > position_tolerance_m)
        first_frame = int(first_divergent[0]) if first_divergent.size else None
        participants: list[str] = []
        if first_frame is not None:
            joint_names = list(left.get("joint_names", [str(i) for i in range(left_pose.shape[1])]))
            participants = [
                joint_names[index]
                for index in np.flatnonzero(joint_distance[first_frame] > position_tolerance_m)
            ]

        left_residual = np.asarray(left.get("constraint_residuals"))
        right_residual = np.asarray(right.get("constraint_residuals"))
        names = list(left.get("constraint_residual_names", []))
        if left_residual.shape != right_residual.shape:
            residual_difference = np.asarray([[float("inf")]])
            residual_max = float("inf")
            residual_frame = 0
            residual_names: list[str] = ["shape_mismatch"]
        else:
            residual_difference = np.abs(left_residual - right_residual)
            if residual_difference.size:
                residual_frame, residual_index = np.unravel_index(
                    np.argmax(residual_difference), residual_difference.shape
                )
                residual_max = float(residual_difference[residual_frame, residual_index])
                residual_names = [names[int(residual_index)]] if names else [str(int(residual_index))]
            else:
                residual_max = 0.0
                residual_frame = 0
                residual_names = []

        boundaries = sorted(
            set(left.get("boundary_frames", [])) | set(right.get("boundary_frames", []))
        )
        location = None
        if first_frame is not None and boundaries:
            distance = _distance_to_boundary(first_frame, boundaries)
            if distance is not None and distance <= overlap_frames:
                location = {
                    "kind": "window_or_anchor_boundary_region",
                    "nearest_boundary_frame": min(
                        boundaries, key=lambda candidate: abs(candidate - first_frame)
                    ),
                    "distance_frames": distance,
                }
            else:
                location = {
                    "kind": "within_window_or_anchor_interval",
                    "nearest_boundary_frame": min(
                        boundaries, key=lambda candidate: abs(candidate - first_frame)
                    ),
                    "distance_frames": distance,
                }

        consistent = bool(
            np.max(frame_max) <= position_tolerance_m and residual_max <= residual_tolerance
        )
        all_consistent = all_consistent and consistent
        pair_reports.append(
            {
                "left": left_name,
                "right": right_name,
                "consistent": consistent,
                "max_joint_position_difference_m": float(np.max(frame_max)),
                "max_difference_frame": max_frame,
                "first_divergence_frame": first_frame,
                "first_divergence_joints": participants,
                "max_constraint_residual_difference": residual_max,
                "max_constraint_residual_difference_frame": int(residual_frame),
                "max_constraint_residual_difference_constraints": residual_names,
                "divergence_location": location,
                "position_tolerance_m": float(position_tolerance_m),
                "constraint_residual_tolerance": float(residual_tolerance),
            }
        )
    return {
        "consistent": all_consistent,
        "strategies_compared": sorted(results),
        "pairs": pair_reports,
    }
