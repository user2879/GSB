"""Pose constraint model and deterministic vectorised projection solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .errors import BusinessError, ErrorCode
from .skeleton import EPS, Skeleton, angle_between_batch, safe_unit


@dataclass
class JointLimit:
    joint: str
    min_deg: float
    max_deg: float


@dataclass
class FootContact:
    joint: str
    point: tuple[float, float, float]
    period: int = 90
    phase: int = 0
    duration: int = 14

    def active(self, frames: int) -> np.ndarray:
        if self.period <= 0 or self.duration <= 0:
            return np.zeros((frames,), dtype=bool)
        indexes = np.arange(frames, dtype=np.int64)
        return ((indexes - int(self.phase)) % int(self.period)) < int(self.duration)


@dataclass
class RootHeight:
    min_z: float
    max_z: float


@dataclass
class SymmetryConstraint:
    pairs: tuple[tuple[str, str], ...]
    metric: str
    tolerance: float


@dataclass
class ConstraintSet:
    joint_limits: list[JointLimit] = field(default_factory=list)
    foot_contacts: list[FootContact] = field(default_factory=list)
    root_height: RootHeight | None = None
    symmetries: list[SymmetryConstraint] = field(default_factory=list)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "joint_limits": [vars(item) for item in self.joint_limits],
            "foot_contacts": [
                {
                    "joint": item.joint,
                    "point": list(item.point),
                    "period": item.period,
                    "phase": item.phase,
                    "duration": item.duration,
                }
                for item in self.foot_contacts
            ],
            "root_height": None if self.root_height is None else vars(self.root_height),
            "symmetries": [
                {
                    "pairs": [list(pair) for pair in item.pairs],
                    "metric": item.metric,
                    "tolerance": item.tolerance,
                }
                for item in self.symmetries
            ],
        }

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> "ConstraintSet":
        root_payload = payload.get("root_height")
        return cls(
            joint_limits=[JointLimit(**item) for item in payload.get("joint_limits", [])],
            foot_contacts=[
                FootContact(
                    joint=str(item["joint"]),
                    point=tuple(float(x) for x in item["point"]),
                    period=int(item.get("period", 90)),
                    phase=int(item.get("phase", 0)),
                    duration=int(item.get("duration", 14)),
                )
                for item in payload.get("foot_contacts", [])
            ],
            root_height=None if root_payload is None else RootHeight(**root_payload),
            symmetries=[
                SymmetryConstraint(
                    pairs=tuple(tuple(pair) for pair in item["pairs"]),
                    metric=str(item["metric"]),
                    tolerance=float(item["tolerance"]),
                )
                for item in payload.get("symmetries", [])
            ],
        )


def default_constraints(target: Skeleton) -> ConstraintSet:
    lookup = target.index
    left_ankle = target.rest_world[lookup["l_ankle"]].copy()
    right_ankle = target.rest_world[lookup["r_ankle"]].copy()
    # Leave a small kinematic margin so the fixed contact remains reachable
    # under root translation and noise correction.
    left_ankle[2] += 0.080
    right_ankle[2] += 0.080
    root_z = float(target.rest_world[lookup["hips"], 2])
    return ConstraintSet(
        joint_limits=[
            JointLimit("l_elbow", 25.0, 135.0),
            JointLimit("r_elbow", 25.0, 135.0),
            JointLimit("l_knee", 15.0, 170.0),
            JointLimit("r_knee", 15.0, 170.0),
        ],
        foot_contacts=[
            FootContact("l_ankle", tuple(float(x) for x in left_ankle), period=90, phase=0, duration=14),
            FootContact("r_ankle", tuple(float(x) for x in right_ankle), period=90, phase=45, duration=14),
        ],
        root_height=RootHeight(root_z - 0.015, root_z + 0.030),
        symmetries=[
            SymmetryConstraint((("l_elbow", "r_elbow"), ("l_knee", "r_knee")), "angle_deg", 10.0),
            SymmetryConstraint((("l_shoulder", "r_shoulder"), ("l_hip", "r_hip")), "position", 0.035),
        ],
    )


def _joint_axis_geometry(
    pose: np.ndarray, skeleton: Skeleton, joint: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parent = int(skeleton.parents[joint])
    children = skeleton.children(joint)
    if not children:
        raise BusinessError(ErrorCode.E_INPUT, "angle constraint joint has no child", joint=skeleton.joints[joint])
    child = children[0]
    u = pose[:, joint] - pose[:, parent]
    v = pose[:, child] - pose[:, joint]
    angle = angle_between_batch(u, v)
    axis = np.cross(safe_unit(u), safe_unit(v))
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    fallback = np.cross(safe_unit(u), np.asarray((0.0, 0.0, 1.0)))
    fallback_norm = np.linalg.norm(fallback, axis=-1, keepdims=True)
    fallback2 = np.cross(safe_unit(u), np.asarray((0.0, 1.0, 0.0)))
    fallback = np.where(fallback_norm > EPS, fallback, fallback2)
    axis = np.where(axis_norm > EPS, axis / np.maximum(axis_norm, EPS), safe_unit(fallback))
    return u, v, angle, axis


def _rotate_joint_descendants(
    pose: np.ndarray,
    skeleton: Skeleton,
    joint: int,
    angles: np.ndarray,
    axis: np.ndarray,
    mask: np.ndarray,
) -> None:
    from .skeleton import rotations_about_axes

    active = np.asarray(mask, dtype=bool)
    if not np.any(active):
        return
    descendants = skeleton.subtree_indices(joint)
    if descendants.size == 0:
        return
    rot = rotations_about_axes(axis[active], angles[active])
    pivot = pose[active, joint][:, None, :]
    local = pose[np.ix_(active, descendants)] - pivot
    pose[np.ix_(active, descendants)] = np.einsum("nij,nkj->nki", rot, local) + pivot


def project_root_height(pose: np.ndarray, constraints: ConstraintSet) -> None:
    if constraints.root_height is None:
        return
    current = pose[:, 0, 2]
    target = np.clip(current, constraints.root_height.min_z, constraints.root_height.max_z)
    delta = target - current
    active = np.abs(delta) > 1e-15
    if np.any(active):
        pose[active] += delta[active, None, None]


def project_joint_limits(pose: np.ndarray, skeleton: Skeleton, constraints: ConstraintSet) -> None:
    lookup = skeleton.index
    for limit in constraints.joint_limits:
        if limit.joint not in lookup:
            raise BusinessError(ErrorCode.E_INPUT, "unknown joint in limit", joint=limit.joint)
        joint = lookup[limit.joint]
        _, _, angle, axis = _joint_axis_geometry(pose, skeleton, joint)
        lo = np.deg2rad(limit.min_deg)
        hi = np.deg2rad(limit.max_deg)
        upper = angle > hi
        lower = angle < lo
        delta = np.zeros_like(angle)
        delta[upper] = hi - angle[upper]
        delta[lower] = lo - angle[lower]
        mask = upper | lower
        _rotate_joint_descendants(pose, skeleton, joint, delta, axis, mask)


def project_foot_contacts(
    pose: np.ndarray,
    skeleton: Skeleton,
    constraints: ConstraintSet,
    frame_indices: np.ndarray | None = None,
) -> tuple[list[str], list[np.ndarray]]:
    """Place active ankles exactly on contacts using a two-bone IK branch."""
    lookup = skeleton.index
    conflicts: list[str] = []
    unreachable_masks: list[np.ndarray] = []
    for contact in constraints.foot_contacts:
        if contact.joint not in lookup:
            raise BusinessError(ErrorCode.E_INPUT, "unknown foot-contact joint", joint=contact.joint)
        ankle = lookup[contact.joint]
        knee = int(skeleton.parents[ankle])
        if knee <= 0:
            raise BusinessError(ErrorCode.E_INPUT, "foot contact requires a two-bone leg", joint=contact.joint)
        hip = int(skeleton.parents[knee])
        active = _active_contact_mask(contact, pose.shape[0], frame_indices)
        if not np.any(active):
            continue
        h = pose[active, hip]
        k = pose[active, knee]
        target_ankle = np.broadcast_to(np.asarray(contact.point, dtype=np.float64), (int(active.sum()), 3))
        d_vec = target_ankle - h
        d = np.linalg.norm(d_vec, axis=-1)
        l1 = skeleton.bone_lengths[knee]
        l2 = skeleton.bone_lengths[ankle]
        reachable = (d <= l1 + l2 + 1e-10) & (d >= abs(l1 - l2) - 1e-10) & (d > 1e-10)
        if not np.all(reachable):
            bad = np.zeros_like(active)
            bad[np.flatnonzero(active)[~reachable]] = True
            unreachable_masks.append(bad)
            conflicts.append(f"foot_contact:{contact.joint}")
            continue
        d_hat = d_vec / d[:, None]
        along = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d)
        height_sq = np.maximum(l1 * l1 - along * along, 0.0)
        height = np.sqrt(height_sq)
        base = h + along[:, None] * d_hat
        rest_hip = skeleton.rest_world[hip]
        rest_knee = skeleton.rest_world[knee]
        rest_ankle = skeleton.rest_world[ankle]
        rest_line = safe_unit((rest_ankle - rest_hip)[None, :])[0]
        rest_bend = rest_knee - rest_hip - np.dot(rest_knee - rest_hip, rest_line) * rest_line
        if np.linalg.norm(rest_bend) <= EPS:
            rest_bend = np.asarray((0.0, 1.0, 0.0))
        reference = np.broadcast_to(safe_unit(rest_bend)[None, :], d_hat.shape)
        perp = reference - np.sum(reference * d_hat, axis=-1, keepdims=True) * d_hat
        perp_norm = np.linalg.norm(perp, axis=-1, keepdims=True)
        current_perp = k - base
        current_perp -= np.sum(current_perp * d_hat, axis=-1, keepdims=True) * d_hat
        current_norm = np.linalg.norm(current_perp, axis=-1, keepdims=True)
        fallback = np.cross(d_hat, np.asarray((0.0, 0.0, 1.0)))
        fallback_norm = np.linalg.norm(fallback, axis=-1, keepdims=True)
        fallback2 = np.cross(d_hat, np.asarray((0.0, 1.0, 0.0)))
        fallback = np.where(fallback_norm > EPS, fallback, fallback2)
        perp_hat = np.where(
            perp_norm > EPS,
            perp / np.maximum(perp_norm, EPS),
            np.where(current_norm > EPS, current_perp / np.maximum(current_norm, EPS), safe_unit(fallback)),
        )
        new_k = base + height[:, None] * perp_hat
        pose[np.flatnonzero(active), knee] = new_k
        pose[np.flatnonzero(active), ankle] = target_ankle
    return conflicts, unreachable_masks


def _project_angle_pair(
    pose: np.ndarray,
    skeleton: Skeleton,
    left: int,
    right: int,
    target_left: np.ndarray,
    target_right: np.ndarray,
    mask: np.ndarray,
) -> None:
    for joint, target in ((left, target_left), (right, target_right)):
        _, _, angle, axis = _joint_axis_geometry(pose, skeleton, joint)
        delta = target - angle
        _rotate_joint_descendants(pose, skeleton, joint, delta, axis, mask)


def _active_contact_mask(
    contact: FootContact, local_frames: int, frame_indices: np.ndarray | None
) -> np.ndarray:
    if frame_indices is None:
        return contact.active(local_frames)
    indexes = np.asarray(frame_indices, dtype=np.int64)
    if indexes.shape != (local_frames,):
        raise BusinessError(ErrorCode.E_INPUT, "frame_indices must match pose frame count")
    if indexes.size == 0:
        return np.zeros((0,), dtype=bool)
    return contact.active(int(np.max(indexes)) + 1)[indexes]


def project_symmetry(
    pose: np.ndarray,
    skeleton: Skeleton,
    constraints: ConstraintSet,
    frame_indices: np.ndarray | None = None,
) -> None:
    lookup = skeleton.index
    contact_active = [
        _active_contact_mask(contact, pose.shape[0], frame_indices)
        for contact in constraints.foot_contacts
    ]

    def anchored_by_contact(joint: int) -> np.ndarray:
        anchored = np.zeros((pose.shape[0],), dtype=bool)
        descendants = skeleton.subtree_indices(joint, include_self=True)
        descendant_set = set(int(item) for item in descendants)
        for contact, active in zip(constraints.foot_contacts, contact_active):
            ankle = lookup[contact.joint]
            if ankle in descendant_set:
                anchored |= active
        return anchored

    for symmetry in constraints.symmetries:
        pair_indexes = []
        for left_name, right_name in symmetry.pairs:
            if left_name not in lookup or right_name not in lookup:
                raise BusinessError(
                    ErrorCode.E_INPUT, "unknown symmetry pair", pair=[left_name, right_name]
                )
            pair_indexes.append((lookup[left_name], lookup[right_name]))

        if symmetry.metric == "angle_deg":
            tolerance = np.deg2rad(float(symmetry.tolerance))
            for left, right in pair_indexes:
                _, _, left_angle, _ = _joint_axis_geometry(pose, skeleton, left)
                _, _, right_angle, _ = _joint_axis_geometry(pose, skeleton, right)
                difference = left_angle - right_angle
                mask = np.abs(difference) > tolerance + 1e-15
                if not np.any(mask):
                    continue
                midpoint = 0.5 * (left_angle + right_angle)
                target_left = np.clip(left_angle, midpoint - tolerance / 2.0, midpoint + tolerance / 2.0)
                target_right = np.clip(right_angle, midpoint - tolerance / 2.0, midpoint + tolerance / 2.0)
                left_fixed = anchored_by_contact(left)
                right_fixed = anchored_by_contact(right)
                left_only = mask & left_fixed & ~right_fixed
                right_only = mask & right_fixed & ~left_fixed
                if np.any(left_only):
                    target_left[left_only] = left_angle[left_only]
                    target_right[left_only] = np.where(
                        left_angle[left_only] > right_angle[left_only],
                        left_angle[left_only] - tolerance,
                        left_angle[left_only] + tolerance,
                    )
                if np.any(right_only):
                    target_right[right_only] = right_angle[right_only]
                    target_left[right_only] = np.where(
                        right_angle[right_only] > left_angle[right_only],
                        right_angle[right_only] - tolerance,
                        right_angle[right_only] + tolerance,
                    )
                _project_angle_pair(pose, skeleton, left, right, target_left, target_right, mask)
        elif symmetry.metric == "position":
            tolerance = float(symmetry.tolerance)
            for left, right in pair_indexes:
                left_parent = int(skeleton.parents[left])
                right_parent = int(skeleton.parents[right])
                origin = pose[:, left_parent].copy() if left_parent == right_parent else pose[:, 0, :].copy()
                mirror = pose[:, right] - origin
                mirror[:, 0] *= -1.0
                mirror += origin
                residual = pose[:, left] - mirror
                residual_norm = np.linalg.norm(residual, axis=-1)
                correction_scale = np.maximum(1.0 - tolerance / np.maximum(residual_norm, 1e-15), 0.0)
                correction = residual * correction_scale[:, None]
                mask = residual_norm > tolerance + 1e-15
                if np.any(mask):
                    same_radius = (
                        left_parent == right_parent
                        and abs(skeleton.bone_lengths[left] - skeleton.bone_lengths[right]) < 1e-6
                    )
                    if same_radius and left_parent >= 0:
                        parent_position = pose[mask, left_parent]
                        left_ray = pose[mask, left] - parent_position
                        right_ray = pose[mask, right] - parent_position
                        right_ray[:, 0] *= -1.0
                        averaged = safe_unit(left_ray + right_ray)
                        radius = skeleton.bone_lengths[left]
                        new_left = parent_position + averaged * radius
                        new_right_local = averaged.copy()
                        new_right_local[:, 0] *= -1.0
                        new_right = parent_position + new_right_local * radius
                        pose[mask, left] = new_left
                        pose[mask, right] = new_right
                    else:
                        split = correction * 0.5
                        pose[mask, left] -= split[mask]
                        delta_right = split[mask].copy()
                        delta_right[:, 0] *= -1.0
                        pose[mask, right] += delta_right
        else:
            raise BusinessError(
                ErrorCode.E_INPUT, "unsupported symmetry metric", metric=symmetry.metric
            )


def project_bone_lengths(pose: np.ndarray, skeleton: Skeleton) -> None:
    """Restore exact target bone lengths while moving whole child subtrees."""
    for node in range(1, skeleton.joint_count):
        parent = int(skeleton.parents[node])
        offset = pose[:, node] - pose[:, parent]
        length = np.linalg.norm(offset, axis=-1)
        valid = length > EPS
        if not np.any(valid):
            continue
        direction = np.zeros_like(offset)
        direction[valid] = offset[valid] / length[valid, None]
        delta = direction * (skeleton.bone_lengths[node] - length)[:, None]
        active = np.abs(skeleton.bone_lengths[node] - length) > 1e-12
        if not np.any(active):
            continue
        subtree = skeleton.subtree_indices(node, include_self=True)
        pose[np.ix_(active, subtree)] += delta[active, None, :]


def constraint_names(skeleton: Skeleton, constraints: ConstraintSet) -> list[str]:
    names = [f"joint_limit:{item.joint}" for item in constraints.joint_limits]
    names.extend(f"foot_contact:{item.joint}" for item in constraints.foot_contacts)
    if constraints.root_height is not None:
        names.append("root_height")
    for symmetry in constraints.symmetries:
        for left, right in symmetry.pairs:
            names.append(f"symmetry_{symmetry.metric}:{left}/{right}")
    return names


def evaluate_constraints(
    pose: np.ndarray,
    skeleton: Skeleton,
    constraints: ConstraintSet,
    frame_indices: np.ndarray | None = None,
) -> np.ndarray:
    frames = pose.shape[0]
    values: list[np.ndarray] = []
    lookup = skeleton.index
    for limit in constraints.joint_limits:
        joint = lookup[limit.joint]
        _, _, angle, _ = _joint_axis_geometry(pose, skeleton, joint)
        angle_deg = np.rad2deg(angle)
        values.append(np.maximum(np.maximum(limit.min_deg - angle_deg, angle_deg - limit.max_deg), 0.0))
    for contact in constraints.foot_contacts:
        ankle = lookup[contact.joint]
        active = _active_contact_mask(contact, frames, frame_indices)
        difference = np.linalg.norm(pose[:, ankle] - np.asarray(contact.point), axis=-1)
        values.append(np.where(active, difference, 0.0))
    if constraints.root_height is not None:
        z = pose[:, 0, 2]
        values.append(
            np.maximum(
                np.maximum(constraints.root_height.min_z - z, z - constraints.root_height.max_z), 0.0
            )
        )
    for symmetry in constraints.symmetries:
        for left_name, right_name in symmetry.pairs:
            left = lookup[left_name]
            right = lookup[right_name]
            if symmetry.metric == "angle_deg":
                _, _, left_angle, _ = _joint_axis_geometry(pose, skeleton, left)
                _, _, right_angle, _ = _joint_axis_geometry(pose, skeleton, right)
                excess = np.abs(np.rad2deg(left_angle - right_angle)) - symmetry.tolerance
                values.append(np.maximum(excess, 0.0))
            elif symmetry.metric == "position":
                left_parent = int(skeleton.parents[left])
                right_parent = int(skeleton.parents[right])
                origin = pose[:, left_parent] if left_parent == right_parent else pose[:, 0]
                mirrored = pose[:, right] - origin
                mirrored[:, 0] *= -1.0
                mirrored += origin
                residual = pose[:, left] - mirrored
                excess = np.linalg.norm(residual, axis=-1) - symmetry.tolerance
                values.append(np.maximum(excess, 0.0))
            else:
                raise BusinessError(ErrorCode.E_INPUT, "unsupported symmetry metric", metric=symmetry.metric)
    if not values:
        return np.zeros((frames, 0), dtype=np.float64)
    return np.stack(values, axis=1)


def solve_constraints(
    retargeted_pose: np.ndarray,
    skeleton: Skeleton,
    constraints: ConstraintSet,
    iterations: int = 18,
    feasibility_tolerance: float = 1e-4,
    frame_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    if frame_indices is not None:
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if frame_indices.shape != (retargeted_pose.shape[0],):
            raise BusinessError(ErrorCode.E_INPUT, "frame_indices must match the pose slice")
    anchor = np.asarray(retargeted_pose, dtype=np.float64)
    pose = anchor.copy()
    conflict_notes: set[str] = set()
    unreachable_masks: list[np.ndarray] = []
    for _ in range(max(1, iterations)):
        project_joint_limits(pose, skeleton, constraints)
        project_root_height(pose, constraints)
        conflicts, masks = project_foot_contacts(pose, skeleton, constraints, frame_indices)
        conflict_notes.update(conflicts)
        unreachable_masks.extend(masks)
        project_symmetry(pose, skeleton, constraints, frame_indices)
        project_bone_lengths(pose, skeleton)
        # Contacts are hard constraints; restore them once more after symmetry/length work.
        conflicts, masks = project_foot_contacts(pose, skeleton, constraints, frame_indices)
        conflict_notes.update(conflicts)
        unreachable_masks.extend(masks)

    residual = evaluate_constraints(pose, skeleton, constraints, frame_indices)
    names = constraint_names(skeleton, constraints)
    bone_residual = np.linalg.norm(pose[:, 1:] - pose[:, skeleton.parents[1:]], axis=-1)
    bone_error = np.abs(bone_residual - skeleton.bone_lengths[None, 1:])
    infeasible_frames = np.flatnonzero(np.any(residual > feasibility_tolerance, axis=1))
    if infeasible_frames.size:
        violations: list[dict[str, Any]] = []
        for frame in infeasible_frames[:50]:
            active = [names[i] for i in np.flatnonzero(residual[frame] > feasibility_tolerance)]
            violations.append({"frame": int(frame), "constraints": active})
        raise BusinessError(
            ErrorCode.E_CONSTRAINT_UNSAT,
            "pose constraints are infeasible; no tolerance was relaxed",
            infeasible_frame_count=int(infeasible_frames.size),
            first_infeasible_frames=infeasible_frames[:20].tolist(),
            violation_examples=violations,
            conflict_constraints=sorted(conflict_notes),
            unreachable_contact_frames=[
                int(np.flatnonzero(mask)[0]) for mask in unreachable_masks if np.any(mask)
            ][:20],
        )
    if np.any(bone_error > 2e-4):
        frame, local = np.unravel_index(np.argmax(bone_error), bone_error.shape)
        raise BusinessError(
            ErrorCode.E_CONSTRAINT_UNSAT,
            "constraint projection could not retain target bone lengths",
            frame=int(frame),
            joint=skeleton.joints[int(local) + 1],
            bone_error=float(bone_error[frame, local]),
            conflict_constraints=sorted(conflict_notes) + ["bone_length"],
        )

    deviation = np.linalg.norm(pose - anchor, axis=-1)
    return {
        "pose": pose,
        "residuals": residual,
        "residual_names": names,
        "iterations": int(iterations),
        "feasible": True,
        "max_deviation_m": float(np.max(deviation)) if deviation.size else 0.0,
        "mean_deviation_m": float(np.mean(deviation)) if deviation.size else 0.0,
        "max_bone_length_error_m": float(np.max(bone_error)) if bone_error.size else 0.0,
    }
