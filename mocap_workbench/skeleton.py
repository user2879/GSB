"""Skeleton definitions and deterministic rigid-body rotation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .errors import BusinessError, ErrorCode


EPS = 1e-12

JOINT_NAMES = (
    "hips",
    "spine",
    "chest",
    "neck",
    "head",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
    "l_hip",
    "l_knee",
    "l_ankle",
    "r_hip",
    "r_knee",
    "r_ankle",
)

PARENTS = np.asarray((-1, 0, 1, 2, 3, 2, 5, 6, 2, 8, 9, 0, 11, 12, 0, 14, 15), dtype=np.int64)

LEFT_RIGHT_PAIRS = (
    ("l_shoulder", "r_shoulder"),
    ("l_elbow", "r_elbow"),
    ("l_wrist", "r_wrist"),
    ("l_hip", "r_hip"),
    ("l_knee", "r_knee"),
    ("l_ankle", "r_ankle"),
)


def source_rest_positions() -> np.ndarray:
    return np.asarray(
        [
            (0.00, 0.00, 1.00),
            (0.00, 0.00, 1.13),
            (0.00, 0.00, 1.35),
            (0.00, 0.00, 1.54),
            (0.00, 0.00, 1.68),
            (0.18, 0.00, 1.48),
            (0.46, 0.00, 1.48),
            (0.71, 0.00, 1.48),
            (-0.18, 0.00, 1.48),
            (-0.46, 0.00, 1.48),
            (-0.71, 0.00, 1.48),
            (0.12, 0.00, 1.00),
            (0.12, 0.00, 0.58),
            (0.12, 0.00, 0.18),
            (-0.12, 0.00, 1.00),
            (-0.12, 0.00, 0.58),
            (-0.12, 0.00, 0.18),
        ],
        dtype=np.float64,
    )


def target_rest_positions() -> np.ndarray:
    """A taller target with shortened, lowered arms and longer legs."""
    rest = source_rest_positions()
    out = rest.copy()
    out[:, 2] *= 1.07
    out[:, 0] *= 0.96
    # Lower the target arms by ~24 degrees relative to the source T-pose.
    out[5] = (0.174, 0.00, 1.585)
    out[6] = (0.174 + 0.285 * np.cos(np.deg2rad(24)), 0.00, 1.585 - 0.285 * np.sin(np.deg2rad(24)))
    out[7] = (out[6, 0] + 0.245 * np.cos(np.deg2rad(24)), 0.00, out[6, 2] - 0.245 * np.sin(np.deg2rad(24)))
    out[8] = (-out[5, 0], 0.00, out[5, 2])
    out[9] = (-out[6, 0], 0.00, out[6, 2])
    out[10] = (-out[7, 0], 0.00, out[7, 2])
    # Slightly longer legs, with a small sagittal knee offset.
    out[11] = (0.119, 0.00, 1.07)
    out[12] = (0.115, 0.018, 0.626)
    out[13] = (0.115, -0.012, 0.193)
    out[14] = (-0.119, 0.00, 1.07)
    out[15] = (-0.115, 0.018, 0.626)
    out[16] = (-0.115, -0.012, 0.193)
    return out


def rest_height(rest: np.ndarray) -> float:
    z = np.asarray(rest)[:, 2]
    return float(np.max(z) - np.min(z))


@dataclass(frozen=True)
class Skeleton:
    name: str
    joints: tuple[str, ...]
    parents: np.ndarray
    rest_local: np.ndarray
    rest_world: np.ndarray
    bone_lengths: np.ndarray

    @classmethod
    def from_positions(
        cls, name: str, joints: Sequence[str], parents: Sequence[int], rest_world: np.ndarray
    ) -> "Skeleton":
        joints_tuple = tuple(str(j) for j in joints)
        parent_array = np.asarray(parents, dtype=np.int64)
        rest = np.asarray(rest_world, dtype=np.float64)
        if len(joints_tuple) != len(parent_array) or rest.shape != (len(joints_tuple), 3):
            raise BusinessError(ErrorCode.E_INPUT, "skeleton shape is inconsistent")
        local = np.zeros_like(rest)
        lengths = np.zeros((len(joints_tuple),), dtype=np.float64)
        for idx, parent in enumerate(parent_array):
            if idx == 0:
                if parent != -1:
                    raise BusinessError(ErrorCode.E_INPUT, "root joint must have parent -1")
                continue
            if parent < 0 or parent >= idx:
                raise BusinessError(
                    ErrorCode.E_INPUT,
                    "joints must be topologically ordered with parents before children",
                    joint=joints_tuple[idx],
                )
            local[idx] = rest[idx] - rest[parent]
            lengths[idx] = float(np.linalg.norm(local[idx]))
            if lengths[idx] <= 1e-9:
                raise BusinessError(ErrorCode.E_INPUT, "zero-length bone", joint=joints_tuple[idx])
        return cls(
            name=name,
            joints=joints_tuple,
            parents=parent_array,
            rest_local=local,
            rest_world=rest,
            bone_lengths=lengths,
        )

    @property
    def joint_count(self) -> int:
        return len(self.joints)

    @property
    def index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.joints)}

    def children(self, joint_index: int) -> list[int]:
        return [i for i, parent in enumerate(self.parents) if parent == joint_index]

    def subtree_indices(self, joint_index: int, include_self: bool = False) -> np.ndarray:
        result: list[int] = []
        stack = list(self.children(joint_index))
        while stack:
            node = stack.pop()
            result.append(node)
            stack.extend(self.children(node))
        if include_self:
            result.append(joint_index)
        return np.asarray(sorted(result), dtype=np.int64)

    def to_manifest(self) -> dict:
        return {
            "name": self.name,
            "joints": list(self.joints),
            "parents": self.parents.tolist(),
            "rest_world": self.rest_world,
            "rest_local": self.rest_local,
            "bone_lengths": self.bone_lengths,
        }

    @classmethod
    def from_manifest(cls, payload: dict) -> "Skeleton":
        return cls(
            name=str(payload["name"]),
            joints=tuple(payload["joints"]),
            parents=np.asarray(payload["parents"], dtype=np.int64),
            rest_local=np.asarray(payload["rest_local"], dtype=np.float64),
            rest_world=np.asarray(payload["rest_world"], dtype=np.float64),
            bone_lengths=np.asarray(payload["bone_lengths"], dtype=np.float64),
        )


def default_source_skeleton() -> Skeleton:
    return Skeleton.from_positions("source", JOINT_NAMES, PARENTS, source_rest_positions())


def default_target_skeleton() -> Skeleton:
    return Skeleton.from_positions("target", JOINT_NAMES, PARENTS, target_rest_positions())


def _normalise(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    safe = np.where(norm > EPS, vector / np.maximum(norm, EPS), 0.0)
    if fallback is not None:
        safe = np.where(norm > EPS, safe, fallback)
    return safe


def rotation_between_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotations taking each unit vector ``a`` to ``b``.

    Inputs have shape (..., 3); output has shape (..., 3, 3). A deterministic
    fallback axis is used for antiparallel vectors.
    """
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if av.shape != bv.shape or av.shape[-1] != 3:
        raise ValueError("rotation_between_batch expects matching (..., 3) arrays")
    u = _normalise(av, fallback=np.asarray((1.0, 0.0, 0.0)))
    v = _normalise(bv, fallback=np.asarray((1.0, 0.0, 0.0)))
    cross = np.cross(u, v)
    dot = np.sum(u * v, axis=-1)
    sin_norm = np.linalg.norm(cross, axis=-1)
    shape = u.shape[:-1] + (3, 3)
    out = np.broadcast_to(np.eye(3, dtype=np.float64), shape).copy()

    normal = sin_norm > 1e-10
    if np.any(normal):
        axis = cross[normal] / sin_norm[normal, None]
        c = dot[normal]
        s = sin_norm[normal]
        x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
        K = np.zeros(axis.shape[:-1] + (3, 3), dtype=np.float64)
        K[..., 0, 1] = -z
        K[..., 0, 2] = y
        K[..., 1, 0] = z
        K[..., 1, 2] = -x
        K[..., 2, 0] = -y
        K[..., 2, 1] = x
        factor = (1.0 - c)[..., None, None]
        out[normal] = np.eye(3) + s[..., None, None] * K + K @ K * factor

    antiparallel = (~normal) & (dot < 0.0)
    if np.any(antiparallel):
        base = u[antiparallel]
        fallback_axis = np.asarray((0.0, 0.0, 1.0))
        parallel = np.abs(base[:, 2]) > 0.9
        if np.any(parallel):
            fallback_axis = np.asarray((0.0, 1.0, 0.0))
        axis = np.cross(base, fallback_axis)
        axis = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), EPS)
        K = np.zeros(axis.shape[:-1] + (3, 3), dtype=np.float64)
        x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
        K[..., 0, 1] = -z
        K[..., 0, 2] = y
        K[..., 1, 0] = z
        K[..., 1, 2] = -x
        K[..., 2, 0] = -y
        K[..., 2, 1] = x
        out[antiparallel] = np.eye(3) + 2.0 * (K @ K)
    return out


def rotation_between(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    return rotation_between_batch(
        np.asarray(a, dtype=np.float64)[None, :], np.asarray(b, dtype=np.float64)[None, :]
    )[0]


def rotations_about_axes(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rodrigues rotations for arrays of unit axes and angles (radians)."""
    unit = np.asarray(axes, dtype=np.float64)
    theta = np.asarray(angles, dtype=np.float64)
    shape = np.broadcast_shapes(unit.shape[:-1], theta.shape) + (3, 3)
    unit = np.broadcast_to(unit, shape[:-2] + (3,))
    theta = np.broadcast_to(theta, shape[:-2])
    x, y, z = unit[..., 0], unit[..., 1], unit[..., 2]
    c = np.cos(theta)
    s = np.sin(theta)
    one_c = 1.0 - c
    out = np.empty(shape, dtype=np.float64)
    out[..., 0, 0] = c + x * x * one_c
    out[..., 0, 1] = x * y * one_c - z * s
    out[..., 0, 2] = x * z * one_c + y * s
    out[..., 1, 0] = y * x * one_c + z * s
    out[..., 1, 1] = c + y * y * one_c
    out[..., 1, 2] = y * z * one_c - x * s
    out[..., 2, 0] = z * x * one_c - y * s
    out[..., 2, 1] = z * y * one_c + x * s
    out[..., 2, 2] = c + z * z * one_c
    return out


def rotate_vectors(rotations: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...j->...i", rotations, vectors)


def rotation_vectors_from_matrices(rotations: np.ndarray) -> np.ndarray:
    """Convert a batch of rotation matrices to Rodrigues rotation vectors."""
    r = np.asarray(rotations, dtype=np.float64)
    if r.shape[-2:] != (3, 3):
        raise ValueError("rotation_vectors_from_matrices expects (..., 3, 3) input")
    trace = np.trace(r, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cosine)
    axis = np.stack(
        (
            r[..., 2, 1] - r[..., 1, 2],
            r[..., 0, 2] - r[..., 2, 0],
            r[..., 1, 0] - r[..., 0, 1],
        ),
        axis=-1,
    )
    sine = np.sin(angle)
    normal = sine > 1e-8
    vectors = np.zeros_like(r[..., 0])
    vectors[normal] = angle[normal, None] * axis[normal] / (2.0 * sine[normal, None])
    near_pi = (~normal) & (cosine < -0.999999)
    if np.any(near_pi):
        diagonal = np.clip((np.diagonal(r, axis1=-2, axis2=-1) + 1.0) * 0.5, 0.0, None)
        local_axis = np.sqrt(diagonal)
        largest = np.argmax(local_axis, axis=-1)
        for index in np.flatnonzero(near_pi.reshape(-1)):
            multi = np.unravel_index(index, near_pi.shape)
            ax = int(largest[multi])
            vector = local_axis[multi]
            if vector[ax] > 1e-12:
                for component in range(3):
                    if component != ax:
                        vector[component] = r[multi + (ax, component)] / (2.0 * vector[ax])
            norm = np.linalg.norm(vector)
            vectors[multi] = np.pi * vector / max(norm, 1e-12)
    return vectors


def rotate_points_about(
    points: np.ndarray, pivot: np.ndarray, axes: np.ndarray, angles: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Rotate selected points (N,3) around pivots in-place-safe form."""
    result = np.array(points, copy=True)
    active = np.asarray(mask, dtype=bool)
    if not np.any(active):
        return result
    rot = rotations_about_axes(np.asarray(axes)[active], np.asarray(angles)[active])
    local = result[active] - np.asarray(pivot)[active]
    result[active] = rotate_vectors(rot, local) + np.asarray(pivot)[active]
    return result


def forward_kinematics(
    skeleton: Skeleton, local_rotations: np.ndarray, root_positions: np.ndarray, root_rotations: np.ndarray | None = None
) -> np.ndarray:
    """FK for (F,J,3,3) local rotations and (F,3) root positions."""
    frames = local_rotations.shape[0]
    global_rot = np.tile(np.eye(3), (frames, skeleton.joint_count, 1, 1))
    pose = np.zeros((frames, skeleton.joint_count, 3), dtype=np.float64)
    if root_rotations is None:
        root_rotations = np.tile(np.eye(3), (frames, 1, 1))
    pose[:, 0] = root_positions
    global_rot[:, 0] = root_rotations
    for idx in range(1, skeleton.joint_count):
        parent = int(skeleton.parents[idx])
        global_rot[:, idx] = global_rot[:, parent] @ local_rotations[:, idx]
        pose[:, idx] = pose[:, parent] + rotate_vectors(global_rot[:, parent], skeleton.rest_local[idx])
    return pose


def bone_direction_residuals(pose: np.ndarray, skeleton: Skeleton) -> np.ndarray:
    residuals = np.zeros(pose.shape[:2], dtype=np.float64)
    for idx in range(1, skeleton.joint_count):
        parent = int(skeleton.parents[idx])
        actual = np.linalg.norm(pose[:, idx] - pose[:, parent], axis=-1)
        residuals[:, idx] = np.abs(actual - skeleton.bone_lengths[idx])
    return residuals


def safe_unit(vectors: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    return _normalise(np.asarray(vectors, dtype=np.float64), fallback=fallback)


def angle_between_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    u = safe_unit(a)
    v = safe_unit(b)
    cosine = np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)
    return np.arccos(cosine)


def subtree_mask(skeleton: Skeleton) -> np.ndarray:
    mask = np.zeros((skeleton.joint_count, skeleton.joint_count), dtype=bool)
    for idx in range(skeleton.joint_count):
        mask[idx, skeleton.subtree_indices(idx)] = True
    return mask


def paired_indices(skeleton: Skeleton, pairs: Iterable[Sequence[str]]) -> list[tuple[int, int]]:
    lookup = skeleton.index
    result: list[tuple[int, int]] = []
    for left, right in pairs:
        if left not in lookup or right not in lookup:
            raise BusinessError(ErrorCode.E_INPUT, "unknown symmetry pair", pair=[left, right])
        result.append((lookup[left], lookup[right]))
    return result
