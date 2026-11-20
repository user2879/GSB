"""Skeleton data model and deterministic procedural sample generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hashutil import canonical_json_bytes, sha256_hex
from .quat import conjugate, multiply, normalize


def _rng(seed: int) -> np.random.Generator:
    # PCG64 is stable across processes and platform bitness for a fixed seed.
    return np.random.Generator(np.random.PCG64(seed))


def _random_rotation(rng: np.random.Generator, spread: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = float(rng.uniform(-spread, spread))
    half = angle * 0.5
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])


HIERARCHY: list[tuple[str, int]] = [
    ("root", -1),
    ("pelvis", 0),
    ("spine", 1),
    ("neck", 2),
    ("head", 3),
    ("l_shoulder", 2),
    ("l_elbow", 5),
    ("r_shoulder", 2),
    ("r_elbow", 7),
]


@dataclass(frozen=True)
class Skeleton:
    name: str
    parents: tuple[int, ...]
    names: tuple[str, ...]
    rest_rotations: np.ndarray  # (J, 4) wxyz local rest rotations
    rest_translations: np.ndarray  # (J, 3) local rest offsets
    bind_matrices: np.ndarray  # (J, 4, 4) inverse global rest transforms
    bone_lengths: np.ndarray  # (J,) root length is 0.0

    @property
    def joint_count(self) -> int:
        return len(self.names)

    def index_of(self, joint_name: str) -> int:
        return self.names.index(joint_name)

    def global_rest_rotations(self) -> np.ndarray:
        """Accumulate local rest rotations down the hierarchy."""

        out = np.zeros_like(self.rest_rotations)
        for joint, parent in enumerate(self.parents):
            local = self.rest_rotations[joint]
            if parent < 0:
                out[joint] = local
            else:
                out[joint] = multiply(out[parent], local)
        return out

    def parameter_version(self) -> str:
        payload = {
            "name": self.name,
            "parents": list(self.parents),
            "names": list(self.names),
            "rest_rotations": self.rest_rotations.tolist(),
            "rest_translations": self.rest_translations.tolist(),
            "bone_lengths": self.bone_lengths.tolist(),
        }
        return sha256_hex(canonical_json_bytes(payload))[:16]


def _build_bind_matrices(
    parents: tuple[int, ...],
    rest_rotations: np.ndarray,
    rest_translations: np.ndarray,
) -> np.ndarray:
    joint_count = len(parents)
    globals_ = np.zeros((joint_count, 4, 4), dtype=np.float64)
    for joint, parent in enumerate(parents):
        local = np.eye(4, dtype=np.float64)
        local[:3, :3] = _rotation_matrix(rest_rotations[joint])
        local[:3, 3] = rest_translations[joint]
        if parent < 0:
            globals_[joint] = local
        else:
            globals_[joint] = globals_[parent] @ local
    binds = np.zeros_like(globals_)
    for joint in range(joint_count):
        binds[joint] = np.linalg.inv(globals_[joint])
    return binds


def _rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def generate_skeleton(name: str, seed: int, *, length_scale: float = 1.0) -> Skeleton:
    """Produce a fully deterministic skeleton for a fixed integer seed."""

    rng = _rng(seed)
    names = tuple(item[0] for item in HIERARCHY)
    parents = tuple(item[1] for item in HIERARCHY)
    joint_count = len(names)

    rest_rotations = np.zeros((joint_count, 4), dtype=np.float64)
    rest_translations = np.zeros((joint_count, 3), dtype=np.float64)
    bone_lengths = np.zeros(joint_count, dtype=np.float64)

    for joint, parent in enumerate(parents):
        rest_rotations[joint] = normalize(_random_rotation(rng, 0.35))
        if parent >= 0:
            length = float(rng.uniform(0.08, 0.32)) * length_scale
            bone_lengths[joint] = length
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            rest_translations[joint] = direction * length

    binds = _build_bind_matrices(parents, rest_rotations, rest_translations)
    return Skeleton(
        name=name,
        parents=parents,
        names=names,
        rest_rotations=rest_rotations,
        rest_translations=rest_translations,
        bind_matrices=binds,
        bone_lengths=bone_lengths,
    )


def rest_pose_delta(source: Skeleton, target: Skeleton) -> np.ndarray:
    """Global-frame rotation difference ``Tg * Sg^-1`` for every joint."""

    if source.names != target.names:
        raise ValueError("source and target skeletons must share joint names")
    source_global = source.global_rest_rotations()
    target_global = target.global_rest_rotations()
    delta = np.zeros_like(source_global)
    for joint in range(source.joint_count):
        delta[joint] = multiply(target_global[joint], conjugate(source_global[joint]))
    return delta


def root_bone_scale(source: Skeleton, target: Skeleton) -> float:
    """Mean child-bone length ratio (root joint itself has no length)."""

    child_mask = np.array(source.parents, dtype=np.int64) >= 0
    ratios = target.bone_lengths[child_mask] / source.bone_lengths[child_mask]
    return float(np.mean(ratios))
