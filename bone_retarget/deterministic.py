"""Small deterministic PRNG and benchmark data generators."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .math3d import normalize_quaternion, quat_from_axis_angle, quat_multiply
from .models import Animation, Joint, Skeleton

MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = int(seed) & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) & MASK64

    def unit(self) -> float:
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self.unit()

    def signed_unit(self) -> float:
        return self.uniform(-1.0, 1.0)

    def direction(self) -> np.ndarray:
        while True:
            vector = np.array(
                [self.signed_unit(), self.signed_unit(), self.signed_unit()],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(vector))
            if norm > 1e-6:
                return vector / norm


JOINT_SPEC: tuple[tuple[str, str | None, float], ...] = (
    ("root", None, 0.36),
    ("pelvis", "root", 0.18),
    ("spine_01", "pelvis", 0.22),
    ("spine_02", "spine_01", 0.22),
    ("neck", "spine_02", 0.12),
    ("head", "neck", 0.18),
    ("shoulder_L", "spine_02", 0.15),
    ("upper_arm_L", "shoulder_L", 0.27),
    ("forearm_L", "upper_arm_L", 0.24),
    ("hand_L", "forearm_L", 0.09),
    ("shoulder_R", "spine_02", 0.15),
    ("upper_arm_R", "shoulder_R", 0.27),
    ("forearm_R", "upper_arm_R", 0.24),
    ("hand_R", "forearm_R", 0.09),
    ("thigh_L", "pelvis", 0.42),
    ("shin_L", "thigh_L", 0.40),
    ("foot_L", "shin_L", 0.16),
    ("thigh_R", "pelvis", 0.42),
    ("shin_R", "thigh_R", 0.40),
    ("foot_R", "shin_R", 0.16),
)


def _parent_lengths() -> dict[str, float]:
    base_lengths = {name: length for name, _parent, length in JOINT_SPEC}
    return {
        name: base_lengths[parent]
        for name, parent, _length in JOINT_SPEC
        if parent is not None
    }


def generate_skeleton(seed: int, name: str) -> Skeleton:
    rng = SplitMix64(seed)
    global_scale = rng.uniform(0.82, 1.28)
    lengths: dict[str, float] = {}
    for joint_name, _parent, base_length in JOINT_SPEC:
        lengths[joint_name] = base_length * global_scale * rng.uniform(0.88, 1.12)
    parent_length = _parent_lengths()

    temporary: list[Joint] = []
    for joint_name, parent, _base_length in JOINT_SPEC:
        if parent is None:
            translation = np.array([0.0, 0.92 * global_scale, 0.0], dtype=np.float64)
        else:
            translation = np.array([0.0, parent_length[joint_name] * global_scale, 0.0])
            translation *= lengths[parent] / (parent_length[joint_name] * global_scale)
        angle = rng.uniform(-0.62, 0.62)
        axis = rng.direction()
        rotation = canonical_signed(quat_from_axis_angle(axis, angle))
        temporary.append(
            Joint(
                name=joint_name,
                parent=parent,
                length=lengths[joint_name],
                rest_translation=translation,
                rest_rotation=rotation,
                inverse_bind_matrix=np.eye(4, dtype=np.float64),
            )
        )

    skeleton = Skeleton(name=name, joints=tuple(temporary))
    world_matrices = skeleton.world_rest_matrices()
    final_joints: list[Joint] = []
    for index, joint in enumerate(skeleton.joints):
        final_joints.append(
            Joint(
                name=joint.name,
                parent=joint.parent,
                length=joint.length,
                rest_translation=joint.rest_translation,
                rest_rotation=joint.rest_rotation,
                inverse_bind_matrix=np.linalg.inv(world_matrices[index]),
            )
        )
    return Skeleton(name=name, joints=tuple(final_joints))


def canonical_signed(quat: np.ndarray) -> np.ndarray:
    value = normalize_quaternion(quat)
    first = next((index for index in range(4) if abs(float(value[index])) > 1e-12), 0)
    if value[first] < 0.0:
        value = -value
    return value


def _joint_motion_parameters(seed: int, joint_names: Iterable[str]) -> dict[str, tuple[np.ndarray, float, float]]:
    parameters: dict[str, tuple[np.ndarray, float, float]] = {}
    for index, joint_name in enumerate(joint_names):
        rng = SplitMix64(seed * 1009 + index * 7919)
        parameters[joint_name] = (
            rng.direction(),
            rng.uniform(0.08, 0.42),
            rng.uniform(-math.pi, math.pi),
        )
    return parameters


def generate_animation(
    skeleton: Skeleton,
    seed: int,
    *,
    frame_count: int = 7,
    fps: float = 12.0,
) -> Animation:
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    times = np.arange(frame_count, dtype=np.float64) / float(fps)
    parameters = _joint_motion_parameters(seed, skeleton.joint_names)
    rotations = np.empty((frame_count, skeleton.joint_count, 4), dtype=np.float64)
    for frame_index, time_value in enumerate(times):
        for joint_index, joint in enumerate(skeleton.joints):
            axis, amplitude, phase = parameters[joint.name]
            envelope = math.sin((math.pi * frame_index) / max(frame_count - 1, 1))
            angle = amplitude * envelope * math.sin(1.7 * time_value * math.tau + phase)
            delta = quat_from_axis_angle(axis, angle)
            rotations[frame_index, joint_index] = quat_multiply(joint.rest_rotation, delta)

    root_scale = float(next(joint.length for joint in skeleton.joints if joint.parent is None))
    phase = SplitMix64(seed * 31 + 17).uniform(-math.pi, math.pi)
    roots = np.empty((frame_count, 3), dtype=np.float64)
    for frame_index, time_value in enumerate(times):
        roots[frame_index] = np.array(
            [
                root_scale * 0.09 * math.sin(math.tau * 0.35 * time_value + phase),
                root_scale * 0.035 * (time_value / max(times[-1], 1e-9)),
                root_scale * 0.06 * math.cos(math.tau * 0.27 * time_value + phase),
            ],
            dtype=np.float64,
        )
    return Animation(
        joint_names=skeleton.joint_names,
        times=times,
        rotations=rotations,
        root_translation=roots,
        interpolation="slerp",
        fps=fps,
    )
