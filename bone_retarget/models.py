"""Core skeleton and animation data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .errors import ErrorCode, PipelineError
from .math3d import (
    canonical_hemisphere,
    canonicalize_quaternion_array,
    normalize_quaternion,
    quat_multiply,
    rigid_matrix,
    rotate_vector,
)


def _require_mapping(value: Any, field: str, *, joint: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            f"field {field} must be an object",
            field=field,
            joint=joint,
        )
    return value


def _required(mapping: dict[str, Any], key: str, *, field: str | None = None, joint: str | None = None) -> Any:
    if key not in mapping:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            f"required field {field or key} is missing",
            field=field or key,
            joint=joint,
        )
    return mapping[key]


def _float_vector(value: Any, size: int, field: str, *, joint: str | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            f"field {field} must contain {size} numeric values",
            field=field,
            joint=joint,
        ) from exc
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            f"field {field} must contain {size} finite numeric values",
            field=field,
            joint=joint,
        )
    return array


@dataclass
class Joint:
    name: str
    parent: str | None
    length: float
    rest_translation: np.ndarray
    rest_rotation: np.ndarray
    inverse_bind_matrix: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise PipelineError(ErrorCode.E_FIELD_MISSING, "joint name must be non-empty", field="name")
        if self.parent is not None and (not isinstance(self.parent, str) or not self.parent):
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "joint parent must be null or a non-empty string",
                field="parent",
                joint=self.name,
            )
        length = float(self.length)
        if not np.isfinite(length) or length <= 0.0:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "joint length must be finite and positive",
                field="length",
                joint=self.name,
            )
        self.length = length
        self.rest_translation = _float_vector(
            self.rest_translation, 3, "rest_translation", joint=self.name
        )
        self.rest_rotation = canonical_hemisphere(
            normalize_quaternion(self.rest_rotation, joint=self.name)
        )
        matrix = np.asarray(self.inverse_bind_matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "inverse_bind_matrix must be a finite 4x4 matrix",
                field="inverse_bind_matrix",
                joint=self.name,
            )
        self.inverse_bind_matrix = matrix.copy()


@dataclass
class Skeleton:
    name: str
    joints: tuple[Joint, ...]

    def __post_init__(self) -> None:
        self.joints = tuple(self.joints)
        if not isinstance(self.name, str) or not self.name:
            raise PipelineError(ErrorCode.E_FIELD_MISSING, "skeleton name must be non-empty", field="name")
        if not self.joints:
            raise PipelineError(ErrorCode.E_FIELD_MISSING, "skeleton has no joints", field="joints")
        seen: dict[str, int] = {}
        for index, joint in enumerate(self.joints):
            if joint.name in seen:
                raise PipelineError(
                    ErrorCode.E_INDEX_OVERFLOW,
                    "duplicate joint name",
                    joint=joint.name,
                    field="joints",
                    details={"first_index": seen[joint.name], "second_index": index},
                )
            if joint.parent is not None and joint.parent not in seen:
                raise PipelineError(
                    ErrorCode.E_FIELD_MISSING,
                    "parent joint must appear before its child",
                    joint=joint.name,
                    field="parent",
                    details={"missing_parent": joint.parent},
                )
            seen[joint.name] = index
        roots = [joint.name for joint in self.joints if joint.parent is None]
        if len(roots) != 1:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "skeleton must have exactly one root joint",
                field="joints",
                details={"roots": roots},
            )

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    @property
    def joint_count(self) -> int:
        return len(self.joints)

    @property
    def root_name(self) -> str:
        return next(joint.name for joint in self.joints if joint.parent is None)

    def joint_index(self, name: str) -> int:
        for index, joint in enumerate(self.joints):
            if joint.name == name:
                return index
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "joint is missing from skeleton",
            joint=name,
            field="joints",
        )

    def parent_index(self, index: int) -> int | None:
        parent = self.joints[index].parent
        return None if parent is None else self.joint_index(parent)

    def world_rest_rotations(self) -> np.ndarray:
        output = np.empty((len(self.joints), 4), dtype=np.float64)
        for index, joint in enumerate(self.joints):
            parent = self.parent_index(index)
            output[index] = (
                joint.rest_rotation
                if parent is None
                else quat_multiply(output[parent], joint.rest_rotation)
            )
            output[index] = canonical_hemisphere(normalize_quaternion(output[index]))
        return output

    def world_rest_matrices(self) -> np.ndarray:
        rotations = self.world_rest_rotations()
        translations = np.empty((len(self.joints), 3), dtype=np.float64)
        matrices = np.empty((len(self.joints), 4, 4), dtype=np.float64)
        for index, joint in enumerate(self.joints):
            parent = self.parent_index(index)
            if parent is None:
                translations[index] = joint.rest_translation
            else:
                translations[index] = translations[parent] + rotate_vector(
                    rotations[parent], joint.rest_translation
                )
            matrices[index] = rigid_matrix(translations[index], rotations[index])
        return matrices

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "joints": [
                {
                    "name": joint.name,
                    "parent": joint.parent,
                    "length": joint.length,
                    "rest_translation": [float(value) for value in joint.rest_translation],
                    "rest_rotation_wxyz": [float(value) for value in joint.rest_rotation],
                    "inverse_bind_matrix": [
                        [float(value) for value in row] for row in joint.inverse_bind_matrix
                    ],
                }
                for joint in self.joints
            ],
        }

    @classmethod
    def from_dict(cls, document: Any) -> "Skeleton":
        mapping = _require_mapping(document, "skeleton")
        name = _required(mapping, "name", field="skeleton.name")
        raw_joints = _required(mapping, "joints", field="skeleton.joints")
        if not isinstance(raw_joints, list) or not raw_joints:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "skeleton.joints must be a non-empty array",
                field="skeleton.joints",
            )
        joints: list[Joint] = []
        for index, raw_joint in enumerate(raw_joints):
            joint_mapping = _require_mapping(raw_joint, f"joints[{index}]")
            joint_name = _required(joint_mapping, "name", field=f"joints[{index}].name")
            if not isinstance(joint_name, str) or not joint_name:
                raise PipelineError(
                    ErrorCode.E_FIELD_MISSING,
                    "joint name must be a non-empty string",
                    field=f"joints[{index}].name",
                )
            parent = joint_mapping.get("parent")
            length = _required(joint_mapping, "length", field=f"joints[{index}].length", joint=joint_name)
            translation = _float_vector(
                _required(
                    joint_mapping,
                    "rest_translation",
                    field=f"joints[{index}].rest_translation",
                    joint=joint_name,
                ),
                3,
                f"joints[{index}].rest_translation",
                joint=joint_name,
            )
            rotation = normalize_quaternion(
                _required(
                    joint_mapping,
                    "rest_rotation_wxyz",
                    field=f"joints[{index}].rest_rotation_wxyz",
                    joint=joint_name,
                ),
                joint=joint_name,
            )
            bind = np.asarray(
                _required(
                    joint_mapping,
                    "inverse_bind_matrix",
                    field=f"joints[{index}].inverse_bind_matrix",
                    joint=joint_name,
                ),
                dtype=np.float64,
            )
            joints.append(
                Joint(
                    name=joint_name,
                    parent=parent,
                    length=float(length),
                    rest_translation=translation,
                    rest_rotation=rotation,
                    inverse_bind_matrix=bind,
                )
            )
        return cls(name=str(name), joints=tuple(joints))


@dataclass
class Animation:
    joint_names: tuple[str, ...]
    times: np.ndarray
    rotations: np.ndarray
    root_translation: np.ndarray
    interpolation: str = "slerp"
    fps: float | None = None
    source_schema_version: int | None = None

    def __post_init__(self) -> None:
        self.joint_names = tuple(self.joint_names)
        if not self.joint_names or any(not isinstance(name, str) or not name for name in self.joint_names):
            raise PipelineError(ErrorCode.E_FIELD_MISSING, "joint names must be non-empty", field="joints")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise PipelineError(ErrorCode.E_INDEX_OVERFLOW, "joint names are not unique", field="joints")
        times = np.asarray(self.times, dtype=np.float64)
        if times.ndim != 1 or times.size == 0:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "timeline must be a non-empty one-dimensional array",
                field="times",
            )
        if not np.all(np.isfinite(times)):
            raise PipelineError(ErrorCode.E_FIELD_MISSING, "timeline contains non-finite values", field="times")
        if times.size > 1:
            violations = np.nonzero(np.diff(times) <= 0.0)[0]
            if violations.size:
                frame = int(violations[0] + 1)
                raise PipelineError(
                    ErrorCode.E_TIME_NONMONOTONIC,
                    "timeline timestamps must be strictly increasing",
                    frame=frame,
                    field="times",
                    details={"previous": float(times[frame - 1]), "current": float(times[frame])},
                )
        rotations = np.asarray(self.rotations, dtype=np.float64)
        expected = (times.size, len(self.joint_names), 4)
        if rotations.shape != expected:
            raise PipelineError(
                ErrorCode.E_QUAT_INVALID,
                f"rotation count does not match frames and joints; expected {expected}, got {rotations.shape}",
                field="rotations",
            )
        roots = np.asarray(self.root_translation, dtype=np.float64)
        if roots.shape != (times.size, 3):
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                f"root_translation must have shape ({times.size}, 3), got {roots.shape}",
                field="root_translation",
            )
        if not np.all(np.isfinite(roots)):
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "root_translation contains non-finite values",
                field="root_translation",
            )
        if self.interpolation not in {"slerp", "linear", "step"}:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                f"unsupported interpolation method: {self.interpolation}",
                field="interpolation",
            )
        if self.fps is not None:
            fps = float(self.fps)
            if not np.isfinite(fps) or fps <= 0.0:
                raise PipelineError(
                    ErrorCode.E_FIELD_MISSING,
                    "fps must be finite and positive",
                    field="fps",
                )
            self.fps = fps
        self.times = times.copy()
        self.root_translation = roots.copy()
        self.rotations = canonicalize_quaternion_array(rotations, joint_names=self.joint_names)

    @property
    def frame_count(self) -> int:
        return int(self.times.size)

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    def joint_index(self, name: str) -> int:
        try:
            return self.joint_names.index(name)
        except ValueError as exc:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "joint is missing from animation",
                joint=name,
                field="joints",
            ) from exc

    def copy(self) -> "Animation":
        return Animation(
            joint_names=self.joint_names,
            times=self.times.copy(),
            rotations=self.rotations.copy(),
            root_translation=self.root_translation.copy(),
            interpolation=self.interpolation,
            fps=self.fps,
            source_schema_version=self.source_schema_version,
        )

    def frame_global_matrices(self, skeleton: Skeleton, frame_index: int) -> np.ndarray:
        if skeleton.joint_names != self.joint_names:
            missing = [name for name in skeleton.joint_names if name not in self.joint_names]
            if missing:
                raise PipelineError(
                    ErrorCode.E_FIELD_MISSING,
                    "animation is missing skeleton joints",
                    joint=missing[0],
                    field="joints",
                )
            extra = [name for name in self.joint_names if name not in skeleton.joint_names]
            if extra:
                raise PipelineError(
                    ErrorCode.E_INDEX_OVERFLOW,
                    "animation contains joints not present in skeleton",
                    joint=extra[0],
                    field="joints",
                )
        if self.joint_names == skeleton.joint_names:
            local_rotations = self.rotations[frame_index]
        else:
            local_rotations = np.stack(
                [
                    self.rotations[frame_index, self.joint_index(joint.name)]
                    for joint in skeleton.joints
                ],
                axis=0,
            )
        world_rotations = np.empty((skeleton.joint_count, 4), dtype=np.float64)
        for index, _joint in enumerate(skeleton.joints):
            parent = skeleton.parent_index(index)
            world_rotations[index] = (
                local_rotations[index]
                if parent is None
                else quat_multiply(world_rotations[parent], local_rotations[index])
            )
            world_rotations[index] = normalize_quaternion(world_rotations[index])
        translations = np.empty((skeleton.joint_count, 3), dtype=np.float64)
        matrices = np.empty((skeleton.joint_count, 4, 4), dtype=np.float64)
        for index, joint in enumerate(skeleton.joints):
            parent = skeleton.parent_index(index)
            local_translation = joint.rest_translation.copy()
            if parent is None:
                local_translation = local_translation + self.root_translation[frame_index]
                translations[index] = local_translation
            else:
                translations[index] = translations[parent] + rotate_vector(
                    world_rotations[parent], local_translation
                )
            matrices[index] = rigid_matrix(translations[index], world_rotations[index])
        return matrices
