"""Quaternion and rigid-transform helpers.

All quaternions stored by the pipeline use scalar-first ``wxyz`` ordering.
External formats may declare ``xyzw`` and are reordered at their boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .errors import ErrorCode, PipelineError

QUAT_COMPONENT_ORDERS = ("wxyz", "xyzw")
NORM_EPSILON = 1e-12
FINITE_EPSILON = 1e-12



def normalize_quaternion(
    value: Any,
    *,
    joint: str | None = None,
    frame: int | None = None,
) -> np.ndarray:
    try:
        quat = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            ErrorCode.E_QUAT_INVALID,
            "quaternion is not numeric",
            joint=joint,
            frame=frame,
        ) from exc
    if quat.shape != (4,):
        raise PipelineError(
            ErrorCode.E_QUAT_INVALID,
            f"quaternion must have exactly 4 components, got shape {quat.shape}",
            joint=joint,
            frame=frame,
        )
    if not np.all(np.isfinite(quat)):
        raise PipelineError(
            ErrorCode.E_QUAT_INVALID,
            "quaternion contains NaN or infinity",
            joint=joint,
            frame=frame,
        )
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= NORM_EPSILON:
        raise PipelineError(
            ErrorCode.E_QUAT_INVALID,
            f"quaternion norm is degenerate ({norm:.17g})",
            joint=joint,
            frame=frame,
        )
    if abs(norm - 1.0) <= 8.0 * np.finfo(np.float64).eps:
        return quat
    return quat / norm


def canonical_hemisphere(quat: np.ndarray) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64).copy()
    if value[0] < 0.0:
        value = -value
    elif abs(value[0]) <= FINITE_EPSILON:
        first_nonzero = next((idx for idx in range(4) if abs(value[idx]) > FINITE_EPSILON), None)
        if first_nonzero is not None and value[first_nonzero] < 0.0:
            value = -value
    return value


def align_quaternion_to_previous(quat: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64)
    if previous is None:
        return canonical_hemisphere(value)
    if float(np.dot(value, previous)) < 0.0:
        return -value
    return value


def canonicalize_quaternion_array(
    rotations: Any,
    *,
    joint_names: Sequence[str] | None = None,
) -> np.ndarray:
    array = np.asarray(rotations, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 4:
        raise PipelineError(
            ErrorCode.E_QUAT_INVALID,
            f"rotation tensor must be (frames, joints, 4), got {array.shape}",
        )
    output = np.empty_like(array, dtype=np.float64)
    for joint_index in range(array.shape[1]):
        previous: np.ndarray | None = None
        joint_name = joint_names[joint_index] if joint_names is not None else None
        for frame_index in range(array.shape[0]):
            quat = normalize_quaternion(
                array[frame_index, joint_index],
                joint=joint_name,
                frame=frame_index,
            )
            quat = align_quaternion_to_previous(quat, previous)
            output[frame_index, joint_index] = quat
            previous = quat
    return output


def quat_multiply(left: Any, right: Any) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def quat_conjugate(quat: Any) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64)
    result = value.copy()
    result[..., 1:] *= -1.0
    return result


def quat_inverse(quat: Any) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64)
    norm_squared = np.sum(value * value, axis=-1, keepdims=True)
    return quat_conjugate(value) / norm_squared


def quat_dot(left: Any, right: Any) -> np.ndarray:
    return np.sum(np.asarray(left) * np.asarray(right), axis=-1)


def quat_angle_between(left: Any, right: Any) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    delta = quat_multiply(a, quat_conjugate(b))
    delta = delta / np.linalg.norm(delta, axis=-1, keepdims=True)
    vector_norm = np.linalg.norm(delta[..., 1:], axis=-1)
    return 2.0 * np.arctan2(vector_norm, np.abs(delta[..., 0]))


def quat_from_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= NORM_EPSILON:
        raise PipelineError(ErrorCode.E_QUAT_INVALID, "axis-angle axis is degenerate")
    vector = vector / norm
    half = 0.5 * float(angle)
    return np.concatenate(([np.cos(half)], vector * np.sin(half)))


def quat_to_matrix(quat: Any) -> np.ndarray:
    value = normalize_quaternion(quat)
    w, x, y, z = value
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rigid_matrix(translation: Sequence[float], quat: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_to_matrix(quat)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def rotate_vector(quat: Any, vector: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    return quat_multiply(quat_multiply(q, np.concatenate(([0.0], v))), quat_conjugate(q))[1:]


def reorder_quaternion(value: Any, source_order: str, *, context: str = "") -> np.ndarray:
    order = source_order.lower()
    if order not in QUAT_COMPONENT_ORDERS:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            f"unsupported quaternion component order: {source_order}",
            field=f"{context}quat_order" if context else "quat_order",
        )
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,):
        raise PipelineError(
            ErrorCode.E_QUAT_INVALID,
            f"quaternion must have exactly 4 components, got shape {array.shape}",
            field=f"{context}quat_order" if context else "quat_order",
        )
    if order == "wxyz":
        return array
    return array[[3, 0, 1, 2]]


def format_quaternion(quat: Any, order: str) -> list[float]:
    value = np.asarray(quat, dtype=np.float64)
    if value.shape != (4,):
        raise PipelineError(
            ErrorCode.E_QUAT_INVALID,
            f"quaternion must have exactly 4 components, got shape {value.shape}",
            field="quat_order",
        )
    if order == "wxyz":
        return [float(item) for item in value]
    if order == "xyzw":
        return [float(value[1]), float(value[2]), float(value[3]), float(value[0])]
    raise PipelineError(
        ErrorCode.E_FIELD_MISSING,
        f"unsupported quaternion component order: {order}",
        field="quat_order",
    )


def interpolate_quaternions(
    left: np.ndarray,
    right: np.ndarray,
    alpha: float,
    method: str,
) -> np.ndarray:
    q0 = normalize_quaternion(left)
    q1 = normalize_quaternion(right)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    if method == "step" or alpha <= 0.0:
        return q0
    if alpha >= 1.0:
        return q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if method == "linear":
        value = (1.0 - alpha) * q0 + alpha * q1
    elif method == "slerp":
        if dot > 0.9995:
            value = (1.0 - alpha) * q0 + alpha * q1
        else:
            theta = np.arccos(dot)
            sin_theta = np.sin(theta)
            value = (
                np.sin((1.0 - alpha) * theta) / sin_theta * q0
                + np.sin(alpha * theta) / sin_theta * q1
            )
    else:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            f"unsupported interpolation method: {method}",
            field="interpolation",
        )
    return normalize_quaternion(value)
