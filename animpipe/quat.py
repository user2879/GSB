"""Quaternion normalization, sign alignment and interpolation helpers.

Conventions
-----------
* Quaternions are stored as ``[w, x, y, z]`` numpy arrays (float64).
* A quaternion is invalid when it contains NaN/Inf or its L2 norm is zero.
* Sign alignment walks the keyframe sequence and flips a quaternion when
  its dot product with the previous keyframe is negative.  Interpolation
  is invariant to that choice: ``slerp(q, r)`` and ``slerp(q, -r)``
  describe the same rigid rotation.
"""

from __future__ import annotations

import numpy as np

from .errors import E_QUAT_INVALID

QUAT_ORDERS = ("wxyz", "xyzw")


def check_quat(q: np.ndarray, *, joint: int = -1, frame: int = -1) -> np.ndarray:
    """Validate a single quaternion, raising ``E_QUAT_INVALID`` on bad data."""

    arr = np.asarray(q, dtype=np.float64)
    if arr.shape != (4,):
        raise E_QUAT_INVALID(
            f"quaternion must have 4 components, got shape {arr.shape}",
            {"joint": joint, "frame": frame, "shape": list(arr.shape)},
        )
    if not np.all(np.isfinite(arr)):
        raise E_QUAT_INVALID(
            "quaternion contains NaN or Inf",
            {"joint": joint, "frame": frame, "components": arr.tolist()},
        )
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise E_QUAT_INVALID(
            "zero-norm quaternion",
            {"joint": joint, "frame": frame, "norm": 0.0},
        )
    return arr


def normalize(q: np.ndarray) -> np.ndarray:
    """Return the unit quaternion for ``q`` (validates zero-norm / NaN)."""

    arr = check_quat(q)
    return arr / np.linalg.norm(arr)


def align_sign(q: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip ``q`` so its dot product with ``reference`` is non-negative."""

    if float(np.dot(q, reference)) < 0.0:
        return -q
    return q


def normalize_frames(
    rotations: np.ndarray, *, frame_offset: int = 0
) -> np.ndarray:
    """Validate and normalize an ``(F, J, 4)`` rotation array.

    Raises ``E_QUAT_INVALID`` locating the offending joint and absolute
    frame index.
    """

    arr = np.asarray(rotations, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise E_QUAT_INVALID(
            "rotation array must have shape (frames, joints, 4)",
            {"shape": list(arr.shape)},
        )
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))
        first = bad[0]
        raise E_QUAT_INVALID(
            "rotation contains NaN or Inf",
            {"frame": int(first[0]) + frame_offset, "joint": int(first[1])},
        )
    norms = np.linalg.norm(arr, axis=-1)
    if np.any(norms == 0.0):
        bad = np.argwhere(norms == 0.0)
        first = bad[0]
        raise E_QUAT_INVALID(
            "zero-norm quaternion",
            {
                "frame": int(first[0]) + frame_offset,
                "joint": int(first[1]),
                "norm": 0.0,
            },
        )
    return arr / norms[..., None]


def align_sequence(rotations: np.ndarray) -> np.ndarray:
    """Normalize then enforce a continuous sign across keyframes per joint.

    The first keyframe of every joint keeps its sign; later frames flip
    when the dot product with their predecessor is negative.
    """

    out = normalize_frames(rotations)
    frames, joints = out.shape[:2]
    for frame in range(1, frames):
        for joint in range(joints):
            out[frame, joint] = align_sign(out[frame, joint], out[frame - 1, joint])
    return out


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two unit quaternions (w, x, y, z)."""

    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def conjugate(q: np.ndarray) -> np.ndarray:
    result = np.array(q, dtype=np.float64).copy()
    result[1:] = -result[1:]
    return result


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Spherical linear interpolation, invariant to quaternion sign."""

    q0 = normalize(q0)
    q1 = normalize(q1)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + u * (q1 - q0)
        return result / np.linalg.norm(result)
    theta0 = np.arccos(dot)
    theta = theta0 * u
    sin_theta0 = np.sin(theta0)
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta0
    s1 = np.sin(theta) / sin_theta0
    return s0 * q0 + s1 * q1


def nlerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Normalized linear interpolation (also sign-invariant after norm)."""

    q0 = normalize(q0)
    q1 = normalize(q1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    result = (1.0 - u) * q0 + u * q1
    return result / np.linalg.norm(result)


def angular_distance(q0: np.ndarray, q1: np.ndarray) -> float:
    """Geodesic angle in radians between two rotations."""

    q0 = normalize(q0)
    q1 = normalize(q1)
    dot = abs(float(np.dot(q0, q1)))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def reorder(q: np.ndarray, source_order: str, target_order: str = "wxyz") -> np.ndarray:
    """Convert quaternion component ordering, e.g. xyzw -> wxyz."""

    if source_order not in QUAT_ORDERS or target_order not in QUAT_ORDERS:
        raise E_QUAT_INVALID(
            "unsupported quaternion component order",
            {"source_order": source_order, "target_order": target_order},
        )
    arr = np.asarray(q, dtype=np.float64)
    if source_order == target_order:
        return arr.copy()
    if source_order == "xyzw":  # -> wxyz
        return np.concatenate([arr[..., 3:4], arr[..., 0:3]], axis=-1)
    return np.concatenate([arr[..., 1:4], arr[..., 0:1]], axis=-1)  # wxyz -> xyzw
