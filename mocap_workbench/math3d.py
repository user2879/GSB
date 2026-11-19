"""Quaternion and vector helpers with explicit shortest-path semantics."""

from __future__ import annotations

import math

import numpy as np


EPSILON = 1e-12


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= EPSILON:
        return q.copy()
    return q / norm


def align_sign(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    q = np.asarray(candidate, dtype=np.float64)
    if float(np.dot(reference, q)) < 0.0:
        return -q
    return q


def quat_dot_abs(left: np.ndarray, right: np.ndarray) -> float:
    return abs(float(np.dot(left, right)))


def quat_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    dot = min(1.0, max(-1.0, quat_dot_abs(left, right)))
    return math.degrees(2.0 * math.acos(dot))


def slerp(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-path SLERP with right-sided sign alignment to the left endpoint."""

    q0 = normalize_quat(np.asarray(left, dtype=np.float64))
    q1 = align_sign(q0, normalize_quat(np.asarray(right, dtype=np.float64)))
    dot = max(-1.0, min(1.0, float(np.dot(q0, q1))))
    if dot > 0.9999995:
        result = q0 + alpha * (q1 - q0)
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        result = (
            math.sin((1.0 - alpha) * theta) / sin_theta * q0
            + math.sin(alpha * theta) / sin_theta * q1
        )
    return normalize_quat(result)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def quat_log(q: np.ndarray) -> np.ndarray:
    q = align_sign(np.array([1.0, 0.0, 0.0, 0.0]), normalize_quat(q))
    vector = q[1:]
    length = float(np.linalg.norm(vector))
    if length <= EPSILON:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(length, max(-1.0, min(1.0, q[0])))
    return vector * (angle / length)


def relative_rotvec(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    return quat_log(quat_multiply(quat_conjugate(previous), current))


def angle_between_vectors_deg(left: np.ndarray, right: np.ndarray) -> float:
    ln = float(np.linalg.norm(left))
    rn = float(np.linalg.norm(right))
    if ln <= EPSILON or rn <= EPSILON:
        return 0.0
    cosine = float(np.dot(left, right) / (ln * rn))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

