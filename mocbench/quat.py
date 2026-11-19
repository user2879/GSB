import math

import numpy as np

EPS = 1e-12


def normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def is_unit(q, tol=1e-6):
    return abs(float(np.dot(q, q)) - 1.0) <= tol


def mul(a, b):
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


def conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def rotate(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u = q[1:]
    s = q[0]
    return 2.0 * np.dot(u, v) * u + (s * s - np.dot(u, u)) * v + 2.0 * s * np.cross(u, v)


def slerp(q0, q1, t):
    q0 = normalize(np.asarray(q0, dtype=np.float64))
    q1 = normalize(np.asarray(q1, dtype=np.float64))
    d = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 1.0 - EPS:
        return normalize(q0 + t * (q1 - q0))
    theta0 = math.acos(d)
    sin_theta0 = math.sin(theta0)
    theta = theta0 * t
    w0 = math.cos(theta) - d * math.sin(theta) / sin_theta0
    w1 = math.sin(theta) / sin_theta0
    return normalize(w0 * q0 + w1 * q1)


def axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < EPS or abs(angle) < EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / n
    half = angle * 0.5
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def rotation_between(a, b):
    a = normalize_vec(np.asarray(a, dtype=np.float64))
    b = normalize_vec(np.asarray(b, dtype=np.float64))
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    axis = np.cross(a, b)
    sin_ang = np.linalg.norm(axis)
    if sin_ang < EPS:
        if c > 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        alt = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            alt = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = normalize_vec(np.cross(a, alt))
        return axis_angle(axis, math.pi)
    angle = math.atan2(sin_ang, c)
    return axis_angle(axis, angle)


def normalize_vec(v):
    n = np.linalg.norm(v)
    if n < EPS:
        return np.zeros(3, dtype=np.float64)
    return v / n


def relative_rotation(parent_world, child_world):
    return mul(conj(parent_world), child_world)
