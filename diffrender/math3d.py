"""Small deterministic 3D helpers.

The project intentionally uses plain Python floats rather than relying on a
GPU or BLAS implementation.  IEEE-754 binary64 operations are deterministic for
the operations used here, and all containers emitted to disk are sorted.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


def vadd(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vmul(a: Sequence[float], s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Sequence[float]) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Sequence[float]) -> Vec3:
    n = norm(a)
    if n <= 1.0e-15:
        raise ValueError("cannot normalize a zero-length vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def mmul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(
        tuple(a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c] for c in range(3))
        for r in range(3)
    )  # type: ignore[return-value]


def mcol(m: Mat3, c: int) -> Vec3:
    return (m[0][c], m[1][c], m[2][c])


def _rotation_y(angle: float) -> Mat3:
    c, s = math.cos(angle), math.sin(angle)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def _rotation_y_derivative(angle: float) -> Mat3:
    c, s = math.cos(angle), math.sin(angle)
    return ((-s, 0.0, c), (0.0, 0.0, 0.0), (-c, 0.0, -s))


def _rotation_x(angle: float) -> Mat3:
    c, s = math.cos(angle), math.sin(angle)
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def _rotation_x_derivative(angle: float) -> Mat3:
    c, s = math.cos(angle), math.sin(angle)
    return ((0.0, 0.0, 0.0), (0.0, -s, -c), (0.0, c, -s))


def _rotation_z(angle: float) -> Mat3:
    c, s = math.cos(angle), math.sin(angle)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _rotation_z_derivative(angle: float) -> Mat3:
    c, s = math.cos(angle), math.sin(angle)
    return ((-s, -c, 0.0), (c, -s, 0.0), (0.0, 0.0, 0.0))


def camera_rotation(yaw: float, pitch: float, roll: float) -> Mat3:
    """World-from-camera rotation for yaw (Y), pitch (X), then roll (Z)."""

    return mmul(mmul(_rotation_y(yaw), _rotation_x(pitch)), _rotation_z(roll))


def camera_basis(
    yaw: float,
    pitch: float,
    roll: float,
    with_derivatives: bool = False,
):
    """Return right/up/forward and optionally their analytic derivatives.

    The derivative tuple is ``(d_yaw, d_pitch, d_roll)`` and each item contains
    derivative vectors in the same right/up/forward order.
    """

    ry, rx, rz = _rotation_y(yaw), _rotation_x(pitch), _rotation_z(roll)
    r = mmul(mmul(ry, rx), rz)
    right, up = mcol(r, 0), mcol(r, 1)
    forward = (-r[0][2], -r[1][2], -r[2][2])
    if not with_derivatives:
        return right, up, forward

    dry = _rotation_y_derivative(yaw)
    drx = _rotation_x_derivative(pitch)
    drz = _rotation_z_derivative(roll)
    derivs = (
        mmul(mmul(dry, rx), rz),
        mmul(mmul(ry, drx), rz),
        mmul(mmul(ry, rx), drz),
    )
    dbasis: list[tuple[Vec3, Vec3, Vec3]] = []
    for d in derivs:
        dbasis.append(
            (mcol(d, 0), mcol(d, 1), (-d[0][2], -d[1][2], -d[2][2]))
        )
    return right, up, forward, tuple(dbasis)


def component_basis(component: int) -> Vec3:
    if component == 0:
        return (1.0, 0.0, 0.0)
    if component == 1:
        return (0.0, 1.0, 0.0)
    if component == 2:
        return (0.0, 0.0, 1.0)
    raise ValueError("component must be 0, 1, or 2")


def rotate_transpose(vectors: Iterable[Vec3], vector: Sequence[float]) -> Vec3:
    """Dot a vector with each basis vector in order."""

    return tuple(dot(vector, axis) for axis in vectors)  # type: ignore[return-value]

