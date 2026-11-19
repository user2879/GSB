"""三维向量工具（numpy float64 数组，运算顺序固定）。"""

from __future__ import annotations

import numpy as np

from .errors import RayDegenerateError

EPS = np.float64(1e-9)
RAY_EPS = np.float64(1e-7)
ZERO_NORM_TOL = np.float64(1e-15)


def as_vec3(values) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float64)
    if vec.shape != (3,):
        raise ValueError(f"expected length-3 vector, got shape {vec.shape}")
    return vec


def dot(a: np.ndarray, b: np.ndarray) -> np.float64:
    return np.float64(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def norm(a: np.ndarray) -> np.float64:
    return np.sqrt(np.float64(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]))


def normalize(a: np.ndarray, *, strict: bool = True) -> np.ndarray:
    """归一化；方向向量长度退化时按 E_RAY_DEGENERATE 归类。"""
    length = norm(a)
    if length <= ZERO_NORM_TOL:
        if strict:
            raise RayDegenerateError(
                "cannot normalize a zero-length direction vector",
                {"vector": [float(x) for x in a], "norm": float(length)},
            )
        return np.zeros(3, dtype=np.float64)
    return np.array(
        [float(a[0]) / float(length), float(a[1]) / float(length), float(a[2]) / float(length)],
        dtype=np.float64,
    )


def reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """镜面反射：入射方向 d 与朝向入射侧的法线 n。"""
    factor = np.float64(2.0) * dot(direction, normal)
    return np.array(
        [
            direction[0] - factor * normal[0],
            direction[1] - factor * normal[1],
            direction[2] - factor * normal[2],
        ],
        dtype=np.float64,
    )
