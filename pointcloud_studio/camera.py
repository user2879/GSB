"""Camera pose and row-major projection matrices shared by render/selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    eye: tuple[float, float, float] = (8.0, 4.5, 8.0)
    target: tuple[float, float, float] = (0.0, 1.0, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_y_degrees: float = 52.0
    near: float = 0.05
    far: float = 80.0


def perspective(fov_y_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_degrees) / 2.0)
    result = np.zeros((4, 4), dtype=np.float64)
    result[0, 0] = f / max(aspect, 1.0e-9)
    result[1, 1] = f
    result[2, 2] = (far + near) / (near - far)
    result[2, 3] = (2.0 * far * near) / (near - far)
    result[3, 2] = -1.0
    return result


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    true_up = np.cross(side, forward)
    result = np.eye(4, dtype=np.float64)
    result[0, :3] = side
    result[1, :3] = true_up
    result[2, :3] = -forward
    result[0, 3] = -float(np.dot(side, eye))
    result[1, 3] = -float(np.dot(true_up, eye))
    result[2, 3] = float(np.dot(forward, eye))
    return result

