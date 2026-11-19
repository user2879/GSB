"""Camera model with explicitly typed, fixed-order float64 matrix arithmetic.

Coordinate system: right-handed, +X east, +Y up, +Z south. The view-projection
matrix is always built in the order

    P (float64) * V (float64, Rz * Ry * Rx convention) * T(-eye)

and points are transformed as ``M @ [x, y, z, 1]``. The numeric type and the
operation order are part of the deterministic contract; reordering them changes
rounding and is treated as an algorithm change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    TRACK_SEED, WORLD_EXTENT, MAX_ELEVATION, F32_REL_EPS, ABS_FLOOR,
    FRUSTUM_NEAR, FRUSTUM_FAR, VERTICAL_FOV_RAD,
)


def perspective(fov_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.float64(fov_rad) * 0.5)
    p = np.zeros((4, 4), dtype=np.float64)
    p[0, 0] = f / np.float64(aspect)
    p[1, 1] = f
    p[2, 2] = (np.float64(far) + np.float64(near)) / (np.float64(near) - np.float64(far))
    p[2, 3] = (2.0 * np.float64(far) * np.float64(near)) / (np.float64(near) - np.float64(far))
    p[3, 2] = -1.0
    return p


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    forward = forward / np.maximum(np.linalg.norm(forward), ABS_FLOOR)
    right = np.cross(forward, up)
    right = right / np.maximum(np.linalg.norm(right), ABS_FLOOR)
    true_up = np.cross(right, forward)
    view = np.eye(4, dtype=np.float64)
    view[0, 0:3] = right
    view[1, 0:3] = true_up
    view[2, 0:3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(true_up, eye)
    view[2, 3] = np.dot(forward, eye)
    return view


@dataclass(frozen=True)
class CameraFrame:
    index: int
    eye: tuple
    target: tuple
    view: np.ndarray
    projection: np.ndarray
    view_projection: np.ndarray

    def to_param_dict(self) -> dict:
        return {
            "index": self.index,
            "eye": [float(v) for v in self.eye],
            "target": [float(v) for v in self.target],
            "view_projection": [[float(v) for v in row] for row in self.view_projection],
        }


class Camera:
    def __init__(self, width: int, height: int,
                 near: float = FRUSTUM_NEAR, far: float = FRUSTUM_FAR,
                 fov_rad: float = VERTICAL_FOV_RAD):
        self.width = int(width)
        self.height = int(height)
        self.near = float(near)
        self.far = float(far)
        self.fov_rad = float(fov_rad)
        self._eye = np.zeros(3, dtype=np.float64)
        self.projection = perspective(self.fov_rad, self.width / self.height,
                                      self.near, self.far)

    def frame(self, index: int, eye: np.ndarray, target: np.ndarray) -> CameraFrame:
        view = look_at(eye, target, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        # Fixed multiplication order: P * V.
        vp = self.projection @ view
        self._eye = np.asarray(eye, dtype=np.float64).copy()
        return CameraFrame(index=index,
                           eye=tuple(float(v) for v in np.asarray(eye)),
                           target=tuple(float(v) for v in np.asarray(target)),
                           view=view, projection=self.projection, view_projection=vp)

    def world_to_ndc(self, vp: np.ndarray, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
        clip = homog @ vp.T
        w = clip[:, 3:4]
        # Guard divide-by-zero explicitly rather than relying on NaN propagation.
        safe_w = np.where(np.abs(w) < ABS_FLOOR, np.sign(w) * ABS_FLOOR + ABS_FLOOR, w)
        return clip[:, :3] / safe_w

    def world_depth(self, view: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Positive camera-space linear depth (forward axis), float64."""
        pts = np.asarray(points, dtype=np.float64)
        homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
        view_pts = homog @ view.T
        return -view_pts[:, 2]


def generate_track(frame_count: int, size: int, seed: int = TRACK_SEED) -> list[CameraFrame]:
    """A fixed-seed roaming spline: two control loops blended deterministically."""
    rng = np.random.Generator(np.random.PCG64(seed + size))
    extent = WORLD_EXTENT
    # Control points stay inside the map and above the highest possible hill.
    ctrl_eye = np.stack([
        np.linspace(extent * 0.15, extent * 0.85, 8),
        rng.uniform(MAX_ELEVATION * 0.9, MAX_ELEVATION * 1.4, 8),
        rng.uniform(extent * 0.1, extent * 0.9, 8),
    ], axis=1)
    ctrl_target = np.stack([
        ctrl_eye[:, 0] + rng.uniform(-300, 300, 8),
        rng.uniform(40.0, MAX_ELEVATION * 0.5, 8),
        ctrl_eye[:, 2] + rng.uniform(-300, 300, 8),
    ], axis=1)

    def catmull_rom(points: np.ndarray, t: float) -> np.ndarray:
        seg = len(points) - 1
        pos = min(max(t, 0.0), 1.0) * seg
        i0 = int(np.floor(pos))
        local = pos - i0
        if i0 >= seg:
            return points[-1].copy()
        p0 = points[max(i0 - 1, 0)]
        p1 = points[i0]
        p2 = points[i0 + 1]
        p3 = points[min(i0 + 2, seg)]
        t2 = local * local
        t3 = t2 * local
        return 0.5 * ((2 * p1) + (-p0 + p2) * local
                      + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                      + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)

    camera = Camera(192, 128)
    frames: list[CameraFrame] = []
    for i in range(frame_count):
        t = i / max(frame_count - 1, 1)
        eye = catmull_rom(ctrl_eye, t)
        target = catmull_rom(ctrl_target, t)
        frames.append(camera.frame(i, eye, target))
    return frames


def projection_error_bounds(center_distance: float, radius: float) -> dict:
    """Float32-equivalent absolute error bounds for a transformed AABB sphere.

    The bound follows the standard first-order model
    ``|err| <= n * eps * (|M| |x| + ...)`` simplified to a relative distance
    term plus an absolute floor, and is intentionally conservative.
    """
    center_distance = max(float(center_distance), ABS_FLOOR)
    radius = max(float(radius), 0.0)
    matrix_abs = F32_REL_EPS * (center_distance + radius)
    point_abs = F32_REL_EPS * radius
    abs_error = max(matrix_abs + point_abs, ABS_FLOOR)
    return {
        "center_distance": center_distance,
        "radius": radius,
        "matrix_abs_error": matrix_abs,
        "point_abs_error": point_abs,
        "abs_error": abs_error,
        "relative_error": abs_error / center_distance,
    }
