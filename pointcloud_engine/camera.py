"""Pinhole camera construction and point projection.

Coordinate convention
---------------------
World coordinates are right-handed. The camera looks down its local **-Z**
axis (OpenGL convention): ``+Z_cam`` points *behind* the camera so visible
points have a negative camera-space z. Camera-space x points right and y
points up.

Depth used for occlusion is the positive distance along the view axis,
``depth = -z_cam`` (a world-units quantity). All attenuation is applied in
the same world units, which keeps occlusion/grayscale ordering invariant
under a uniform change of world scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import EngineError


@dataclass(frozen=True)
class Camera:
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    vfov_deg: float
    aspect: float
    near: float
    reference_distance: float
    basis: np.ndarray  # 3x3 rows [right, up_cam, forward]
    tan_half_v: float
    tan_half_h: float

    @classmethod
    def build(cls, position, target, up, vfov_deg: float, aspect: float, near: float = 1e-6) -> "Camera":
        position = np.asarray(position, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)

        forward = target - position
        forward_norm = np.linalg.norm(forward)
        if not (forward_norm > 0.0):
            raise EngineError("E_CAMERA", "camera position and target coincide")
        forward = forward / forward_norm

        up_norm_value = np.linalg.norm(up)
        if not (up_norm_value > 0.0):
            raise EngineError("E_CAMERA", "camera up vector is the zero vector")
        up = up / up_norm_value

        # Parallel (or anti-parallel) up and view direction is degenerate.
        alignment = abs(float(np.dot(up, forward)))
        if alignment > 1.0 - 1e-10:
            raise EngineError("E_CAMERA", "up direction is parallel to the view direction")

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up_cam = np.cross(right, forward)
        up_cam = up_cam / np.linalg.norm(up_cam)
        # Camera looks down its local -Z, so the third row is -forward.
        basis = np.vstack([right, up_cam, -forward])

        half_v = 0.5 * math.radians(vfov_deg)
        tan_half_v = math.tan(half_v)
        tan_half_h = aspect * tan_half_v
        reference_distance = float(np.linalg.norm(target - position))
        return cls(position, target, up, float(vfov_deg), float(aspect), float(near),
                   reference_distance, basis, float(tan_half_v), float(tan_half_h))

    def to_camera(self, world_points: np.ndarray) -> np.ndarray:
        """Transform world points to camera space (visible points have z<0)."""
        return (world_points - self.position) @ self.basis.T

    def project(self, points_cam: np.ndarray):
        """Project camera-space points onto normalized device coordinates.

        Returns ``(ndc_x, ndc_y, depth)`` where ndc coordinates are in the
        closed-ish range [-1, 1] for points inside the frustum and depth is
        the positive view-axis distance in world units.
        """
        depth = -points_cam[:, 2]
        safe_depth = np.where(depth > 0, depth, np.nan)
        ndc_x = points_cam[:, 0] / (self.tan_half_h * safe_depth)
        ndc_y = points_cam[:, 1] / (self.tan_half_v * safe_depth)
        return ndc_x, ndc_y, depth

    def screen_radius(self, radius_world: float, depth: np.ndarray,
                      grid_half_height: float) -> np.ndarray:
        """Project an isotropic world-space point radius to supersample pixels.

        A sphere of radius ``r`` at view distance ``d`` subtends a vertical
        half-angle ``atan(r/d)``. The vertical half-FOV spans
        ``grid_half_height`` supersample cells, so dividing by the angular
        size of one half-cell yields the footprint radius in cells.
        """
        angle = np.arctan(radius_world / depth)
        per_half_cell = math.radians(self.vfov_deg) * 0.5 / float(grid_half_height)
        return angle / per_half_cell
