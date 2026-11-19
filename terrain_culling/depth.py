"""Deterministic software depth rendering and conservative Hi-Z pyramid.

The depth pass ray-marches each pixel against the resident heightfield and
stores *linear view-space depth*. The Hi-Z down-sample always takes the maximum
depth of the 2x2 footprint; combined with an added per-level error bound this is
conservative for the "is this block behind already drawn geometry?" test.

The software path is the byte-reproducible reference used by the demo. A live
EGL/GLES backend is probed separately in ``gl_context.py`` and never changes the
numeric reference output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    ABS_FLOOR, F32_REL_EPS, MAX_DEPTH_BIAS,
)
from .errors import HiZLevelError, DepthBiasError


def frame_forward(camera_frame) -> np.ndarray:
    view = np.asarray(camera_frame.view, dtype=np.float64)
    # Camera looks down its own -Z, so world forward is the negated row 2.
    return -view[2, 0:3]
from .terrain import height_to_world, WORLD_EXTENT, MAX_ELEVATION


@dataclass(frozen=True)
class HiZLevelInfo:
    level: int
    width: int
    height: int
    texel_world_x: float
    texel_world_z: float
    depth_error_bound: float
    sampling_footprint_world: float
    pixel_half_angle_rad: float = 0.0

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "width": self.width,
            "height": self.height,
            "texel_world_x": self.texel_world_x,
            "texel_world_z": self.texel_world_z,
            "depth_error_bound": self.depth_error_bound,
            "sampling_footprint_world": self.sampling_footprint_world,
            "pixel_half_angle_rad": self.pixel_half_angle_rad,
        }


class DepthRenderer:
    def __init__(self, width: int, height: int, raymarch_steps: int,
                 max_levels: int, max_depth_bias: float = MAX_DEPTH_BIAS):
        self.width = int(width)
        self.height = int(height)
        self.raymarch_steps = int(raymarch_steps)
        self.max_levels = int(max_levels)
        self.max_depth_bias = float(max_depth_bias)

    def render_depth(self, dataset, camera_frame, resident_ids: set[str]) -> np.ndarray:
        """Ray-march linear depth; non-resident cells contribute no occluder."""
        heights = dataset.heights
        size = dataset.size
        cell = WORLD_EXTENT / size
        eye = np.asarray(camera_frame.eye, dtype=np.float64)
        vp_inv = np.linalg.inv(camera_frame.view_projection)

        xs = (np.arange(self.width, dtype=np.float64) + 0.5) / self.width * 2.0 - 1.0
        # Row 0 is the top of the image, i.e. NDC y = +1, so the row axis runs
        # from +1 at the top to -1 at the bottom. Getting this sign wrong flips
        # every ray vertically and produces false occlusion.
        ys = 1.0 - (np.arange(self.height, dtype=np.float64) + 0.5) / self.height * 2.0
        px, py = np.meshgrid(xs, ys)
        ndc = np.stack([px.ravel(), py.ravel(), -np.ones(px.size)], axis=1)
        homog = np.concatenate([ndc, np.ones((ndc.shape[0], 1))], axis=1)
        rays = homog @ vp_inv.T
        rays = rays[:, :3] / np.maximum(rays[:, 3:4], ABS_FLOOR)
        directions = rays - eye
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), ABS_FLOOR)

        # Resident mask in height-sample grid (union of resident block windows).
        resident_grid = np.zeros((size, size), dtype=bool)
        for block_id in resident_ids:
            block = dataset.get_block(block_id)
            resident_grid[block.y0:block.y1, block.x0:block.x1] = True

        # Bound each ray's reach and choose a step size that never exceeds one
        # horizontal grid cell. A vertical-only bound lets near-horizontal rays
        # leap over ridges, so the cell constraint is the dominant one.
        horizontal = np.maximum(
            np.linalg.norm(directions[:, [0, 2]], axis=1), ABS_FLOOR)
        max_t = (WORLD_EXTENT * 2.5) / np.maximum(np.abs(directions[:, 1]), 0.15)
        cell_step_t = cell / horizontal
        dt = np.minimum(max_t / self.raymarch_steps, cell_step_t)
        depth = np.full(directions.shape[0], np.inf, dtype=np.float64)
        hit_t = np.full(directions.shape[0], np.inf, dtype=np.float64)
        forward = directions @ frame_forward(camera_frame)

        def ground_at(points: np.ndarray):
            gx = np.floor(points[:, 0] / cell).astype(np.int64)
            gy = np.floor(points[:, 2] / cell).astype(np.int64)
            inside = (gx >= 0) & (gx < size) & (gy >= 0) & (gy < size)
            values = np.full(points.shape[0], np.inf, dtype=np.float64)
            covered = np.zeros(points.shape[0], dtype=bool)
            values[inside] = height_to_world(heights[gy[inside], gx[inside]])
            covered[inside] = resident_grid[gy[inside], gx[inside]]
            return values, covered

        prev = np.repeat(eye[None, :], directions.shape[0], axis=0)
        prev_ground, _ = ground_at(prev)
        # Fixed step count; a hit is a ground crossing between segment ends so
        # a large vertical step can never tunnel through the terrain.
        for step in range(self.raymarch_steps):
            t = (step + 1) * dt
            active = np.isinf(hit_t)
            if not active.any():
                break
            pts = eye + directions * t[:, None]
            ground, covered = ground_at(pts)
            front_facing = forward > 0.0
            crossed = (active & front_facing & covered
                       & (pts[:, 1] <= ground)
                       & (prev[:, 1] > prev_ground))
            hit_idx = np.where(crossed)[0]
            if hit_idx.size:
                denom = prev[hit_idx, 1] - pts[hit_idx, 1]
                denom = np.where(np.abs(denom) < ABS_FLOOR, ABS_FLOOR, denom)
                frac = np.clip(
                    (prev[hit_idx, 1] - prev_ground[hit_idx]) / denom, 0.0, 1.0)
                t_hit = (step + frac) * dt[hit_idx]
                hit_t[hit_idx] = t_hit
                # Camera-space linear depth = ray distance * forward component.
                depth[hit_idx] = t_hit * forward[hit_idx]
            prev = pts
            prev_ground = ground
        return depth.reshape(self.height, self.width)

    def build_hiz(self, depth: np.ndarray, fov_rad: float | None = None
                  ) -> tuple[list[np.ndarray], list[HiZLevelInfo]]:
        """Build max-depth mip pyramid and a recomputable per-level error table."""
        levels = [np.nan_to_num(depth, nan=np.inf, posinf=np.inf).astype(np.float64)]
        infos: list[HiZLevelInfo] = []
        cur = levels[0]
        world_per_pixel_x = WORLD_EXTENT / self.width
        world_per_pixel_z = WORLD_EXTENT / self.height
        # Base level: raymarch step overshoot plus nearest-grid sampling bound.
        base_footprint = max(world_per_pixel_x, world_per_pixel_z)
        # Half-angle subtended by one base-level pixel (vertical FOV). At a
        # candidate depth D the pixel footprint spans D*tan(half-angle); this is
        # the angular aliasing term that lets a grazing block share a pixel with
        # a nearer occluder and must be added to the decision error.
        base_half_angle = (float(fov_rad) / self.height
                           if fov_rad is not None else 0.0)
        for level in range(self.max_levels):
            h, w = cur.shape
            footprint = base_footprint * (2 ** level)
            # Error grows with the conservative 2x2 max footprint and float32
            # rounding of the depth compare chain.
            # Raymarch step overshoot scales with level-0 footprint; each 2x2
            # max down-sample adds one more conservative footprint term, and a
            # float32 rounding term bounds the depth compare chain.
            half_angle = base_half_angle * (2 ** level)
            depth_error = (base_footprint * 0.05 * (level + 1)
                           + footprint * F32_REL_EPS + ABS_FLOOR)
            infos.append(HiZLevelInfo(
                level=level, width=w, height=h,
                texel_world_x=world_per_pixel_x * (2 ** level),
                texel_world_z=world_per_pixel_z * (2 ** level),
                depth_error_bound=float(depth_error),
                sampling_footprint_world=float(footprint),
                pixel_half_angle_rad=float(half_angle),
            ))
            if h <= 2 or w <= 2 or level == self.max_levels - 1:
                break
            padded_h = h if h % 2 == 0 else h + 1
            padded_w = w if w % 2 == 0 else w + 1
            padded = np.full((padded_h, padded_w), -np.inf, dtype=np.float64)
            padded[:h, :w] = cur
            # Max-depth keeps occlusion conservative (farther stored depth).
            nxt = np.maximum(np.maximum(padded[0::2, 0::2], padded[1::2, 0::2]),
                             np.maximum(padded[0::2, 1::2], padded[1::2, 1::2]))
            nxt = np.where(np.isneginf(nxt), np.inf, nxt)
            levels.append(nxt)
            cur = nxt
        return levels, infos

    def check_level(self, level: int, infos: list[HiZLevelInfo]) -> None:
        if not isinstance(level, int) or level < 0 or level >= len(infos):
            raise HiZLevelError(
                f"Hi-Z level {level} out of range [0, {len(infos) - 1}]",
                context={"level": level, "max_level": len(infos) - 1},
            )

    def check_depth_bias(self, bias: float) -> None:
        if not np.isfinite(bias) or bias < 0.0 or bias > self.max_depth_bias:
            raise DepthBiasError(
                f"depth bias {bias} exceeds limit {self.max_depth_bias}",
                context={"bias": float(bias), "limit": self.max_depth_bias},
            )


def groundtruth_block_visible(dataset, camera, camera_frame, block,
                              march_steps: int = 1200) -> bool:
    """Independent conservative ground-truth visibility test.

    A block is visible if any in-frustum surface point has an unobstructed
    line of sight to the camera. It uses the full heightfield (not the resident
    subset) and a fine step count, so it is independent of the Hi-Z path and
    serves as the no-false-culling oracle in tests.
    """
    from .culling import block_surface_points
    heights = dataset.heights
    size = dataset.size
    cell = WORLD_EXTENT / size
    eye = np.asarray(camera_frame.eye, dtype=np.float64)
    # Test the actual block surface (not the tall AABB corners): a high corner
    # can peek over a ridge while every renderable surface point stays hidden.
    points = block_surface_points(block, heights, grid_step=12)
    ndc = camera.world_to_ndc(camera_frame.view_projection, points)
    out_x = np.all(ndc[:, 0] > 1.0) or np.all(ndc[:, 0] < -1.0)
    out_y = np.all(ndc[:, 1] > 1.0) or np.all(ndc[:, 1] < -1.0)
    out_z = np.all(ndc[:, 2] > 1.0) or np.all(ndc[:, 2] < -1.0)
    if out_x or out_y or out_z:
        return False
    forward = frame_forward(camera_frame)
    in_frustum = ((np.abs(ndc[:, 0]) <= 1.0)
                  & (np.abs(ndc[:, 1]) <= 1.0)
                  & (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0))
    points = points[in_frustum]
    if points.shape[0] == 0:
        return False
    for point in points:
        direction = point - eye
        length = float(np.linalg.norm(direction))
        if length < ABS_FLOOR:
            return True
        direction /= length
        point_depth = float((point - eye) @ forward)
        blocked = False
        for step in range(1, march_steps):
            t = length * step / march_steps
            sample = eye + direction * t
            gx = int(np.floor(sample[0] / cell))
            gy = int(np.floor(sample[2] / cell))
            if not (0 <= gx < size and 0 <= gy < size):
                break
            ground = float(height_to_world(heights[gy, gx]))
            sample_depth = float((sample - eye) @ forward)
            # Nearest-grid sampling can underestimate ground by a sub-cell
            # amount; treat near-grazing rays (within one vertical cell) as
            # blocked so the oracle stays conservative about visibility.
            grazing_tolerance = cell * 0.5
            if (sample[1] <= ground + grazing_tolerance
                    and sample_depth < point_depth - cell):
                blocked = True
                break
        if not blocked:
            return True
    return False
