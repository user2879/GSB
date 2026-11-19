from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .spec import Camera, TaskSpec, VolumeSpec

_EPS = 1e-12
SAMPLE_BUFFER_BYTES_PER_VALUE = 64


@dataclass(frozen=True)
class CameraRays:
    origin: np.ndarray
    directions: np.ndarray  # (height, width, 3), normalized
    t_near: np.ndarray  # (height, width), millimeters, clamped to >= 0
    t_far: np.ndarray  # (height, width), millimeters
    hit: np.ndarray  # (height, width), boolean
    max_depth_mm: float

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.directions.shape[0]), int(self.directions.shape[1])


@dataclass(frozen=True)
class BlockPlan:
    index: int
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int
    pixel_count: int
    voxel_bounds_min: tuple[int, int, int] | None
    voxel_bounds_max: tuple[int, int, int] | None
    ray_hit_count: int
    max_steps: int
    sample_buffer_bytes: int

    @property
    def pixel_bounds(self) -> dict[str, list[int]]:
        return {"x": [self.x0, self.x1], "y": [self.y0, self.y1]}

    @property
    def voxel_bounds(self) -> dict[str, list[int]] | None:
        if self.voxel_bounds_min is None or self.voxel_bounds_max is None:
            return None
        return {
            "min": list(self.voxel_bounds_min),
            "max": list(self.voxel_bounds_max),
        }


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPS:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / norm


def build_camera_rays(camera: Camera, width: int, height: int, volume: VolumeSpec) -> CameraRays:
    origin = np.asarray(camera.position, dtype=np.float64)
    target = np.asarray(camera.target, dtype=np.float64)
    up = np.asarray(camera.up, dtype=np.float64)
    forward = _normalized(target - origin)
    right = _normalized(np.cross(forward, up))
    true_up = _normalized(np.cross(right, forward))

    tan_half = math.tan(math.radians(camera.fov_y_deg) * 0.5)
    px = ((np.arange(width, dtype=np.float64) + 0.5) / width) * 2.0 - 1.0
    py = 1.0 - ((np.arange(height, dtype=np.float64) + 0.5) / height) * 2.0
    screen_x = px * camera.aspect * tan_half
    screen_y = py * tan_half
    directions = (
        forward[None, None, :]
        + screen_x[None, :, None] * right[None, None, :]
        + screen_y[:, None, None] * true_up[None, None, :]
    )
    directions = directions / np.linalg.norm(directions, axis=2, keepdims=True)
    t_near, t_far, hit = ray_box_intersections(origin, directions, volume)
    corners = []
    for x in (0.0, volume.dims[0] * volume.spacing_mm[0]):
        for y in (0.0, volume.dims[1] * volume.spacing_mm[1]):
            for z in (0.0, volume.dims[2] * volume.spacing_mm[2]):
                corners.append((x, y, z))
    max_depth = max(float(np.linalg.norm(np.asarray(corner) - origin)) for corner in corners)
    return CameraRays(
        origin=origin,
        directions=directions,
        t_near=t_near,
        t_far=t_far,
        hit=hit,
        max_depth_mm=max_depth,
    )


def ray_box_intersections(
    origin: np.ndarray, directions: np.ndarray, volume: VolumeSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    box_max = np.asarray(volume.dims, dtype=np.float64) * np.asarray(
        volume.spacing_mm, dtype=np.float64
    )
    shape = directions.shape[:-1]
    t_enter = np.full(shape, -np.inf, dtype=np.float64)
    t_exit = np.full(shape, np.inf, dtype=np.float64)
    valid = np.ones(shape, dtype=bool)

    for axis in range(3):
        direction = directions[..., axis]
        parallel = np.abs(direction) <= _EPS
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = (0.0 - origin[axis]) / direction
            t1 = (box_max[axis] - origin[axis]) / direction
        lower = np.minimum(t0, t1)
        upper = np.maximum(t0, t1)
        t_enter = np.where(parallel, t_enter, np.maximum(t_enter, lower))
        t_exit = np.where(parallel, t_exit, np.minimum(t_exit, upper))
        valid &= (~parallel) | ((origin[axis] >= 0.0) & (origin[axis] <= box_max[axis]))

    t_enter = np.maximum(t_enter, 0.0)
    hit = valid & (t_exit >= t_enter)
    return t_enter, t_exit, hit


def make_block_plans(task: TaskSpec, rays: CameraRays) -> list[BlockPlan]:
    plans: list[BlockPlan] = []
    index = 0
    for row in range(task.block_rows):
        for col in range(task.block_cols):
            x0 = col * task.block_width
            x1 = x0 + task.block_width
            y0 = row * task.block_height
            y1 = y0 + task.block_height
            hit = rays.hit[y0:y1, x0:x1]
            t0 = rays.t_near[y0:y1, x0:x1]
            t1 = rays.t_far[y0:y1, x0:x1]
            hit_count = int(np.count_nonzero(hit))
            bounds_min: tuple[int, int, int] | None = None
            bounds_max: tuple[int, int, int] | None = None
            max_steps = 0
            if hit_count:
                dirs = rays.directions[y0:y1, x0:x1][hit]
                starts = t0[hit]
                ends = t1[hit]
                p0 = rays.origin[None, :] + dirs * starts[:, None]
                p1 = rays.origin[None, :] + dirs * ends[:, None]
                spacing = np.asarray(task.volume.spacing_mm, dtype=np.float64)
                u0 = p0 / spacing[None, :]
                u1 = p1 / spacing[None, :]
                low_float = np.minimum(u0.min(axis=0), u1.min(axis=0))
                high_float = np.maximum(u0.max(axis=0), u1.max(axis=0))
                dims = np.asarray(task.volume.dims, dtype=np.int64)
                low = np.floor(low_float + 1e-10).astype(np.int64)
                high = np.floor(high_float - 1e-10).astype(np.int64)
                low = np.clip(low, 0, dims - 1)
                high = np.clip(high, 0, dims - 1)
                bounds_min = (int(low[0]), int(low[1]), int(low[2]))
                bounds_max = (int(high[0]), int(high[1]), int(high[2]))
                deltas = ends - starts
                counts = np.floor(deltas / task.sampling_step_mm + 1e-12).astype(np.int64) + 1
                max_steps = max(0, int(counts.max(initial=0)))

            pixel_count = task.block_width * task.block_height
            buffer_bytes = pixel_count * max(1, max_steps) * SAMPLE_BUFFER_BYTES_PER_VALUE
            buffer_bytes += pixel_count * 128
            plans.append(
                BlockPlan(
                    index=index,
                    row=row,
                    col=col,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    pixel_count=pixel_count,
                    voxel_bounds_min=bounds_min,
                    voxel_bounds_max=bounds_max,
                    ray_hit_count=hit_count,
                    max_steps=max_steps,
                    sample_buffer_bytes=buffer_bytes,
                )
            )
            index += 1
    return plans


def count_rays_hitting_changed(
    task: TaskSpec, rays: CameraRays, plan: BlockPlan, changed: np.ndarray
) -> int:
    if plan.voxel_bounds_min is None:
        return 0
    dims = np.asarray(task.volume.dims, dtype=np.int64)
    low = np.maximum(np.asarray(plan.voxel_bounds_min, dtype=np.int64) - 1, 0)
    high = np.minimum(np.asarray(plan.voxel_bounds_max, dtype=np.int64) + 1, dims - 1)
    x0, y0, z0 = (int(low[0]), int(low[1]), int(low[2]))
    x1, y1, z1 = (int(high[0]), int(high[1]), int(high[2]))
    changed_slice = changed[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
    if not bool(changed_slice.any()):
        return 0
    coords = np.argwhere(changed_slice)
    coords[:, 0] += z0
    coords[:, 1] += y0
    coords[:, 2] += x0
    spacing = np.asarray(task.volume.spacing_mm, dtype=np.float64)
    # Trilinear interpolation at voxel-center coordinate u uses voxel i when
    # u is in [i-0.5, i+1.5]. These support boxes are intentionally a little
    # wider than one cell; this is the actual geometric dependency footprint.
    lower = (coords[:, [2, 1, 0]].astype(np.float64) - 0.5) * spacing[None, :]
    upper = (coords[:, [2, 1, 0]].astype(np.float64) + 1.5) * spacing[None, :]

    hit = rays.hit[plan.y0 : plan.y1, plan.x0 : plan.x1].reshape(-1)
    directions = rays.directions[plan.y0 : plan.y1, plan.x0 : plan.x1].reshape(-1, 3)[hit]
    starts = rays.t_near[plan.y0 : plan.y1, plan.x0 : plan.x1].reshape(-1)[hit]
    ends = rays.t_far[plan.y0 : plan.y1, plan.x0 : plan.x1].reshape(-1)[hit]
    if directions.size == 0:
        return 0

    any_hit = np.zeros(directions.shape[0], dtype=bool)
    # Bound temporary slab-test memory for very large edits.
    chunk_size = 2048
    for start in range(0, lower.shape[0], chunk_size):
        stop = min(start + chunk_size, lower.shape[0])
        chunk_lower = lower[start:stop]
        chunk_upper = upper[start:stop]
        enter = np.broadcast_to(starts[:, None], (directions.shape[0], stop - start)).copy()
        leave = np.broadcast_to(ends[:, None], (directions.shape[0], stop - start)).copy()
        inside = np.ones_like(enter, dtype=bool)
        for axis in range(3):
            direction = directions[:, axis, None]
            parallel = np.abs(direction) <= _EPS
            with np.errstate(divide="ignore", invalid="ignore"):
                t0 = (chunk_lower[None, :, axis] - rays.origin[axis]) / direction
                t1 = (chunk_upper[None, :, axis] - rays.origin[axis]) / direction
            axis_enter = np.minimum(t0, t1)
            axis_leave = np.maximum(t0, t1)
            enter = np.where(parallel, enter, np.maximum(enter, axis_enter))
            leave = np.where(parallel, leave, np.minimum(leave, axis_leave))
            inside &= (~parallel) | (
                (rays.origin[axis] >= chunk_lower[None, :, axis])
                & (rays.origin[axis] <= chunk_upper[None, :, axis])
            )
        any_hit |= np.any(inside & (leave >= enter), axis=1)
    return int(np.count_nonzero(any_hit))


def changed_support_voxels_in_bounds(plan: BlockPlan, changed: np.ndarray, dims: tuple[int, int, int]) -> int:
    if plan.voxel_bounds_min is None:
        return 0
    dims_np = np.asarray(dims, dtype=np.int64)
    low = np.maximum(np.asarray(plan.voxel_bounds_min, dtype=np.int64) - 1, 0)
    high = np.minimum(np.asarray(plan.voxel_bounds_max, dtype=np.int64) + 1, dims_np - 1)
    x0, y0, z0 = (int(low[0]), int(low[1]), int(low[2]))
    x1, y1, z1 = (int(high[0]), int(high[1]), int(high[2]))
    return int(np.count_nonzero(changed[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]))


def changed_voxels_in_bounds(plan: BlockPlan, changed: np.ndarray) -> int:
    if plan.voxel_bounds_min is None:
        return 0
    x0, y0, z0 = plan.voxel_bounds_min
    x1, y1, z1 = plan.voxel_bounds_max  # type: ignore[misc]
    return int(np.count_nonzero(changed[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]))


def sample_trilinear(volume: np.ndarray, index_positions: np.ndarray) -> np.ndarray:
    """Clamped trilinear interpolation using voxel-center coordinates."""

    if volume.ndim != 3:
        raise ValueError("volume must be three-dimensional")
    result = np.empty(index_positions.shape[:-1], dtype=np.float64)
    q = index_positions - 0.5
    # index_positions is (x, y, z), while the array is indexed as [z, y, x].
    axes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for position_axis, array_axis in ((0, 2), (1, 1), (2, 0)):
        q_axis = q[..., position_axis]
        n = volume.shape[array_axis]
        i0 = np.floor(q_axis).astype(np.int64)
        fraction = q_axis - i0
        at_or_below = q_axis <= 0.0
        at_or_above = q_axis >= (n - 1)
        i0 = np.where(at_or_below, 0, i0)
        i0 = np.where(at_or_above, n - 1, i0)
        i1 = np.minimum(i0 + 1, n - 1)
        fraction = np.where(at_or_below | at_or_above, 0.0, fraction)
        axes.append((i0, i1, fraction))

    x0, x1, fx = axes[0]
    y0, y1, fy = axes[1]
    z0, z1, fz = axes[2]

    c000 = volume[z0, y0, x0]
    c100 = volume[z0, y0, x1]
    c010 = volume[z0, y1, x0]
    c110 = volume[z0, y1, x1]
    c001 = volume[z1, y0, x0]
    c101 = volume[z1, y0, x1]
    c011 = volume[z1, y1, x0]
    c111 = volume[z1, y1, x1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    result[...] = c0 * (1.0 - fz) + c1 * fz
    return result
