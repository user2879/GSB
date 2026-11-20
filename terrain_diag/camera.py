from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .heightfield import sample_bilinear
from .quadtree import Bounds


WORLD_DTYPE = "float64"
DEPTH_DTYPE = "float32"
MATRIX_ORDER = "row-major; clip = projection @ view @ [x,y,z,1]; column-vector convention"
NUMERIC_POLICY = {
    "world_position_dtype": WORLD_DTYPE,
    "matrix_dtype": WORLD_DTYPE,
    "depth_buffer_dtype": DEPTH_DTYPE,
    "height_input_dtype": "uint16",
    "operation_order": MATRIX_ORDER,
    "evaluation": "NumPy IEEE-754, fixed branch order, no transcendental calls in per-block tests",
}
@dataclass(frozen=True)
class CameraPose:
    frame: int
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov_y_deg: float
    aspect: float
    near: float
    far: float
    view_matrix: tuple[float, ...]
    projection_matrix: tuple[float, ...]
    view_projection_matrix: tuple[float, ...]
    track_id: str

    def to_report(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "track_id": self.track_id,
            "eye": list(self.eye),
            "target": list(self.target),
            "up": list(self.up),
            "fov_y_deg": self.fov_y_deg,
            "aspect": self.aspect,
            "near": self.near,
            "far": self.far,
            "view_matrix": list(self.view_matrix),
            "projection_matrix": list(self.projection_matrix),
            "view_projection_matrix": list(self.view_projection_matrix),
            "numeric_policy": NUMERIC_POLICY,
        }


@dataclass(frozen=True)
class ProjectionResult:
    block_id: str
    intersects_frustum: bool
    screen_bbox: tuple[int, int, int, int]
    raw_screen_bbox: tuple[float, float, float, float]
    near_depth: float
    far_depth: float
    center_distance: float
    error_components_px: dict[str, float]
    projection_error_px: float
    any_corner_behind_near: bool

    def to_report(self) -> dict[str, Any]:
        return {
            "intersects_frustum": self.intersects_frustum,
            "screen_bbox": list(self.screen_bbox),
            "raw_screen_bbox": list(self.raw_screen_bbox),
            "near_depth": self.near_depth,
            "far_depth": self.far_depth,
            "center_distance": self.center_distance,
            "projection_error_px": self.projection_error_px,
            "projection_error_components_px": self.error_components_px,
            "any_corner_behind_near": self.any_corner_behind_near,
        }


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.sqrt(np.dot(vector, vector)))
    if length <= 1e-15:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / length


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, up))
    camera_up = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, :3] = right
    matrix[1, :3] = camera_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -float(np.dot(right, eye))
    matrix[1, 3] = -float(np.dot(camera_up, eye))
    matrix[2, 3] = float(np.dot(forward, eye))
    return matrix


def _perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    focal = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = focal / aspect
    matrix[1, 1] = focal
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def _flatten(matrix: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in matrix.reshape(-1))


def make_camera_pose(
    frame: int,
    eye: Sequence[float],
    target: Sequence[float],
    up: Sequence[float],
    fov_y_deg: float,
    aspect: float,
    near: float,
    far: float,
    track_id: str,
) -> CameraPose:
    eye_array = np.asarray(eye, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    up_array = np.asarray(up, dtype=np.float64)
    view = _look_at(eye_array, target_array, up_array)
    projection = _perspective(fov_y_deg, aspect, near, far)
    view_projection = projection @ view
    return CameraPose(
        frame=int(frame),
        eye=tuple(float(value) for value in eye_array),
        target=tuple(float(value) for value in target_array),
        up=tuple(float(value) for value in up_array),
        fov_y_deg=float(fov_y_deg),
        aspect=float(aspect),
        near=float(near),
        far=float(far),
        view_matrix=_flatten(view),
        projection_matrix=_flatten(projection),
        view_projection_matrix=_flatten(view_projection),
        track_id=track_id,
    )


def generate_trajectory(
    heights: np.ndarray,
    frame_count: int = 24,
    variant: str = "canonical",
) -> tuple[CameraPose, ...]:
    size = int(heights.shape[0])
    frames: list[CameraPose] = []
    yaw_offsets = {
        "canonical": 0.0,
        "yaw_offset": math.radians(9.0),
        "height_offset": 0.0,
    }
    if variant not in yaw_offsets:
        raise ValueError(f"unknown camera trajectory variant: {variant}")

    for frame in range(frame_count):
        t = frame / float(max(1, frame_count - 1))
        phase = 2.0 * math.pi * t + math.pi * 0.17
        base_x = 0.5 * size + 0.30 * size * math.sin(phase)
        base_z = 0.10 * size + 0.80 * size * t
        base_y = sample_bilinear(heights, base_x, base_z)
        height_boost = 0.015 * size if variant == "height_offset" else 0.0
        eye = np.array(
            [
                base_x,
                base_y + 0.025 * size + height_boost + 0.004 * size * math.sin(3.0 * phase),
                base_z,
            ],
            dtype=np.float64,
        )

        forward_x = 0.10 * size * math.cos(phase)
        forward_z = 0.18 * size
        yaw = math.atan2(forward_x, forward_z) + yaw_offsets[variant]
        target_x = eye[0] + 0.22 * size * math.sin(yaw)
        target_z = eye[2] + 0.22 * size * math.cos(yaw)
        target_y = sample_bilinear(heights, target_x, target_z) + 0.020 * size
        target = np.array([target_x, target_y, target_z], dtype=np.float64)

        frame_pose = make_camera_pose(
            frame=frame,
            eye=eye,
            target=target,
            up=(0.0, 1.0, 0.0),
            fov_y_deg=60.0,
            aspect=1.0,
            near=0.5,
            far=max(128.0, float(size) * 2.5),
            track_id=variant,
        )
        frames.append(frame_pose)
    return tuple(frames)


def project_bounds(
    block_id: str,
    bounds: Bounds,
    pose: CameraPose,
    width: int,
    height: int,
) -> ProjectionResult:
    corners = np.array(
        [
            [x, y, z, 1.0]
            for z in (bounds.zmin, bounds.zmax)
            for y in (bounds.ymin, bounds.ymax)
            for x in (bounds.xmin, bounds.xmax)
        ],
        dtype=np.float64,
    )
    view_projection = np.asarray(pose.view_projection_matrix, dtype=np.float64).reshape(4, 4)
    clip = corners @ view_projection.T
    w = clip[:, 3]
    front = w > 1e-12
    any_behind = bool(np.any(~front))

    outside_tests = (
        bool(np.all(clip[:, 0] < -w)),
        bool(np.all(clip[:, 0] > w)),
        bool(np.all(clip[:, 1] < -w)),
        bool(np.all(clip[:, 1] > w)),
        bool(np.all(clip[:, 2] < -w)),
        bool(np.all(clip[:, 2] > w)),
        bool(np.all(w <= 0.0)),
    )
    intersects = not any(outside_tests)

    center = np.array(
        [
            0.5 * (bounds.xmin + bounds.xmax),
            0.5 * (bounds.ymin + bounds.ymax),
            0.5 * (bounds.zmin + bounds.zmax),
        ],
        dtype=np.float64,
    )
    distance = float(np.sqrt(np.sum((center - np.asarray(pose.eye, dtype=np.float64)) ** 2)))

    if np.any(front):
        ndc_x = clip[front, 0] / w[front]
        ndc_y = clip[front, 1] / w[front]
        depths = clip[front, 2] / w[front] * 0.5 + 0.5
        screen_x = (ndc_x * 0.5 + 0.5) * width
        screen_y = (0.5 - ndc_y * 0.5) * height
        raw_left = float(screen_x.min())
        raw_right = float(screen_x.max())
        raw_top = float(screen_y.min())
        raw_bottom = float(screen_y.max())
        near_depth = float(depths.min())
        far_depth = float(depths.max())
    else:
        raw_left, raw_top, raw_right, raw_bottom = 0.0, 0.0, float(width), float(height)
        near_depth, far_depth = 0.0, 1.0

    if not intersects:
        left = top = right = bottom = 0
        screen_width = 0.0
        screen_height = 0.0
    else:
        left = max(0, min(width, int(math.floor(raw_left))))
        right = max(0, min(width, int(math.ceil(raw_right))))
        top = max(0, min(height, int(math.floor(raw_top))))
        bottom = max(0, min(height, int(math.ceil(raw_bottom))))
        screen_width = max(0.0, raw_right - raw_left)
        screen_height = max(0.0, raw_bottom - raw_top)
        if any_behind:
            left, top, right, bottom = 0, 0, width, height
            screen_width = float(width)
            screen_height = float(height)

    max_abs_coord = float(np.max(np.abs(corners[:, :3])))
    focal = 0.5 * height / math.tan(math.radians(pose.fov_y_deg) * 0.5)
    float_roundoff_px = 16.0 * np.finfo(np.float64).eps * max(1.0, max_abs_coord)
    pixel_snap_px = 1.0
    min_axis = max(1e-6, min(screen_width, screen_height))
    footprint_px = 1.0 / max(0.25, min_axis)
    depth_range = max(0.0, far_depth - near_depth)
    perspective_px = (
        focal
        * pose.near
        * depth_range
        / max(1.0, distance * distance + pose.near * pose.near)
    )
    near_clip_px = 2.0 if any_behind and intersects else 0.0
    error_components = {
        "float_roundoff_px": float_roundoff_px,
        "pixel_snap_px": pixel_snap_px,
        "subpixel_footprint_px": footprint_px,
        "perspective_depth_sensitivity_px": perspective_px,
        "near_clip_px": near_clip_px,
    }
    total_error = float(sum(error_components.values()))
    return ProjectionResult(
        block_id=block_id,
        intersects_frustum=intersects,
        screen_bbox=(left, top, right, bottom),
        raw_screen_bbox=(raw_left, raw_top, raw_right, raw_bottom),
        near_depth=near_depth,
        far_depth=far_depth,
        center_distance=distance,
        error_components_px=error_components,
        projection_error_px=total_error,
        any_corner_behind_near=any_behind,
    )
