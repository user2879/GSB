"""针孔相机：逐像素主光线生成与透明体投影覆盖真值。

光线方向的构造顺序固定（先相机基轴、后像素偏移），保证浮点序列确定。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Ray, intersect_slab
from .scene import Camera, TransparentBody
from .vectors import as_vec3, normalize

if TYPE_CHECKING:
    from .scene import Scene


@dataclass(frozen=True)
class CameraFrame:
    eye: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    half_height: float
    half_width: float


def build_camera_frame(camera: Camera) -> CameraFrame:
    forward = normalize(as_vec3(camera.look_at) - as_vec3(camera.eye))
    up_raw = as_vec3(camera.up)
    right = normalize(np.cross(forward, up_raw))
    up = np.cross(right, forward)
    aspect = float(camera.width) / float(camera.height)
    half_height = math.tan(math.radians(camera.fov_y_deg) * 0.5)
    half_width = half_height * aspect
    return CameraFrame(as_vec3(camera.eye), forward, right, up, half_height, half_width)


def primary_ray(frame: CameraFrame, camera: Camera, row: int, col: int) -> Ray:
    """行优先像素坐标 (row, col) -> 单位方向主光线。"""
    x_norm = (float(col) + 0.5 - 0.5 * float(camera.width)) / (0.5 * float(camera.width))
    y_norm = (0.5 * float(camera.height) - (float(row) + 0.5)) / (0.5 * float(camera.height))
    direction = (
        frame.forward
        + frame.right * (x_norm * frame.half_width)
        + frame.up * (y_norm * frame.half_height)
    )
    return Ray(frame.eye, normalize(direction))


def _pixel_ray_direction(frame: CameraFrame, camera: Camera, row: int, col: int) -> np.ndarray:
    x_norm = (float(col) + 0.5 - 0.5 * float(camera.width)) / (0.5 * float(camera.width))
    y_norm = (0.5 * float(camera.height) - (float(row) + 0.5)) / (0.5 * float(camera.height))
    direction = (
        frame.forward
        + frame.right * (x_norm * frame.half_width)
        + frame.up * (y_norm * frame.half_height)
    )
    return normalize(direction)


def project_sphere_bounds(frame: CameraFrame, camera: Camera, body: TransparentBody) -> tuple[int, int, int, int]:
    """逐像素针孔投影判定：主光线与球体有实根的像素包围盒（闭区间）。"""
    rows: list[int] = []
    cols: list[int] = []
    center = as_vec3(body.center)
    radius2 = float(body.radius) ** 2
    for row in range(camera.height):
        for col in range(camera.width):
            direction = _pixel_ray_direction(frame, camera, row, col)
            offset = frame.eye - center
            b = float(np.dot(offset, direction))
            c = float(np.dot(offset, offset)) - radius2
            if b * b - c >= 0.0:
                rows.append(row)
                cols.append(col)
    if not rows:
        return (0, -1, 0, -1)
    return (min(rows), max(rows), min(cols), max(cols))


def analytic_coverage(scene: 'Scene', frame: CameraFrame) -> dict[str, dict[str, int]]:
    """每个透明体的解析投影覆盖像素数（与实测命中覆盖同口径核对）。"""
    coverage: dict[str, dict[str, int]] = {}
    for body in scene.bodies:
        if body.kind != "sphere":
            # 有限平板：逐像素判定主光线是否落在板面内。
            count = 0
            bounds_rows: list[int] = []
            bounds_cols: list[int] = []
            for row in range(scene.camera.height):
                for col in range(scene.camera.width):
                    direction = _pixel_ray_direction(frame, scene.camera, row, col)
                    ray = Ray(frame.eye, direction)
                    if intersect_slab(ray, body) is not None:
                        count += 1
                        bounds_rows.append(row)
                        bounds_cols.append(col)
            coverage[body.body_id] = {
                "analytic_bbox_pixels": count,
                "bbox": ([min(bounds_rows), max(bounds_rows),
                          min(bounds_cols), max(bounds_cols)] if count else [0, -1, 0, -1]),
            }
            continue
        r0, r1, c0, c1 = project_sphere_bounds(frame, scene.camera, body)
        count = max(0, (r1 - r0 + 1) * (c1 - c0 + 1)) if r1 >= r0 else 0
        # 精确逐像素计数（包围盒内仍需逐根光线判定）。
        exact = 0
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                direction = _pixel_ray_direction(frame, scene.camera, row, col)
                offset = frame.eye - as_vec3(body.center)
                b = float(np.dot(offset, direction))
                c = float(np.dot(offset, offset)) - float(body.radius) ** 2
                if b * b - c >= 0.0:
                    exact += 1
        coverage[body.body_id] = {
            "analytic_bbox_pixels": exact,
            "bbox": [r0, r1, c0, c1],
        }
    return coverage
