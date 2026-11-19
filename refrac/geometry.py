"""光线与透明体（球体、平板玻璃层）及不透明背景平面的求交。

约定：
- 光线用 (origin, direction) 表示，direction 为单位向量。
- 求交返回最近的正 t（t > RAY_EPS），命中点与朝向入射侧的几何法线。
- 退化输入（半径为负、方向为零等）抛出可归类业务异常，绝不静默丢弃。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .errors import IntersectDegenerateError, RayDegenerateError
from .vectors import RAY_EPS, ZERO_NORM_TOL, as_vec3, dot, norm, normalize


@dataclass(frozen=True)
class Ray:
    origin: np.ndarray
    direction: np.ndarray

    def point_at(self, t: float) -> np.ndarray:
        return np.array(
            [
                self.origin[0] + t * self.direction[0],
                self.origin[1] + t * self.direction[1],
                self.origin[2] + t * self.direction[2],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Hit:
    t: float
    point: np.ndarray
    normal: np.ndarray
    body_id: str
    kind: str  # "sphere" | "slab" | "background"
    side: str  # "enter" | "exit"（透明体）；背景平面为 "enter"


def make_ray(origin, direction, *, strict: bool = True) -> Ray:
    origin_vec = as_vec3(origin)
    dir_vec = as_vec3(direction)
    length = norm(dir_vec)
    if length <= ZERO_NORM_TOL:
        if strict:
            raise RayDegenerateError(
                "ray direction is a zero-length vector",
                {"origin": [float(x) for x in origin_vec], "norm": float(length)},
            )
    dir_unit = np.array(
        [dir_vec[0] / length, dir_vec[1] / length, dir_vec[2] / length],
        dtype=np.float64,
    )
    return Ray(origin_vec, dir_unit)


def _nearest_root(roots: tuple[float, float]) -> Optional[float]:
    candidates = [t for t in roots if t > RAY_EPS]
    if not candidates:
        return None
    return min(candidates)


def intersect_sphere(ray: Ray, body) -> Optional[Hit]:
    """球体求交。半径为负归类为 E_INTERSECT_DEGENERATE。"""
    if body.radius <= 0.0:
        raise IntersectDegenerateError(
            f"sphere '{body.body_id}' has non-positive radius",
            {"body_id": body.body_id, "radius": float(body.radius)},
        )
    center = as_vec3(body.center)
    offset = np.array(
        [ray.origin[0] - center[0], ray.origin[1] - center[1], ray.origin[2] - center[2]],
        dtype=np.float64,
    )
    b = dot(offset, ray.direction)
    c = dot(offset, offset) - np.float64(body.radius) * np.float64(body.radius)
    disc = b * b - c
    if disc < 0.0:
        return None
    root = np.sqrt(disc)
    t = _nearest_root((float(-b - root), float(-b + root)))
    if t is None:
        return None
    point = ray.point_at(t)
    normal = np.array(
        [
            (point[0] - center[0]) / np.float64(body.radius),
            (point[1] - center[1]) / np.float64(body.radius),
            (point[2] - center[2]) / np.float64(body.radius),
        ],
        dtype=np.float64,
    )
    side = "enter" if dot(ray.direction, normal) < 0.0 else "exit"
    return Hit(float(t), point, normal, body.body_id, "sphere", side)


def intersect_slab(ray: Ray, body) -> Optional[Hit]:
    """平板玻璃层求交（两个平行平面夹成的介质层）。

    支持 axis 对齐平板与通过 slab_normal 声明的倾斜平板（法线真值由场景给出）。
    """
    half = np.float64(body.thickness) * np.float64(0.5)
    if half <= 0.0:
        raise IntersectDegenerateError(
            f"slab '{body.body_id}' has non-positive thickness",
            {"body_id": body.body_id, "thickness": float(body.thickness)},
        )
    center = as_vec3(body.center)
    declared_normal = getattr(body, "slab_normal", None)
    if declared_normal is not None:
        normal_axis = normalize(as_vec3(declared_normal))
        axis = int(np.argmax(np.abs(normal_axis)))
    else:
        normal_axis = np.zeros(3, dtype=np.float64)
        axis = int(body.axis)
        normal_axis[axis] = 1.0

    denom = dot(ray.direction, normal_axis)
    if abs(float(denom)) <= float(ZERO_NORM_TOL):
        return None
    offset = dot(center - ray.origin, normal_axis)
    t_near = float((offset - half) / denom)
    t_far = float((offset + half) / denom)
    if t_near > t_far:
        t_near, t_far = t_far, t_near
    t = _nearest_root((t_near, t_far))
    if t is None:
        return None
    point = ray.point_at(t)
    # 有限平板：切向坐标必须落在 half_extent 内，否则视为未命中。
    extent = np.float64(getattr(body, "half_extent", 0.0) or 0.0)
    if extent > 0.0:
        tangent = point - center
        tangent = tangent - normal_axis * dot(tangent, normal_axis)
        if float(np.max(np.abs(tangent))) > float(extent):
            return None
    entering = abs(t - t_near) <= abs(t - t_far)
    # 几何外法线：前表面朝 -normal_axis、后表面朝 +normal_axis，
    # 再按光线方向翻转到“朝向入射侧”。
    geometric = normal_axis if not entering else -normal_axis
    if dot(ray.direction, geometric) > 0.0:
        geometric = -geometric
    return Hit(
        float(t),
        point,
        geometric,
        body.body_id,
        "slab",
        "enter" if entering else "exit",
    )


def intersect_background(ray: Ray, plane) -> Optional[Hit]:
    """不透明背景平面（法线 n，方程 n·(p - point)=0）。"""
    normal = as_vec3(plane.normal)
    denom = dot(ray.direction, normal)
    if abs(float(denom)) <= float(ZERO_NORM_TOL):
        return None
    t = float(dot(as_vec3(plane.point) - ray.origin, normal) / denom)
    if t <= float(RAY_EPS):
        return None
    point = ray.point_at(t)
    oriented = normal if denom < 0.0 else -normal
    return Hit(float(t), point, oriented, plane.body_id, "background", "enter")


def nearest_transparent_hit(ray: Ray, bodies) -> Optional[Hit]:
    """按 body_id 字典序遍历求交，取最近命中（确定性遍历顺序）。"""
    best: Optional[Hit] = None
    for body in sorted(bodies, key=lambda item: item.body_id):
        if body.kind == "sphere":
            hit = intersect_sphere(ray, body)
        elif body.kind == "slab":
            hit = intersect_slab(ray, body)
        else:  # pragma: no cover - 场景 schema 已保证
            raise IntersectDegenerateError(
                f"unknown transparent body kind '{body.kind}'",
                {"body_id": body.body_id, "kind": body.kind},
            )
        if hit is not None and (best is None or hit.t < best.t):
            best = hit
    return best
