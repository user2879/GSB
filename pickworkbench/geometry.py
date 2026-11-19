"""Deterministic geometry, projection, picking, and snapping primitives."""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable

import numpy as np

from .canonical import to_builtin
from .errors import (
    E_INJECT_CONFIG,
    E_NORMAL_ZERO,
    E_RAY_DEGENERATE,
    E_SNAP_SPACING,
    make_error,
)

FLOAT_EPS = 1.0e-12
INTERSECTION_EPS = 1.0e-9
RAY_DIRECTION_EPS = 1.0e-12


def vec3(value: Iterable[float], *, name: str = "vector") -> np.ndarray:
    array = np.asarray(list(value), dtype=np.float64)
    if array.shape != (3,):
        raise make_error(E_INJECT_CONFIG, f"{name} must contain exactly three numbers.", field=name)
    return array


def normalize(value: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(value))
    if not math.isfinite(length) or length <= RAY_DIRECTION_EPS:
        return None
    return value / length


def euler_rotation_deg(rotation_deg: Iterable[float]) -> np.ndarray:
    x, y, z = np.deg2rad(vec3(rotation_deg, name="rotation_deg"))
    cx, sx = math.cos(float(x)), math.sin(float(x))
    cy, sy = math.cos(float(y)), math.sin(float(y))
    cz, sz = math.cos(float(z)), math.sin(float(z))
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def object_rotation(obj: dict[str, Any]) -> np.ndarray:
    return euler_rotation_deg(obj.get("transform", {}).get("rotation_deg", [0.0, 0.0, 0.0]))


def object_translation(obj: dict[str, Any]) -> np.ndarray:
    return vec3(obj.get("transform", {}).get("translation", [0.0, 0.0, 0.0]), name="translation")


def object_aabb(obj: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    aabb = obj["bounding_volume"]["aabb"]
    return (
        vec3(aabb["min"], name="aabb.min"),
        vec3(aabb["max"], name="aabb.max"),
    )


def compute_object_bounds(obj: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return world AABB and bounding sphere for a generated object."""

    kind = obj["kind"]
    translation = object_translation(obj)
    rotation = object_rotation(obj)
    if kind == "sphere":
        radius = float(obj["geometry"]["radius"])
        minimum = translation - radius
        maximum = translation + radius
        return minimum, maximum, translation.copy(), radius
    if kind == "aabb":
        half = vec3(obj["geometry"]["half_extents"], name="half_extents")
        minimum = translation - half
        maximum = translation + half
        radius = float(np.linalg.norm(half))
        return minimum, maximum, translation.copy(), radius
    if kind == "convex_polyhedron":
        local_vertices = np.asarray(obj["geometry"]["vertices"], dtype=np.float64)
        world_vertices = (rotation @ local_vertices.T).T + translation
        minimum = np.min(world_vertices, axis=0)
        maximum = np.max(world_vertices, axis=0)
        center = (minimum + maximum) * 0.5
        radius = float(np.max(np.linalg.norm(world_vertices - center, axis=1)))
        return minimum, maximum, center, radius
    raise make_error(E_INJECT_CONFIG, f"Unsupported object kind: {kind!r}.", object_id=obj.get("id"))


def ray_aabb_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    *,
    include_origin: bool = True,
) -> tuple[float, float] | None:
    t_enter = -math.inf
    t_exit = math.inf
    for axis in range(3):
        o = float(origin[axis])
        d = float(direction[axis])
        lo = float(minimum[axis])
        hi = float(maximum[axis])
        if abs(d) <= FLOAT_EPS:
            if o < lo - FLOAT_EPS or o > hi + FLOAT_EPS:
                return None
            continue
        t0 = (lo - o) / d
        t1 = (hi - o) / d
        if t0 > t1:
            t0, t1 = t1, t0
        t_enter = max(t_enter, t0)
        t_exit = min(t_exit, t1)
        if t_enter > t_exit + INTERSECTION_EPS:
            return None
    if t_exit < -INTERSECTION_EPS:
        return None
    if not include_origin:
        t_enter = max(t_enter, 0.0)
    return t_enter, t_exit


def ray_intersects_aabb(origin: np.ndarray, direction: np.ndarray, obj: dict[str, Any]) -> bool:
    minimum, maximum = object_aabb(obj)
    result = ray_aabb_intersection(origin, direction, minimum, maximum)
    if result is None:
        return False
    t_enter, t_exit = result
    return t_exit >= 0.0 and (t_enter >= 0.0 or t_exit >= 0.0)


def _hit_record(
    obj: dict[str, Any],
    origin: np.ndarray,
    direction: np.ndarray,
    distance: float,
    method: str,
) -> dict[str, Any]:
    point = origin + direction * distance
    return {
        "object_id": str(obj["id"]),
        "pick_priority": int(obj["pick_priority"]),
        "hit_distance": float(distance),
        "hit_point": [float(point[0]), float(point[1]), float(point[2])],
        "intersection_method": method,
    }


def intersect_sphere(
    origin: np.ndarray,
    direction: np.ndarray,
    obj: dict[str, Any],
) -> dict[str, Any] | None:
    translation = object_translation(obj)
    rotation = object_rotation(obj)
    local_origin = rotation.T @ (origin - translation)
    local_direction = rotation.T @ direction
    radius = float(obj["geometry"]["radius"])
    a = float(np.dot(local_direction, local_direction))
    b = 2.0 * float(np.dot(local_origin, local_direction))
    c = float(np.dot(local_origin, local_origin)) - radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < -INTERSECTION_EPS:
        return None
    root = math.sqrt(max(0.0, discriminant))
    candidates = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
    positive = [value for value in candidates if value >= -INTERSECTION_EPS]
    if not positive:
        return None
    distance = max(0.0, min(positive))
    return _hit_record(obj, origin, direction, distance, "sphere_quadratic")


def intersect_aabb(
    origin: np.ndarray,
    direction: np.ndarray,
    obj: dict[str, Any],
) -> dict[str, Any] | None:
    minimum, maximum = object_aabb(obj)
    result = ray_aabb_intersection(origin, direction, minimum, maximum, include_origin=False)
    if result is None:
        return None
    t_enter, t_exit = result
    if t_exit < 0.0 or t_enter < -INTERSECTION_EPS:
        return None
    return _hit_record(obj, origin, direction, max(0.0, t_enter), "aabb_slab")


def world_planes(obj: dict[str, Any]) -> list[tuple[np.ndarray, float]]:
    rotation = object_rotation(obj)
    translation = object_translation(obj)
    planes: list[tuple[np.ndarray, float]] = []
    for plane in obj["geometry"].get("planes", []):
        local_normal = vec3(plane["normal"], name="plane.normal")
        unit_normal = normalize(local_normal)
        if unit_normal is None:
            raise make_error(
                E_NORMAL_ZERO,
                "Convex polyhedron plane normal has zero length.",
                object_id=str(obj.get("id", "")),
                plane_index=int(plane.get("plane_index", len(planes))),
            )
        world_normal = rotation @ unit_normal
        offset = float(plane["offset"])
        world_offset = offset + float(np.dot(world_normal, translation))
        planes.append((world_normal, world_offset))
    return planes


def intersect_convex(
    origin: np.ndarray,
    direction: np.ndarray,
    obj: dict[str, Any],
) -> dict[str, Any] | None:
    t_enter = -math.inf
    t_exit = math.inf
    for normal, offset in world_planes(obj):
        denominator = float(np.dot(normal, direction))
        distance_from_plane = float(np.dot(normal, origin)) - offset
        if abs(denominator) <= FLOAT_EPS:
            if distance_from_plane > INTERSECTION_EPS:
                return None
            continue
        crossing = -distance_from_plane / denominator
        if denominator > 0.0:
            t_exit = min(t_exit, crossing)
        else:
            t_enter = max(t_enter, crossing)
        if t_enter > t_exit + INTERSECTION_EPS:
            return None
    if t_exit < 0.0 or t_enter < -INTERSECTION_EPS:
        return None
    if t_enter < 0.0:
        # The ray starts inside the convex object; use the origin as the first
        # pickable point so the result remains finite and deterministic.
        t_enter = 0.0
    return _hit_record(obj, origin, direction, max(0.0, t_enter), "convex_halfspace_clip")


def intersect_object(
    origin: np.ndarray,
    direction: np.ndarray,
    obj: dict[str, Any],
) -> dict[str, Any] | None:
    kind = obj["kind"]
    if kind == "sphere":
        return intersect_sphere(origin, direction, obj)
    if kind == "aabb":
        return intersect_aabb(origin, direction, obj)
    if kind == "convex_polyhedron":
        return intersect_convex(origin, direction, obj)
    raise make_error(E_INJECT_CONFIG, f"Unsupported object kind: {kind!r}.", object_id=obj.get("id"))


def selection_key(hit: dict[str, Any]) -> tuple[int, float, str]:
    """Higher priority wins; then nearest hit; then object ID ascending."""

    return (
        -int(hit["pick_priority"]),
        float(hit["hit_distance"]),
        str(hit["object_id"]),
    )


def select_winner(hits: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    materialized = list(hits)
    if not materialized:
        return None
    return min(materialized, key=selection_key)


def all_object_hits(
    origin: np.ndarray,
    direction: np.ndarray,
    objects: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for obj in sorted(objects, key=lambda item: (str(item["id"]), int(item.get("object_index", 0)))):
        hit = intersect_object(origin, direction, obj)
        if hit is not None:
            hits.append(hit)
    return hits


def ray_from_screen(
    camera: dict[str, Any],
    screen_x: float,
    screen_y: float,
    viewport_width: int | None = None,
    viewport_height: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    position = vec3(camera["position"], name="camera.position")
    target = vec3(camera["target"], name="camera.target")
    up = vec3(camera["up"], name="camera.up")
    width = int(viewport_width if viewport_width is not None else camera["viewport"]["width"])
    height = int(viewport_height if viewport_height is not None else camera["viewport"]["height"])
    forward = normalize(target - position)
    if forward is None:
        raise make_error(E_INJECT_CONFIG, "Camera target coincides with its position.", field="camera")
    right = normalize(np.cross(forward, up))
    if right is None:
        raise make_error(E_INJECT_CONFIG, "Camera up vector is parallel to the view direction.", field="camera")
    true_up = np.cross(right, forward)
    focal_y = height / (2.0 * math.tan(math.radians(float(camera["fov_y_deg"])) / 2.0))
    camera_x = (float(screen_x) - width / 2.0) / focal_y
    camera_y = (height / 2.0 - float(screen_y)) / focal_y
    direction = normalize(forward + right * camera_x + true_up * camera_y)
    if direction is None:
        raise make_error(E_RAY_DEGENERATE, "Derived camera ray has zero length.", query_type="click")
    return position, direction


def project_point(camera: dict[str, Any], point: np.ndarray) -> dict[str, float] | None:
    position = vec3(camera["position"], name="camera.position")
    target = vec3(camera["target"], name="camera.target")
    up = vec3(camera["up"], name="camera.up")
    forward = normalize(target - position)
    if forward is None:
        raise make_error(E_INJECT_CONFIG, "Camera target coincides with its position.", field="camera")
    right = normalize(np.cross(forward, up))
    if right is None:
        raise make_error(E_INJECT_CONFIG, "Camera up vector is parallel to the view direction.", field="camera")
    true_up = np.cross(right, forward)
    relative = point - position
    depth = float(np.dot(relative, forward))
    if depth <= float(camera.get("near", 1.0e-4)):
        return None
    width = float(camera["viewport"]["width"])
    height = float(camera["viewport"]["height"])
    focal_y = height / (2.0 * math.tan(math.radians(float(camera["fov_y_deg"])) / 2.0))
    camera_x = float(np.dot(relative, right))
    camera_y = float(np.dot(relative, true_up))
    return {
        "x": width / 2.0 + camera_x * focal_y / depth,
        "y": height / 2.0 - camera_y * focal_y / depth,
        "depth": depth,
        "focal_px": focal_y,
    }


def object_screen_bounds(camera: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any] | None:
    kind = obj["kind"]
    if kind == "sphere":
        center = object_translation(obj)
        projected = project_point(camera, center)
        if projected is None:
            return None
        radius_px = float(obj["geometry"]["radius"]) * projected["focal_px"] / projected["depth"]
        return {
            "shape": "circle",
            "cx": projected["x"],
            "cy": projected["y"],
            "radius_px": radius_px,
            "depth": projected["depth"],
        }

    if kind == "aabb":
        minimum, maximum = object_aabb(obj)
        points = [
            np.array([x, y, z], dtype=np.float64)
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ]
    elif kind == "convex_polyhedron":
        rotation = object_rotation(obj)
        translation = object_translation(obj)
        local_vertices = np.asarray(obj["geometry"]["vertices"], dtype=np.float64)
        points = list((rotation @ local_vertices.T).T + translation)
    else:
        raise make_error(E_INJECT_CONFIG, f"Unsupported object kind: {kind!r}.", object_id=obj.get("id"))

    projected_points = [projected for point in points if (projected := project_point(camera, point)) is not None]
    if not projected_points:
        return None
    xs = [item["x"] for item in projected_points]
    ys = [item["y"] for item in projected_points]
    depths = [item["depth"] for item in projected_points]
    return {
        "shape": "rect",
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
        "depth": min(depths),
    }


def screen_bounds_contain(bounds: dict[str, Any], x: float, y: float, tolerance_px: float) -> bool:
    if tolerance_px < 0.0:
        raise make_error(E_INJECT_CONFIG, "Click tolerance must be non-negative.", tolerance_px=tolerance_px)
    if bounds["shape"] == "circle":
        dx = x - float(bounds["cx"])
        dy = y - float(bounds["cy"])
        return math.hypot(dx, dy) <= float(bounds["radius_px"]) + tolerance_px
    return (
        float(bounds["min_x"]) - tolerance_px <= x <= float(bounds["max_x"]) + tolerance_px
        and float(bounds["min_y"]) - tolerance_px <= y <= float(bounds["max_y"]) + tolerance_px
    )


def screen_candidates(
    camera: dict[str, Any],
    objects: Iterable[dict[str, Any]],
    screen_x: float,
    screen_y: float,
    tolerance_px: float,
) -> list[str]:
    candidates: list[str] = []
    for obj in sorted(objects, key=lambda item: (str(item["id"]), int(item.get("object_index", 0)))):
        bounds = object_screen_bounds(camera, obj)
        if bounds is not None and screen_bounds_contain(bounds, screen_x, screen_y, tolerance_px):
            candidates.append(str(obj["id"]))
    return candidates


def ray_plane_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    unit_normal = normalize(plane_normal)
    if unit_normal is None:
        raise make_error(E_NORMAL_ZERO, "Drag plane normal has zero length.", plane="drag")
    denominator = float(np.dot(unit_normal, direction))
    if abs(denominator) <= FLOAT_EPS:
        raise make_error(
            E_INJECT_CONFIG,
            "Drag ray is parallel to the drag plane.",
            plane="drag",
        )
    distance = float(np.dot(unit_normal, plane_point - origin)) / denominator
    if distance < -INTERSECTION_EPS:
        raise make_error(E_INJECT_CONFIG, "Drag plane lies behind the camera.", plane="drag")
    return origin + direction * max(0.0, distance)


def _valid_spacing_scalar(spacing_value: Any) -> float:
    if isinstance(spacing_value, (int, float)) and not isinstance(spacing_value, bool):
        spacing = float(spacing_value)
        if math.isfinite(spacing) and spacing > 0.0:
            return spacing
    if isinstance(spacing_value, (list, tuple)) and len(spacing_value) == 3:
        values = [float(item) for item in spacing_value]
        if (
            all(math.isfinite(value) and value > 0.0 for value in values)
            and max(values) - min(values) <= 1.0e-12
        ):
            return values[0]
    raise make_error(
        E_SNAP_SPACING,
        "Snap spacing needs one positive uniform scalar (non-equidistant spacing is rejected).",
        snap_spacing=to_builtin(spacing_value),
    )


def validate_snap_spacing(spacing_value: Any) -> float:
    return _valid_spacing_scalar(spacing_value)


def snap_position(
    position: np.ndarray,
    spacing_value: Any,
    axes: Iterable[str],
) -> tuple[np.ndarray, float]:
    spacing = _valid_spacing_scalar(spacing_value)
    axis_map = {"x": 0, "y": 1, "z": 2}
    snapped = position.copy()
    for axis in axes:
        if axis not in axis_map:
            raise make_error(E_INJECT_CONFIG, f"Unknown snap axis: {axis!r}.", axis=axis)
        index = axis_map[axis]
        snapped[index] = round(float(position[index]) / spacing) * spacing
    return snapped, spacing


def perform_drag(camera: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    plane = query["plane"]
    plane_point = vec3(plane["point"], name="plane.point")
    plane_normal = vec3(plane["normal"], name="plane.normal")
    if normalize(plane_normal) is None:
        raise make_error(E_NORMAL_ZERO, "Drag plane normal has zero length.", plane="drag")
    start_world = vec3(query["start_world"], name="start_world")
    orientation_before = vec3(query.get("orientation", [0.0, 0.0, 0.0]), name="orientation")
    start_origin, start_direction = ray_from_screen(
        camera,
        float(query["start_screen"][0]),
        float(query["start_screen"][1]),
    )
    end_origin, end_direction = ray_from_screen(
        camera,
        float(query["end_screen"][0]),
        float(query["end_screen"][1]),
    )
    world_start = ray_plane_intersection(start_origin, start_direction, plane_point, plane_normal)
    world_end = ray_plane_intersection(end_origin, end_direction, plane_point, plane_normal)
    delta = world_end - world_start
    unsnapped = start_world + delta
    snap_enabled = bool(query.get("snap_enabled", False))
    spacing_value = query.get("snap_spacing", 0.25)
    axes = query.get("snap_axes", ["x", "z"])
    snapped, spacing = snap_position(unsnapped, spacing_value, axes) if snap_enabled else (
        unsnapped.copy(),
        _valid_spacing_scalar(spacing_value),
    )
    orientation_after = orientation_before.copy()
    return to_builtin(
        {
            "screen": {
                "start": list(query["start_screen"]),
                "end": list(query["end_screen"]),
            },
            "plane": {
                "point": plane_point,
                "normal": plane_normal,
            },
            "world_delta": delta,
            "start_world": start_world,
            "unsnapped_position": unsnapped,
            "final_position": snapped,
            "snap_enabled": snap_enabled,
            "snap_spacing": spacing,
            "snap_axes": list(axes),
            "orientation_before": orientation_before,
            "orientation_after": orientation_after,
            "orientation_unchanged": bool(np.array_equal(orientation_before, orientation_after)),
            "decision_method": "screen_ray_plane_then_uniform_grid_snap"
            if snap_enabled
            else "screen_ray_plane",
        }
    )


def replay_drag_without_snap(camera: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    replay = copy.deepcopy(query)
    replay["snap_enabled"] = False
    return perform_drag(camera, replay)
