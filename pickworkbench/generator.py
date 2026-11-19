"""Seeded, deterministic scene and query generation."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from .canonical import to_builtin
from .geometry import (
    compute_object_bounds,
    euler_rotation_deg,
    normalize,
    project_point,
    ray_from_screen,
)

MASK64 = (1 << 64) - 1


class SplitMix64:
    """Small cross-runtime deterministic PRNG with an explicit state stream."""

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
        value = (value ^ (value >> 27)) * 0x94D049BB133111EB & MASK64
        return (value ^ (value >> 31)) & MASK64

    def uniform(self, low: float, high: float) -> float:
        unit = (self.next_u64() >> 11) * (1.0 / float(1 << 53))
        return float(low) + (float(high) - float(low)) * unit

    def randrange(self, start: int, stop: int) -> int:
        if stop <= start:
            raise ValueError("randrange requires stop > start")
        return start + self.next_u64() % (stop - start)


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    seed: int = 20261119
    object_count: int = 12
    ray_count: int = 10
    click_count: int = 8
    drag_count: int = 4
    tolerance_px: float = 5.0
    viewport_width: int = 1280
    viewport_height: int = 720

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _plane(normal: Iterable[float], offset: float) -> dict[str, Any]:
    vector = np.asarray(list(normal), dtype=np.float64)
    length = float(np.linalg.norm(vector))
    return {
        "normal": [float(value / length) for value in vector],
        "offset": float(offset / length),
    }


def _polyhedron_template(scale: float) -> tuple[list[list[float]], list[dict[str, Any]]]:
    raw_planes = [
        ((1.0, 0.0, 0.0), 1.0),
        ((-1.0, 0.0, 0.0), 1.0),
        ((0.0, 1.0, 0.0), 1.0),
        ((0.0, -1.0, 0.0), 1.0),
        ((0.0, 0.0, 1.0), 1.0),
        ((0.0, 0.0, -1.0), 1.0),
        ((1.0, 1.0, 0.0), 1.40),
        ((0.0, 1.0, 1.0), 1.50),
    ]
    planes: list[dict[str, Any]] = []
    for index, (normal, offset) in enumerate(raw_planes):
        item = _plane(normal, offset * scale)
        item["plane_index"] = index
        planes.append(item)

    normal_arrays = [np.asarray(item["normal"], dtype=np.float64) for item in planes]
    offsets = [float(item["offset"]) for item in planes]
    vertices: list[list[float]] = []
    for triple in itertools.combinations(range(len(planes)), 3):
        matrix = np.vstack([normal_arrays[index] for index in triple])
        if abs(float(np.linalg.det(matrix))) <= 1.0e-10:
            continue
        point = np.linalg.solve(matrix, np.asarray([offsets[index] for index in triple]))
        if all(
            float(np.dot(normal_arrays[index], point)) <= offsets[index] + 1.0e-9
            for index in range(len(planes))
        ):
            vertices.append([round(float(value), 12) for value in point])

    unique: dict[tuple[float, float, float], list[float]] = {}
    for vertex in vertices:
        key = tuple(0.0 if abs(value) < 1.0e-12 else value for value in vertex)
        unique[key] = [float(value) for value in key]
    ordered_vertices = [unique[key] for key in sorted(unique)]
    return ordered_vertices, planes


def generate_scene(config: GeneratorConfig | None = None) -> dict[str, Any]:
    config = config or GeneratorConfig()
    rng = SplitMix64(config.seed ^ 0xC0FFEE)
    camera = {
        "position": [0.0, 3.4, 11.5],
        "target": [0.0, 1.35, 0.0],
        "up": [0.0, 1.0, 0.0],
        "fov_y_deg": 43.0,
        "near": 0.1,
        "far": 100.0,
        "viewport": {"width": config.viewport_width, "height": config.viewport_height},
    }

    kinds = ("sphere", "aabb", "convex_polyhedron")
    priorities = (0, 10, 20, 30, 40, 50)
    objects: list[dict[str, Any]] = []
    for index in range(config.object_count):
        column = index % 4
        row = index // 4
        translation = [
            -3.35 + column * 2.23 + rng.uniform(-0.12, 0.12),
            0.55 + row * 1.22 + rng.uniform(-0.08, 0.08),
            -0.35 + rng.uniform(-0.25, 0.25),
        ]
        kind = kinds[index % len(kinds)]
        rotation = [0.0, 0.0, 0.0]
        geometry: dict[str, Any]
        if kind == "sphere":
            geometry = {"radius": rng.uniform(0.58, 0.84)}
        elif kind == "aabb":
            geometry = {
                "half_extents": [
                    rng.uniform(0.52, 0.78),
                    rng.uniform(0.48, 0.72),
                    rng.uniform(0.52, 0.78),
                ]
            }
        else:
            geometry = {}
            local_vertices, local_planes = _polyhedron_template(rng.uniform(0.64, 0.82))
            geometry = {"vertices": local_vertices, "planes": local_planes}
            rotation = [0.0, rng.uniform(-18.0, 18.0), 0.0]

        obj = {
            "id": f"obj-{index:03d}",
            "object_index": index,
            "kind": kind,
            "transform": {"translation": translation, "rotation_deg": rotation},
            "geometry": geometry,
            "pick_priority": priorities[rng.randrange(0, len(priorities))],
        }
        minimum, maximum, center, radius = compute_object_bounds(obj)
        obj["bounding_volume"] = {
            "aabb": {
                "min": [float(value) for value in minimum],
                "max": [float(value) for value in maximum],
            },
            "sphere": {
                "center": [float(value) for value in center],
                "radius": float(radius),
            },
        }
        objects.append(obj)

    global_min = np.min(
        np.asarray([obj["bounding_volume"]["aabb"]["min"] for obj in objects], dtype=np.float64),
        axis=0,
    )
    global_max = np.max(
        np.asarray([obj["bounding_volume"]["aabb"]["max"] for obj in objects], dtype=np.float64),
        axis=0,
    )
    scene = {
        "schema_version": 1,
        "generator": {
            "name": "pickworkbench.generator",
            "version": 1,
            "config": config.as_dict(),
        },
        "seed": config.seed,
        "priority_range": {"min": 0, "max": 50},
        "camera": camera,
        "objects": objects,
        "bounds": {
            "min": [float(value) for value in global_min],
            "max": [float(value) for value in global_max],
        },
        "object_count": len(objects),
    }
    return to_builtin(scene)


def _camera_object_ray(
    camera: dict[str, Any],
    obj: dict[str, Any],
    jitter_x: float,
    jitter_y: float,
) -> tuple[list[float], list[float], dict[str, float]]:
    center = np.asarray(obj["transform"]["translation"], dtype=np.float64)
    projected = project_point(camera, center)
    if projected is None:
        raise ValueError(f"Object {obj['id']} is behind the generated camera")
    origin, direction = ray_from_screen(
        camera,
        projected["x"] + jitter_x,
        projected["y"] + jitter_y,
    )
    return (
        [float(value) for value in origin],
        [float(value) for value in direction],
        projected,
    )


def generate_queries(
    scene: dict[str, Any],
    config: GeneratorConfig | None = None,
) -> dict[str, Any]:
    config = config or GeneratorConfig()
    rng = SplitMix64(config.seed ^ 0x51C0DE)
    camera = scene["camera"]
    objects = scene["objects"]
    count = len(objects)
    if count == 0:
        raise ValueError("Query generation needs at least one object")
    if config.ray_count + config.click_count + config.drag_count > 10_000:
        raise ValueError("Query count is unreasonably large")

    queries: list[dict[str, Any]] = []
    query_index = 0

    for offset in range(config.ray_count):
        obj = objects[(offset * 2 + offset // 4) % count]
        origin, direction, _ = _camera_object_ray(
            camera,
            obj,
            rng.uniform(-1.8, 1.8),
            rng.uniform(-1.8, 1.8),
        )
        queries.append(
            {
                "query_index": query_index,
                "query_id": f"q-{query_index:03d}-ray",
                "kind": "ray",
                "origin": origin,
                "direction": direction,
                "target_object_id": obj["id"],
            }
        )
        query_index += 1

    click_indices = [0, 1, 3, 5, 7, 9, 10, 11]
    for offset in range(config.click_count):
        obj = objects[click_indices[offset % len(click_indices)] % count]
        center = np.asarray(obj["transform"]["translation"], dtype=np.float64)
        projected = project_point(camera, center)
        if projected is None:
            raise ValueError(f"Object {obj['id']} is behind the generated camera")
        queries.append(
            {
                "query_index": query_index,
                "query_id": f"q-{query_index:03d}-click",
                "kind": "click",
                "screen": {
                    "x": projected["x"] + rng.uniform(-1.5, 1.5),
                    "y": projected["y"] + rng.uniform(-1.5, 1.5),
                },
                "viewport": {"width": config.viewport_width, "height": config.viewport_height},
                "tolerance_px": config.tolerance_px,
                "target_object_id": obj["id"],
            }
        )
        query_index += 1

    drag_indices = [4, 6, 8, 10]
    drag_deltas = [(36.0, -14.0), (-28.0, 18.0), (22.0, 22.0), (-40.0, -12.0)]
    for offset in range(config.drag_count):
        obj = objects[drag_indices[offset % len(drag_indices)] % count]
        center = np.asarray(obj["transform"]["translation"], dtype=np.float64)
        projected = project_point(camera, center)
        if projected is None:
            raise ValueError(f"Object {obj['id']} is behind the generated camera")
        delta_x, delta_y = drag_deltas[offset % len(drag_deltas)]
        queries.append(
            {
                "query_index": query_index,
                "query_id": f"q-{query_index:03d}-drag",
                "kind": "drag",
                "plane": {
                    "point": [float(value) for value in center],
                    "normal": [0.0, 1.0, 0.0],
                },
                "start_world": [float(value) for value in center],
                "start_screen": [projected["x"], projected["y"]],
                "end_screen": [projected["x"] + delta_x, projected["y"] + delta_y],
                "orientation": list(obj["transform"]["rotation_deg"]),
                "snap_enabled": True,
                "snap_spacing": 0.25,
                "snap_axes": ["x", "z"],
                "target_object_id": obj["id"],
            }
        )
        query_index += 1

    bundle = {
        "schema_version": 1,
        "seed": config.seed,
        "query_count": len(queries),
        "counts": {
            "ray": config.ray_count,
            "click": config.click_count,
            "drag": config.drag_count,
        },
        "queries": queries,
    }
    return to_builtin(bundle)


def generate_sample(
    config: GeneratorConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = config or GeneratorConfig()
    scene = generate_scene(config)
    return scene, generate_queries(scene, config)


def generate_sample_with_config(config: GeneratorConfig | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    config = config or GeneratorConfig()
    scene = generate_scene(config)
    return scene, generate_queries(scene, config)


def summarize_scene(scene: dict[str, Any], query_bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(scene["seed"]),
        "object_count": int(scene["object_count"]),
        "object_kinds": {
            kind: sum(1 for obj in scene["objects"] if obj["kind"] == kind)
            for kind in ("sphere", "aabb", "convex_polyhedron")
        },
        "bounds": scene["bounds"],
        "query_count": int(query_bundle["query_count"]),
        "query_counts": query_bundle["counts"],
    }
