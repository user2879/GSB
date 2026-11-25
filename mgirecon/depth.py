"""Full-resolution quantized depth maps with explicit conflict handling."""

from __future__ import annotations

from typing import Any

from .fixed import digest, quantize
from .geometry import projection


def quantum_for_level(fixed_bits: int, level: int) -> int:
    return level * (1 << max(0, fixed_bits - 4))


def _pixel_key(view_id: str, x: int, y: int, quantum: int, source_key: str | None) -> str:
    return digest(
        {
            "stage": "depth_pixel",
            "view": view_id,
            "coord": [x, y],
            "quantum": quantum,
            "source_point_key": source_key,
            "rule": "nearest-then-lower-depth-half-quantum-v1",
        }
    )


def build_depth_map(
    view: dict[str, Any],
    points: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    fixed_bits: int,
    quantum: int,
    quantization_level: int,
) -> dict[str, Any]:
    view_id = view["id"]
    sources: list[dict[str, Any]] = []
    for point in points:
        projected = projection(point["xyz"], view["intrinsics"], view["extrinsics"], fixed_bits)
        if not projected["valid"]:
            continue
        u = int(projected["u"])
        v = int(projected["v"])
        if not (0 <= u < width and 0 <= v < height):
            continue
        depth_value, depth_direction = quantize(int(projected["depth"]), quantum)
        sources.append(
            {
                "point_id": point["id"],
                "point_key": point["cache_key"],
                "u": u,
                "v": v,
                "depth": depth_value,
                "rounding": depth_direction,
            }
        )
    sources.sort(key=lambda item: (item["v"], item["u"], item["point_id"]))

    depths = [0] * (width * height)
    directions = ["exact"] * (width * height)
    boundary = [0] * (width * height)
    source_points: list[str | None] = [None] * (width * height)
    pixel_keys = [""] * (width * height)
    depth_refs_by_point: dict[str, list[str]] = {}
    for source in sources:
        pixel_id = f"{view_id}:{source['u']}:{source['v']}"
        depth_refs_by_point.setdefault(source["point_id"], []).append(pixel_id)

    if not sources:
        for y in range(height):
            for x in range(width):
                index = y * width + x
                pixel_keys[index] = _pixel_key(view_id, x, y, quantum, None)
        return {
            "view_id": view_id,
            "width": width,
            "height": height,
            "quantization_level": quantization_level,
            "quantum": quantum,
            "cache_key": digest(
                {
                    "stage": "depth_map",
                    "view": view_id,
                    "camera_hash": view["camera_hash"],
                    "quantum": quantum,
                    "pixel_keys": pixel_keys,
                    "rule": "nearest-then-lower-depth-half-quantum-v1",
                }
            ),
            "depth": depths,
            "rounding": directions,
            "boundary": boundary,
            "source_points": source_points,
            "pixel_keys": pixel_keys,
            "depth_refs_by_point": depth_refs_by_point,
            "visible_point_keys": [],
            "populated_pixels": 0,
            "boundary_pixels": 0,
        }

    for y in range(height):
        for x in range(width):
            ranked: list[tuple[int, str, dict[str, Any]]] = []
            for source in sources:
                distance = (source["u"] - x) ** 2 + (source["v"] - y) ** 2
                ranked.append((distance, source["point_id"], source))
            ranked.sort(key=lambda item: (item[0], item[1]))
            best_distance, best_id, best = ranked[0]
            chosen = best
            conflict = 0
            if len(ranked) > 1:
                second_distance, second_id, second = ranked[1]
                if second_distance - best_distance <= 1:
                    half_quantum = quantum // 2
                    if abs(second["depth"] - best["depth"]) <= half_quantum:
                        conflict = 1
                        options = sorted(
                            [(best["depth"], best_id, best), (second["depth"], second_id, second)],
                            key=lambda item: (item[0], item[1]),
                        )
                        chosen = options[0][2]
            index = y * width + x
            depths[index] = int(chosen["depth"])
            directions[index] = str(chosen["rounding"])
            boundary[index] = conflict
            source_points[index] = chosen["point_id"]
            pixel_keys[index] = _pixel_key(view_id, x, y, quantum, chosen["point_key"])

    populated = sum(1 for value in source_points if value is not None)
    boundary_count = sum(boundary)
    visible_point_keys = sorted({source["point_key"] for source in sources})
    return {
        "view_id": view_id,
        "width": width,
        "height": height,
        "quantization_level": quantization_level,
        "quantum": quantum,
        "cache_key": digest(
            {
                "stage": "depth_map",
                "view": view_id,
                "camera_hash": view["camera_hash"],
                "quantum": quantum,
                "pixel_keys": pixel_keys,
                "rule": "nearest-then-lower-depth-half-quantum-v1",
            }
        ),
        "depth": depths,
        "rounding": directions,
        "boundary": boundary,
        "source_points": source_points,
        "pixel_keys": pixel_keys,
        "depth_refs_by_point": depth_refs_by_point,
        "visible_point_keys": visible_point_keys,
        "populated_pixels": populated,
        "boundary_pixels": boundary_count,
    }
