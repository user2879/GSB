from __future__ import annotations

import math
from typing import Any

from .depth import depth_ref_id
from .hashing import sha256_object


def voxel_parameter_fingerprint(config: dict[str, Any], strategy: dict[str, Any]) -> str:
    return sha256_object(
        {
            "stage": "voxel_parameters",
            "algorithm_version": config.get("algorithm_version", "mvrv-1"),
            "residual_threshold_q": int(config["residual_threshold_q"]),
            "voxel_size_q": int(config["voxel_size"]) * 1024,
            "max_subdivision_depth": int(config["max_subdivision_depth"]),
            "depth_quantization_q": int(config["depth_quantization"]) * 1024,
            "strategy_fingerprint": strategy["fingerprint"],
            "weights": {"residual": 3, "depth": 1},
        }
    )


def root_dimensions(config: dict[str, Any]) -> tuple[int, int, int]:
    bounds = config["world_bounds"]
    size = int(config["voxel_size"])
    return tuple(
        math.ceil((bounds["max"][i] - bounds["min"][i]) / size)
        for i in range(3)
    )  # type: ignore[return-value]


def root_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    bounds = config["world_bounds"]
    size_real = int(config["voxel_size"])
    size_q = size_real * 1024
    nx, ny, nz = root_dimensions(config)
    specs = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                min_q = [
                    bounds["min"][0] * 1024 + ix * size_q,
                    bounds["min"][1] * 1024 + iy * size_q,
                    bounds["min"][2] * 1024 + iz * size_q,
                ]
                voxel_id = f"V:L0:{ix:02d},{iy:02d},{iz:02d}"
                specs.append(
                    {
                        "voxel_id": voxel_id,
                        "level": 0,
                        "indices": [ix, iy, iz],
                        "min_q": min_q,
                        "size_q": [size_q, size_q, size_q],
                        "parent_id": None,
                        "octant": None,
                        "sort_key": [0, ix, iy, iz],
                    }
                )
    return sorted(specs, key=lambda item: tuple(item["sort_key"]))


def point_root_index(point: dict[str, Any], config: dict[str, Any]) -> tuple[int, int, int]:
    bounds = config["world_bounds"]
    size_q = int(config["voxel_size"]) * 1024
    nx, ny, nz = root_dimensions(config)
    indices = []
    for axis in range(3):
        raw = (int(point["coords_q"][axis]) - bounds["min"][axis] * 1024) // size_q
        indices.append(max(0, min((nx, ny, nz)[axis] - 1, raw)))
    return tuple(indices)  # type: ignore[return-value]


def child_spec(spec: dict[str, Any], octant: int) -> dict[str, Any]:
    half = [value // 2 for value in spec["size_q"]]
    bits = [octant & 1, (octant >> 1) & 1, (octant >> 2) & 1]
    min_q = [spec["min_q"][axis] + bits[axis] * half[axis] for axis in range(3)]
    indices = list(spec["indices"])
    path = list(spec.get("path", [])) + [octant]
    return {
        "voxel_id": spec["voxel_id"] + f".o{octant}",
        "level": int(spec["level"]) + 1,
        "indices": indices,
        "path": path,
        "min_q": min_q,
        "size_q": half,
        "parent_id": spec["voxel_id"],
        "octant": octant,
        "sort_key": [int(spec["level"]) + 1, *indices, *path],
    }


def make_voxel_cache_key(
    spec: dict[str, Any],
    point_refs: list[dict[str, Any]],
    depth_pixels: list[dict[str, Any]],
    parameter_fingerprint: str,
) -> str:
    point_keys = sorted(
        (
            {"point_id": point["point_id"], "cache_key": point["cache_key"]}
            for point in point_refs
        ),
        key=lambda item: item["point_id"],
    )
    depth_keys = sorted(
        (
            {"depth_ref": pixel["depth_ref"], "cache_key": pixel["cache_key"]}
            for pixel in depth_pixels
        ),
        key=lambda item: item["depth_ref"],
    )
    return sha256_object(
        {
            "stage": "voxel",
            "voxel_id": spec["voxel_id"],
            "geometry": {"min_q": spec["min_q"], "size_q": spec["size_q"]},
            "parameter_fingerprint": parameter_fingerprint,
            "points": point_keys,
            "depth_pixels": depth_keys,
        }
    )


def build_voxel_record(
    spec: dict[str, Any],
    point_refs: list[dict[str, Any]],
    depth_lookup: dict[tuple[int, int, int], dict[str, Any]],
    config: dict[str, Any],
    strategy: dict[str, Any],
    parameter_fingerprint: str,
) -> dict[str, Any]:
    depth_ref_keys = sorted(
        {
            (obs["view_index"], obs["xy"][0], obs["xy"][1])
            for point in point_refs
            for obs in point["observations"]
        }
    )
    depth_pixels = [depth_lookup[key] for key in depth_ref_keys if key in depth_lookup]
    cache_key = make_voxel_cache_key(spec, point_refs, depth_pixels, parameter_fingerprint)
    point_ids = sorted(point["point_id"] for point in point_refs)
    depth_refs = sorted(depth_ref_id(*key) for key in depth_ref_keys)

    record: dict[str, Any] = {
        "voxel_id": spec["voxel_id"],
        "level": spec["level"],
        "resolution_level": spec["level"],
        "edge_size_q": spec["size_q"][0],
        "indices": spec["indices"],
        "path": spec.get("path", []),
        "parent_id": spec["parent_id"],
        "octant": spec["octant"],
        "min_q": spec["min_q"],
        "size_q": spec["size_q"],
        "sort_key": spec["sort_key"],
        "point_refs": point_ids,
        "depth_refs": depth_refs,
        "cache_key": cache_key,
        "strategy_id": strategy["id"],
        "subdivision_order": strategy["child_order"],
    }

    if not point_refs:
        record.update(
            {
                "status": "empty",
                "occupied": False,
                "score_q": 0,
                "threshold_q": int(config["residual_threshold_q"]),
                "decision_basis": "no_triangulated_points",
            }
        )
        return record

    residual_mean_q = sum(int(p["residual_q"]) for p in point_refs) // len(point_refs)
    depth_values = [int(pixel["depth_q"]) for pixel in depth_pixels if pixel["depth_q"] > 0]
    depth_spread_q = max(depth_values) - min(depth_values) if depth_values else 0
    boundary_count = sum(1 for pixel in depth_pixels if pixel.get("boundary"))
    step_q = int(config["depth_quantization"]) * 1024
    boundary_penalty_q = boundary_count * step_q // (8 * max(1, len(depth_pixels)))
    depth_inconsistency_q = depth_spread_q + boundary_penalty_q
    score_q = (3 * residual_mean_q + depth_inconsistency_q) // 4
    threshold_q = int(config["residual_threshold_q"])
    max_depth = int(config["max_subdivision_depth"])

    record.update(
        {
            "residual_mean_q": residual_mean_q,
            "depth_spread_q": depth_spread_q,
            "boundary_ref_count": boundary_count,
            "depth_inconsistency_q": depth_inconsistency_q,
            "score_q": score_q,
            "threshold_q": threshold_q,
            "point_count": len(point_refs),
            "depth_pixel_count": len(depth_pixels),
        }
    )

    if score_q >= threshold_q and int(spec["level"]) < max_depth:
        record.update(
            {
                "status": "subdivided",
                "occupied": False,
                "decision_basis": "score_meets_residual_threshold",
                "subdivision_reason": {
                    "residual_mean_q": residual_mean_q,
                    "depth_inconsistency_q": depth_inconsistency_q,
                    "score_q": score_q,
                    "threshold_q": threshold_q,
                    "order_axis": strategy["order_axis"],
                },
                "child_octants": list(strategy["child_order"]),
            }
        )
    else:
        basis = (
            "max_depth_threshold_exceeded"
            if score_q >= threshold_q
            else "score_below_residual_threshold"
        )
        record.update(
            {
                "status": "occupied",
                "occupied": True,
                "occupancy_basis": basis,
                "decision_basis": basis,
            }
        )
    return record


def assign_child_points(
    spec: dict[str, Any], octant: int, points: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    child = child_spec(spec, octant)
    result = []
    for point in points:
        coords = point["coords_q"]
        inside = True
        for axis in range(3):
            lower = child["min_q"][axis]
            upper = lower + child["size_q"][axis]
            if not (lower <= coords[axis] < upper):
                inside = False
                break
        if inside:
            result.append(point)
    return result
