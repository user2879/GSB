"""Residual-driven octree voxel subdivision and occupancy decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .constants import ENGINE_VERSION
from .depth import depth_pixel_identifier
from .fixed import quantize, stable_hash
from .models import ReconstructionConfig, ValidatedView
from .triangulation import project_triangle


@dataclass
class VoxelBuild:
    stage: dict[str, Any]
    events: list[dict[str, Any]]
    math: dict[str, int]


def _octant_bounds(
    bounds: tuple[int, int, int, int, int, int],
    octant: int,
) -> tuple[int, int, int, int, int, int]:
    x0, y0, z0, x1, y1, z1 = bounds
    mx = (x0 + x1) // 2
    my = (y0 + y1) // 2
    mz = (z0 + z1) // 2
    x_low, x_high = (x0, mx) if not (octant & 4) else (mx, x1)
    y_low, y_high = (y0, my) if not (octant & 2) else (my, y1)
    z_low, z_high = (z0, mz) if not (octant & 1) else (mz, z1)
    return (x_low, y_low, z_low, x_high, y_high, z_high)


def _contains(bounds_fixed: tuple[int, int, int, int, int, int], point: list[int]) -> bool:
    x0, y0, z0, x1, y1, z1 = bounds_fixed
    return x0 <= point[0] < x1 and y0 <= point[1] < y1 and z0 <= point[2] < z1


def _evaluation_order(
    triangles: list[dict[str, Any]], order_name: str
) -> list[dict[str, Any]]:
    if order_name == "depth_first":
        return sorted(
            triangles,
            key=lambda item: (-item["residual_fixed"], item["id"], item["key"]),
        )
    if order_name == "breadth":
        return sorted(
            triangles,
            key=lambda item: (item["residual_fixed"], item["id"], item["key"]),
        )
    return sorted(triangles, key=lambda item: (item["id"], item["key"]))


class _Builder:
    def __init__(
        self,
        triangles: dict[str, dict[str, Any]],
        views: dict[int, ValidatedView],
        depth_views: dict[int, dict[str, Any]],
        config: ReconstructionConfig,
        image_width: int,
        image_height: int,
        strategy: dict[str, Any],
        cache_by_key: dict[str, dict[str, Any]],
        old_by_path: dict[str, dict[str, Any]],
    ) -> None:
        self.triangles = triangles
        self.views = views
        self.config = config
        self.image_width = image_width
        self.image_height = image_height
        self.order_name = strategy["subdivision_order"]
        self.cache_by_key = cache_by_key
        self.old_by_path = old_by_path
        self.scale = config.scale
        self.nodes: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.math = {
            "triangle_intersection_checks": 0,
            "depth_projection_checks": 0,
            "multiply_adds": 0,
        }
        self.pixel_by_view_coordinate: dict[int, dict[tuple[int, int], dict[str, Any]]] = {
            int(view_id): {
                (pixel["x"], pixel["y"]): pixel
                for pixel in depth_views[view_id]["pixels"]
            }
            for view_id in sorted(depth_views)
        }

    def _triangles_in_bounds(
        self, bounds_fixed: tuple[int, int, int, int, int, int]
    ) -> list[dict[str, Any]]:
        return [
            triangle
            for triangle_id, triangle in sorted(self.triangles.items())
            if _contains(bounds_fixed, triangle["xyz_q"])
        ]

    def _depth_dependencies(
        self, triangles: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], int, int]:
        observations: list[dict[str, Any]] = []
        pixels: list[dict[str, Any]] = []
        pixel_keys: set[str] = set()
        for triangle in triangles:
            for view_id in sorted(self.views):
                projection = project_triangle(
                    triangle, self.views[view_id].camera, self.config
                )
                self.math["depth_projection_checks"] += 1
                if projection is None:
                    continue
                x, y, depth_raw = projection
                if x < 0 or y < 0 or x >= self.image_width or y >= self.image_height:
                    continue
                pixel = self.pixel_by_view_coordinate[view_id].get((x, y))
                if pixel is None:
                    continue
                depth_q, _, _ = quantize(depth_raw, self.config.quant_step)
                difference = abs(pixel["depth_q"] - depth_q)
                observations.append(
                    {
                        "triangle_id": triangle["id"],
                        "view_id": view_id,
                        "pixel_id": pixel["id"],
                        "pixel_key": pixel["key"],
                        "difference_fixed": difference,
                        "tolerance_fixed": self.config.quant_step * 2,
                    }
                )
                pixels.append(pixel)
                pixel_keys.add(pixel["key"])
        unique_pixels = sorted(pixels, key=lambda item: (item["id"], item["key"]))
        merged_pixels: list[dict[str, Any]] = []
        seen: set[str] = set()
        for pixel in unique_pixels:
            if pixel["id"] in seen or pixel["key"] in seen:
                continue
            seen.add(pixel["id"])
            seen.add(pixel["key"])
            merged_pixels.append(pixel)
        return observations, merged_pixels, sorted(pixel_keys), len(observations), len(pixel_keys)

    def _base_key(
        self,
        path: str,
        bounds: tuple[int, int, int, int, int, int],
        triangle_keys: list[str],
        depth_keys: list[str],
    ) -> str:
        return stable_hash(
            {
                "engine_version": ENGINE_VERSION,
                "stage": "voxel",
                "path": path,
                "bounds": list(bounds),
                "bits": self.config.bits,
                "quant_level": self.config.quant_level,
                "quant_step": self.config.quant_step,
                "voxel_size": self.config.voxel_size,
                "residual_threshold": self.config.residual_threshold,
                "max_subdiv_depth": self.config.max_subdiv_depth,
                "max_voxels_per_parent": self.config.max_voxels_per_parent,
                "world_bounds": list(self.config.world_bounds),
                "subdivision_order": self.order_name,
                "triangle_keys": triangle_keys,
                "depth_pixel_keys": depth_keys,
            }
        )

    def _collect_cached_subtree(self, record: dict[str, Any], root_path: str) -> bool:
        pending = [record]
        collected: list[dict[str, Any]] = []
        while pending:
            current = pending.pop(0)
            path = current["path"]
            if path in self.nodes:
                continue
            cached = self.old_by_path.get(path)
            if cached is None or cached.get("key") != current.get("key"):
                return False
            collected.append(cached)
            for child_id in current.get("children", []):
                child_path = child_id[2:] if child_id.startswith("v:") else child_id
                child = self.old_by_path.get(child_path)
                if child is None:
                    return False
                pending.append(child)
        for node in collected:
            self.nodes[node["path"]] = node
            self.events.append(
                {
                    "stage": "voxels",
                    "artifact": node["id"],
                    "sort_key": node["path"],
                    "action": "reused",
                    "reason": f"cached_parent_key:{root_path}",
                    "key": node["key"],
                }
            )
        return True

    def _record(self, path: str, record: dict[str, Any]) -> None:
        self.nodes[path] = record
        previous = self.old_by_path.get(path)
        if previous is None:
            action = "computed"
            reason = "no_prior_artifact"
        elif previous.get("key") != record["key"]:
            action = "invalidated"
            reason = "voxel_dependency_or_parameter_key_changed"
        else:
            action = "reused"
            reason = "cache_key_match"
        self.events.append(
            {
                "stage": "voxels",
                "artifact": record["id"],
                "sort_key": path,
                "action": action,
                "reason": reason,
                "key": record["key"],
            }
        )

    def visit(
        self,
        path: str,
        bounds: tuple[int, int, int, int, int, int],
        parent_id: str,
        depth: int,
        is_root: bool,
    ) -> dict[str, Any]:
        bounds_fixed = (
            bounds[0] * self.scale,
            bounds[1] * self.scale,
            bounds[2] * self.scale,
            bounds[3] * self.scale,
            bounds[4] * self.scale,
            bounds[5] * self.scale,
        )
        triangles = self._triangles_in_bounds(bounds_fixed)
        self.math["triangle_intersection_checks"] += len(self.triangles)
        triangles = _evaluation_order(triangles, self.order_name)
        triangle_keys = sorted(triangle["key"] for triangle in triangles)
        observations, depth_pixels, depth_keys, observation_count, pixel_count = (
            self._depth_dependencies(triangles)
        )
        self.math["multiply_adds"] += len(triangles) * len(self.views) * 3 + observation_count
        base_key = self._base_key(path, bounds, triangle_keys, depth_keys)
        cached = self.cache_by_key.get(base_key)
        if cached is not None and self._collect_cached_subtree(cached, path):
            return cached

        residual_mean = (
            sum(triangle["residual_fixed"] for triangle in triangles) // len(triangles)
            if triangles
            else 0
        )
        depth_inconsistency = (
            sum(observation["difference_fixed"] for observation in observations)
            // observation_count
            if observation_count
            else 0
        )
        consistent = sum(
            1
            for observation in observations
            if observation["difference_fixed"] <= observation["tolerance_fixed"]
        )
        consistency_permille = (
            consistent * 1000 // observation_count if observation_count else 0
        )
        combined = residual_mean + depth_inconsistency
        threshold = self.config.residual_threshold * self.scale
        reaches_threshold = bool(triangles) and combined >= threshold
        blocker = ""
        if not triangles:
            blocker = "no_triangle_support"
        elif not reaches_threshold:
            blocker = "below_threshold"
        elif depth >= self.config.max_subdiv_depth:
            blocker = "max_subdivision_depth"
        subdivide = bool(triangles) and reaches_threshold and not blocker
        child_records: list[dict[str, Any]] = []
        if subdivide:
            child_candidates = []
            for octant in range(self.config.max_voxels_per_parent):
                child_path = f"{path}/{octant}"
                child_bounds = _octant_bounds(bounds, octant)
                if (
                    child_bounds[0] >= child_bounds[3]
                    or child_bounds[1] >= child_bounds[4]
                    or child_bounds[2] >= child_bounds[5]
                ):
                    continue
                child_candidates.append((child_path, child_bounds))
            for child_path, child_bounds in child_candidates:
                child = self.visit(
                    child_path,
                    child_bounds,
                    "",
                    depth + 1,
                    False,
                )
                child_records.append(child)
        occupied = None
        if not subdivide:
            occupied = bool(triangles) and consistency_permille >= 500
        children_ids = [child["id"] for child in sorted(child_records, key=lambda item: item["path"])]
        record = {
            "id": f"v:{path}",
            "path": path,
            "key": base_key,
            "parent_id": parent_id,
            "depth": depth,
            "is_root": is_root,
            "bounds": list(bounds),
            "triangle_ids": sorted(triangle["id"] for triangle in triangles),
            "triangle_keys": triangle_keys,
            "depth_pixel_ids": sorted(pixel["id"] for pixel in depth_pixels),
            "depth_pixel_keys": [pixel["key"] for pixel in depth_pixels],
            "observation_count": observation_count,
            "consistent_observations": consistent,
            "consistency_permille": consistency_permille,
            "residual_mean_fixed": residual_mean,
            "depth_inconsistency_mean_fixed": depth_inconsistency,
            "combined_score_fixed": combined,
            "residual_threshold_fixed": threshold,
            "subdivide": subdivide,
            "subdivision_basis": {
                "rule": "residual_mean_plus_depth_inconsistency_ge_threshold",
                "reaches_threshold": reaches_threshold,
                "blocker": blocker or "subdivided",
                "evaluated_triangle_ids": [triangle["id"] for triangle in triangles],
            },
            "children": children_ids,
            "occupied": occupied,
            "occupancy_basis": (
                "not_leaf"
                if subdivide
                else (
                    "triangle_support_and_depth_consistency_ge_500_permille"
                    if occupied
                    else "insufficient_support_or_consistency"
                )
            ),
        }
        self._record(path, record)
        for child in child_records:
            child["parent_id"] = record["id"]
        return record

    def build(self) -> VoxelBuild:
        x0, y0, z0, x1, y1, z1 = self.config.world_bounds
        roots: list[str] = []
        for ix, x_low in enumerate(range(x0, x1, self.config.voxel_size)):
            for iy, y_low in enumerate(range(y0, y1, self.config.voxel_size)):
                for iz, z_low in enumerate(range(z0, z1, self.config.voxel_size)):
                    path = f"r:{ix:04d}:{iy:04d}:{iz:04d}"
                    bounds = (
                        x_low,
                        y_low,
                        z_low,
                        min(x_low + self.config.voxel_size, x1),
                        min(y_low + self.config.voxel_size, y1),
                        min(z_low + self.config.voxel_size, z1),
                    )
                    self.visit(path, bounds, "", 0, True)
                    roots.append(path)
        nodes = {path: self.nodes[path] for path in sorted(self.nodes)}
        leaf_count = sum(1 for node in nodes.values() if not node["subdivide"])
        occupied_count = sum(1 for node in nodes.values() if node["occupied"])
        parent_count = sum(1 for node in nodes.values() if node["subdivide"])
        stage = {
            "nodes": nodes,
            "order": sorted(nodes),
            "roots": sorted(roots),
            "leaf_count": leaf_count,
            "occupied_count": occupied_count,
            "parent_count": parent_count,
            "node_count": len(nodes),
        }
        self.events.sort(key=lambda item: item["sort_key"])
        return VoxelBuild(stage=stage, events=self.events, math=self.math)


def build_voxels(
    triangles: dict[str, dict[str, Any]],
    views: dict[int, ValidatedView],
    depth_views: dict[int, dict[str, Any]],
    config: ReconstructionConfig,
    image_width: int,
    image_height: int,
    strategy: dict[str, Any],
    cache_by_key: dict[str, dict[str, Any]],
    old_by_path: dict[str, dict[str, Any]],
) -> VoxelBuild:
    builder = _Builder(
        triangles=triangles,
        views=views,
        depth_views=depth_views,
        config=config,
        image_width=image_width,
        image_height=image_height,
        strategy=strategy,
        cache_by_key=cache_by_key,
        old_by_path=old_by_path,
    )
    return builder.build()
