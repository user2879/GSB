"""Content-keyed indexes over the most recent persisted stage payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheIndex:
    feature_by_key: dict[str, dict[str, Any]] = field(default_factory=dict)
    feature_by_id: dict[int, dict[str, Any]] = field(default_factory=dict)
    match_by_key: dict[str, dict[str, Any]] = field(default_factory=dict)
    match_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    triangle_by_key: dict[str, dict[str, Any]] = field(default_factory=dict)
    triangle_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    depth_pixel_by_key: dict[str, dict[str, Any]] = field(default_factory=dict)
    depth_pixel_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    voxel_by_key: dict[str, dict[str, Any]] = field(default_factory=dict)
    voxel_by_path: dict[str, dict[str, Any]] = field(default_factory=dict)
    depth_views: dict[int, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_stages(cls, stages: dict[str, Any]) -> "CacheIndex":
        index = cls()
        feature_stage = stages.get("features", {})
        for suite in feature_stage.get("views", {}).values():
            if "key" in suite:
                index.feature_by_key[suite["key"]] = suite
                index.feature_by_id[int(suite["view_id"])] = suite
        match_stage = stages.get("matches", {})
        for suite in match_stage.get("pairs", {}).values():
            if "key" in suite:
                index.match_by_key[suite["key"]] = suite
                index.match_by_id[suite["pair_id"]] = suite
        triangle_stage = stages.get("triangulation", {})
        for point in triangle_stage.get("points", {}).values():
            index.triangle_by_key[point["key"]] = point
            index.triangle_by_id[point["id"]] = point
        depth_stage = stages.get("depth", {})
        for view_id_string, view in depth_stage.get("views", {}).items():
            view_id = int(view_id_string)
            index.depth_views[view_id] = view
            for pixel in view.get("pixels", []):
                index.depth_pixel_by_key[pixel["key"]] = pixel
                index.depth_pixel_by_id[pixel["id"]] = pixel
        voxel_stage = stages.get("voxels", {})
        for path, node in voxel_stage.get("nodes", {}).items():
            index.voxel_by_key[node["key"]] = node
            index.voxel_by_path[path] = node
        return index


def empty_cache() -> CacheIndex:
    return CacheIndex()

