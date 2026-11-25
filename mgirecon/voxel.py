"""Residual-driven adaptive voxel occupancy with precise cache reuse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .fixed import digest, mean_int
from .geometry import projection
from .scheduler import TrackingExecutor


@dataclass(frozen=True)
class VoxelContext:
    fixed_bits: int
    quantum: int
    quantization_level: int
    residual_threshold: int
    min_occupancy_points: int
    max_depth: int
    split_order: tuple[int, ...]
    bounds_min_fixed: tuple[int, int, int]
    bounds_max_fixed: tuple[int, int, int]
    root_size_fixed: tuple[int, int, int]
    views: dict[str, dict[str, Any]]
    depth_maps: dict[str, dict[str, Any]]


def _format_index(value: int) -> str:
    return f"{value:04d}"


def _root_path(z: int, y: int, x: int) -> str:
    return "/".join((_format_index(z), _format_index(y), _format_index(x)))


def _child_path(parent: str, octant: int) -> str:
    return f"{parent}/{octant}"


def _in_bounds(value: int, minimum: int, maximum: int, is_global_max: bool) -> bool:
    return minimum <= value < maximum or (is_global_max and value == maximum)


def _point_in_bounds(point: dict[str, Any], bounds_min: tuple[int, int, int], bounds_max: tuple[int, int, int], context: VoxelContext) -> bool:
    return all(
        _in_bounds(
            int(point["xyz"][axis]),
            bounds_min[axis],
            bounds_max[axis],
            bounds_max[axis] == context.bounds_max_fixed[axis],
        )
        for axis in range(3)
    )


def _prepare_point_depth_refs(points: list[dict[str, Any]], context: VoxelContext) -> None:
    for point in points:
        refs: list[dict[str, Any]] = []
        for view_id in sorted(context.views):
            view = context.views[view_id]
            depth_map = context.depth_maps.get(view_id)
            if depth_map is None:
                continue
            projected = projection(point["xyz"], view["intrinsics"], view["extrinsics"], context.fixed_bits)
            if not projected["valid"]:
                continue
            u = int(projected["u"])
            v = int(projected["v"])
            if not (0 <= u < depth_map["width"] and 0 <= v < depth_map["height"]):
                continue
            index = v * depth_map["width"] + u
            refs.append(
                {
                    "view": view_id,
                    "pixel": f"{view_id}:{u}:{v}",
                    "pixel_key": depth_map["pixel_keys"][index],
                    "depth": depth_map["depth"][index],
                    "boundary": depth_map["boundary"][index],
                }
            )
        point["depth_refs"] = sorted(refs, key=lambda item: item["pixel"])


def _geometry_path_key(path: str) -> str:
    return path


def _base_payload(
    path: str,
    parent: str | None,
    depth: int,
    bounds_min: tuple[int, int, int],
    bounds_max: tuple[int, int, int],
    point_keys: list[str],
    depth_pixel_keys: list[str],
    context: VoxelContext,
) -> dict[str, Any]:
    return {
        "stage": "voxel",
        "path": path,
        "parent": parent,
        "depth": depth,
        "bounds_min": list(bounds_min),
        "bounds_max": list(bounds_max),
        "point_keys": point_keys,
        "depth_pixel_keys": depth_pixel_keys,
        "residual_threshold": context.residual_threshold,
        "quantum": context.quantum,
        "quantization_level": context.quantization_level,
        "min_occupancy_points": context.min_occupancy_points,
        "max_depth": context.max_depth,
        "split_order": list(context.split_order),
        "rule": "residual-mean-plus-depth-spread-v1",
    }


def _score_node(
    points: list[dict[str, Any]],
    depth_refs: list[dict[str, Any]],
    context: VoxelContext,
) -> tuple[int, int, int]:
    if not points:
        return 0, 0, 0
    residual_mean = mean_int(int(point["residual"]) for point in points)
    if not depth_refs:
        return residual_mean, residual_mean, 0
    depth_values = [int(ref["depth"]) for ref in depth_refs]
    depth_spread = (max(depth_values) - min(depth_values)) // context.quantum
    return residual_mean + depth_spread, residual_mean, depth_spread


def _empty_ledger() -> dict[str, list[dict[str, Any]]]:
    return {"reused": [], "invalidated": []}


def _reuse_subtree(
    path: str,
    old_map: dict[str, dict[str, Any]],
    *,
    basis: dict[str, Any],
    ledger: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    prefix = path + "/"
    paths = [candidate for candidate in old_map if candidate == path or candidate.startswith(prefix)]
    paths.sort(key=_geometry_path_key)
    copied: list[dict[str, Any]] = []
    for candidate in paths:
        record = dict(old_map[candidate])
        record.setdefault("run_state", {})["reuse"] = "hit"
        copied.append(record)
        ledger["reused"].append(
            {
                "stage": "voxel",
                "key": candidate,
                "cache_key": record["cache_key"],
                "basis": {
                    "path": candidate,
                    "point_key_digest": digest(record.get("point_keys", [])),
                    "depth_key_digest": digest(record.get("depth_pixel_keys", [])),
                    "parent": record.get("parent"),
                    "children": record.get("children", []),
                    "subtree_root_basis": basis["cache_key"],
                },
            }
        )
    return copied


def _invalidation_relations(
    old: dict[str, Any] | None,
    *,
    point_keys: list[str],
    depth_keys: list[str],
    children: list[str],
    parent: str | None,
    hierarchy_changed: bool,
) -> list[str]:
    if old is None:
        return ["hierarchy"]
    relations: list[str] = []
    if old.get("point_keys", []) != point_keys:
        relations.append("point")
    if old.get("depth_pixel_keys", []) != depth_keys:
        relations.append("depth")
    if hierarchy_changed or old.get("children", []) != children or old.get("parent") != parent:
        relations.append("hierarchy")
    if not relations:
        relations.append("parameter")
    return relations


def _build_node(
    path: str,
    parent: str | None,
    depth: int,
    bounds_min: tuple[int, int, int],
    bounds_max: tuple[int, int, int],
    all_points: list[dict[str, Any]],
    context: VoxelContext,
    old_map: dict[str, dict[str, Any]],
    ledger: dict[str, list[dict[str, Any]]],
    *,
    force_hierarchy_invalid: bool = False,
) -> list[dict[str, Any]]:
    points = [point for point in all_points if _point_in_bounds(point, bounds_min, bounds_max, context)]
    points.sort(key=lambda point: point["stable_key"])
    point_keys = [point["cache_key"] for point in points]
    depth_refs: list[dict[str, Any]] = []
    for point in points:
        depth_refs.extend(point["depth_refs"])
    depth_refs.sort(key=lambda ref: ref["pixel"])
    depth_keys = [ref["pixel_key"] for ref in depth_refs]
    base_payload = _base_payload(
        path,
        parent,
        depth,
        bounds_min,
        bounds_max,
        point_keys,
        depth_keys,
        context,
    )
    base_key = digest(base_payload)
    old = old_map.get(path)
    can_reuse = (
        not force_hierarchy_invalid
        and old is not None
        and old.get("base_key") == base_key
    )
    if can_reuse:
        return _reuse_subtree(path, old_map, basis={**base_payload, "cache_key": base_key}, ledger=ledger)

    score, residual_mean, depth_spread = _score_node(points, depth_refs, context)
    should_subdivide = bool(points) and depth < context.max_depth and score >= context.residual_threshold
    children: list[str] = []
    records: list[dict[str, Any]] = []
    structure_changed = old is not None and old.get("decision") == "SUBDIVIDED" and not should_subdivide
    if should_subdivide:
        sizes = tuple((bounds_max[axis] - bounds_min[axis]) // 2 for axis in range(3))
        octants = list(range(8))
        order_axis = context.split_order[0] if context.split_order else 0
        octants.sort(key=lambda octant: (((octant >> order_axis) & 1), octant))
        for octant in octants:
            child_min = list(bounds_min)
            child_max = list(bounds_max)
            for axis in range(3):
                if (octant >> axis) & 1:
                    child_min[axis] = bounds_min[axis] + sizes[axis]
                else:
                    child_max[axis] = bounds_min[axis] + sizes[axis]
            child = _child_path(path, octant)
            children.append(child)
            records.extend(
                _build_node(
                    child,
                    path,
                    depth + 1,
                    tuple(child_min),
                    tuple(child_max),
                    points,
                    context,
                    old_map,
                    ledger,
                    force_hierarchy_invalid=force_hierarchy_invalid or structure_changed,
                )
            )
        children.sort(key=_geometry_path_key)
        decision = "SUBDIVIDED"
        direct_occupied: bool | None = None
        descendant_solid = sum(int(record.get("descendant_solid_leaf_count", 0)) for record in records if record.get("parent") == path)
    else:
        direct_occupied = len(points) >= max(1, context.min_occupancy_points)
        decision = "SOLID" if direct_occupied else "EMPTY"
        descendant_solid = 1 if direct_occupied else 0

    child_keys = sorted(record["cache_key"] for record in records if record.get("parent") == path)
    cache_payload = {**base_payload, "base_key": base_key, "children": children, "child_keys": child_keys}
    cache_key = digest(cache_payload)
    record = {
        "path": path,
        "parent": parent,
        "children": children,
        "depth": depth,
        "bounds_min": list(bounds_min),
        "bounds_max": list(bounds_max),
        "voxel_size_fixed": [bounds_max[i] - bounds_min[i] for i in range(3)],
        "voxel_size_world": [(bounds_max[i] - bounds_min[i]) // (1 << context.fixed_bits) for i in range(3)],
        "point_refs": [point["id"] for point in points],
        "point_keys": point_keys,
        "depth_refs": [ref["pixel"] for ref in depth_refs],
        "depth_pixel_keys": depth_keys,
        "residual_mean": residual_mean,
        "depth_spread": depth_spread,
        "score": score,
        "score_basis": {
            "residual_mean": residual_mean,
            "depth_spread_quanta": depth_spread,
            "residual_threshold": context.residual_threshold,
        },
        "decision": decision,
        "direct_occupied": direct_occupied,
        "descendant_solid_leaf_count": descendant_solid,
        "occupancy_rule": "leaf-direct-only; parent aggregate is reporting-only",
        "base_key": base_key,
        "cache_key": cache_key,
        "cache_basis": {
            "base_key": base_key,
            "point_key_digest": digest(point_keys),
            "depth_key_digest": digest(depth_keys),
            "child_keys": child_keys,
            "split_order": list(context.split_order),
        },
        "run_state": {"reuse": "miss"},
    }
    relations = _invalidation_relations(
        old,
        point_keys=point_keys,
        depth_keys=depth_keys,
        children=children,
        parent=parent,
        hierarchy_changed=structure_changed or force_hierarchy_invalid,
    )
    if old is not None and old.get("cache_key") == cache_key:
        record["run_state"]["reuse"] = "hit"
        ledger["reused"].append(
            {
                "stage": "voxel",
                "key": path,
                "cache_key": cache_key,
                "basis": {
                    "point_key_digest": digest(point_keys),
                    "depth_key_digest": digest(depth_keys),
                    "child_keys": child_keys,
                    "decision": decision,
                },
            }
        )
    else:
        ledger["invalidated"].append(
            {
                "stage": "voxel",
                "key": path,
                "relations": relations,
                "reason": "cache_key_changed" if old is not None else "new_voxel",
                "old_cache_key": old.get("cache_key") if old else None,
                "new_cache_key": cache_key,
                "decision": decision,
                "score": score,
            }
        )
    records.append(record)
    records.sort(key=lambda item: item["path"])
    return records


def _leaf_adjacency(voxels: list[dict[str, Any]]) -> list[list[str]]:
    leaves = [voxel for voxel in voxels if voxel["decision"] != "SUBDIVIDED"]
    edges: list[list[str]] = []
    for index, left in enumerate(leaves):
        for right in leaves[index + 1 :]:
            shared_faces = 0
            overlap = True
            for axis in range(3):
                left_min, left_max = left["bounds_min"][axis], left["bounds_max"][axis]
                right_min, right_max = right["bounds_min"][axis], right["bounds_max"][axis]
                touching = left_max == right_min or right_max == left_min
                overlapping = min(left_max, right_max) > max(left_min, right_min)
                if touching:
                    shared_faces += 1
                elif not overlapping:
                    overlap = False
                    break
            if overlap and shared_faces == 1:
                edges.append(sorted([left["path"], right["path"]]))
    edges.sort(key=lambda edge: (edge[0], edge[1]))
    return edges


def _root_jobs(context: VoxelContext) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    counts = [
        (context.bounds_max_fixed[axis] - context.bounds_min_fixed[axis]) // context.root_size_fixed[axis]
        for axis in range(3)
    ]
    order = list(context.split_order)
    ranges = [range(counts[axis]) for axis in order]
    for values in _cartesian_product(ranges):
        index = [0, 0, 0]
        for axis, value in zip(order, values):
            index[axis] = value
        bounds_min = tuple(
            context.bounds_min_fixed[axis] + index[axis] * context.root_size_fixed[axis] for axis in range(3)
        )
        bounds_max = tuple(
            context.bounds_min_fixed[axis] + (index[axis] + 1) * context.root_size_fixed[axis] for axis in range(3)
        )
        jobs.append(
            {
                "path": _root_path(index[2], index[1], index[0]),
                "bounds_min": bounds_min,
                "bounds_max": bounds_max,
            }
        )
    jobs.sort(key=lambda job: job["path"])
    return jobs


def _cartesian_product(ranges):
    if not ranges:
        yield ()
        return
    first, rest = ranges[0], ranges[1:]
    for value in first:
        for suffix in _cartesian_product(rest):
            yield (value,) + suffix


def build_voxels(
    accepted_points: list[dict[str, Any]],
    *,
    context: VoxelContext,
    old_voxels: list[dict[str, Any]] | None,
    max_workers: int,
) -> dict[str, Any]:
    _prepare_point_depth_refs(accepted_points, context)
    old_map = {voxel["path"]: voxel for voxel in (old_voxels or [])}
    ledger = _empty_ledger()
    root_jobs = _root_jobs(context)

    def run_root(job: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, int]]:
        local_ledger = _empty_ledger()
        records = _build_node(
            job["path"],
            None,
            0,
            job["bounds_min"],
            job["bounds_max"],
            accepted_points,
            context,
            old_map,
            local_ledger,
        )
        return records, local_ledger, {"roots": 1}

    with TrackingExecutor(max_workers) as executor:
        results = executor.map_sorted(root_jobs, key=lambda job: job["path"], function=run_root)
        executor_stats = executor.stats()

    voxels: list[dict[str, Any]] = []
    for records, local_ledger, _ in results:
        voxels.extend(records)
        ledger["reused"].extend(local_ledger["reused"])
        ledger["invalidated"].extend(local_ledger["invalidated"])
    voxels.sort(key=lambda voxel: voxel["path"])
    ledger["reused"].sort(key=lambda entry: entry["key"])
    ledger["invalidated"].sort(key=lambda entry: entry["key"])
    produced_paths = {voxel["path"] for voxel in voxels}
    for path in sorted(set(old_map) - produced_paths):
        ledger["invalidated"].append(
            {
                "stage": "voxel",
                "key": path,
                "relations": ["hierarchy"],
                "reason": "removed_by_parent_structure",
                "old_cache_key": old_map[path].get("cache_key"),
                "new_cache_key": None,
                "decision": None,
                "score": None,
            }
        )
    ledger["invalidated"].sort(key=lambda entry: entry["key"])
    leaves = [voxel for voxel in voxels if voxel["decision"] != "SUBDIVIDED"]
    depth_distribution: dict[str, int] = {}
    for voxel in leaves:
        key = str(voxel["depth"])
        depth_distribution[key] = depth_distribution.get(key, 0) + 1
    adjacency = _leaf_adjacency(voxels)
    return {
        "voxels": voxels,
        "voxel_map": {voxel["path"]: voxel for voxel in voxels},
        "adjacency": adjacency,
        "ledger": ledger,
        "stats": {
            "voxel_count": len(voxels),
            "leaf_count": len(leaves),
            "subdivided_count": len(voxels) - len(leaves),
            "solid_leaf_count": sum(1 for voxel in leaves if voxel["decision"] == "SOLID"),
            "empty_leaf_count": sum(1 for voxel in leaves if voxel["decision"] == "EMPTY"),
            "max_depth_used": max((voxel["depth"] for voxel in voxels), default=0),
            "depth_distribution": {key: depth_distribution[key] for key in sorted(depth_distribution, key=int)},
            "adjacency_edges": len(adjacency),
        },
        "executor": executor_stats,
    }
