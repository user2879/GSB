from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .depth import backproject_pixel_q
from .utils import fingerprint, round_div


def grid_spec(config: dict[str, Any]) -> dict[str, Any]:
    bounds = config["voxel_bounds"]
    size = int(config["voxel_size"])
    origin = [int(bounds[f"min_{axis}"]) for axis in ("x", "y", "z")]
    upper = [int(bounds[f"max_{axis}"]) for axis in ("x", "y", "z")]
    dims = [max(1, math.ceil((upper[i] - origin[i]) / size)) for i in range(3)]
    return {"origin": origin, "upper": upper, "size": size, "dims": dims}


def build_spatial_refs(
    triangulation: dict[str, Any],
    depth_maps: dict[str, dict[str, Any]],
    validated_views: list[dict[str, Any]],
    cameras: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    bits = int(config["frac_bits"])
    scale = 1 << bits
    spec = grid_spec(config)
    origin_q = [value * scale for value in spec["origin"]]
    root_size_q = int(spec["size"]) * scale
    dims = spec["dims"]

    tri_by_root: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    depth_by_root: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    out_of_bounds = {"tri": 0, "depth": 0}

    def root_index(world_q: list[int]) -> tuple[int, int, int] | None:
        indices = [(world_q[i] - origin_q[i]) // root_size_q for i in range(3)]
        if any(indices[i] < 0 or indices[i] >= dims[i] for i in range(3)):
            return None
        return int(indices[0]), int(indices[1]), int(indices[2])

    for point in triangulation["points"]:
        index = root_index(point["point_q"])
        if index is None:
            out_of_bounds["tri"] += 1
            continue
        tri_by_root.setdefault(index, []).append(point)

    for view in validated_views:
        view_id = view["view_id"]
        depth_map = depth_maps[view_id]
        camera = cameras[view_id]
        for row in depth_map["pixels"]:
            x, y, depth_q = int(row[0]), int(row[1]), int(row[2])
            world_q = backproject_pixel_q(view_id, x, y, depth_q, camera, config)
            if world_q is None:
                continue
            index = root_index(world_q)
            if index is None:
                out_of_bounds["depth"] += 1
                continue
            ref = {
                "pixel_ref": f"{view_id}:{x}:{y}",
                "view_id": view_id,
                "x": x,
                "y": y,
                "depth_q": depth_q,
                "flag": int(row[4]),
                "source_point_id": row[5],
                "depth_cache_key": depth_map["cache_key"],
                "world_q": world_q,
            }
            depth_by_root.setdefault(index, []).append(ref)

    return {
        "spec": spec,
        "origin_q": origin_q,
        "root_size_q": root_size_q,
        "tri_by_root": tri_by_root,
        "depth_by_root": depth_by_root,
        "out_of_bounds": out_of_bounds,
    }


def root_id(index: tuple[int, int, int]) -> str:
    return f"v{index[0]}_{index[1]}_{index[2]}"


def _contains(world_q: list[int], bounds_q: list[list[int]]) -> bool:
    low, high = bounds_q
    return all(low[i] <= world_q[i] < high[i] for i in range(3))


def evaluate_voxel(
    voxel_id: str,
    parent_id: str | None,
    parent_cache_key: str | None,
    sort_path: list[int],
    level: int,
    bounds_q: list[list[int]],
    tri_refs: list[dict[str, Any]],
    depth_refs: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    bits = int(config["frac_bits"])
    scale = 1 << bits
    shift = int(config["quant_shifts"][config["quant_level"]])
    threshold = int(config["residual_threshold"])
    max_depth = int(config["max_subdivision_depth"])
    weights = config["score_weights"]

    tri_sorted = sorted(tri_refs, key=lambda p: p["point_id"])
    depth_sorted = sorted(depth_refs, key=lambda r: r["pixel_ref"])
    tri_links = [{"point_id": p["point_id"], "cache_key": p["cache_key"]} for p in tri_sorted]
    depth_links = [r["pixel_ref"] for r in depth_sorted]
    depth_link_keys = [
        {
            "pixel_ref": r["pixel_ref"],
            "depth_q": int(r["depth_q"]),
            "depth_cache_key": r["depth_cache_key"],
        }
        for r in depth_sorted
    ]

    if tri_sorted:
        residual_total = sum(int(p["residual_q"]) for p in tri_sorted)
        residual_mean_q, residual_dir = round_div(residual_total, len(tri_sorted))
    else:
        residual_mean_q, residual_dir = 0, "exact"

    depth_values = [int(r["depth_q"]) for r in depth_sorted]
    if depth_values:
        depth_mean_q, _ = round_div(sum(depth_values), len(depth_values))
        mad_total = sum(abs(value - depth_mean_q) for value in depth_values)
        depth_mad_q, _ = round_div(mad_total, len(depth_values))
        conflict_count = sum(1 for ref in depth_sorted if int(ref["flag"]) == 1)
        disagreement_count = sum(1 for ref in depth_sorted if int(ref["flag"]) == 2)
        conflict_penalty_q = conflict_count * max(1, scale >> 1)
        depth_inconsistency_q = depth_mad_q + conflict_penalty_q
        depth_consistency_score_q = max(0, 8 * scale - depth_inconsistency_q)
    else:
        depth_mean_q = 0
        depth_mad_q = 0
        conflict_count = 0
        disagreement_count = 0
        conflict_penalty_q = 0
        depth_inconsistency_q = 0
        depth_consistency_score_q = 8 * scale

    combined_score_q = (
        int(weights["residual"]) * residual_mean_q
        + int(weights["depth"]) * depth_inconsistency_q
    )
    evidence = 5 * len(tri_sorted) + len(depth_sorted)
    occupied = (
        len(tri_sorted) >= int(config["occupancy_min_tri"])
        or len(depth_sorted) >= int(config["occupancy_min_depth"])
    )

    subdivision_reason = "score_above_threshold" if combined_score_q > threshold else "score_at_or_below_threshold"
    wants_subdivision = level < max_depth and combined_score_q > threshold and evidence > 0
    if level >= max_depth and combined_score_q > threshold:
        subdivision_reason = "max_subdivision_depth_reached"

    if wants_subdivision:
        provisional_status = "split_candidate"
        confidence_q = scale
    elif occupied:
        provisional_status = "occupied"
        confidence_q, _ = round_div(scale * evidence, evidence + int(config["occupancy_min_depth"]))
    else:
        provisional_status = "free"
        confidence_q, _ = round_div(
            scale * int(config["occupancy_min_depth"]),
            int(config["occupancy_min_depth"]) + min(evidence, int(config["occupancy_min_depth"])),
        )

    cache_key = fingerprint(
        {
            "stage": "voxel-evaluation",
            "kernel_version": 1,
            "voxel_id": voxel_id,
            "parent_id": parent_id,
            "parent_cache_key": parent_cache_key,
            "sort_path": sort_path,
            "level": level,
            "bounds_q": bounds_q,
            "tri_links": tri_links,
            "depth_links": depth_link_keys,
            "voxel_size": config["voxel_size"],
            "threshold": threshold,
            "max_depth": max_depth,
            "frac_bits": bits,
            "quant_level": config["quant_level"],
            "quant_shift": shift,
            "score_weights": weights,
        }
    )

    return {
        "stage": "voxel",
        "voxel_id": voxel_id,
        "parent_id": parent_id,
        "parent_cache_key": parent_cache_key,
        "sort_path": sort_path,
        "depth_level": level,
        "bounds_q": {"min": bounds_q[0], "max": bounds_q[1]},
        "tri_point_count": len(tri_sorted),
        "tri_point_refs": tri_links,
        "depth_pixel_count": len(depth_sorted),
        "depth_pixel_refs": depth_links,
        "depth_pixel_links": depth_link_keys,
        "residual_mean_q": residual_mean_q,
        "residual_round_dir": residual_dir,
        "depth_mean_q": depth_mean_q,
        "depth_mad_q": depth_mad_q,
        "depth_conflict_count": conflict_count,
        "depth_disagreement_count": disagreement_count,
        "depth_conflict_penalty_q": conflict_penalty_q,
        "depth_inconsistency_q": depth_inconsistency_q,
        "depth_consistency_score_q": depth_consistency_score_q,
        "combined_score_q": combined_score_q,
        "residual_threshold_q": threshold,
        "subdivision_basis": {
            "rule": subdivision_reason,
            "residual_mean_q": residual_mean_q,
            "depth_consistency_score_q": depth_consistency_score_q,
            "depth_inconsistency_q": depth_inconsistency_q,
            "combined_score_q": combined_score_q,
            "threshold_q": threshold,
            "wants_subdivision": wants_subdivision,
        },
        "provisional_status": provisional_status,
        "status": provisional_status,
        "confidence_q": int(confidence_q),
        "cache_key": cache_key,
        "child_ids": [],
        "_tri_refs": tri_sorted,
        "_depth_refs": depth_sorted,
    }


def make_children(node: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    low = list(node["bounds_q"]["min"])
    high = list(node["bounds_q"]["max"])
    mid = [(low[i] + high[i]) // 2 for i in range(3)]
    children: list[dict[str, Any]] = []
    parent_tri = node["_tri_refs"]
    parent_depth = node["_depth_refs"]
    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                bit_path = (ox << 2) | (oy << 1) | oz
                bits = (ox, oy, oz)
                child_low = [mid[i] if bits[i] else low[i] for i in range(3)]
                child_high = [high[i] if bits[i] else mid[i] for i in range(3)]
                child_bounds = [child_low, child_high]
                child_id = f"{node['voxel_id']}.{bit_path}"
                child_path = node["sort_path"] + [bit_path]
                tri_refs = [p for p in parent_tri if _contains(p["point_q"], child_bounds)]
                depth_refs = [r for r in parent_depth if _contains(r["world_q"], child_bounds)]
                child = evaluate_voxel(
                    child_id,
                    node["voxel_id"],
                    node["cache_key"],
                    child_path,
                    int(node["depth_level"]) + 1,
                    child_bounds,
                    tri_refs,
                    depth_refs,
                    config,
                )
                children.append(child)
    children.sort(key=lambda item: tuple(item["sort_path"]))
    return children


def run_voxel_stage(
    spatial: dict[str, Any],
    config: dict[str, Any],
    max_subdivisions: int | None = None,
    max_workers: int = 1,
    parallel: bool = True,
    priority_pending: list[str] | None = None,
) -> dict[str, Any]:
    spec = spatial["spec"]
    origin_q = spatial["origin_q"]
    root_size_q = spatial["root_size_q"]
    dims = spec["dims"]
    budget = int(config["budgets"]["max_subdivisions_per_batch"] if max_subdivisions is None else max_subdivisions)

    roots_context: list[tuple[tuple[int, int, int], list[list[int]], list[dict[str, Any]], list[dict[str, Any]]]] = []
    for ix in range(dims[0]):
        for iy in range(dims[1]):
            for iz in range(dims[2]):
                index = (ix, iy, iz)
                voxel_id = root_id(index)
                lo = [origin_q[i] + index[i] * root_size_q for i in range(3)]
                hi = [value + root_size_q for value in lo]
                roots_context.append(
                    (
                        index,
                        [lo, hi],
                        spatial["tri_by_root"].get(index, []),
                        spatial["depth_by_root"].get(index, []),
                    )
                )

    def evaluate_root(context):
        index, bounds, tri_refs, depth_refs = context
        return evaluate_voxel(root_id(index), None, None, list(index), 0, bounds, tri_refs, depth_refs, config)

    if parallel and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            current_nodes = list(executor.map(evaluate_root, roots_context))
    else:
        current_nodes = [evaluate_root(context) for context in roots_context]

    nodes: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, str]] = []
    split_events: list[dict[str, Any]] = []
    budget_used = 0
    priority = set(priority_pending or [])

    while current_nodes:
        current_nodes.sort(key=lambda node: tuple(node["sort_path"]))
        for node in current_nodes:
            nodes[node["voxel_id"]] = node

        next_nodes: list[dict[str, Any]] = []
        for node in current_nodes:
            if node["provisional_status"] != "split_candidate":
                node["status"] = node["provisional_status"]
                continue
            if budget_used >= budget:
                node["status"] = "pending"
                node["pending_code"] = "E_BUDGET_PENDING"
                pending.append(
                    {
                        "unit_type": "voxel_subdivision",
                        "unit_id": node["voxel_id"],
                        "stable_key": canonical_unit_key(node),
                        "reason": "max_subdivisions_per_batch exhausted",
                        "priority": "1" if node["voxel_id"] in priority else "0",
                    }
                )
                continue

            node["status"] = "subdivided"
            budget_used += 1
            split_events.append(
                {
                    "voxel_id": node["voxel_id"],
                    "cache_key": node["cache_key"],
                    "basis": node["subdivision_basis"],
                    "budget_sequence": budget_used,
                }
            )
            children = make_children(node, config)
            node["child_ids"] = [child["voxel_id"] for child in children]
            next_nodes.extend(children)

        current_nodes = next_nodes

    pending.sort(key=lambda item: (item["priority"] != "1", item["stable_key"]))
    all_nodes = [_strip_node(nodes[voxel_id]) for voxel_id in sorted(nodes, key=lambda vid: tuple(nodes[vid]["sort_path"]))]
    output_nodes = {node["voxel_id"]: node for node in all_nodes}
    counts = {
        "total": len(all_nodes),
        "subdivided": sum(1 for n in all_nodes if n["status"] == "subdivided"),
        "occupied": sum(1 for n in all_nodes if n["status"] == "occupied"),
        "free": sum(1 for n in all_nodes if n["status"] == "free"),
        "pending": sum(1 for n in all_nodes if n["status"] == "pending"),
    }
    forest_key = fingerprint(
        {
            "stage": "voxel-forest",
            "nodes": [
                {k: v for k, v in node.items() if not k.startswith("_")}
                for node in sorted(all_nodes, key=lambda n: n["voxel_id"])
            ],
            "counts": counts,
        }
    )
    return {
        "stage": "voxels",
        "cache_key": forest_key,
        "grid_spec": spatial["spec"],
        "out_of_bounds": spatial["out_of_bounds"],
        "budget": {"limit": budget, "used": budget_used, "pending_count": len(pending)},
        "counts": counts,
        "pending": pending,
        "split_events": split_events,
        "nodes": output_nodes,
        "node_list": all_nodes,
    }


def canonical_unit_key(node: dict[str, Any]) -> str:
    return "/".join(str(value) for value in node["sort_path"])


def _strip_node(node: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in node.items() if not key.startswith("_")}
