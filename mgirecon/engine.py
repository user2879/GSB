"""Incremental reconstruction orchestrator."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .config import validate_config
from .depth import build_depth_map, quantum_for_level
from .errors import E_RECON_BUDGET_EXCEEDED, diagnostic
from .features import extract_features, feature_params_key
from .fixed import digest
from .geometry import point_cache_payload, point_stable_key, projection, triangulate_observation
from .matching import candidate_pairs, match_pair, match_params_key, pair_id, select_matching_pairs
from .pngio import decode_png
from .scheduler import TrackingExecutor
from .strategy import select_strategy
from .voxel import VoxelContext, build_voxels


def _empty_ledger() -> dict[str, list[dict[str, Any]]]:
    return {"reused": [], "invalidated": []}


def _add_reuse(ledger: dict[str, list[dict[str, Any]]], stage: str, key: str, cache_key: str, basis: dict[str, Any]) -> None:
    ledger["reused"].append({"stage": stage, "key": key, "cache_key": cache_key, "basis": basis})


def _add_invalidation(
    ledger: dict[str, list[dict[str, Any]]],
    stage: str,
    key: str,
    *,
    relations: list[str],
    reason: str,
    old_cache_key: str | None,
    new_cache_key: str | None,
) -> None:
    ledger["invalidated"].append(
        {
            "stage": stage,
            "key": key,
            "relations": sorted(relations),
            "reason": reason,
            "old_cache_key": old_cache_key,
            "new_cache_key": new_cache_key,
        }
    )


def _root_voxel_count(config: dict[str, Any]) -> int:
    reconstruction = config["reconstruction"]
    bounds = reconstruction["bounds"]
    voxel_size = reconstruction["voxel_size"]
    extents = [bounds["max"][i] - bounds["min"][i] for i in range(3)]
    if any(voxel_size[i] <= 0 or extents[i] % voxel_size[i] != 0 for i in range(3)):
        return 0
    count = 1
    for axis in range(3):
        count *= extents[axis] // voxel_size[axis]
    return count


def _view_map(validation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {view["id"]: view for view in validation["valid_views"]}


def _feature_stage(
    project_dir: Path,
    validation: dict[str, Any],
    strategy: dict[str, Any],
    previous_state: dict[str, Any],
    ledger: dict[str, list[dict[str, Any]]],
    max_workers: int,
) -> dict[str, Any]:
    config = validation["config"]
    width = config["resolution"]["width"]
    height = config["resolution"]["height"]
    fixed_bits = config["fixed_bits"]
    block_size = strategy["block_size"]
    descriptor_shift = strategy["descriptor_shift"]
    old_features = previous_state.get("features", {})
    reused: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for view in validation["valid_views"]:
        params_key = feature_params_key(
            view_id=view["id"],
            image_sha256=view["image_sha256"],
            width=width,
            height=height,
            block_size=block_size,
            descriptor_shift=descriptor_shift,
            fixed_bits=fixed_bits,
        )
        old = old_features.get(view["id"])
        if old is not None and old.get("params_key") == params_key:
            record = copy.deepcopy(old)
            reused[view["id"]] = record
            _add_reuse(
                ledger,
                "feature",
                view["id"],
                record["cache_key"],
                {
                    "image_sha256": view["image_sha256"],
                    "params_key": params_key,
                    "feature_count": record["feature_count"],
                },
            )
        else:
            pending.append(view)
            relations = ["content"] if old is not None else ["new"]
            _add_invalidation(
                ledger,
                "feature",
                view["id"],
                relations=relations,
                reason="image_or_parameter_changed" if old is not None else "new_view",
                old_cache_key=old.get("cache_key") if old else None,
                new_cache_key=params_key,
            )
    for view_id in sorted(set(old_features) - {view["id"] for view in validation["valid_views"]}):
        _add_invalidation(
            ledger,
            "feature",
            view_id,
            relations=["removed"],
            reason="view_missing_or_invalid",
            old_cache_key=old_features[view_id].get("cache_key"),
            new_cache_key=None,
        )

    def run_feature(view: dict[str, Any]) -> dict[str, Any]:
        image = decode_png(view["image_path"])
        return extract_features(
            image,
            view_id=view["id"],
            block_size=block_size,
            descriptor_shift=descriptor_shift,
            fixed_bits=fixed_bits,
        )

    computed: list[dict[str, Any]] = []
    if pending:
        with TrackingExecutor(max_workers) as executor:
            computed = executor.map_sorted(pending, key=lambda view: view["id"], function=run_feature)
            stats = executor.stats()
    else:
        stats = {"max_workers": max_workers, "max_active": 0, "submitted": 0}
    for record in computed:
        reused[record["view_id"]] = record
    ordered = {view_id: reused[view_id] for view_id in sorted(reused)}
    return {"maps": ordered, "executor": stats}


def _matching_stage(
    validation: dict[str, Any],
    features: dict[str, dict[str, Any]],
    strategy: dict[str, Any],
    config: dict[str, Any],
    previous_state: dict[str, Any],
    ledger: dict[str, list[dict[str, Any]]],
    max_workers: int,
) -> dict[str, Any]:
    reconstruction = config["reconstruction"]
    matching = config["matching"]
    valid_view_ids = sorted(features)
    pairs = candidate_pairs(valid_view_ids)
    old_pairs = {pair["pair_id"]: pair for pair in previous_state.get("match_pairs", [])}
    results: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, str, dict[str, Any]]] = []
    for left, right in pairs:
        identifier = pair_id(left, right)
        left_map = features[left]
        right_map = features[right]
        params_key = match_params_key(
            left_map=left_map,
            right_map=right_map,
            min_disparity=reconstruction["min_disparity"],
            residual_threshold=reconstruction["residual_threshold"],
            max_match_distance=strategy["max_match_distance"],
            quantization_level=reconstruction["quantization_level"],
            min_inliers=matching["min_inliers"],
        )
        old = old_pairs.get(identifier)
        if old is not None and old.get("params_key") == params_key:
            record = copy.deepcopy(old)
            record["selected"] = False
            record["tied"] = False
            record["tie_winner"] = False
            record["decision_rank"] = -1
            results[identifier] = record
            _add_reuse(
                ledger,
                "match_pair",
                identifier,
                record["cache_key"],
                {
                    "params_key": params_key,
                    "left_feature_key": left_map["cache_key"],
                    "right_feature_key": right_map["cache_key"],
                    "inlier_count": record["inlier_count"],
                },
            )
        else:
            pending.append((identifier, params_key, old))
            relations = ["feature"] if old is not None else ["new"]
            _add_invalidation(
                ledger,
                "match_pair",
                identifier,
                relations=relations,
                reason="upstream_feature_or_matching_parameter_changed" if old is not None else "new_pair",
                old_cache_key=old.get("cache_key") if old else None,
                new_cache_key=params_key,
            )

    def run_match(item: tuple[str, str, dict[str, Any] | None]) -> dict[str, Any]:
        identifier, _, _ = item
        left, right = identifier.split("-", 1)
        return match_pair(
            features[left],
            features[right],
            min_disparity=reconstruction["min_disparity"],
            residual_threshold=reconstruction["residual_threshold"],
            max_match_distance=strategy["max_match_distance"],
            quantization_level=reconstruction["quantization_level"],
            min_inliers=matching["min_inliers"],
        )

    computed: list[dict[str, Any]] = []
    if pending:
        with TrackingExecutor(max_workers) as executor:
            computed = executor.map_sorted(pending, key=lambda item: item[0], function=run_match)
            stats = executor.stats()
    else:
        stats = {"max_workers": max_workers, "max_active": 0, "submitted": 0}
    for record in computed:
        results[record["pair_id"]] = record
    for identifier in sorted(set(old_pairs) - set(results)):
        _add_invalidation(
            ledger,
            "match_pair",
            identifier,
            relations=["removed"],
            reason="feature_set_changed",
            old_cache_key=old_pairs[identifier].get("cache_key"),
            new_cache_key=None,
        )
    selected_and_sorted = select_matching_pairs(
        list(results.values()),
        min_inliers=matching["min_inliers"],
        max_pairs=matching["max_pairs"],
    )
    return {"pairs": selected_and_sorted, "executor": stats}


def _point_stage(
    validation: dict[str, Any],
    features: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    config: dict[str, Any],
    previous_state: dict[str, Any],
    ledger: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    reconstruction = config["reconstruction"]
    fixed_bits = config["fixed_bits"]
    level = reconstruction["quantization_level"]
    quantum = quantum_for_level(fixed_bits, level)
    cameras = _view_map(validation)
    old_points = {point["stable_key"]: point for point in previous_state.get("points", [])}
    points: list[dict[str, Any]] = []
    produced_keys: set[str] = set()
    selected = [pair for pair in pairs if pair["selected"]]
    for pair in sorted(selected, key=lambda item: item["pair_id"]):
        left_id = pair["left"]
        right_id = pair["right"]
        left_features = {feature["id"]: feature for feature in features[left_id]["features"]}
        right_features = {feature["id"]: feature for feature in features[right_id]["features"]}
        for match in sorted(
            pair["matches"],
            key=lambda item: (item["left_coord"][0], item["left_coord"][1], item["right_coord"], item["right_feature"]),
        ):
            stable_key = point_stable_key(pair["pair_id"], match["left_feature"], match["right_feature"])
            produced_keys.add(stable_key)
            payload = point_cache_payload(
                pair,
                cameras[left_id],
                cameras[right_id],
                stable_key,
                fixed_bits=fixed_bits,
                quantum=quantum,
                quantization_level=level,
            )
            expected_cache_key = digest(payload)
            old = old_points.get(stable_key)
            if old is not None and old.get("cache_key") == expected_cache_key:
                point = copy.deepcopy(old)
                point["residual_pass"] = point["residual"] <= reconstruction["residual_threshold"]
                point["budget_accepted"] = False
                point["budget_status"] = "pending"
                point["depth_refs"] = []
                points.append(point)
                _add_reuse(
                    ledger,
                    "triangulated_point",
                    stable_key,
                    point["cache_key"],
                    {
                        "pair_cache_key": pair["cache_key"],
                        "left_camera_key": cameras[left_id]["camera_hash"],
                        "right_camera_key": cameras[right_id]["camera_hash"],
                        "quantum": quantum,
                    },
                )
                continue
            left_feature = left_features[match["left_feature"]]
            right_feature = right_features[match["right_feature"]]
            triangulated = triangulate_observation(
                left_feature["coord"],
                right_feature["coord"],
                cameras[left_id],
                cameras[right_id],
                fixed_bits,
                quantum,
            )
            point_id = "p" + digest(stable_key)[:20]
            point = {
                "id": point_id,
                "stable_key": stable_key,
                "sort_key": stable_key,
                "pair_id": pair["pair_id"],
                "left_view": left_id,
                "right_view": right_id,
                "left_feature": left_feature["id"],
                "right_feature": right_feature["id"],
                "left_coord": left_feature["coord"],
                "right_coord": right_feature["coord"],
                "raw_xyz": triangulated["raw_xyz"],
                "xyz": triangulated["xyz"],
                "residual": triangulated["residual"],
                "residual_pass": triangulated["residual"] <= reconstruction["residual_threshold"],
                "solve_rounding": triangulated["solve_rounding"],
                "quantization_rounding": triangulated["quantization_rounding"],
                "determinant": triangulated["determinant"],
                "reprojections": triangulated["reprojections"],
                "cache_key": expected_cache_key,
                "cache_basis": payload,
                "budget_accepted": False,
                "budget_status": "pending",
                "budget_owner_views": [left_id, right_id],
                "depth_refs": [],
            }
            points.append(point)
            _add_invalidation(
                ledger,
                "triangulated_point",
                stable_key,
                relations=["pair", "camera", "quantization"] if old else ["new"],
                reason="upstream_match_camera_or_quantization_changed" if old else "new_point",
                old_cache_key=old.get("cache_key") if old else None,
                new_cache_key=expected_cache_key,
            )

    for stable_key in sorted(set(old_points) - produced_keys):
        _add_invalidation(
            ledger,
            "triangulated_point",
            stable_key,
            relations=["removed"],
            reason="matching_pair_or_inlier_removed",
            old_cache_key=old_points[stable_key].get("cache_key"),
            new_cache_key=None,
        )

    points.sort(key=lambda point: point["sort_key"])
    budget = config["scheduler"]["global_point_budget"]
    quota = config["scheduler"]["per_view_point_quota"]
    consumed = 0
    view_usage = {view_id: 0 for view_id in sorted(features)}
    pending: list[dict[str, Any]] = []
    for point in points:
        if not point["residual_pass"]:
            point["budget_status"] = "residual_rejected"
            continue
        views = sorted(set(point["budget_owner_views"]))
        if consumed >= budget:
            point["budget_status"] = "pending_global_budget"
            pending.append(
                {
                    "kind": "triangulated_point",
                    "key": point["stable_key"],
                    "views": views,
                    "reason": "global_point_budget",
                    "sort_key": point["sort_key"],
                }
            )
            continue
        if any(view_usage[view_id] >= quota for view_id in views):
            point["budget_status"] = "pending_view_quota"
            pending.append(
                {
                    "kind": "triangulated_point",
                    "key": point["stable_key"],
                    "views": views,
                    "reason": "per_view_point_quota",
                    "sort_key": point["sort_key"],
                }
            )
            continue
        point["budget_accepted"] = True
        point["budget_status"] = "accepted"
        consumed += 1
        for view_id in views:
            view_usage[view_id] += 1
    return {
        "points": points,
        "pending": pending,
        "budget": {
            "global_limit": budget,
            "per_view_quota": quota,
            "global_consumed": consumed,
            "per_view_consumed": view_usage,
            "pending_count": len(pending),
        },
        "quantum": quantum,
    }


def _expected_visible_points(
    accepted_points: list[dict[str, Any]],
    view: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    width = config["resolution"]["width"]
    height = config["resolution"]["height"]
    keys: set[str] = set()
    for point in accepted_points:
        projected = projection(point["xyz"], view["intrinsics"], view["extrinsics"], config["fixed_bits"])
        if not projected["valid"]:
            continue
        u = int(projected["u"])
        v = int(projected["v"])
        if 0 <= u < width and 0 <= v < height:
            keys.add(point["cache_key"])
    return sorted(keys)


def _depth_stage(
    validation: dict[str, Any],
    points: list[dict[str, Any]],
    config: dict[str, Any],
    previous_state: dict[str, Any],
    ledger: dict[str, list[dict[str, Any]]],
    quantum: int,
) -> dict[str, Any]:
    accepted = [point for point in points if point["budget_accepted"]]
    old_maps = previous_state.get("depth_maps", {})
    maps: dict[str, dict[str, Any]] = {}
    width = config["resolution"]["width"]
    height = config["resolution"]["height"]
    level = config["reconstruction"]["quantization_level"]
    for view in validation["valid_views"]:
        view_id = view["id"]
        expected_visible = _expected_visible_points(accepted, view, config)
        old = old_maps.get(view_id)
        can_reuse = (
            old is not None
            and old.get("quantum") == quantum
            and old.get("width") == width
            and old.get("height") == height
            and old.get("camera_hash") == view["camera_hash"]
            and old.get("visible_point_keys") == expected_visible
        )
        if can_reuse:
            record = copy.deepcopy(old)
            record.setdefault("run_state", {})["reuse"] = "map_hit"
            maps[view_id] = record
            _add_reuse(
                ledger,
                "depth_map",
                view_id,
                record["cache_key"],
                {
                    "camera_hash": view["camera_hash"],
                    "quantum": quantum,
                    "visible_point_key_digest": digest(expected_visible),
                    "pixel_count": width * height,
                },
            )
            continue
        record = build_depth_map(
            view,
            accepted,
            width=width,
            height=height,
            fixed_bits=config["fixed_bits"],
            quantum=quantum,
            quantization_level=level,
        )
        record["camera_hash"] = view["camera_hash"]
        record["fixed_bits"] = config["fixed_bits"]
        old_pixels = old.get("pixel_keys", []) if old else []
        if old_pixels and len(old_pixels) == len(record["pixel_keys"]):
            reused_pixels = sum(1 for left, right in zip(old_pixels, record["pixel_keys"]) if left == right)
        else:
            reused_pixels = 0
        record["run_state"] = {
            "reuse": "recomputed",
            "reused_pixels": reused_pixels,
            "invalidated_pixels": width * height - reused_pixels,
        }
        maps[view_id] = record
        relations: list[str] = []
        if old is None or old.get("visible_point_keys") != expected_visible:
            relations.append("point")
        if old is None or old.get("quantum") != quantum:
            relations.append("quantization")
        if old is not None and old.get("camera_hash") != view["camera_hash"]:
            relations.append("camera")
        if not relations:
            relations.append("depth")
        _add_invalidation(
            ledger,
            "depth_map",
            view_id,
            relations=relations,
            reason="upstream_points_quantization_or_camera_changed" if old else "new_depth_map",
            old_cache_key=old.get("cache_key") if old else None,
            new_cache_key=record["cache_key"],
        )
    for view_id in sorted(set(old_maps) - set(maps)):
        _add_invalidation(
            ledger,
            "depth_map",
            view_id,
            relations=["removed"],
            reason="view_missing",
            old_cache_key=old_maps[view_id].get("cache_key"),
            new_cache_key=None,
        )
    return {"maps": maps, "reused_pixels": sum(map(lambda item: int(item.get("run_state", {}).get("reused_pixels", 0)), maps.values()))}


def reconstruct(
    project_dir: str | Path,
    config: dict[str, Any],
    *,
    previous_state: dict[str, Any] | None = None,
    inherited_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(project_dir)
    validation = validate_config(config, root)
    errors = list(inherited_errors or []) + validation["errors"]
    ledger = _empty_ledger()
    previous = previous_state or {}
    valid_views = validation["valid_views"]
    pair_count = len(candidate_pairs([view["id"] for view in valid_views]))
    root_count = _root_voxel_count(validation["config"])
    strategy = select_strategy(
        len(valid_views),
        pair_count,
        validation["config"]["reconstruction"]["quantization_level"],
        root_count,
    )
    max_workers = validation["config"]["scheduler"]["max_workers"]

    feature_result = _feature_stage(root, validation, strategy, previous, ledger, max_workers)
    features = feature_result["maps"]
    matching_result = _matching_stage(validation, features, strategy, validation["config"], previous, ledger, max_workers)
    pairs = matching_result["pairs"]
    point_result = _point_stage(validation, features, pairs, validation["config"], previous, ledger)
    points = point_result["points"]
    pending = list(point_result["pending"])
    if point_result["pending"]:
        errors.append(
            diagnostic(
                E_RECON_BUDGET_EXCEEDED,
                "reconstruction point budget or per-view quota exhausted; pending work retained",
                stage="schedule",
                detail={
                    "global_limit": validation["config"]["scheduler"]["global_point_budget"],
                    "per_view_quota": validation["config"]["scheduler"]["per_view_point_quota"],
                    "pending_count": len(point_result["pending"]),
                    "first_pending": point_result["pending"][0]["key"],
                },
            )
        )
    depth_result = _depth_stage(
        validation,
        points,
        validation["config"],
        previous,
        ledger,
        point_result["quantum"],
    )
    depth_maps = depth_result["maps"]
    voxel_result: dict[str, Any] = {
        "voxels": [],
        "adjacency": [],
        "stats": {
            "voxel_count": 0,
            "leaf_count": 0,
            "subdivided_count": 0,
            "solid_leaf_count": 0,
            "empty_leaf_count": 0,
            "max_depth_used": 0,
            "depth_distribution": {},
            "adjacency_edges": 0,
        },
        "executor": {"max_workers": max_workers, "max_active": 0, "submitted": 0},
    }
    if validation["voxel_aligned"]:
        reconstruction = validation["config"]["reconstruction"]
        fixed_bits = validation["config"]["fixed_bits"]
        scale = 1 << fixed_bits
        bounds = reconstruction["bounds"]
        voxel_size = reconstruction["voxel_size"]
        strategy_depth = int(strategy["effective_max_depth"])
        max_depth = min(int(reconstruction["voxel_max_depth"]), strategy_depth)
        context = VoxelContext(
            fixed_bits=fixed_bits,
            quantum=point_result["quantum"],
            quantization_level=reconstruction["quantization_level"],
            residual_threshold=reconstruction["residual_threshold"],
            min_occupancy_points=reconstruction["min_occupancy_points"],
            max_depth=max_depth,
            split_order=tuple(strategy["split_order"]),
            bounds_min_fixed=tuple(value * scale for value in bounds["min"]),
            bounds_max_fixed=tuple(value * scale for value in bounds["max"]),
            root_size_fixed=tuple(value * scale for value in voxel_size),
            views=_view_map(validation),
            depth_maps=depth_maps,
        )
        accepted = [point for point in points if point["budget_accepted"]]
        voxel_result = build_voxels(
            accepted,
            context=context,
            old_voxels=previous.get("voxels"),
            max_workers=max_workers,
        )
        ledger["reused"].extend(voxel_result["ledger"]["reused"])
        ledger["invalidated"].extend(voxel_result["ledger"]["invalidated"])
        pending.extend(
            {
                "kind": "voxel",
                "key": voxel["path"],
                "reason": "awaiting_budget_for_upstream_points",
                "sort_key": voxel["path"],
            }
            for voxel in voxel_result["voxels"]
            if voxel["decision"] == "EMPTY" and point_result["pending"]
        )

    ledger["reused"].sort(key=lambda entry: (entry["stage"], entry["key"]))
    ledger["invalidated"].sort(key=lambda entry: (entry["stage"], entry["key"]))
    scheduler_stats = {
        "max_workers": max_workers,
        "features": feature_result["executor"],
        "matching": matching_result["executor"],
        "voxels": voxel_result["executor"],
        "max_active_observed": max(
            feature_result["executor"].get("max_active", 0),
            matching_result["executor"].get("max_active", 0),
            voxel_result["executor"].get("max_active", 0),
        ),
    }
    return {
        "schema_version": 1,
        "config": validation["config"],
        "config_hash": validation["config_hash"],
        "camera_hash": validation["camera_hash"],
        "validation": {
            "valid_view_ids": [view["id"] for view in valid_views],
            "images": validation["images"],
            "voxel_aligned": validation["voxel_aligned"],
            "errors": validation["errors"],
            "warnings": validation["warnings"],
        },
        "strategy": strategy,
        "features": features,
        "match_pairs": pairs,
        "points": points,
        "depth_maps": depth_maps,
        "voxels": voxel_result["voxels"],
        "adjacency": voxel_result["adjacency"],
        "voxel_stats": voxel_result["stats"],
        "ledger": ledger,
        "pending": sorted(pending, key=lambda item: (item["kind"], item["sort_key"], item["reason"])),
        "budget": point_result["budget"],
        "depth_reuse": {"reused_pixels": depth_result["reused_pixels"]},
        "scheduler": scheduler_stats,
        "errors": errors,
        "status": "ok" if not errors else "completed_with_errors",
    }
