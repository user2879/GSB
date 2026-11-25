from __future__ import annotations

from typing import Any

from .depth import depth_parameter_fingerprint, compute_depth_pixel, pixel_cache_key
from .features import extract_view_features, feature_cache_key
from .geometry import triangulate_fixed
from .hashing import sha256_object, file_sha256
from .matching import match_pair, pair_cache_key
from .scheduler import StableTask, run_stable
from .strategy import estimate_subvoxels, select_strategy
from .voxels import (
    build_voxel_record,
    make_voxel_cache_key,
    point_root_index,
    root_specs,
    voxel_parameter_fingerprint,
)


def camera_fingerprint(camera: dict[str, Any]) -> str:
    return sha256_object(
        {
            "index": camera["index"],
            "K_q": camera["K_q"],
            "R_q": camera["R_q"],
            "t_q": camera["t_q"],
        }
    )


def _solver_fingerprint(config: dict[str, Any]) -> str:
    return sha256_object(
        {
            "stage": "fixed_triangulation",
            "algorithm_version": config.get("algorithm_version", "mvrv-1"),
            "scale": 1024,
            "method": "normal_equations_cramer_v1",
        }
    )


def _prev_index(previous: dict[str, Any] | None, key: str) -> dict[Any, dict[str, Any]]:
    if not previous:
        return {}
    return {item.get("id", item.get(key)): item for item in previous.get(key, [])}


def build_reconstruction(
    loaded: Any,
    previous_state: dict[str, Any] | None = None,
    *,
    global_budget: int | None = None,
    view_quota: int | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    if not loaded.ok:
        raise ValueError("cannot build reconstruction from failed validation result")

    config = loaded.config
    views = sorted(loaded.views, key=lambda item: int(item["index"]))
    view_indices = [int(view["index"]) for view in views]
    cameras_by_index = {int(view["index"]): view for view in views}
    image_hashes = {int(view["index"]): file_sha256(view["image_abs"]) for view in views}
    camera_keys = {index: camera_fingerprint(cameras_by_index[index]) for index in view_indices}
    camera_hash = sha256_object([camera_keys[index] for index in sorted(camera_keys)])
    config_fingerprint = sha256_object(config)

    pair_indices = [(view_indices[a], view_indices[b]) for a in range(len(view_indices)) for b in range(a + 1, len(view_indices))]
    estimated_children = estimate_subvoxels(config, len(view_indices), len(pair_indices))
    strategy = select_strategy(len(view_indices), len(pair_indices), estimated_children)
    worker_count = int(workers if workers is not None else config.get("workers", 1))

    prev_features = {item["view_index"]: item for item in (previous_state or {}).get("views", [])}
    prev_pairs = {tuple(item["pair"]): item for item in (previous_state or {}).get("pairs", [])}
    prev_points = {item["point_id"]: item for item in (previous_state or {}).get("points", [])}
    prev_depth_pixels = {
        item["depth_ref"]: item
        for depth_map in (previous_state or {}).get("depth_maps", [])
        for item in depth_map.get("pixels", [])
    }
    prev_voxels = {item["voxel_id"]: item for item in (previous_state or {}).get("voxels", [])}

    ledger: dict[str, Any] = {
        "mode": "incremental" if previous_state else "full",
        "features": {"recomputed": [], "reused": []},
        "pairs": {"recomputed": [], "reused": []},
        "points": {"recomputed": [], "reused": [], "gate_changed": [], "rejected": [], "removed": []},
        "depth_pixels": {"recomputed": [], "reused_count": 0},
        "voxels": {"recomputed": [], "reused": [], "removed": [], "reused_subtrees": []},
        "budget": {},
        "invalidated": [],
    }
    schedule_reports: list[dict[str, Any]] = []

    feature_tasks = [
        StableTask(f"feature:{index:02d}", (index,), "feature", index)
        for index in view_indices
    ]

    def handle_feature(task: StableTask) -> dict[str, Any]:
        index = int(task.payload)
        expected_key = feature_cache_key(index, image_hashes[index], strategy)
        previous = prev_features.get(index)
        if previous and previous.get("cache_key") == expected_key:
            return {"record": previous, "status": "reused"}
        record = extract_view_features(cameras_by_index[index], image_hashes[index], strategy)
        return {"record": record, "status": "recomputed"}

    feature_outputs, schedule = run_stable(feature_tasks, handle_feature, worker_count)
    schedule_reports.append({"stage": "features", **schedule})
    feature_sets: list[dict[str, Any]] = []
    for output in feature_outputs:
        record = output["record"]
        feature_sets.append(record)
        ledger["features"][output["status"]].append(record["view_index"])

    feature_by_index = {item["view_index"]: item for item in feature_sets}

    pair_tasks = [
        StableTask(f"pair:{i:02d}-{j:02d}", (i, j), "pair", (i, j))
        for i, j in pair_indices
    ]

    def handle_pair(task: StableTask) -> dict[str, Any]:
        i, j = task.payload
        left = feature_by_index[i]
        right = feature_by_index[j]
        expected_key = pair_cache_key(left, right, strategy)
        previous = prev_pairs.get((i, j))
        if previous and previous.get("cache_key") == expected_key:
            return {"record": previous, "status": "reused"}
        return {"record": match_pair(left, right, strategy), "status": "recomputed"}

    pair_outputs, schedule = run_stable(pair_tasks, handle_pair, worker_count)
    schedule_reports.append({"stage": "matching", **schedule})
    pair_records = []
    for output in pair_outputs:
        record = output["record"]
        pair_records.append(record)
        ledger["pairs"][output["status"]].append(record["pair_id"])

    global_rank = 0
    for pair in pair_records:
        for match in pair["matches"]:
            global_rank += 1
            match["global_inlier_rank"] = global_rank
            match["global_cumulative_inliers"] = global_rank

    candidates: list[dict[str, Any]] = []
    for pair in pair_records:
        i, j = pair["pair"]
        for match in pair["matches"]:
            lx, ly = match["left_xy"]
            rx, ry = match["right_xy"]
            point_id = f"T:{i:02d}-{j:02d}:{lx:02d},{ly:02d}"
            candidates.append(
                {
                    "point_id": point_id,
                    "pair": [i, j],
                    "match": match,
                    "left_xy": [lx, ly],
                    "right_xy": [rx, ry],
                    "sort_key": [i, j, lx, ly],
                }
            )

    budget_total = int(global_budget if global_budget is not None else config["global_point_budget"])
    quota = int(view_quota if view_quota is not None else config["view_point_quota"])
    remaining_global = budget_total
    remaining_view = {index: quota for index in view_indices}
    admitted: list[dict[str, Any]] = []
    denied: dict[str, str] = {}
    stop_reason: str | None = None
    consumed_views: dict[int, int] = {index: 0 for index in view_indices}

    for position, candidate in enumerate(candidates):
        i, j = candidate["pair"]
        if stop_reason is not None:
            denied[candidate["point_id"]] = "pending_budget"
            continue
        if remaining_global <= 0 or remaining_view[i] <= 0 or remaining_view[j] <= 0:
            stop_reason = (
                "global_point_budget_exhausted"
                if remaining_global <= 0
                else f"view_quota_exhausted:{i if remaining_view[i] <= 0 else j}"
            )
            denied[candidate["point_id"]] = "aborted_budget"
            continue
        admitted.append(candidate)
        remaining_global -= 1
        remaining_view[i] -= 1
        remaining_view[j] -= 1
        consumed_views[i] += 1
        consumed_views[j] += 1

    todo_queue = [point_id for point_id, status in denied.items() if status == "pending_budget"]
    aborted = [point_id for point_id, status in denied.items() if status == "aborted_budget"]
    ledger["budget"] = {
        "global_point_budget": budget_total,
        "view_point_quota": quota,
        "consumed_global": len(admitted),
        "consumed_by_view": consumed_views,
        "remaining_global": remaining_global,
        "remaining_by_view": remaining_view,
        "stop_reason": stop_reason,
        "aborted": aborted,
        "pending_count": len(todo_queue),
    }

    solver_fp = _solver_fingerprint(config)
    min_parallax_q = int(config["min_parallax"]) * 1024

    point_tasks = [
        StableTask(f"point:{candidate['point_id']}", tuple(candidate["sort_key"]), "triangulation", candidate)
        for candidate in admitted
    ]

    def handle_point(task: StableTask) -> dict[str, Any]:
        candidate = task.payload
        point_id = candidate["point_id"]
        i, j = candidate["pair"]
        lx, ly = candidate["left_xy"]
        rx, ry = candidate["right_xy"]
        match = candidate["match"]
        parallax_q = abs(lx - rx) * 1024
        geometry_key = sha256_object(
            {
                "stage": "tri_point_geometry",
                "point_id": point_id,
                "match_fingerprint": match["match_fingerprint"],
                "camera_keys": {str(i): camera_keys[i], str(j): camera_keys[j]},
                "solver_fingerprint": solver_fp,
            }
        )
        gate_key = sha256_object(
            {
                "stage": "tri_point_gate",
                "geometry_key": geometry_key,
                "min_parallax_q": min_parallax_q,
                "parallax_q": parallax_q,
            }
        )
        previous = prev_points.get(point_id)

        if parallax_q < min_parallax_q:
            record = {
                "point_id": point_id,
                "pair": [i, j],
                "match_id": match["match_id"],
                "left_xy": [lx, ly],
                "right_xy": [rx, ry],
                "parallax_q": parallax_q,
                "status": "rejected_parallax",
                "computed": False,
                "geometry_key": geometry_key,
                "cache_key": gate_key,
                "upstream": {
                    "match_fingerprint": match["match_fingerprint"],
                    "camera_keys": [camera_keys[i], camera_keys[j]],
                    "min_parallax_q": min_parallax_q,
                },
                "sort_key": candidate["sort_key"],
            }
            status = "reused" if previous and previous.get("cache_key") == gate_key else "gate_changed"
            return {"record": record, "status": status}

        if previous and previous.get("status") == "ok" and previous.get("geometry_key") == geometry_key:
            return {"record": previous, "status": "reused"}

        result = triangulate_fixed(
            [(lx, ly, cameras_by_index[i]), (rx, ry, cameras_by_index[j])]
        )
        if not result.get("ok"):
            record = {
                "point_id": point_id,
                "pair": [i, j],
                "status": "error",
                "computed": False,
                "error": result.get("error", "triangulation_failed"),
                "geometry_key": geometry_key,
                "cache_key": sha256_object({"error": result, "geometry_key": geometry_key}),
                "sort_key": candidate["sort_key"],
            }
            return {"record": record, "status": "recomputed"}

        depths = result["view_depth_q"]
        record = {
            "point_id": point_id,
            "pair": [i, j],
            "match_id": match["match_id"],
            "left_xy": [lx, ly],
            "right_xy": [rx, ry],
            "parallax_q": parallax_q,
            "status": "ok",
            "computed": True,
            "coords_q": result["coords_q"],
            "rounding": result["rounding"],
            "residual_q": result["residual_q"],
            "depth_q": result["depth_q"],
            "view_depths": {str(i): depths[0], str(j): depths[1]},
            "observations": [
                {"view_index": i, "xy": [lx, ly], "depth_q": depths[0]},
                {"view_index": j, "xy": [rx, ry], "depth_q": depths[1]},
            ],
            "geometry_key": geometry_key,
            "cache_key": geometry_key,
            "upstream": {
                "match_fingerprint": match["match_fingerprint"],
                "camera_keys": [camera_keys[i], camera_keys[j]],
                "solver_fingerprint": solver_fp,
            },
            "sort_key": candidate["sort_key"],
        }
        return {"record": record, "status": "recomputed"}

    point_outputs, schedule = run_stable(point_tasks, handle_point, worker_count)
    schedule_reports.append({"stage": "triangulation", **schedule})
    outputs_by_id = {output["record"]["point_id"]: output for output in point_outputs}
    point_records: list[dict[str, Any]] = []
    for candidate in candidates:
        point_id = candidate["point_id"]
        if point_id in denied:
            status = denied[point_id]
            record = {
                "point_id": point_id,
                "pair": candidate["pair"],
                "match_id": candidate["match"]["match_id"],
                "left_xy": candidate["left_xy"],
                "right_xy": candidate["right_xy"],
                "status": status,
                "computed": False,
                "cache_key": sha256_object({"budget_pending": point_id, "budget": ledger["budget"]}),
                "sort_key": candidate["sort_key"],
            }
            point_records.append(record)
            continue
        output = outputs_by_id[point_id]
        record = output["record"]
        point_records.append(record)
        if output["status"] == "recomputed":
            ledger["points"]["recomputed"].append(point_id)
        elif output["status"] == "reused":
            ledger["points"]["reused"].append(point_id)
        else:
            ledger["points"]["gate_changed"].append(point_id)
        if record.get("status") == "rejected_parallax":
            ledger["points"]["rejected"].append(point_id)

    ok_points = [point for point in point_records if point.get("status") == "ok"]

    depth_parameter_fp = depth_parameter_fingerprint(config)
    source_map: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for point in ok_points:
        for observation in point["observations"]:
            key = (int(observation["view_index"]), int(observation["xy"][0]), int(observation["xy"][1]))
            source_map.setdefault(key, []).append(point)

    depth_tasks = [
        StableTask(f"depth-view:{index:02d}", (index,), "depth", index)
        for index in view_indices
    ]

    def handle_depth_view(task: StableTask) -> dict[str, Any]:
        index = int(task.payload)
        pixels = []
        recomputed_refs = []
        for y in range(48):
            for x in range(64):
                sources = source_map.get((index, x, y), [])
                expected_key = pixel_cache_key(index, x, y, sources, depth_parameter_fp)
                previous = prev_depth_pixels.get(f"D:{index:02d}:{x:02d},{y:02d}")
                if previous and previous.get("cache_key") == expected_key:
                    pixel = previous
                    status = "reused"
                else:
                    pixel = compute_depth_pixel(index, x, y, sources, depth_parameter_fp, config)
                    status = "recomputed"
                    recomputed_refs.append(pixel["depth_ref"])
                pixels.append(pixel)
        return {"view_index": index, "pixels": pixels, "recomputed": recomputed_refs}

    depth_outputs, schedule = run_stable(depth_tasks, handle_depth_view, worker_count)
    schedule_reports.append({"stage": "depth", **schedule})
    depth_maps = []
    depth_recomputed: list[str] = []
    depth_reused_count = 0
    for output in depth_outputs:
        depth_maps.append(
            {
                "view_index": output["view_index"],
                "width": 64,
                "height": 48,
                "quantization_q": int(config["depth_quantization"]) * 1024,
                "pixels": output["pixels"],
            }
        )
        depth_recomputed.extend(output["recomputed"])
        depth_reused_count += 64 * 48 - len(output["recomputed"])
    ledger["depth_pixels"]["recomputed"] = depth_recomputed
    ledger["depth_pixels"]["reused_count"] = depth_reused_count
    depth_lookup = {
        (int(pixel["view_index"]), int(pixel["xy"][0]), int(pixel["xy"][1])): pixel
        for depth_map in depth_maps
        for pixel in depth_map["pixels"]
    }

    points_by_root: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for point in ok_points:
        root_index = point_root_index(point, config)
        points_by_root.setdefault(root_index, []).append(point)

    voxel_parameter_fp = voxel_parameter_fingerprint(config, strategy)
    roots = root_specs(config)
    voxel_records: dict[str, dict[str, Any]] = {}
    prev_children: dict[str, list[str]] = {}
    for voxel in (previous_state or {}).get("voxels", []):
        parent = voxel.get("parent_id")
        if parent:
            prev_children.setdefault(parent, []).append(voxel["voxel_id"])

    def collect_prev_subtree(root_id: str) -> list[dict[str, Any]]:
        result = []
        stack = [root_id]
        seen = set()
        while stack:
            current = stack.pop()
            if current in seen or current not in prev_voxels:
                continue
            seen.add(current)
            record = prev_voxels[current]
            result.append(record)
            stack.extend(prev_children.get(current, []))
        return result

    initial_voxel_tasks = []
    for spec in roots:
        root_index = tuple(spec["indices"])
        initial_voxel_tasks.append(
            StableTask(
                f"voxel:{spec['voxel_id']}",
                tuple(spec["sort_key"]),
                "voxel",
                {"spec": spec, "points": points_by_root.get(root_index, [])},
            )
        )

    pending_voxel_tasks = initial_voxel_tasks
    voxel_schedule_index = 0
    while pending_voxel_tasks:
        def handle_voxel(task: StableTask) -> dict[str, Any]:
            payload = task.payload
            spec = payload["spec"]
            point_refs = payload["points"]
            depth_keys = sorted(
                {
                    (int(obs["view_index"]), int(obs["xy"][0]), int(obs["xy"][1]))
                    for point in point_refs
                    for obs in point["observations"]
                }
            )
            depth_pixels = [depth_lookup[key] for key in depth_keys if key in depth_lookup]
            expected_key = make_voxel_cache_key(spec, point_refs, depth_pixels, voxel_parameter_fp)
            previous = prev_voxels.get(spec["voxel_id"])
            if previous and previous.get("cache_key") == expected_key:
                subtree = collect_prev_subtree(spec["voxel_id"])
                if subtree and subtree[0]["cache_key"] == expected_key:
                    return {
                        "status": "reused_subtree",
                        "record": subtree[0],
                        "subtree": subtree,
                        "reused_ids": [item["voxel_id"] for item in subtree],
                    }
            record = build_voxel_record(
                spec, point_refs, depth_lookup, config, strategy, voxel_parameter_fp
            )
            next_tasks = []
            if record["status"] == "subdivided":
                from .voxels import assign_child_points, child_spec

                for octant in record["child_octants"]:
                    child = child_spec(spec, int(octant))
                    child_points = assign_child_points(spec, int(octant), point_refs)
                    next_tasks.append(
                        StableTask(
                            f"voxel:{child['voxel_id']}",
                            tuple(child["sort_key"]),
                            "voxel",
                            {"spec": child, "points": child_points},
                        )
                    )
            return {"status": "recomputed", "record": record, "next_tasks": next_tasks}

        outputs, schedule = run_stable(pending_voxel_tasks, handle_voxel, worker_count)
        schedule_reports.append({"stage": f"voxels-level-{voxel_schedule_index}", **schedule})
        voxel_schedule_index += 1
        next_tasks: list[StableTask] = []
        for output in outputs:
            if output["status"] == "reused_subtree":
                for record in output["subtree"]:
                    voxel_records[record["voxel_id"]] = record
                ledger["voxels"]["reused"].extend(output["reused_ids"])
                ledger["voxels"]["reused_subtrees"].append(output["record"]["voxel_id"])
            else:
                record = output["record"]
                voxel_records[record["voxel_id"]] = record
                ledger["voxels"]["recomputed"].append(record["voxel_id"])
                next_tasks.extend(output.get("next_tasks", []))
        pending_voxel_tasks = sorted(next_tasks, key=lambda task: task.sort_key)

    voxel_list = sorted(voxel_records.values(), key=lambda record: tuple(record["sort_key"]))
    current_point_ids = {point["point_id"] for point in point_records}
    ledger["points"]["removed"] = sorted(set(prev_points) - current_point_ids)
    current_voxel_ids = {voxel["voxel_id"] for voxel in voxel_list}
    ledger["voxels"]["removed"] = sorted(set(prev_voxels) - current_voxel_ids)

    invalidated = set()
    invalidated.update(f"feature:{index}" for index in ledger["features"]["recomputed"])
    invalidated.update(ledger["pairs"]["recomputed"])
    invalidated.update(ledger["points"]["recomputed"])
    invalidated.update(ledger["points"]["gate_changed"])
    invalidated.update(ledger["points"]["removed"])
    invalidated.update(ledger["depth_pixels"]["recomputed"])
    invalidated.update(ledger["voxels"]["recomputed"])
    invalidated.update(ledger["voxels"]["removed"])
    ledger["invalidated"] = sorted(invalidated)

    boundary_pixels = sum(1 for dm in depth_maps for p in dm["pixels"] if p.get("boundary"))
    conflict_pixels = sum(1 for dm in depth_maps for p in dm["pixels"] if p.get("conflict"))
    occupied = sum(1 for v in voxel_list if v.get("status") == "occupied")
    subdivided = sum(1 for v in voxel_list if v.get("status") == "subdivided")
    empty = sum(1 for v in voxel_list if v.get("status") == "empty")
    max_voxel_level = max((int(v["level"]) for v in voxel_list), default=0)
    metrics = {
        "view_count": len(view_indices),
        "pair_count": len(pair_indices),
        "feature_count": sum(item["feature_count"] for item in feature_sets),
        "match_count": global_rank,
        "triangulated_points": len(ok_points),
        "rejected_parallax": len(ledger["points"]["rejected"]),
        "budget_aborted": len(aborted),
        "budget_pending": len(todo_queue),
        "depth_pixel_count": len(view_indices) * 64 * 48,
        "depth_boundary_pixels": boundary_pixels,
        "depth_conflict_pixels": conflict_pixels,
        "voxel_count": len(voxel_list),
        "occupied_voxels": occupied,
        "subdivided_voxels": subdivided,
        "empty_voxels": empty,
        "max_voxel_level": max_voxel_level,
        "strategy_id": strategy["id"],
        "estimated_subvoxels": estimated_children,
    }

    state = {
        "schema_version": "mvrv-state-1",
        "config_fingerprint": config_fingerprint,
        "camera_hash": camera_hash,
        "strategy": strategy,
        "views": feature_sets,
        "pairs": pair_records,
        "points": point_records,
        "depth_maps": depth_maps,
        "voxels": voxel_list,
        "todo_queue": [*aborted, *todo_queue],
        "metrics": metrics,
    }
    ledger["counts"] = {
        "features_recomputed": len(ledger["features"]["recomputed"]),
        "pairs_recomputed": len(ledger["pairs"]["recomputed"]),
        "points_recomputed": len(ledger["points"]["recomputed"]),
        "points_reused": len(ledger["points"]["reused"]),
        "point_gate_changed": len(ledger["points"]["gate_changed"]),
        "depth_recomputed": len(ledger["depth_pixels"]["recomputed"]),
        "depth_reused": ledger["depth_pixels"]["reused_count"],
        "voxels_recomputed": len(ledger["voxels"]["recomputed"]),
        "voxels_reused": len(ledger["voxels"]["reused"]),
        "invalidated_products": len(ledger["invalidated"]),
    }
    return {"state": state, "ledger": ledger, "schedules": schedule_reports}


def canonical_state_blob(state: dict[str, Any]) -> str:
    return sha256_object(state)
