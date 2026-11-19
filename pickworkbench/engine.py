"""Batch picking/space-operation validation, first-failure location, and witnesses."""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from .canonical import content_hash, to_builtin, with_content_hash
from .errors import (
    ANOMALY_ORDER,
    E_INJECT_CONFIG,
    E_NORMAL_ZERO,
    E_OBJECT_DUP_ID,
    E_PICK_MISMATCH,
    E_PRIORITY_RANGE,
    E_RAY_DEGENERATE,
    E_SNAP_SPACING,
    WorkbenchError,
    error_sort_key,
    make_error,
)
from .geometry import (
    all_object_hits,
    normalize,
    object_screen_bounds,
    perform_drag,
    ray_from_screen,
    ray_intersects_aabb,
    replay_drag_without_snap,
    screen_bounds_contain,
    screen_candidates,
    select_winner,
    validate_snap_spacing,
    vec3,
)
from .injection import apply_injection

SELECTION_RULE = "priority_desc,hit_distance_asc,object_id_asc"
QUERY_ORDER = "query_index_asc"
OBJECT_ORDER = "object_id_asc,object_index_asc"


def _query_snapshot(query: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(query)


def _strip_objects_for_trigger(scene: dict[str, Any], indices: Iterable[int]) -> list[dict[str, Any]]:
    unique = sorted(set(int(index) for index in indices))
    return [copy.deepcopy(scene["objects"][index]) for index in unique]


def _duplicate_groups(objects: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, obj in enumerate(objects):
        groups[str(obj["id"])].append(index)
    return {key: value for key, value in groups.items() if len(value) > 1}


def _object_anomalies(scene: dict[str, Any]) -> tuple[dict[int, list[WorkbenchError]], dict[str, list[int]]]:
    objects = scene["objects"]
    duplicate_groups = _duplicate_groups(objects)
    duplicate_index_map: dict[int, str] = {}
    for object_id, indices in duplicate_groups.items():
        for index in indices:
            duplicate_index_map[index] = object_id
    priority_range = scene.get("priority_range", {"min": 0, "max": 99})
    minimum_priority = int(priority_range["min"])
    maximum_priority = int(priority_range["max"])

    anomalies: dict[int, list[WorkbenchError]] = defaultdict(list)
    for object_index, obj in enumerate(objects):
        object_id = str(obj["id"])
        if object_index in duplicate_index_map:
            indices = duplicate_groups[duplicate_index_map[object_index]]
            trigger_objects = _strip_objects_for_trigger(scene, indices)
            anomalies[object_index].append(
                make_error(
                    E_OBJECT_DUP_ID,
                    f"Object ID {object_id!r} appears on {len(indices)} objects.",
                    object_id=object_id,
                    object_indices=indices,
                    object_ids=[str(objects[index]["id"]) for index in indices],
                    objects=trigger_objects,
                )
            )
        priority = int(obj.get("pick_priority", 0))
        if priority < minimum_priority or priority > maximum_priority:
            anomalies[object_index].append(
                make_error(
                    E_PRIORITY_RANGE,
                    f"Pick priority {priority} is outside [{minimum_priority}, {maximum_priority}].",
                    object_id=object_id,
                    object_index=object_index,
                    priority=priority,
                    priority_range={"min": minimum_priority, "max": maximum_priority},
                    object=copy.deepcopy(obj),
                )
            )
        if obj["kind"] == "convex_polyhedron":
            for plane_index, plane in enumerate(obj["geometry"].get("planes", [])):
                if normalize(vec3(plane["normal"], name="plane.normal")) is None:
                    anomalies[object_index].append(
                        make_error(
                            E_NORMAL_ZERO,
                            f"Object {object_id!r} plane {plane_index} has a zero-length normal.",
                            object_id=object_id,
                            object_index=object_index,
                            plane_index=plane_index,
                            plane=copy.deepcopy(plane),
                            object=copy.deepcopy(obj),
                        )
                    )
        anomalies[object_index].sort(key=error_sort_key)
    return dict(anomalies), duplicate_groups


def _query_geometry(
    scene: dict[str, Any],
    query: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    if query["kind"] == "ray":
        origin = vec3(query["origin"], name="ray.origin")
        direction = normalize(vec3(query["direction"], name="ray.direction"))
        if direction is None:
            raise make_error(
                E_RAY_DEGENERATE,
                "Ray direction is a zero vector.",
                query_id=str(query["query_id"]),
                query_index=int(query["query_index"]),
                query=_query_snapshot(query),
            )
        return origin, direction
    if query["kind"] == "click":
        viewport = query.get("viewport", scene["camera"]["viewport"])
        return ray_from_screen(
            scene["camera"],
            float(query["screen"]["x"]),
            float(query["screen"]["y"]),
            int(viewport["width"]),
            int(viewport["height"]),
        )
    return None


def _object_touches_query(
    scene: dict[str, Any],
    query: dict[str, Any],
    ray_geometry: tuple[np.ndarray, np.ndarray] | None,
    obj: dict[str, Any],
) -> bool:
    if query["kind"] == "drag":
        return False
    if ray_geometry is not None:
        origin, direction = ray_geometry
        if ray_intersects_aabb(origin, direction, obj):
            return True
    if query["kind"] == "click":
        bounds = object_screen_bounds(scene["camera"], obj)
        if bounds is not None and screen_bounds_contain(
            bounds,
            float(query["screen"]["x"]),
            float(query["screen"]["y"]),
            float(query.get("tolerance_px", 0.0)),
        ):
            return True
    return False


def find_query_anomaly(
    scene: dict[str, Any],
    query: dict[str, Any],
    anomalies: dict[int, list[WorkbenchError]] | None = None,
) -> WorkbenchError | None:
    """Return the first input anomaly affecting this query in contract order."""

    if anomalies is None:
        anomalies, _ = _object_anomalies(scene)

    try:
        ray_geometry = _query_geometry(scene, query)
    except WorkbenchError as error:
        return error

    if query["kind"] == "drag":
        plane_normal = vec3(query["plane"]["normal"], name="plane.normal")
        if normalize(plane_normal) is None:
            return make_error(
                E_NORMAL_ZERO,
                "Drag plane normal has zero length.",
                query_id=str(query["query_id"]),
                query_index=int(query["query_index"]),
                query=_query_snapshot(query),
            )
        if bool(query.get("snap_enabled", False)):
            try:
                validate_snap_spacing(query.get("snap_spacing"))
            except WorkbenchError as error:
                error.trigger.setdefault("query_id", str(query["query_id"]))
                error.trigger.setdefault("query_index", int(query["query_index"]))
                error.trigger.setdefault("query", _query_snapshot(query))
                return error
        return None

    objects = scene["objects"]
    ordered_indices = sorted(
        range(len(objects)),
        key=lambda index: (str(objects[index]["id"]), int(objects[index].get("object_index", index))),
    )
    for index in ordered_indices:
        obj = objects[index]
        if not _object_touches_query(scene, query, ray_geometry, obj):
            continue
        object_errors = anomalies.get(index, [])
        if object_errors:
            source_error = object_errors[0]
            error = make_error(
                source_error.code,
                source_error.message,
                **copy.deepcopy(source_error.trigger),
            )
            error.trigger.setdefault("query_id", str(query["query_id"]))
            error.trigger.setdefault("query_index", int(query["query_index"]))
            error.trigger.setdefault("query", _query_snapshot(query))
            return error
    return None


def _error_record(
    scene: dict[str, Any],
    query: dict[str, Any],
    error: WorkbenchError,
) -> dict[str, Any]:
    record = {
        "query_id": str(query["query_id"]),
        "query_index": int(query["query_index"]),
        "query_type": str(query["kind"]),
        "status": "error",
        "error": error.to_dict(),
        "decision_method": "input_validation",
        "selection_rule": SELECTION_RULE,
    }
    return with_content_hash(record)


def _top_level_hit_fields(winner: dict[str, Any] | None) -> dict[str, Any]:
    if winner is None:
        return {
            "hit_object_id": None,
            "hit_distance": None,
            "hit_point": None,
            "intersection_method": None,
        }
    return {
        "hit_object_id": str(winner["object_id"]),
        "hit_distance": float(winner["hit_distance"]),
        "hit_point": list(winner["hit_point"]),
        "intersection_method": str(winner["intersection_method"]),
    }


def run_ray_query(scene: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    origin = vec3(query["origin"], name="ray.origin")
    direction = normalize(vec3(query["direction"], name="ray.direction"))
    if direction is None:
        raise make_error(
            E_RAY_DEGENERATE,
            "Ray direction is a zero vector.",
            query_id=str(query["query_id"]),
            query_index=int(query["query_index"]),
            query=_query_snapshot(query),
        )
    candidate_objects = [
        obj for obj in scene["objects"] if ray_intersects_aabb(origin, direction, obj)
    ]
    candidate_objects.sort(
        key=lambda item: (str(item["id"]), int(item.get("object_index", 0)))
    )
    hits = all_object_hits(origin, direction, candidate_objects)
    winner = select_winner(hits)
    record = {
        "query_id": str(query["query_id"]),
        "query_index": int(query["query_index"]),
        "query_type": "ray",
        "status": "ok",
        "input": {
            "origin": [float(value) for value in origin],
            "direction": [float(value) for value in direction],
        },
        "decision_method": "ray_object_intersection",
        "selection_rule": SELECTION_RULE,
        "candidate_hits": hits,
        "winner": winner,
        "no_hit": winner is None,
        **_top_level_hit_fields(winner),
    }
    return with_content_hash(record)


def _hits_for_objects(
    origin: np.ndarray,
    direction: np.ndarray,
    objects: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(objects, key=lambda item: (str(item["id"]), int(item.get("object_index", 0))))
    return all_object_hits(origin, direction, ordered)


def _equivalent_result(winner: dict[str, Any] | None) -> dict[str, Any] | None:
    if winner is None:
        return None
    return {
        "object_id": winner["object_id"],
        "hit_distance": winner["hit_distance"],
        "hit_point": list(winner["hit_point"]),
        "intersection_method": winner["intersection_method"],
    }


def _results_match(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        str(left["object_id"]) == str(right["object_id"])
        and abs(float(left["hit_distance"]) - float(right["hit_distance"])) <= 1.0e-9
        and all(
            abs(float(a) - float(b)) <= 1.0e-9
            for a, b in zip(left["hit_point"], right["hit_point"])
        )
    )


def run_click_query(scene: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    camera = scene["camera"]
    viewport = query.get("viewport", camera["viewport"])
    screen_x = float(query["screen"]["x"])
    screen_y = float(query["screen"]["y"])
    tolerance = float(query.get("tolerance_px", 0.0))
    origin, direction = ray_from_screen(
        camera,
        screen_x,
        screen_y,
        int(viewport["width"]),
        int(viewport["height"]),
    )

    candidate_ids = screen_candidates(camera, scene["objects"], screen_x, screen_y, tolerance)
    candidate_by_id = {str(obj["id"]): obj for obj in scene["objects"] if str(obj["id"]) in candidate_ids}
    candidate_objects = list(candidate_by_id.values())
    hits = _hits_for_objects(origin, direction, candidate_objects)
    winner = select_winner(hits)

    full_hits = _hits_for_objects(origin, direction, scene["objects"])
    equivalent_winner = select_winner(full_hits)
    matches = _results_match(winner, equivalent_winner)

    alternate_tolerance = tolerance + 8.0
    alternate_ids = screen_candidates(
        camera,
        scene["objects"],
        screen_x,
        screen_y,
        alternate_tolerance,
    )
    alternate_by_id = {
        str(obj["id"]): obj for obj in scene["objects"] if str(obj["id"]) in alternate_ids
    }
    alternate_hits = _hits_for_objects(origin, direction, alternate_by_id.values())
    alternate_winner = select_winner(alternate_hits)

    if not matches:
        raise make_error(
            E_PICK_MISMATCH,
            "Screen-space click winner differs from the equivalent ray winner.",
            query_id=str(query["query_id"]),
            query_index=int(query["query_index"]),
            screen_winner=_equivalent_result(winner),
            ray_winner=_equivalent_result(equivalent_winner),
            tolerance_px=tolerance,
        )

    record = {
        "query_id": str(query["query_id"]),
        "query_index": int(query["query_index"]),
        "query_type": "click",
        "status": "ok",
        "input": {
            "screen": {"x": screen_x, "y": screen_y},
            "viewport": {"width": int(viewport["width"]), "height": int(viewport["height"])},
            "tolerance_px": tolerance,
            "equivalent_ray": {
                "origin": [float(value) for value in origin],
                "direction": [float(value) for value in direction],
            },
        },
        "decision_method": "screen_space_tolerance_then_ray_intersection",
        "selection_rule": SELECTION_RULE,
        "candidate_object_ids": sorted(candidate_ids),
        "candidate_hits": hits,
        "winner": winner,
        "equivalent_ray_winner": equivalent_winner,
        "equivalence": {
            "matches": True,
            "comparison": "object_id,hit_distance,hit_point",
            "tolerance_px": tolerance,
        },
        "tolerance_probe": {
            "alternate_tolerance_px": alternate_tolerance,
            "alternate_candidate_object_ids": sorted(alternate_ids),
            "candidate_set_changed": sorted(candidate_ids) != sorted(alternate_ids),
            "winner_unchanged": _results_match(winner, alternate_winner),
            "alternate_winner": _equivalent_result(alternate_winner),
        },
        "no_hit": winner is None,
        **_top_level_hit_fields(winner),
    }
    return with_content_hash(record)


def run_drag_query(scene: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    operation = perform_drag(scene["camera"], query)
    replay = replay_drag_without_snap(scene["camera"], query)
    unsnapped_replay_matches = all(
        abs(float(a) - float(b)) <= 1.0e-9
        for a, b in zip(operation["unsnapped_position"], replay["final_position"])
    )
    if not operation["orientation_unchanged"]:
        raise make_error(
            E_INJECT_CONFIG,
            "Snap operation changed orientation.",
            query_id=str(query["query_id"]),
            query_index=int(query["query_index"]),
        )
    record = {
        "query_id": str(query["query_id"]),
        "query_index": int(query["query_index"]),
        "query_type": "drag",
        "status": "ok",
        "input": _query_snapshot(query),
        "decision_method": operation["decision_method"],
        "operation": operation,
        "replay_without_snap": {
            "final_position": replay["final_position"],
            "matches_unsnapped_position": unsnapped_replay_matches,
        },
        "orientation_unchanged": True,
    }
    return with_content_hash(record)


def run_batch(scene: dict[str, Any], query_bundle: dict[str, Any]) -> dict[str, Any]:
    anomalies, duplicate_groups = _object_anomalies(scene)
    queries = sorted(
        query_bundle["queries"],
        key=lambda item: (int(item["query_index"]), str(item["query_id"])),
    )
    records: list[dict[str, Any]] = []
    for query in queries:
        anomaly = find_query_anomaly(scene, query, anomalies)
        if anomaly is not None:
            records.append(_error_record(scene, query, anomaly))
            continue
        try:
            if query["kind"] == "ray":
                record = run_ray_query(scene, query)
            elif query["kind"] == "click":
                record = run_click_query(scene, query)
            elif query["kind"] == "drag":
                record = run_drag_query(scene, query)
            else:
                raise make_error(
                    E_INJECT_CONFIG,
                    f"Unsupported query kind: {query['kind']!r}.",
                    query_id=str(query["query_id"]),
                    query_index=int(query["query_index"]),
                )
            records.append(record)
        except WorkbenchError as error:
            records.append(_error_record(scene, query, error))
        except Exception as exc:  # isolate an unexpected query failure, never pass it
            wrapped = make_error(
                E_INJECT_CONFIG,
                f"Unexpected query failure: {type(exc).__name__}: {exc}",
                query_id=str(query["query_id"]),
                query_index=int(query["query_index"]),
                query=_query_snapshot(query),
            )
            records.append(_error_record(scene, query, wrapped))

    records.sort(key=lambda item: (int(item["query_index"]), str(item["query_id"])))
    error_counts = Counter(
        record["error"]["code"] for record in records if record["status"] == "error"
    )
    failed = sum(1 for record in records if record["status"] != "ok")
    first_record = next((record for record in records if record["status"] == "error"), None)
    first_affected = None
    if first_record is not None:
        first_error = first_record["error"]
        first_affected = {
            "query_id": first_record["query_id"],
            "query_index": first_record["query_index"],
            "query_type": first_record["query_type"],
            "error_code": first_error["code"],
            "message": first_error["message"],
            "trigger": first_error["trigger"],
            "traversal_order": {"queries": QUERY_ORDER, "objects": OBJECT_ORDER},
        }

    summary = {
        "query_count": len(records),
        "passed": len(records) - failed,
        "failed": failed,
        "status": "pass" if failed == 0 else "fail",
        "error_counts": dict(sorted(error_counts.items())),
        "duplicate_id_groups": {
            object_id: indices for object_id, indices in sorted(duplicate_groups.items())
        },
    }
    return {
        "schema_version": 1,
        "summary": summary,
        "first_affected": first_affected,
        "records": records,
        "query_hashes": [
            {
                "query_id": record["query_id"],
                "query_index": record["query_index"],
                "status": record["status"],
                "error_code": record.get("error", {}).get("code"),
                "content_hash": record["content_hash"],
            }
            for record in records
        ],
        "records_hash": content_hash(records),
    }


def _kind_counts(queries: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"ray": 0, "click": 0, "drag": 0}
    for query in queries:
        counts[str(query["kind"])] += 1
    return counts


def _mini_scene(scene: dict[str, Any], indices: Iterable[int]) -> dict[str, Any]:
    selected = [copy.deepcopy(scene["objects"][index]) for index in sorted(set(indices))]
    if selected:
        minima = np.asarray(
            [obj["bounding_volume"]["aabb"]["min"] for obj in selected],
            dtype=np.float64,
        )
        maxima = np.asarray(
            [obj["bounding_volume"]["aabb"]["max"] for obj in selected],
            dtype=np.float64,
        )
        bounds = {
            "min": [float(value) for value in np.min(minima, axis=0)],
            "max": [float(value) for value in np.max(maxima, axis=0)],
        }
    else:
        bounds = {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    return {
        "schema_version": int(scene.get("schema_version", 1)),
        "seed": int(scene.get("seed", 0)),
        "priority_range": copy.deepcopy(scene.get("priority_range", {"min": 0, "max": 99})),
        "camera": copy.deepcopy(scene["camera"]),
        "objects": selected,
        "bounds": bounds,
        "object_count": len(selected),
    }


def _mini_queries(queries: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [copy.deepcopy(query) for query in queries]
    return {
        "schema_version": 1,
        "seed": 0,
        "query_count": len(selected),
        "counts": _kind_counts(selected),
        "queries": selected,
    }


def _indices_for_error(scene: dict[str, Any], error: dict[str, Any]) -> list[int]:
    trigger = error["trigger"]
    code = error["error_code"]
    if code == E_OBJECT_DUP_ID:
        return [int(index) for index in trigger.get("object_indices", [])]
    if "object_index" in trigger:
        return [int(trigger["object_index"])]
    return []


def _reproduction(scene: dict[str, Any], query_bundle: dict[str, Any]) -> dict[str, Any]:
    result = run_batch(scene, query_bundle)
    first = result["first_affected"]
    return {
        "reproduced": first is not None,
        "error_code": None if first is None else first["error_code"],
        "query_id": None if first is None else first["query_id"],
        "query_index": None if first is None else first["query_index"],
        "message": None if first is None else first["message"],
    }


def build_minimal_counterexample(
    scene: dict[str, Any],
    query_bundle: dict[str, Any],
    first_affected: dict[str, Any],
    *,
    source_name: str,
    injection: dict[str, Any],
) -> dict[str, Any]:
    code = str(first_affected["error_code"])
    query_id = str(first_affected["query_id"])
    query = next(item for item in query_bundle["queries"] if item["query_id"] == query_id)
    object_indices = _indices_for_error(scene, first_affected)
    mini_scene = _mini_scene(scene, object_indices)
    mini_queries = _mini_queries([query])
    first_reproduction = _reproduction(mini_scene, mini_queries)
    expected_reproduced = first_reproduction["reproduced"] and first_reproduction["error_code"] == code

    removal_checks: list[dict[str, Any]] = []
    for position, obj in enumerate(mini_scene["objects"]):
        reduced_indices = [
            index
            for index in range(len(mini_scene["objects"]))
            if index != position
        ]
        reduced_scene = _mini_scene_from_objects(mini_scene, reduced_indices)
        check = _reproduction(reduced_scene, mini_queries)
        removal_checks.append(
            {
                "removed_element": f"object:{obj['id']}[{position}]",
                "reproduced": bool(check["reproduced"] and check["error_code"] == code),
                "observed_error_code": check["error_code"],
            }
        )
    if mini_queries["queries"]:
        empty_queries = _mini_queries([])
        check = _reproduction(mini_scene, empty_queries)
        removal_checks.append(
            {
                "removed_element": f"query:{query_id}",
                "reproduced": bool(check["reproduced"] and check["error_code"] == code),
                "observed_error_code": check["error_code"],
            }
        )

    minimal = expected_reproduced and all(not check["reproduced"] for check in removal_checks)
    witness = {
        "schema_version": 1,
        "source_case": source_name,
        "source_hashes": {
            "injected_scene": content_hash(scene),
            "queries": content_hash(query_bundle),
            "injection": content_hash(injection),
        },
        "expected_error_code": code,
        "effective_injected_scene": mini_scene,
        "effective_injected_queries": mini_queries,
        "reproduction": first_reproduction,
        "removal_checks": removal_checks,
        "minimal": minimal,
    }
    witness["content_hash"] = content_hash(witness)
    return witness


def _mini_scene_from_objects(scene: dict[str, Any], indices: Iterable[int]) -> dict[str, Any]:
    selected = [copy.deepcopy(scene["objects"][index]) for index in indices]
    if selected:
        minima = np.asarray(
            [obj["bounding_volume"]["aabb"]["min"] for obj in selected],
            dtype=np.float64,
        )
        maxima = np.asarray(
            [obj["bounding_volume"]["aabb"]["max"] for obj in selected],
            dtype=np.float64,
        )
        bounds = {
            "min": [float(value) for value in np.min(minima, axis=0)],
            "max": [float(value) for value in np.max(maxima, axis=0)],
        }
    else:
        bounds = {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    result = copy.deepcopy(scene)
    result["objects"] = selected
    result["object_count"] = len(selected)
    result["bounds"] = bounds
    return result


def reproduce_witness(witness: dict[str, Any]) -> dict[str, Any]:
    result = run_batch(
        witness["effective_injected_scene"],
        witness["effective_injected_queries"],
    )
    first = result["first_affected"]
    reproduced = first is not None and first["error_code"] == witness["expected_error_code"]
    return {
        "reproduced": reproduced,
        "error_code": None if first is None else first["error_code"],
        "query_id": None if first is None else first["query_id"],
        "report": result,
    }


def execute_case(
    name: str,
    base_scene: dict[str, Any],
    base_queries: dict[str, Any],
    injection: dict[str, Any],
    expected_error_codes: Iterable[str],
) -> dict[str, Any]:
    expected = sorted(set(expected_error_codes))
    try:
        scene, queries, operations = apply_injection(base_scene, base_queries, injection)
    except WorkbenchError as error:
        status = "pass" if error.code in expected else "fail"
        report = {
            "schema_version": 1,
            "case": name,
            "status": status,
            "execution_status": "injection_rejected",
            "expected_error_codes": expected,
            "detected_error_codes": [error.code],
            "expected_met": [error.code] if error.code in expected else [],
            "injection": copy.deepcopy(injection),
            "injection_operations": [],
            "scene_hash": content_hash(base_scene),
            "queries_hash": content_hash(base_queries),
            "injection_hash": content_hash(injection),
            "summary": {
                "query_count": len(base_queries["queries"]),
                "passed": 0,
                "failed": 0,
                "status": "injection_rejected",
                "error_counts": {error.code: 1},
            },
            "first_affected": None,
            "witness": None,
            "query_hashes": [],
            "records": [],
            "error": error.to_dict(),
        }
        report["content_hash"] = content_hash(report)
        return report

    batch = run_batch(scene, queries)
    detected_codes = sorted(batch["summary"]["error_counts"])
    expected_met = sorted(
        code
        for code in detected_codes
        if code in expected or (not expected and batch["summary"]["failed"] == 0)
    )
    if expected:
        status = "pass" if set(expected).issubset(detected_codes) else "fail"
    else:
        status = "pass" if batch["summary"]["failed"] == 0 else "fail"

    witness = None
    if batch["first_affected"] is not None:
        witness = build_minimal_counterexample(
            scene,
            queries,
            batch["first_affected"],
            source_name=name,
            injection=injection,
        )

    report = {
        "schema_version": 1,
        "case": name,
        "status": status,
        "expected_error_codes": expected,
        "detected_error_codes": detected_codes,
        "expected_met": expected_met,
        "injection": copy.deepcopy(injection),
        "injection_operations": operations,
        "scene_hash": content_hash(scene),
        "base_scene_hash": content_hash(base_scene),
        "queries_hash": content_hash(queries),
        "base_queries_hash": content_hash(base_queries),
        "injection_hash": content_hash(injection),
        "summary": batch["summary"],
        "first_affected": batch["first_affected"],
        "witness": witness,
        "query_hashes": batch["query_hashes"],
        "records": batch["records"],
        "records_hash": batch["records_hash"],
    }
    report["content_hash"] = content_hash(report)
    return report
