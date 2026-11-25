from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import STATE_SCHEMA_VERSION, __version__
from .config import config_hash, parameter_fingerprint, quant_scheme_name
from .errors import ErrorCode, make_issue
from .exchange import logical_key
from .hashing import fingerprint
from .imaging import load_and_decode_png, scan_pngs
from .recognition import (
    GRADE_CODE_TO_NAME,
    build_grid,
    compute_cell_stats,
    determine_grade,
    intersection_over_union,
    regress_bbox,
    synthesize_intensity,
)
from .scheduler import run_concurrent, stable_task_key


STAGE_ORDER = {"grid": 1, "stats": 2, "regression": 3, "intensity": 4, "grade": 5, "conversion": 6, "array": 7, "image": 8}


def _cache_lookup(cache: dict[str, Any], key: str) -> dict[str, Any] | None:
    entry = cache.get(key)
    return entry.get("payload") if isinstance(entry, dict) else None


def _event(
    image: str,
    component_id: int | None,
    stage: str,
    action: str,
    key: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "image": image,
        "component_id": component_id,
        "stage": stage,
        "stage_order": STAGE_ORDER[stage],
        "action": action,
        "cache_key": key,
        "reason": reason,
    }


def _judgment_from_component(component: dict[str, Any], target_version: str) -> dict[str, Any]:
    return {
        "source_version": target_version,
        "image": component["image"],
        "component_id": component["component_id"],
        "bbox": component["bbox"],
        "bbox_decimal": [str(v) for v in component["bbox"]],
        "intensity_permille": component["intensity_permille"],
        "intensity_decimal": str(component["intensity"]["intensity_decimal"]),
        "grade_code": component["grade_code"],
        "grade_name": component["grade_name"],
        "quantization_scheme": component["intensity"]["quant_scheme"],
        "missing_fields": [],
        "raw_fields": {},
        "precision_labels": ["PIPELINE_FIXED_POINT_RESULT"],
        "sidecar_key": logical_key(component["image"], component["component_id"]),
    }


def process_component(item: dict[str, Any], cache: dict[str, Any], target_version: str) -> dict[str, Any]:
    image_data = item["image_data"]
    cell = item["cell"]
    config = item["config"]
    image = cell["image"]
    component_id = cell["component_id"]
    writes: list[tuple[str, str, dict[str, Any]]] = []
    events: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = list(item.get("grid_issues", []))

    cell_key = item["cell_key"]
    events.append(
        _event(
            image,
            component_id,
            "grid",
            "REUSED" if item["grid_reused"] else "RECOMPUTED",
            cell_key,
            "CONTENT_KEY_HIT" if item["grid_reused"] else "GRID_PARAMETER_OR_IMAGE_FINGERPRINT_MISS",
        )
    )

    stats_key = item["stats_key"]
    stats = _cache_lookup(cache, stats_key)
    if stats is not None:
        events.append(_event(image, component_id, "stats", "REUSED", stats_key, "STATS_CONTENT_KEY_HIT"))
    else:
        stats, stats_issues = compute_cell_stats(image_data["heat"], cell, config)
        issues.extend(stats_issues)
        writes.append((stats_key, "stats", stats))
        events.append(_event(image, component_id, "stats", "RECOMPUTED", stats_key, "PERCENTILE_OR_CELL_CONTENT_KEY_MISS"))

    regression_param = parameter_fingerprint(config, ["bbox_padding_px"], "BBOX_RULE_v1")
    regression_key = fingerprint(["regression", stats_key, regression_param], "reg:")
    regression = _cache_lookup(cache, regression_key)
    if regression is not None:
        events.append(_event(image, component_id, "regression", "REUSED", regression_key, "REGRESSION_CONTENT_KEY_HIT_NO_RERUN"))
    else:
        regression = regress_bbox(image_data["heat"], cell, stats, config)
        writes.append((regression_key, "regression", regression))
        events.append(_event(image, component_id, "regression", "RECOMPUTED", regression_key, "BBOX_DEPENDENCY_KEY_MISS"))

    intensity_param = parameter_fingerprint(
        config,
        ["fixed_decimals", "quant_levels", "quant_scheme"],
        "INTENSITY_RULE_v1",
    )
    intensity_key = fingerprint(["intensity", regression_key, intensity_param], "int:")
    intensity = _cache_lookup(cache, intensity_key)
    if intensity is not None:
        events.append(_event(image, component_id, "intensity", "REUSED", intensity_key, "INTENSITY_CONTENT_KEY_HIT"))
    else:
        intensity = synthesize_intensity(stats, regression, config)
        writes.append((intensity_key, "intensity", intensity))
        events.append(_event(image, component_id, "intensity", "RECOMPUTED", intensity_key, "FIXED_POINT_OR_QUANT_KEY_MISS"))

    grade_param = parameter_fingerprint(
        config,
        ["grade_thresholds_permille"],
        config["grade_table_version"],
        "EQUAL_THRESHOLD_HIGHER_GRADE_v1",
    )
    grade_key = fingerprint(["grade", intensity_key, grade_param], "grade:")
    grade = _cache_lookup(cache, grade_key)
    degraded = bool(stats.get("degraded"))
    if grade is not None:
        events.append(_event(image, component_id, "grade", "REUSED", grade_key, "GRADE_CONTENT_KEY_HIT"))
    else:
        grade = determine_grade(intensity, config, degraded)
        writes.append((grade_key, "grade", grade))
        events.append(_event(image, component_id, "grade", "RECOMPUTED", grade_key, "THRESHOLD_OR_INTENSITY_KEY_MISS"))

    component = {
        "image": image,
        "component_id": component_id,
        "cell_ref": cell["cell_ref"],
        "row": cell["row"],
        "col": cell["col"],
        "cell_interval": [cell["x0"], cell["y0"], cell["x1"], cell["y1"]],
        "interval_attribution_basis": cell["attribution_basis"],
        "bbox": regression["bbox"],
        "stats": stats,
        "regression": regression,
        "intensity": intensity,
        "grade": grade,
        "fill_permille": regression["fill_permille"],
        "raw_intensity_permille": intensity["raw_intensity_permille"],
        "intensity_permille": intensity["intensity_permille"],
        "intensity_fixed": intensity["intensity_fixed"],
        "quant_integer": intensity["quant_integer"],
        "quant_step_fixed": intensity["quant_step_fixed"],
        "residual_fixed": intensity["residual_fixed"],
        "grade_code": grade["grade_code"],
        "grade_name": grade["grade_name"],
        "threshold_boundary": grade["threshold_boundary"],
        "boundary_threshold_permille": grade["boundary_threshold_permille"],
        "keys": {
            "cell": cell_key,
            "stats": stats_key,
            "regression": regression_key,
            "intensity": intensity_key,
            "grade": grade_key,
        },
        "refs": {
            "cell_interval": [cell["x0"], cell["y0"], cell["x1"], cell["y1"]],
            "stats_key": stats_key,
            "stats_values": {
                "low_percentile_heat": stats["low_percentile_heat"],
                "high_percentile_heat": stats["high_percentile_heat"],
                "hot_area_permille": stats["hot_area_permille"],
                "coverage_permille": stats["coverage_permille"],
                "pixel_count": stats["pixel_count"],
            },
        },
        "issues": issues,
    }

    logical_judgment = _judgment_from_component(component, target_version)
    conversion_key = fingerprint(
        [
            "conversion",
            target_version,
            component["keys"]["grade"],
            intensity["quant_scheme"],
            "EXCHANGE_CONVERSION_v1",
        ],
        "conv:",
    )
    converted = _cache_lookup(cache, conversion_key)
    if converted is not None:
        events.append(_event(image, component_id, "conversion", "REUSED", conversion_key, "EXCHANGE_CONVERSION_KEY_HIT"))
        component["exchange"] = converted
    else:
        converted = {"target_version": target_version, "judgment": logical_judgment}
        writes.append((conversion_key, "conversion", converted))
        component["exchange"] = converted
        events.append(_event(image, component_id, "conversion", "RECOMPUTED", conversion_key, "EXCHANGE_VERSION_OR_RESULT_KEY_MISS"))
    component["keys"]["conversion"] = conversion_key
    return {"component": component, "writes": writes, "events": events, "issues": issues}


def attribute_truth(component: dict[str, Any], truth_by_image: dict[str, dict[str, Any]]) -> dict[str, Any]:
    truth_array = truth_by_image.get(component["image"])
    if not truth_array:
        return {"status": "NO_TRUTH_FOR_IMAGE"}
    match = None
    for item in truth_array["components"]:
        if item.get("component_id") == component["component_id"]:
            match = item
            break
    if match is None:
        return {"status": "NO_TRUTH_COMPONENT", "truth_array_grid": [truth_array["grid_rows"], truth_array["grid_cols"]]}
    truth_bbox = match["bbox"]
    iou = intersection_over_union(component["bbox"], truth_bbox)
    expected_code = match.get("grade_code")
    if expected_code is None and isinstance(match.get("expected_grade"), str):
        expected_code = {name: code for code, name in GRADE_CODE_TO_NAME.items()}.get(match["expected_grade"])
    grade_match = expected_code == component["grade_code"] if expected_code is not None else None
    if component["grade_code"] == 9:
        status = "DEGRADED_CELL"
    elif grade_match is False:
        status = "GRADE_MISMATCH"
    elif iou < 0.50:
        status = "BBOX_DEVIATION"
    else:
        status = "OK"
    return {
        "status": status,
        "truth_bbox": truth_bbox,
        "iou": round(iou, 6),
        "expected_grade_code": expected_code,
        "grade_match": grade_match,
        "error_attribution_only_source": "array_truth.json",
    }


def _array_conclusion(
    image: str,
    components: list[dict[str, Any]],
    cache: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    refs = [logical_key(image, item["component_id"]) for item in components]
    grade_codes = [item["grade_code"] for item in components]
    grade_counts = {name: 0 for name in GRADE_CODE_TO_NAME.values()}
    for code in grade_codes:
        grade_counts[GRADE_CODE_TO_NAME.get(code, "UNKNOWN")] = grade_counts.get(GRADE_CODE_TO_NAME.get(code, "UNKNOWN"), 0) + 1
    max_code = max(grade_codes, default=None)
    payload = {
        "array_id": image.rsplit(".", 1)[0],
        "image": image,
        "status": "complete",
        "component_refs": refs,
        "component_ids": [item["component_id"] for item in components],
        "component_count": len(components),
        "grade_counts": grade_counts,
        "max_grade_code": max_code,
        "max_grade_name": GRADE_CODE_TO_NAME.get(max_code) if max_code is not None else None,
        "max_intensity_permille": max((item["intensity_permille"] for item in components), default=None),
        "boundary_refs": [
            logical_key(image, item["component_id"]) for item in components if item["threshold_boundary"]
        ],
        "critical_refs": [
            logical_key(image, item["component_id"]) for item in components if item["grade_code"] >= 4
        ],
    }
    array_key = fingerprint(
        ["array_conclusion", sorted(item["keys"]["grade"] for item in components), "ARRAY_RULE_v1"],
        "arr:",
    )
    cached = _cache_lookup(cache, array_key)
    if cached is not None:
        events.append(_event(image, None, "array", "REUSED", array_key, "ARRAY_COMPONENT_SET_KEY_HIT"))
        cached["array_key"] = array_key
        return cached
    payload["array_key"] = array_key
    cache[array_key] = {"stage": "array", "payload": deepcopy(payload)}
    events.append(_event(image, None, "array", "RECOMPUTED", array_key, "COMPONENT_SET_OR_CONCLUSION_KEY_MISS"))
    return payload


def run_pipeline(
    input_dir: str,
    config: dict[str, Any],
    truth_by_image: dict[str, dict[str, Any]] | None = None,
    truth_issues: list[dict[str, Any]] | None = None,
    config_issues: list[dict[str, Any]] | None = None,
    previous_state: dict[str, Any] | None = None,
    budget: int | None = None,
    exchange_version: str = "v1",
    worker_runner=run_concurrent,
) -> tuple[dict[str, Any], dict[str, Any]]:
    had_previous = previous_state is not None
    previous_state = previous_state or {}
    current_config_hash = config_hash(config)
    previous_manifest = previous_state.get("manifest", {}) if had_previous else {}
    schema_ok = previous_manifest.get("state_schema_version") == STATE_SCHEMA_VERSION
    previous_config_hash = previous_manifest.get("config_hash")
    config_hash_matches = previous_config_hash == current_config_hash
    if not had_previous:
        recovery = {"mode": "BLANK_FULL_RUN", "reason": "NO_COMMITTED_SNAPSHOT"}
        cache: dict[str, Any] = {}
    elif not schema_ok:
        recovery = {
            "mode": "SCHEMA_RESET",
            "reason": "STATE_SCHEMA_VERSION_MISMATCH",
            "expected_schema_version": STATE_SCHEMA_VERSION,
            "found_schema_version": previous_manifest.get("state_schema_version"),
        }
        cache = {}
    elif config_hash_matches:
        recovery = {
            "mode": "RESUME_SAME_CONFIG",
            "reason": "SCHEMA_AND_CONFIG_HASH_MATCH",
            "global_version": previous_manifest.get("global_version"),
            "config_hash": current_config_hash,
        }
        cache = deepcopy(previous_state.get("cache_entries", {}))
    else:
        recovery = {
            "mode": "CONTENT_KEY_REVALIDATE",
            "reason": "CONFIG_HASH_CHANGED",
            "global_version": previous_manifest.get("global_version"),
            "previous_config_hash": previous_config_hash,
            "config_hash": current_config_hash,
        }
        cache = deepcopy(previous_state.get("cache_entries", {}))
    prior_deductions = list(previous_state.get("budget", {}).get("deductions", [])) if schema_ok else []
    failures: list[dict[str, Any]] = list(config_issues or [])
    failures.extend(truth_issues or [])
    events: list[dict[str, Any]] = []

    png_paths = scan_pngs(input_dir)
    image_fingerprints: dict[str, str] = {}
    image_data_by_name: dict[str, dict[str, Any]] = {}
    failed_image_names: set[str] = set()
    cells_by_image: dict[str, list[dict[str, Any]]] = {}
    tasks: list[dict[str, Any]] = []

    grid_param_fp = parameter_fingerprint(
        config,
        ["fixed_width", "fixed_height", "grid_rows", "grid_cols"],
        "GRID_RULE_v1",
    )
    stats_param_fp = parameter_fingerprint(
        config,
        [
            "percentile_low_permille",
            "percentile_high_permille",
            "min_coverage_permille",
            "min_hotspot_area_permille",
        ],
        "STATS_RULE_v1",
    )

    for path in png_paths:
        image_data, image_issues = load_and_decode_png(path, config)
        failures.extend(image_issues)
        if image_data is None:
            failed_image_names.add(path.name)
            continue
        image = path.name
        image_data_by_name[image] = image_data
        image_fingerprints[image] = image_data["image_sha256"]
        cells = build_grid(image_data, config)
        cells_by_image[image] = cells
        for cell in cells:
            cell_key = fingerprint(
                [
                    "cell_grid",
                    image_data["image_sha256"],
                    grid_param_fp,
                    cell["row"],
                    cell["col"],
                    [cell["x0"], cell["y0"], cell["x1"], cell["y1"]],
                ],
                "cell:",
            )
            cell["cell_key"] = cell_key
            cached_cell = _cache_lookup(cache, cell_key)
            grid_reused = cached_cell is not None
            if not grid_reused:
                cache[cell_key] = {"stage": "grid", "payload": {"cell": deepcopy(cell)}}
            cost = int(cell["pixel_count"]) * int(config["heat_bin_count"])
            stats_key = fingerprint(["stats", cell_key, stats_param_fp], "stats:")
            tasks.append(
                {
                    "scheduler_key": stable_task_key(image, cell["component_id"], "heat_sampling:v1"),
                    "image": image,
                    "component_id": cell["component_id"],
                    "stage_name": "heat_sampling:v1",
                    "cost": cost,
                    "cell": cell,
                    "cell_key": cell_key,
                    "stats_key": stats_key,
                    "grid_reused": grid_reused,
                    "image_data": image_data,
                    "config": config,
                }
            )

    needed_sampling = [task for task in tasks if _cache_lookup(cache, task["stats_key"]) is None]
    budget_value = int(config["sample_budget"] if budget is None else budget)
    ordered_needed = sorted(needed_sampling, key=lambda task: task["scheduler_key"])
    remaining = budget_value
    scheduled_tasks: list[dict[str, Any]] = []
    pending_tasks: list[dict[str, Any]] = []
    new_deductions: list[dict[str, Any]] = []
    stopped = False
    for local_index, task in enumerate(ordered_needed):
        cost = int(task["cost"])
        if stopped or cost > remaining:
            stopped = True
            pending_tasks.append(task)
            continue
        remaining -= cost
        scheduled_tasks.append(task)
        new_deductions.append(
            {
                "sequence_index": len(prior_deductions) + local_index,
                "scheduler_key": task["scheduler_key"],
                "cost": cost,
            }
        )

    scheduled_keys = {task["scheduler_key"] for task in scheduled_tasks}
    processable_tasks = [
        task
        for task in tasks
        if _cache_lookup(cache, task["stats_key"]) is not None or task["scheduler_key"] in scheduled_keys
    ]
    results, concurrency_summary = worker_runner(
        processable_tasks,
        lambda task: process_component(task, cache, exchange_version),
        int(config["max_concurrency"]),
    )

    components: list[dict[str, Any]] = []
    for result in results:
        for key, stage, payload in result["writes"]:
            cache[key] = {"stage": stage, "payload": deepcopy(payload)}
        events.extend(result["events"])
        failures.extend(result["issues"])
        component = result["component"]
        component["truth_attribution"] = attribute_truth(component, truth_by_image or {})
        components.append(component)

    components.sort(key=lambda item: (item["image"], item["component_id"]))
    components_by_image: dict[str, list[dict[str, Any]]] = {}
    for component in components:
        components_by_image.setdefault(component["image"], []).append(component)

    pending_queue = [
        {
            "scheduler_key": task["scheduler_key"],
            "image": task["image"],
            "component_id": task["component_id"],
            "stage_name": task["stage_name"],
            "cost": task["cost"],
            "cell_ref": task["cell"]["cell_ref"],
        }
        for task in pending_tasks
    ]
    if pending_queue:
        failures.append(
            make_issue(
                ErrorCode.SAMPLE_BUDGET_PENDING,
                "sample budget exhausted; work retained in pending queue",
                pending_count=len(pending_queue),
                next_scheduler_key=pending_queue[0]["scheduler_key"],
                next_cost=pending_queue[0]["cost"],
            )
        )

    array_conclusions: list[dict[str, Any]] = []
    image_conclusions: list[dict[str, Any]] = []
    image_outputs: list[dict[str, Any]] = []
    pending_keys = {item["scheduler_key"] for item in pending_queue}

    for path in png_paths:
        image = path.name
        if image in failed_image_names:
            conclusion = {
                "image": image,
                "status": "failed",
                "array_id": image.rsplit(".", 1)[0],
                "component_refs": [],
                "image_key": fingerprint(["image_failed", image, "IMAGE_CONCLUSION_v1"], "img:"),
            }
            image_conclusions.append(conclusion)
            image_outputs.append({"image": image, "status": "failed"})
            continue

        image_components = components_by_image.get(image, [])
        image_tasks = [task for task in tasks if task["image"] == image]
        image_pending = [task for task in image_tasks if task["scheduler_key"] in pending_keys]
        complete = not image_pending and len(image_components) == len(image_tasks)
        if complete:
            array = _array_conclusion(image, image_components, cache, events)
            array_conclusions.append(array)
            image_key = fingerprint(["image_conclusion", array["array_key"], image_fingerprints[image], "IMAGE_v1"], "img:")
            cached_image = _cache_lookup(cache, image_key)
            if cached_image is not None:
                events.append(_event(image, None, "image", "REUSED", image_key, "IMAGE_ARRAY_KEY_HIT"))
                image_conclusion = cached_image
                image_conclusion["image_key"] = image_key
            else:
                image_conclusion = {
                    "image": image,
                    "array_id": array["array_id"],
                    "status": "complete",
                    "component_refs": array["component_refs"],
                    "max_grade_code": array["max_grade_code"],
                    "max_grade_name": array["max_grade_name"],
                    "max_intensity_permille": array["max_intensity_permille"],
                    "failure_count": len([issue for issue in failures if issue.get("image") == image]),
                    "image_key": image_key,
                }
                cache[image_key] = {"stage": "image", "payload": deepcopy(image_conclusion)}
                events.append(_event(image, None, "image", "RECOMPUTED", image_key, "IMAGE_CONCLUSION_KEY_MISS"))
            image_conclusions.append(image_conclusion)
            image_outputs.append(
                {
                    "image": image,
                    "status": "complete",
                    "image_sha256": image_fingerprints[image],
                    "array_conclusion": array,
                    "image_conclusion": image_conclusion,
                }
            )
        else:
            incomplete = {
                "image": image,
                "array_id": image.rsplit(".", 1)[0],
                "status": "sample_pending",
                "completed_components": len(image_components),
                "total_components": len(image_tasks),
                "pending_refs": [task["scheduler_key"] for task in image_pending],
            }
            image_conclusions.append(incomplete)
            image_outputs.append({"image": image, "status": "sample_pending", "conclusion": incomplete})

    events.sort(key=lambda event: (event["image"], event["component_id"] is None, event["component_id"] or 0, event["stage_order"], event["stage"]))
    for index, event in enumerate(events):
        event["event_order"] = index

    boundary_items = [
        {
            "image": item["image"],
            "component_id": item["component_id"],
            "key": logical_key(item["image"], item["component_id"]),
            "intensity_permille": item["intensity_permille"],
            "grade_code": item["grade_code"],
            "grade_name": item["grade_name"],
            "boundary_threshold_permille": item["boundary_threshold_permille"],
        }
        for item in components
        if item["threshold_boundary"]
    ]

    completed_images = [
        image
        for image in image_fingerprints
        if not any(task["image"] == image and task["scheduler_key"] in pending_keys for task in tasks)
        and len(components_by_image.get(image, [])) == len(cells_by_image.get(image, []))
    ]

    ledger = {
        "schema": "pv-ir-ledger-v1",
        "application_version": __version__,
        "config_hash": config_hash(config),
        "quantization_scheme_name": quant_scheme_name(config),
        "exchange_version": exchange_version,
        "recovery": recovery,
        "image_fingerprints": dict(sorted(image_fingerprints.items())),
        "images": image_outputs,
        "cells": [
            {
                "image": item["image"],
                "component_id": item["component_id"],
                "cell_ref": item["cell_ref"],
                "row": item["row"],
                "col": item["col"],
                "interval": item["cell_interval"],
                "interval_attribution_basis": item["interval_attribution_basis"],
                "stats": item["stats"],
                "keys": item["keys"],
            }
            for item in components
        ],
        "components": components,
        "array_conclusions": array_conclusions,
        "image_conclusions": image_conclusions,
        "boundary_grade_items": boundary_items,
        "failures": failures,
        "pending": pending_queue,
        "ledger_events": events,
        "scheduler": {
            "batch_budget": budget_value,
            "new_deductions": new_deductions,
            "cumulative_deduction_count": len(prior_deductions) + len(new_deductions),
            "concurrency": concurrency_summary,
            "pending_count": len(pending_queue),
        },
        "truth_summary": {
            "valid_images": sorted(truth_by_image or {}),
            "issue_count": len(truth_issues or []),
        },
    }

    manifest = {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "global_version": int(previous_manifest.get("global_version", 0)) + 1,
        "config_hash": current_config_hash,
        "colormap_version": config.get("colormap_version"),
        "grade_table_version": config.get("grade_table_version"),
        "completed_images": sorted(completed_images),
        "pending_queue": pending_queue,
    }
    state = {
        "manifest": manifest,
        "cache_entries": cache,
        "budget": {"deductions": prior_deductions + new_deductions},
        "last_ledger_summary": {
            "component_count": len(components),
            "failure_count": len(failures),
            "config_hash": current_config_hash,
        },
    }
    return ledger, state
