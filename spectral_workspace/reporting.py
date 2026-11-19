"""Deterministic, recomputable reports for demos and verification."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .canonical import canonical_bytes, qfloat, sha256_obj
from .injection import injection_hash, validate_injection_config
from .workspace import WorkspaceEngine


REPORT_SCHEMA = "spectral-workspace-report/v1"


def stage_report(engine: WorkspaceEngine) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage_id in sorted(engine.stages):
        stage = engine.stages[stage_id]
        rows.append(
            {
                "stage_id": stage_id,
                "kind": stage["kind"],
                "material_id": stage["material_id"],
                "light_id": stage["light_id"],
                "source_version": stage["computed_revision"],
                "dependency_signature": stage["dependency_signature"],
                "dependency_hash": stage["dependency_hash"],
                "content_hash": stage["content_hash"],
                "product": stage["product"],
            }
        )
    return rows


def color_report(engine: WorkspaceEngine) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage_id in sorted(engine.stages):
        stage = engine.stages[stage_id]
        if stage["kind"] != "material_color":
            continue
        product = stage["product"]
        color = product["color"]
        rows.append(
            {
                "material_id": stage["material_id"],
                "aggregate_xyz": product["aggregate_xyz"],
                "surface": product["surface"],
                "target_color_space": color["space"],
                "linear_rgb": color["linear_rgb"],
                "rgb": color["rgb"],
                "rgb_unclipped": color["rgb_unclipped"],
                "in_gamut": color["in_gamut"],
                "transfer": color["transfer"],
                "light_contributions": product["light_contributions"],
                "stage_content_hash": stage["content_hash"],
            }
        )
    return rows


def operation_trace(engine: WorkspaceEngine) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for delta in sorted(engine.history, key=lambda item: int(item["revision"])):
        rows.append(
            {
                "revision": delta["revision"],
                "operation": delta["operation"],
                "operation_hash": delta["operation_hash"],
                "dirty_closure": delta["dirty_closure"],
                "recomputed_count": delta["recomputed_count"],
                "reused_count": delta["reused_count"],
                "removed_stage_ids": delta["removed_stage_ids"],
                "recomputed": delta["recomputed"],
                "reused": delta["reused"],
            }
        )
    return rows


def run_configuration(engine: WorkspaceEngine) -> dict[str, Any]:
    if engine.scene is None:
        return {"observer": None, "target_color_space": None}
    config = engine.scene.config
    return {
        "observer": config.observer,
        "camera_white_point_xyz": list(config.camera_white_point_xyz),
        "adaptation_method": config.adaptation_method,
        "target_color_space": config.target_color_space,
    }


def build_report(
    engine: WorkspaceEngine,
    *,
    verification: Mapping[str, Any] | None = None,
    injection: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cleaned_injection = validate_injection_config(injection)
    if verification is None:
        verification = engine.verify() if engine.scene is not None else {
            "status": "empty",
            "stage_count": 0,
            "max_abs_diff": 0.0,
        }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "final_scene_sha256": sha256_obj(engine.state.get("scene"))
        if engine.state.get("scene") is not None
        else None,
        "workspace": {
            "revision": engine.revision,
            "snapshot_version": engine.snapshot_version,
            "state_sha256": engine.state_hash,
            "pipeline_version": engine.state.get("pipeline_version"),
        },
        "run_configuration": run_configuration(engine),
        "operations": engine.operations,
        "operation_trace": operation_trace(engine),
        "stages": stage_report(engine),
        "colors": color_report(engine),
        "verification": dict(verification),
        "injection": {
            "config": cleaned_injection if injection is not None else None,
            "hash": injection_hash(injection),
        },
        "recovery": {
            **engine.recovery.to_dict(),
            "forced_termination": engine.recovery.prior_exit_state
            == "forced_termination_recovered",
            "exit_report_fabricated": False,
        },
    }
    if extra:
        report["extra"] = dict(extra)
    return report


def _numeric_difference(
    left: Any, right: Any, path: str = ""
) -> tuple[float, list[dict[str, Any]]]:
    differences: list[dict[str, Any]] = []
    maximum = 0.0
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(
        right, (int, float)
    ) and not isinstance(right, bool):
        difference = abs(float(left) - float(right))
        maximum = difference
        if difference > 0.0:
            differences.append(
                {"field": path, "left": left, "right": right, "abs_diff": qfloat(difference)}
            )
        return maximum, differences
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append({"field": f"{path}/{key}", "presence_mismatch": True})
                maximum = math.inf
                continue
            child_max, child_diff = _numeric_difference(
                left[key], right[key], f"{path}/{key}"
            )
            maximum = max(maximum, child_max)
            differences.extend(child_diff)
        return maximum, differences
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)) and isinstance(
        right, Sequence
    ) and not isinstance(right, (str, bytes)):
        if len(left) != len(right):
            differences.append(
                {
                    "field": path,
                    "length_left": len(left),
                    "length_right": len(right),
                }
            )
            return math.inf, differences
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            child_max, child_diff = _numeric_difference(
                left_item, right_item, f"{path}/{index}"
            )
            maximum = max(maximum, child_max)
            differences.extend(child_diff)
        return maximum, differences
    if left != right:
        differences.append({"field": path, "left": left, "right": right})
        maximum = math.inf
    return maximum, differences


def compare_recovery_reports(
    baseline: Mapping[str, Any], recovered: Mapping[str, Any]
) -> dict[str, Any]:
    color_max, color_differences = _numeric_difference(
        baseline.get("colors"), recovered.get("colors"), "/colors"
    )
    stage_max, stage_differences = _numeric_difference(
        baseline.get("stages"), recovered.get("stages"), "/stages"
    )
    colors_equal = canonical_bytes(baseline.get("colors")) == canonical_bytes(
        recovered.get("colors")
    )
    stages_equal = canonical_bytes(baseline.get("stages")) == canonical_bytes(
        recovered.get("stages")
    )
    return {
        "status": "ok" if colors_equal and stages_equal else "mismatch",
        "colors_byte_equal": colors_equal,
        "stages_byte_equal": stages_equal,
        "color_max_abs_diff": qfloat(color_max),
        "stage_max_abs_diff": qfloat(stage_max),
        "first_color_difference": color_differences[:1],
        "first_stage_difference": stage_differences[:1],
        "tolerance": {"atol": 1e-12, "rtol": 0.0},
    }
