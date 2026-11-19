from __future__ import annotations

import copy
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import canonical_json_bytes, read_json, sha256_bytes, sha256_value, write_json
from .composite import composite_scene
from .diagnostics import classification_records
from .errors import BusinessError, ErrorCode, Location, sorted_issues
from .flatten import flatten_scene
from .inject import (
    INJECTION_SCHEMA,
    apply_injections,
    attach_injection_ids,
    default_render_params,
    empty_injection_config,
)
from .pngio import write_bgra_png, write_coverage_png
from .scene import (
    finalize_scene,
    generate_scene,
    normalized_scene_for_render,
    segment_fragment,
    validate_scene,
    validate_tolerance,
)

PIPELINE_VERSION = "1.0.0"
DEMO_SEED = 0xC0FFEE2026
DEMO_TOLERANCE = 0.25


@dataclass
class CaseExecution:
    case_id: str
    case_dir: Path
    report: dict[str, Any]
    composite: dict[str, Any]


def demo_cases() -> list[dict[str, Any]]:
    base = [
        {"case_id": "normal_cross", "kind": "simple_cross", "seed": DEMO_SEED, "injections": []},
        {"case_id": "self_intersect_loop", "kind": "self_intersect_loop", "seed": DEMO_SEED + 1, "injections": []},
        {"case_id": "overlapping_large", "kind": "overlapping_large", "seed": DEMO_SEED + 2, "injections": []},
        {"case_id": "degenerate_segments", "kind": "degenerate_segments", "seed": DEMO_SEED + 3, "injections": []},
        {
            "case_id": "alpha_overflow_injected",
            "kind": "simple_cross",
            "seed": DEMO_SEED + 4,
            "injections": [
                {
                    "id": "inj-alpha",
                    "type": "fill_alpha_out_of_range",
                    "target": {"path_id": "p0"},
                    "value": 300,
                }
            ],
        },
        {
            "case_id": "invalid_fill_rule_injected",
            "kind": "self_intersect_loop",
            "seed": DEMO_SEED + 5,
            "injections": [
                {
                    "id": "inj-rule",
                    "type": "fill_rule_invalid",
                    "target": {"path_id": "p0"},
                    "value": "winding-clockwise",
                }
            ],
        },
        {
            "case_id": "coordinate_out_of_bounds_injected",
            "kind": "simple_cross",
            "seed": DEMO_SEED + 6,
            "injections": [
                {
                    "id": "inj-coord",
                    "type": "coordinate_out_of_bounds",
                    "target": {"path_id": "p1", "subpath_index": 0, "segment_index": 0, "point": "p1"},
                    "value": {"x": 500.0, "y": -80.0},
                }
            ],
        },
        {
            "case_id": "negative_tolerance_injected",
            "kind": "simple_cross",
            "seed": DEMO_SEED + 7,
            "injections": [
                {"id": "inj-tolerance", "type": "tolerance_negative", "target": {}, "value": -0.5}
            ],
        },
        {
            "case_id": "zero_stroke_injected",
            "kind": "simple_cross",
            "seed": DEMO_SEED + 8,
            "injections": [
                {"id": "inj-width", "type": "stroke_width_zero", "target": {"path_id": "p1"}, "value": 0}
            ],
        },
        {
            "case_id": "verifier_pixel_mismatch_injected",
            "kind": "simple_cross",
            "seed": DEMO_SEED + 9,
            "injections": [
                {
                    "id": "inj-pixel",
                    "type": "layer_pixel_mutation",
                    "target": {"layer_id": "L0000"},
                    "value": {"x": 60, "y": 54, "channel": 0, "delta": 17},
                }
            ],
        },
        {
            "case_id": "injection_conflict_injected",
            "kind": "simple_cross",
            "seed": DEMO_SEED + 10,
            "injections": [
                {"id": "inj-conflict-a", "type": "fill_alpha_out_of_range", "target": {"path_id": "p0"}, "value": 300},
                {"id": "inj-conflict-b", "type": "fill_alpha_out_of_range", "target": {"path_id": "p0"}, "value": 301},
            ],
        },
        {
            "case_id": "layer_hash_mismatch_injected",
            "kind": "simple_cross",
            "seed": DEMO_SEED + 11,
            "injections": [
                {
                    "id": "inj-layer-hash",
                    "type": "layer_hash_mutation",
                    "target": {"layer_id": "L0000"},
                    "value": {"reason": "declared deterministic layer hash corruption"},
                }
            ],
        },
    ]
    for case in base:
        case.setdefault("render_params", {"tolerance": DEMO_TOLERANCE, "subscanlines": 8})
    return base


def demo_spec_hash(cases: list[dict[str, Any]] | None = None) -> str:
    return sha256_value({"schema": "vgsynth.demo/v1", "cases": cases or demo_cases()})


def _used_tolerance(requested: Any) -> tuple[float, str]:
    try:
        value = float(requested)
    except (TypeError, ValueError):
        return DEMO_TOLERANCE, "fallback:default-nonfinite"
    if not np.isfinite(value) or value <= 0.0:
        recovered = abs(value) if np.isfinite(value) and value != 0.0 else DEMO_TOLERANCE
        return float(recovered), "recover:absolute-positive"
    return value, "accepted"


def _first_covered_pixel(layer: dict[str, Any], rule: str | None = None, include_stroke: bool = True) -> tuple[int, int] | None:
    coverage = layer["fill"][rule or layer["fill"]["selected_rule"]].copy()
    if include_stroke:
        coverage = np.maximum(coverage, layer["stroke"]["coverage"])
    indices = np.flatnonzero(coverage.reshape(-1))
    if indices.size == 0:
        return None
    y, x = divmod(int(indices[0]), coverage.shape[1])
    return int(x), int(y)


def enrich_issues(issues: list[BusinessError], scene: dict[str, Any], layers: list[dict[str, Any]]) -> list[BusinessError]:
    layer_by_id = {layer["layer_id"]: layer for layer in layers}
    path_to_layer = {layer["path_id"]: layer for layer in layers}
    for issue in issues:
        layer = None
        if issue.location.layer_id:
            layer = layer_by_id.get(issue.location.layer_id)
        if layer is None and issue.location.path_id:
            layer = path_to_layer.get(issue.location.path_id)
        pixel = None
        if layer is not None:
            pixel = _first_covered_pixel(layer)
        issue.location = Location(
            layer_id=issue.location.layer_id or (layer["layer_id"] if layer else None),
            path_id=issue.location.path_id,
            subpath_index=issue.location.subpath_index,
            segment_index=issue.location.segment_index,
            field=issue.location.field,
            pixel=pixel,
        )
        fragment = segment_fragment(
            scene,
            issue.location.path_id,
            issue.location.subpath_index,
            issue.location.segment_index,
        )
        if fragment is None and issue.location.path_id:
            fragment = segment_fragment(scene, issue.location.path_id, None, None)
        if fragment is not None:
            issue.details["path_fragment"] = fragment
    return sorted_issues(issues)


def _deduplicate_issues(issues: list[BusinessError]) -> list[BusinessError]:
    seen: set[tuple] = set()
    result = []
    for issue in issues:
        key = (
            str(issue.code),
            issue.location.layer_id,
            issue.location.path_id,
            issue.location.subpath_index,
            issue.location.segment_index,
            issue.location.field,
            issue.injection_id,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(issue)
    return result


def _apply_pixel_mutations(
    layers: list[dict[str, Any]],
    mutations: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    stored = [layer["pixels"].copy() for layer in layers]
    by_id = {layer["layer_id"]: index for index, layer in enumerate(layers)}
    applied = []
    for mutation in mutations:
        if mutation["type"] != "layer_pixel_mutation":
            continue
        index = by_id.get(mutation["layer_id"])
        if index is None:
            continue
        value = mutation.get("value", {}) or {}
        try:
            x = int(value.get("x", 0))
            y = int(value.get("y", 0))
            channel = int(value.get("channel", 0))
            delta = int(value.get("delta", 1))
        except (TypeError, ValueError, AttributeError):
            continue
        height, width = stored[index].shape[:2]
        if not (0 <= x < width and 0 <= y < height and 0 <= channel < 4):
            continue
        original = int(stored[index][y, x, channel])
        mutated = (original + delta) % 256
        stored[index][y, x, channel] = mutated
        applied.append(
            {
                "injection_id": mutation["injection_id"],
                "layer_id": mutation["layer_id"],
                "x": x,
                "y": y,
                "channel": channel,
                "original": original,
                "mutated": mutated,
            }
        )
    return stored, applied


def _write_layer_artifacts(
    case_dir: Path,
    layers: list[dict[str, Any]],
    stored_pixels: list[np.ndarray],
) -> list[dict[str, Any]]:
    layer_dir = case_dir / "layers"
    coverage_dir = case_dir / "coverage"
    layer_dir.mkdir(parents=True, exist_ok=True)
    coverage_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (layer, stored) in enumerate(zip(layers, stored_pixels)):
        layer_id = layer["layer_id"]
        layer_path = layer_dir / f"{layer_id}.bgra"
        layer_path.write_bytes(stored.tobytes())
        write_bgra_png(layer_dir / f"{layer_id}.png", stored)
        nonzero_path = coverage_dir / f"{layer_id}.fill_nonzero.raw"
        evenodd_path = coverage_dir / f"{layer_id}.fill_evenodd.raw"
        stroke_path = coverage_dir / f"{layer_id}.stroke.raw"
        nonzero_path.write_bytes(layer["fill"]["nonzero"].tobytes())
        evenodd_path.write_bytes(layer["fill"]["evenodd"].tobytes())
        stroke_path.write_bytes(layer["stroke"]["coverage"].tobytes())
        write_coverage_png(coverage_dir / f"{layer_id}.fill_nonzero.png", layer["fill"]["nonzero"])
        write_coverage_png(coverage_dir / f"{layer_id}.fill_evenodd.png", layer["fill"]["evenodd"])
        write_coverage_png(coverage_dir / f"{layer_id}.stroke.png", layer["stroke"]["coverage"])
        records.append(
            {
                **layer["header"],
                "layer_hash": layer["layer_hash"],
                "pixel_hash": layer["pixel_hash"],
                "stored_pixel_hash": sha256_bytes(stored.tobytes()),
                "bgra_file": f"layers/{layer_id}.bgra",
                "png_file": f"layers/{layer_id}.png",
                "fill_nonzero_file": f"coverage/{layer_id}.fill_nonzero.raw",
                "fill_evenodd_file": f"coverage/{layer_id}.fill_evenodd.raw",
                "fill_selected_file": f"coverage/{layer_id}.fill_{layer['fill']['selected_rule']}.raw",
                "stroke_file": f"coverage/{layer_id}.stroke.raw",
                "fill_nonzero_covered": layer["fill"]["nonzero_info"]["covered_pixels"],
                "fill_evenodd_covered": layer["fill"]["evenodd_info"]["covered_pixels"],
                "fill_selected_covered": layer["fill"][layer["fill"]["selected_rule"] + "_info"]["covered_pixels"],
                "stroke_covered": layer["stroke"]["info"]["covered_pixels"],
                "fill_rule_difference": layer["fill"]["first_difference"],
                "stroke_polygon_count": layer["stroke"]["polygon_count"],
                "stroke_vertex_count": layer["stroke"]["vertex_count"],
            }
        )
    return records


def _case_status(findings: list[BusinessError], verification: dict[str, Any], output_mutations: list[dict[str, Any]]) -> str:
    if verification.get("status") == "mismatch":
        if output_mutations:
            return "expected_verifier_mismatch"
        return "failed"
    if findings:
        return "verified_with_findings"
    return "verified"


def _writing_case_files(
    case: dict[str, Any],
    case_dir: Path,
    scene: dict[str, Any],
    injection_source: dict[str, Any],
    effective_scene: dict[str, Any],
    flat: dict[str, Any],
    composite: dict[str, Any],
    issues: list[BusinessError],
    stored_pixels: list[np.ndarray],
    applied_mutations: list[dict[str, Any]],
    output_mutations: list[dict[str, Any]],
    tolerance_used: float,
    tolerance_action: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_file_hash = write_json(case_dir / "scene.json", scene)
    injection_file_hash = write_json(case_dir / "injection.json", injection_source)
    effective_file_hash = write_json(case_dir / "effective_scene.json", effective_scene)
    generation = {
        "schema": "vgsynth.generation/v1",
        "case_id": case["case_id"],
        "scene_id": scene["scene_id"],
        "kind": scene["kind"],
        "seed": scene["seed"],
        "scene_content_hash": scene["content_hash"],
        "scene_file_sha256": scene_file_hash,
        "seed_audit_sha256": sha256_value(scene["seed_audit"]),
        "seed_audit_entry_count": scene["seed_audit"]["entry_count"],
        "seed_audit": copy.deepcopy(scene["seed_audit"]),
        "injection_file_sha256": injection_file_hash,
        "effective_scene_file_sha256": effective_file_hash,
        "render_params_requested": copy.deepcopy(case.get("render_params", default_render_params())),
        "tolerance_requested": case.get("render_params", {}).get("tolerance", DEMO_TOLERANCE),
        "tolerance_used": tolerance_used,
        "tolerance_action": tolerance_action,
    }
    write_json(case_dir / "generation.json", generation)
    write_json(case_dir / "flatten.json", flat)
    layer_records = _write_layer_artifacts(case_dir, composite["layers"], stored_pixels)
    for mutation in output_mutations:
        if mutation.get("type") != "layer_hash_mutation":
            continue
        record = next((item for item in layer_records if item["layer_id"] == mutation["layer_id"]), None)
        if record is None:
            continue
        original = record["layer_hash"]
        mutated = sha256_bytes((original + "|" + mutation["injection_id"]).encode("utf-8"))
        record["layer_hash"] = mutated
        applied_mutations.append(
            {
                "injection_id": mutation["injection_id"],
                "layer_id": mutation["layer_id"],
                "type": "layer_hash_mutation",
                "x": 0,
                "y": 0,
                "channel": None,
                "original": original,
                "mutated": mutated,
            }
        )
    (case_dir / "final.bgra").write_bytes(composite["canvas"].tobytes())
    write_bgra_png(case_dir / "final.png", composite["canvas"])
    layers_manifest = {
        "schema": "vgsynth.layers/v1",
        "pixel_format": "BGRA8_PREMULTIPLIED",
        "width": composite["width"],
        "height": composite["height"],
        "layers": layer_records,
        "canvas_hash": composite["canvas_hash"],
        "composite_chain_hash": composite["composite_chain_hash"],
        "final_bgra_file": "final.bgra",
        "final_png_file": "final.png",
        "output_mutations": applied_mutations,
        "injection_classification": classification_records(issues),
    }
    write_json(case_dir / "layers.json", layers_manifest)
    issue_records = [issue.to_dict() for issue in issues]
    write_json(
        case_dir / "issues.json",
        {
            "schema": "vgsynth.issues/v1",
            "issues": issue_records,
            "first_issue": issue_records[0] if issue_records else None,
        },
    )
    return layers_manifest, generation


def run_case(
    case: dict[str, Any],
    run_dir: Path,
    *,
    minimize: bool = True,
    verify: bool = True,
) -> CaseExecution:
    case_dir = run_dir / "cases" / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    if "scene" in case:
        scene = finalize_scene(copy.deepcopy(case["scene"]))
    else:
        scene = generate_scene(case["kind"], int(case["seed"]), case.get("scene_id", case["case_id"]))
    injection_source = {
        "schema": INJECTION_SCHEMA,
        "operations": copy.deepcopy(case.get("injections", [])),
        "render_params": copy.deepcopy(case.get("render_params", default_render_params())),
    }
    injection_result = apply_injections(scene, injection_source, injection_source["render_params"])
    tolerance_requested = injection_result.render_params.get("tolerance", DEMO_TOLERANCE)
    tolerance_used, tolerance_action = _used_tolerance(tolerance_requested)
    injected_scene = injection_result.scene
    scene_issues = validate_scene(injected_scene, check_content_hash=False)
    tolerance_issues = validate_tolerance(tolerance_requested)
    all_issues = _deduplicate_issues(injection_result.issues + scene_issues + tolerance_issues)
    attach_injection_ids(all_issues, injection_result.applications)
    normalized_scene = normalized_scene_for_render(injected_scene)
    flat = flatten_scene(normalized_scene, tolerance_used)
    composite = composite_scene(normalized_scene, flat)
    issues = enrich_issues(sorted_issues(all_issues), injected_scene, composite["layers"])
    stored_pixels, applied_mutations = _apply_pixel_mutations(composite["layers"], injection_result.output_mutations)
    layers_manifest, generation = _writing_case_files(
        case,
        case_dir,
        scene,
        injection_source,
        injected_scene,
        flat,
        composite,
        issues,
        stored_pixels,
        applied_mutations,
        injection_result.output_mutations,
        tolerance_used,
        tolerance_action,
    )
    verification: dict[str, Any] = {"status": "skipped", "code": None, "checks": {}}
    if verify:
        from .verifier import verify_case_dir

        verification = verify_case_dir(case_dir)
        write_json(case_dir / "verification.json", verification)

    minimal = None
    if minimize and (issues or verification.get("status") == "mismatch"):
        from .minimize import minimize_case

        minimal = minimize_case(case_dir)
    status = _case_status(issues, verification, applied_mutations)
    flatten_segment_reports = [
        {
            "path_id": path["path_id"],
            "subpath_index": subpath["subpath_index"],
            "segments": [
                {
                    "segment_index": segment["segment_index"],
                    "segment_type": segment["segment_type"],
                    "action": segment["action"],
                    "vertex_count": segment["vertex_count"],
                    "error_estimate": segment["error_estimate"],
                    "error_method": segment["error_method"],
                }
                for segment in subpath["segments"]
            ],
        }
        for path in flat["paths"]
        for subpath in path["subpaths"]
    ]
    report = {
        "case_id": case["case_id"],
        "status": status,
        "scene": {
            "file": "scene.json",
            "content_hash": scene["content_hash"],
            "file_sha256": generation["scene_file_sha256"],
            "seed": scene["seed"],
            "kind": scene["kind"],
            "seed_audit_file": "scene.json",
            "seed_audit_sha256": generation["seed_audit_sha256"],
        },
        "generation": generation,
        "injection": {
            "file": "injection.json",
            "file_sha256": generation["injection_file_sha256"],
            "operation_count": len(injection_source["operations"]),
            "applications": injection_result.applications,
            "output_mutations": applied_mutations,
        },
        "flatten": {
            "file": "flatten.json",
            "hash": flat["flatten_hash"],
            "tolerance_used": tolerance_used,
            "vertex_count": flat["vertex_count"],
            "max_error_estimate": flat["max_error_estimate"],
            "skipped_segment_count": flat["skipped_segment_count"],
            "degraded_segment_count": flat["degraded_segment_count"],
            "segments": flatten_segment_reports,
        },
        "fill": {
            "layers": [
                {
                    "layer_id": layer["layer_id"],
                    "selected_rule": layer["fill"]["selected_rule"],
                    "nonzero_hash": layer["fill"]["nonzero_info"]["content_hash"],
                    "evenodd_hash": layer["fill"]["evenodd_info"]["content_hash"],
                    "first_rule_difference": layer["fill"]["first_difference"],
                }
                for layer in composite["layers"]
            ]
        },
        "stroke": {
            "layers": [
                {
                    "layer_id": layer["layer_id"],
                    "coverage_hash": layer["stroke"]["info"]["content_hash"],
                    "outline_hash": layer["stroke"]["outline_hash"],
                    "polygon_count": layer["stroke"]["polygon_count"],
                }
                for layer in composite["layers"]
            ]
        },
        "composite": {
            "layers_file": "layers.json",
            "layer_hashes": {layer["layer_id"]: layer["layer_hash"] for layer in composite["layers"]},
            "layer_pixel_hashes": {layer["layer_id"]: layer["pixel_hash"] for layer in composite["layers"]},
            "stored_layer_pixel_hashes": {
                layer["layer_id"]: sha256_bytes(stored.tobytes()) for layer, stored in zip(composite["layers"], stored_pixels)
            },
            "canvas_hash": composite["canvas_hash"],
            "composite_chain_hash": composite["composite_chain_hash"],
            "final_bgra_file": "final.bgra",
        },
        "issues": {
            "count": len(issues),
            "items": [issue.to_dict() for issue in issues],
            "first": issues[0].to_dict() if issues else None,
        },
        "verification": verification,
        "minimal_counterexample": minimal,
    }
    write_json(case_dir / "case_report.json", report)
    return CaseExecution(case_id=case["case_id"], case_dir=case_dir, report=report, composite=composite)


def _collect_file_hashes(root: Path) -> dict[str, str]:
    records = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in ("hashes.json", "run_manifest.json"):
            continue
        records[relative] = sha256_bytes(path.read_bytes())
    return records


def run_demo(run_dir: Path, *, minimize: bool = True) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cases = demo_cases()
    executions = [run_case(case, run_dir, minimize=minimize) for case in cases]
    status_counts: dict[str, int] = {}
    for execution in executions:
        status = execution.report["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    failed = [item.report["case_id"] for item in executions if item.report["status"] == "failed"]
    expected_mismatch = [item.report["case_id"] for item in executions if item.report["status"] == "expected_verifier_mismatch"]
    report = {
        "schema": "vgsynth.run-report/v1",
        "pipeline_version": PIPELINE_VERSION,
        "run_id": demo_spec_hash(cases),
        "success": not failed,
        "summary": {
            "case_count": len(executions),
            "status_counts": dict(sorted(status_counts.items())),
            "failed_cases": failed,
            "expected_verifier_mismatch_cases": expected_mismatch,
        },
        "environment": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "pyside6": _pyside_version(),
        },
        "cases": [execution.report for execution in executions],
    }
    report_hash = write_json(run_dir / "report.json", report)
    manifest = {
        "schema": "vgsynth.run-manifest/v1",
        "run_id": report["run_id"],
        "report_file": "report.json",
        "report_sha256": report_hash,
        "case_reports": {item.case_id: sha256_bytes(canonical_json_bytes(item.report)) for item in executions},
        "success": report["success"],
    }
    write_json(run_dir / "run_manifest.json", manifest)
    hashes = _collect_file_hashes(run_dir)
    write_json(run_dir / "hashes.json", {"schema": "vgsynth.file-hashes/v1", "files": hashes})
    return report


def _pyside_version() -> str | None:
    try:
        import PySide6

        return PySide6.__version__
    except Exception:
        return None


def compare_runs(first: Path, second: Path) -> dict[str, Any]:
    first_hashes = read_json(first / "hashes.json")["files"]
    second_hashes = read_json(second / "hashes.json")["files"]
    only_first = sorted(set(first_hashes) - set(second_hashes))
    only_second = sorted(set(second_hashes) - set(first_hashes))
    differing = sorted(path for path in set(first_hashes) & set(second_hashes) if first_hashes[path] != second_hashes[path])
    first_manifest = read_json(first / "run_manifest.json")
    second_manifest = read_json(second / "run_manifest.json")
    equivalent = not only_first and not only_second and not differing and first_manifest == second_manifest
    return {
        "schema": "vgsynth.run-comparison/v1",
        "equivalent": equivalent,
        "file_count_first": len(first_hashes),
        "file_count_second": len(second_hashes),
        "only_first": only_first,
        "only_second": only_second,
        "differing": differing,
        "run_manifest_equal": first_manifest == second_manifest,
    }


def cleanup_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
