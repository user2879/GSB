from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .canonical import read_json, sha256_value, write_json
from .composite import composite_scene
from .errors import BusinessError, sorted_issues
from .flatten import flatten_scene
from .inject import apply_injections, attach_injection_ids
from .pipeline import run_case
from .scene import finalize_scene, normalized_scene_for_render, validate_scene, validate_tolerance


def _target_from_case(case_dir: Path) -> dict[str, Any]:
    verification = read_json(case_dir / "verification.json")
    first_mismatch = verification.get("first_mismatch")
    if verification.get("status") == "mismatch" and first_mismatch:
        layers = read_json(case_dir / "layers.json")
        layer_id = first_mismatch.get("layer_id")
        mutation = next(
            (
                item
                for item in layers.get("output_mutations", [])
                if item.get("layer_id") == layer_id
                and item.get("x") == first_mismatch.get("x")
                and item.get("y") == first_mismatch.get("y")
            ),
            None,
        )
        return {
            "kind": "verifier_mismatch",
            "code": verification.get("code"),
            "injection_id": mutation.get("injection_id") if mutation else None,
            "layer_id": layer_id,
            "path_id": first_mismatch.get("path_id"),
            "x": first_mismatch.get("x"),
            "y": first_mismatch.get("y"),
            "channel": first_mismatch.get("channel"),
            "violation_type": first_mismatch.get("violation_type"),
        }
    issues = read_json(case_dir / "issues.json").get("issues", [])
    if not issues:
        return {"kind": "none"}
    issue = issues[0]
    location = issue.get("location", {})
    return {
        "kind": "business_issue",
        "code": issue.get("code"),
        "injection_id": issue.get("injection_id"),
        "layer_id": location.get("layer_id"),
        "path_id": location.get("path_id"),
        "subpath_index": location.get("subpath_index"),
        "segment_index": location.get("segment_index"),
        "field": location.get("field"),
        "recovery": issue.get("recovery"),
    }


def _target_path(scene: dict[str, Any], target: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    path_id = target.get("path_id")
    if path_id is None and target.get("layer_id"):
        try:
            index = int(str(target["layer_id"])[1:])
        except ValueError:
            return None
        if 0 <= index < len(scene.get("paths", [])):
            return index, scene["paths"][index]
    for index, path in enumerate(scene.get("paths", [])):
        if path.get("path_id") == path_id:
            return index, path
    return None


def _minimal_path(path: dict[str, Any], target: dict[str, Any]) -> tuple[dict[str, Any], bool, bool]:
    minimal = copy.deepcopy(path)
    has_subpath = target.get("subpath_index") is not None and target.get("path_id") is not None
    has_segment = has_subpath and target.get("segment_index") is not None
    if not has_subpath:
        if not path.get("subpaths") or not path["subpaths"][0].get("segments"):
            minimal["subpaths"] = []
            return minimal, False, False
        subpath = copy.deepcopy(path["subpaths"][0])
        subpath["segments"] = [copy.deepcopy(subpath["segments"][0])]
        minimal["subpaths"] = [subpath]
        return minimal, True, True
    subpath = copy.deepcopy(path["subpaths"][int(target["subpath_index"])])
    if not has_segment:
        subpath["segments"] = []
        minimal["subpaths"] = [subpath]
        return minimal, True, False
    segment = copy.deepcopy(subpath["segments"][int(target["segment_index"])])
    subpath["segments"] = [segment]
    minimal["subpaths"] = [subpath]
    return minimal, True, True


def _filter_injections(injection: dict[str, Any], target: dict[str, Any], has_subpath: bool, has_segment: bool) -> dict[str, Any]:
    result = copy.deepcopy(injection)
    target_id = target.get("injection_id")
    operations = []
    for operation in injection.get("operations", []):
        if target_id is not None and operation.get("id") != target_id:
            continue
        if target_id is None:
            continue
        operation = copy.deepcopy(operation)
        op_target = operation.setdefault("target", {})
        if target.get("path_id") is not None and op_target.get("path_id") is not None:
            op_target["path_id"] = target["path_id"]
            if has_subpath:
                op_target["subpath_index"] = 0
            else:
                op_target.pop("subpath_index", None)
            if has_segment:
                op_target["segment_index"] = 0
            else:
                op_target.pop("segment_index", None)
        if target.get("kind") == "verifier_mismatch" and target.get("layer_id") is not None:
            op_target["layer_id"] = target["layer_id"]
            op_target.pop("path_id", None)
        operations.append(operation)
    result["operations"] = operations
    return result


def _build_candidate(
    scene: dict[str, Any], injection: dict[str, Any], target: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_scene = copy.deepcopy(scene)
    candidate_scene["paths"] = []
    has_subpath = False
    has_segment = False
    if target.get("kind") == "business_issue" and target.get("code") == "E_TOLERANCE_RANGE":
        pass
    else:
        found = _target_path(scene, target)
        if found is not None:
            _, path = found
            minimal_path, has_subpath, has_segment = _minimal_path(path, target)
            candidate_scene["paths"] = [minimal_path]
    candidate_scene = finalize_scene(candidate_scene)
    candidate_injection = _filter_injections(injection, target, has_subpath, has_segment)
    return candidate_scene, candidate_injection


def _issues_for(scene: dict[str, Any], injection: dict[str, Any]) -> tuple[list[BusinessError], Any]:
    result = apply_injections(scene, injection, injection.get("render_params", {}))
    issues = result.issues + validate_scene(result.scene, check_content_hash=False) + validate_tolerance(
        result.render_params.get("tolerance", 0.25)
    )
    issues = sorted_issues(issues)
    attach_injection_ids(issues, result.applications)
    return issues, result


def _detect_business(scene: dict[str, Any], injection: dict[str, Any], target: dict[str, Any]) -> bool:
    issues, _ = _issues_for(scene, injection)
    if any(str(issue.code) != target.get("code") for issue in issues):
        return False
    for issue in issues:
        if str(issue.code) != target.get("code"):
            continue
        if target.get("injection_id") is not None and issue.injection_id != target.get("injection_id"):
            continue
        if target.get("path_id") is not None and issue.location.path_id != target.get("path_id"):
            continue
        return True
    return False


def _detect_mismatch(scene: dict[str, Any], injection: dict[str, Any], target: dict[str, Any]) -> bool:
    issues, injected = _issues_for(scene, injection)
    if issues:
        return False
    normalized = normalized_scene_for_render(injected.scene)
    tolerance = float(injected.render_params.get("tolerance", 0.25))
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        tolerance = abs(tolerance) if np.isfinite(tolerance) and tolerance != 0.0 else 0.25
    flat = flatten_scene(normalized, tolerance)
    composite = composite_scene(normalized, flat)
    layer = next((item for item in composite["layers"] if item["layer_id"] == target.get("layer_id")), None)
    if layer is None:
        return False
    if any(
        mutation["type"] == "layer_hash_mutation" and mutation["layer_id"] == target.get("layer_id")
        for mutation in injected.output_mutations
    ):
        return True
    stored = layer["pixels"].copy()
    for mutation in injected.output_mutations:
        if mutation["type"] != "layer_pixel_mutation" or mutation["layer_id"] != target.get("layer_id"):
            continue
        value = mutation.get("value", {})
        x = int(value.get("x", 0))
        y = int(value.get("y", 0))
        channel = int(value.get("channel", 0))
        delta = int(value.get("delta", 1))
        height, width = stored.shape[:2]
        if 0 <= x < width and 0 <= y < height and 0 <= channel < 4:
            stored[y, x, channel] = (int(stored[y, x, channel]) + delta) % 256
    if stored.tobytes() == layer["pixels"].tobytes():
        return False
    if target.get("x") is not None and target.get("y") is not None:
        x, y = int(target["x"]), int(target["y"])
        if not (0 <= x < stored.shape[1] and 0 <= y < stored.shape[0]):
            return False
        if np.array_equal(stored[y, x], layer["pixels"][y, x]):
            return False
    return True


def _detect(scene: dict[str, Any], injection: dict[str, Any], target: dict[str, Any]) -> bool:
    if target.get("kind") == "verifier_mismatch":
        return _detect_mismatch(scene, injection, target)
    if target.get("kind") == "business_issue":
        return _detect_business(scene, injection, target)
    return False


def _observed_codes(scene: dict[str, Any], injection: dict[str, Any]) -> list[str]:
    issues, _ = _issues_for(scene, injection)
    return sorted({str(issue.code) for issue in issues})


def _remove_path(scene: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(scene)
    result["paths"] = []
    return finalize_scene(result)


def _remove_subpath(scene: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(scene)
    for path in result["paths"]:
        path["subpaths"] = []
    return finalize_scene(result)


def _remove_segment(scene: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(scene)
    for path in result["paths"]:
        for subpath in path.get("subpaths", []):
            subpath["segments"] = []
    return finalize_scene(result)


def _remove_injection(injection: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(injection)
    result["operations"] = []
    return result


def _reproduction_match(report: dict[str, Any], target: dict[str, Any]) -> bool:
    if target.get("kind") == "verifier_mismatch":
        verification = report.get("verification", {})
        mismatch = verification.get("first_mismatch") or {}
        return (
            verification.get("status") == "mismatch"
            and mismatch.get("layer_id") == target.get("layer_id")
            and mismatch.get("x") == target.get("x")
            and mismatch.get("y") == target.get("y")
            and mismatch.get("channel") == target.get("channel")
        )
    if target.get("kind") == "business_issue":
        for issue in report.get("issues", {}).get("items", []):
            if issue.get("code") != target.get("code"):
                continue
            if target.get("injection_id") is not None and issue.get("injection_id") != target.get("injection_id"):
                continue
            location = issue.get("location", {})
            if target.get("path_id") is not None and location.get("path_id") != target.get("path_id"):
                continue
            return True
    return False


def minimize_case(case_dir: Path) -> dict[str, Any]:
    case_dir = Path(case_dir)
    scene = read_json(case_dir / "scene.json")
    injection = read_json(case_dir / "injection.json")
    target = _target_from_case(case_dir)
    if target.get("kind") == "none":
        return {"status": "not_needed", "target": target}
    candidate_scene, candidate_injection = _build_candidate(scene, injection, target)
    if not _detect(candidate_scene, candidate_injection, target):
        candidate_scene, candidate_injection = finalize_scene(scene), copy.deepcopy(injection)
    has_path = bool(candidate_scene.get("paths"))
    has_subpath = bool(has_path and candidate_scene["paths"][0].get("subpaths"))
    has_segment = bool(has_subpath and candidate_scene["paths"][0]["subpaths"][0].get("segments"))
    has_operation = bool(candidate_injection.get("operations"))

    elements: list[dict[str, Any]] = []
    if has_path:
        elements.append({"kind": "path", "id": candidate_scene["paths"][0].get("path_id")})
    if has_subpath:
        elements.append({"kind": "subpath", "id": candidate_scene["paths"][0]["subpaths"][0].get("subpath_id")})
    if has_segment:
        elements.append({"kind": "segment", "id": 0})
    if has_operation:
        elements.append({"kind": "injection", "id": candidate_injection["operations"][0].get("id")})

    proof = []
    removers: list[tuple[dict[str, Any], Callable[[], tuple[dict[str, Any], dict[str, Any]]]]] = []
    if has_path:
        removers.append((elements[0], lambda: (_remove_path(candidate_scene), copy.deepcopy(candidate_injection))))
    if has_subpath:
        removers.append((elements[1], lambda: (_remove_subpath(candidate_scene), copy.deepcopy(candidate_injection))))
    if has_segment:
        removers.append((elements[2], lambda: (_remove_segment(candidate_scene), copy.deepcopy(candidate_injection))))
    if has_operation:
        removers.append((elements[-1], lambda: (copy.deepcopy(candidate_scene), _remove_injection(candidate_injection))))

    necessary = []
    for element, remover in removers:
        reduced_scene, reduced_injection = remover()
        still_triggers = _detect(reduced_scene, reduced_injection, target)
        observed_codes = _observed_codes(reduced_scene, reduced_injection)
        proof.append(
            {
                "removed_element": element,
                "still_triggers": still_triggers,
                "result": "violation_persists" if still_triggers else "violation_gone",
                "necessary": not still_triggers,
                "observed_codes_after_removal": observed_codes,
            }
        )
        if not still_triggers:
            necessary.append(element)

    minimal_dir = case_dir / "minimal_counterexample"
    minimal_dir.mkdir(parents=True, exist_ok=True)
    replay_root = minimal_dir / "replay_run"
    if replay_root.exists():
        shutil.rmtree(replay_root)
    replay_case = {
        "case_id": "minimal",
        "scene": candidate_scene,
        "injections": copy.deepcopy(candidate_injection.get("operations", [])),
        "render_params": copy.deepcopy(candidate_injection.get("render_params", {})),
    }
    execution = run_case(replay_case, replay_root, minimize=False, verify=True)
    reproduced = _reproduction_match(execution.report, target)
    replay_case_dir = execution.case_dir
    write_json(minimal_dir / "scene.json", read_json(replay_case_dir / "scene.json"))
    write_json(minimal_dir / "injection.json", read_json(replay_case_dir / "injection.json"))
    write_json(minimal_dir / "removal_proof.json", {"target": target, "elements": elements, "checks": proof})
    report = {
        "schema": "vgsynth.minimal-counterexample/v1",
        "status": "reproduced" if reproduced else "reproduction_failed",
        "target": target,
        "target_policy": "same violation must be isolated; additional business errors do not count as reproduction",
        "elements": elements,
        "necessary_elements": necessary,
        "irreducible": all(item["necessary"] for item in proof) and len(necessary) == len(elements),
        "removal_checks": proof,
        "replay_dir": "replay_run/cases/minimal",
        "reproduced_violation": execution.report.get("verification", {}).get("first_mismatch")
        if target.get("kind") == "verifier_mismatch"
        else execution.report.get("issues", {}).get("first"),
        "counterexample_hash": sha256_value({"scene": candidate_scene, "injection": candidate_injection}),
    }
    write_json(minimal_dir / "counterexample.json", report)
    return report
