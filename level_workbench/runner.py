"""Fixed-sequence orchestration, mismatch detection and deterministic reports."""

from __future__ import annotations

import copy
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .canonical import (
    canonical_json_bytes,
    grid_text,
    sha256_bytes,
    sha256_file,
    sha256_value,
    write_canonical_json,
    write_text,
)
from .counterexample import build_minimal_counterexample
from .errors import InjectionError, SpecError, WorkbenchError
from .generator import generate_level
from .injection import apply_spec_patches, empty_injection, normalize_injection
from .model import LevelSpec
from .samples import demo_cases
from .spec import parse_spec
from .verifier import verify_artifact


def pin_thread_environment() -> dict[str, str]:
    values = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    for key, value in values.items():
        os.environ[key] = value
    return values


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case["id"])
    injection_raw = case.get("injection")
    injection_error: WorkbenchError | None = None
    try:
        injection = normalize_injection(injection_raw)
    except WorkbenchError as exc:
        injection = empty_injection()
        injection_error = exc

    raw_spec = copy.deepcopy(case["spec"])
    patched_spec = raw_spec
    if injection_error is None:
        try:
            patched_spec = apply_spec_patches(raw_spec, injection)
        except WorkbenchError as exc:
            injection_error = exc

    base: dict[str, Any] = {
        "case_id": case_id,
        "kind": case.get("kind"),
        "status": "error",
        "code": "E_INTERNAL",
        "message": "",
        "injection": injection,
        "hashes": {
            "effective_spec_sha256": sha256_value(patched_spec),
            "injection_sha256": sha256_value(injection),
        },
    }
    if injection_error is not None:
        return _finish_error(base, injection_error, "E_INJECTION")

    try:
        spec = parse_spec(patched_spec)
    except SpecError as exc:
        return _finish_error(base, exc, exc.code)

    base["spec"] = spec.normalized
    base["hashes"]["normalized_spec_sha256"] = sha256_value(spec.normalized)
    base["effective_spec"] = patched_spec

    try:
        outcome = generate_level(spec, injection)
    except WorkbenchError as exc:
        return _finish_error(base, exc, exc.code)

    try:
        artifact_tamper = _apply_artifact_tamper(spec, outcome.level, injection, base)
    except WorkbenchError as exc:
        return _finish_error(base, exc, "E_INJECTION")
    level_doc = outcome.level_document()
    if level_doc is not None:
        level_doc["generation"]["artifact_tamper"] = artifact_tamper
        # The content hash must describe the artifact after any tamper step.
        content_core = copy.deepcopy(level_doc)
        content_core.pop("hashes", None)
        level_doc["hashes"]["content_sha256"] = sha256_bytes(canonical_json_bytes(content_core))

    verification: dict[str, Any] | None = None
    counterexample: dict[str, Any] | None = None
    mismatch: dict[str, Any] | None = None
    if level_doc is not None:
        verification = verify_artifact(spec, level_doc)
        mismatch = _compare_conclusions(outcome.evidence, verification)
        if verification["status"] == "fail":
            counterexample = build_minimal_counterexample(spec, level_doc, verification)

    error = outcome.error
    code = "OK"
    message = "all constraints satisfied and independently verified"
    status = "pass"
    if mismatch is not None:
        code = "E_VERIFIER_MISMATCH"
        message = "generator and independent verifier reached different conclusion sets"
        status = "fail"
        error = {
            "code": code,
            "message": message,
            "details": mismatch,
        }
    elif error is not None:
        code = error["code"]
        message = error["message"]
        status = "fail"
    elif verification is not None and verification["status"] == "fail":
        code = "E_VERIFIER_MISMATCH"
        message = "generator reported success but the independent verifier failed the artifact"
        status = "fail"
        error = {"code": code, "message": message, "details": {"verifier": verification}}

    first_violation = _select_first_violation(error, verification)
    base.update(
        {
            "status": status,
            "code": code,
            "message": message,
            "error": error,
            "generation": {
                "status": outcome.status,
                "error": outcome.error,
                "evidence": outcome.evidence,
                "subseeds": outcome.subseeds,
                "stage_log": outcome.stage_log,
                "repair_trace": outcome.repair_trace,
            },
            "verification": verification,
            "mismatch": mismatch,
            "first_violation": first_violation,
            "minimal_counterexample": counterexample,
            "artifact_tamper": artifact_tamper,
            "level_document": level_doc,
        }
    )
    base["hashes"]["generator_evidence_sha256"] = sha256_value(outcome.evidence)
    if outcome.stage_log:
        base["hashes"]["stage_sha256"] = {
            stage["stage"]: sha256_value(stage) for stage in outcome.stage_log
        }
    if outcome.repair_trace:
        base["hashes"]["repair_trace_sha256"] = sha256_value(outcome.repair_trace)
    base["hashes"]["constraint_evidence_sha256"] = {
        entry["constraint"]: sha256_value(entry)
        for entry in outcome.evidence.get("constraints", [])
    }
    if verification is not None:
        base["hashes"]["verifier_sha256"] = sha256_value(verification)
        base["hashes"]["verifier_constraint_sha256"] = {
            entry["constraint"]: sha256_value(entry)
            for entry in verification.get("constraints", [])
        }
    if level_doc is not None:
        base["hashes"]["level_sha256"] = sha256_value(level_doc)
        base["hashes"]["grid_sha256"] = level_doc["hashes"]["grid_sha256"]
    if counterexample is not None:
        base["hashes"]["counterexample_sha256"] = sha256_value(counterexample)
    if first_violation is not None:
        base["hashes"]["first_violation_sha256"] = sha256_value(first_violation)
    return base


def run_demo() -> dict[str, Any]:
    cases = demo_cases()
    first = [run_case(case) for case in cases]
    second = [run_case(case) for case in cases]
    identical = canonical_json_bytes(first) == canonical_json_bytes(second)
    report_cases = {
        result["case_id"]: result
        for result in first
    }
    pass_count = sum(1 for result in first if result["status"] == "pass")
    fail_count = sum(1 for result in first if result["status"] == "fail")
    error_count = sum(1 for result in first if result["status"] == "error")
    report: dict[str, Any] = {
        "schema": "level-workbench/report-v1",
        "workbench_version": __version__,
        "environment": _environment_metadata(),
        "sequence": [case["id"] for case in cases],
        "summary": {
            "case_count": len(first),
            "pass": pass_count,
            "fail": fail_count,
            "error": error_count,
            "expected_failures": [
                "connectivity_unsat",
                "resource_spec_overflow",
                "resource_pipeline_overflow",
                "illegal_adjacent",
                "verifier_mismatch",
            ],
        },
        "determinism": {
            "in_process_repeats": 2,
            "case_reports_byte_identical": identical,
            "case_reports_sha256": sha256_value(first),
        },
        "cases": report_cases,
    }
    return report


def write_demo_artifacts(out_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("specs", "effective_specs", "injections", "levels", "grids", "counterexamples"):
        path = out_dir / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    for name in ("report.json", "report.md", "hash_manifest.json"):
        path = out_dir / name
        if path.exists():
            path.unlink()

    for case in demo_cases():
        case_id = case["id"]
        _write_yaml(out_dir / "specs" / f"{case_id}.yaml", case["spec"])
        injection = normalize_injection(case.get("injection"))
        _write_yaml(out_dir / "injections" / f"{case_id}.yaml", injection)
        result = report["cases"][case_id]
        effective = result.get("effective_spec", apply_spec_patches(case["spec"], injection))
        _write_yaml(out_dir / "effective_specs" / f"{case_id}.yaml", effective)
        level_doc = result.get("level_document")
        if level_doc is not None:
            write_canonical_json(out_dir / "levels" / f"{case_id}.level.json", level_doc)
            write_text(out_dir / "grids" / f"{case_id}.grid.txt", grid_text(level_doc["grid"]["cells"]))
        counterexample = result.get("minimal_counterexample")
        if counterexample is not None:
            write_canonical_json(
                out_dir / "counterexamples" / f"{case_id}.counterexample.json", counterexample
            )

    write_canonical_json(out_dir / "report.json", report)
    write_text(out_dir / "report.md", _render_markdown(report))

    file_hashes: dict[str, str] = {}
    for path in sorted(out_dir.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name == "hash_manifest.json":
            continue
        relative = path.relative_to(out_dir).as_posix()
        file_hashes[relative] = sha256_file(path)
    manifest_core = {
        "schema": "level-workbench/hash-manifest-v1",
        "algorithm": "sha256",
        "files": file_hashes,
    }
    manifest = dict(manifest_core)
    manifest["run_sha256"] = sha256_value(manifest_core)
    write_canonical_json(out_dir / "hash_manifest.json", manifest)
    return manifest


def compare_directories(path_a: Path, path_b: Path) -> dict[str, Any]:
    files_a = _relative_files(path_a)
    files_b = _relative_files(path_b)
    all_files = sorted(set(files_a) | set(files_b))
    differences: list[dict[str, Any]] = []
    for relative in all_files:
        if relative not in files_a:
            differences.append({"path": relative, "status": "missing_in_a"})
            continue
        if relative not in files_b:
            differences.append({"path": relative, "status": "missing_in_b"})
            continue
        bytes_a = (path_a / relative).read_bytes()
        bytes_b = (path_b / relative).read_bytes()
        if bytes_a != bytes_b:
            differences.append(
                {
                    "path": relative,
                    "status": "content_differs",
                    "sha256_a": sha256_bytes(bytes_a),
                    "sha256_b": sha256_bytes(bytes_b),
                }
            )
    return {
        "schema": "level-workbench/comparison-v1",
        "identical": not differences,
        "file_count_a": len(files_a),
        "file_count_b": len(files_b),
        "differences": differences,
        "tree_sha256_a": sha256_value({name: sha256_file(path_a / name) for name in sorted(files_a)}),
        "tree_sha256_b": sha256_value({name: sha256_file(path_b / name) for name in sorted(files_b)}),
    }


def _finish_error(base: dict[str, Any], exc: WorkbenchError, fallback_code: str) -> dict[str, Any]:
    code = exc.code or fallback_code
    base.update(
        {
            "status": "error",
            "code": code,
            "message": exc.message,
            "error": exc.to_dict(),
            "generation": None,
            "verification": None,
            "mismatch": None,
            "first_violation": {
                "code": code,
                "coord": None,
                "path": exc.path,
                "message": exc.message,
                "input_fragment": exc.details,
            },
            "minimal_counterexample": None,
            "artifact_tamper": [],
            "level_document": None,
        }
    )
    base["hashes"]["error_sha256"] = sha256_value(exc.to_dict())
    return base


def _apply_artifact_tamper(
    spec: LevelSpec,
    level: Any,
    injection: dict[str, Any],
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    tamper = injection.get("artifact_tamper", {})
    if level is None:
        return []
    changes: list[dict[str, Any]] = []
    for entry in tamper.get("force_terrain", []):
        x, y = entry["coord"]
        coord = (x, y)
        terrain = entry["terrain"]
        if terrain not in spec.terrain_weights:
            raise InjectionError(
                "E_INJECTION",
                "artifact tamper terrain is not declared by the specification",
                "artifact_tamper.force_terrain",
            )
        if not level.in_bounds(coord):
            raise InjectionError(
                "E_INJECTION",
                "artifact tamper coordinate is outside the grid",
                "artifact_tamper.force_terrain",
            )
        old = level.terrain_at(coord)
        if old != terrain:
            level.set_terrain(coord, terrain)
            changes.append(
                {"stage": "artifact_tamper", "coord": [x, y], "from": old, "to": terrain}
            )
    override = tamper.get("resource_count_override")
    if override is not None:
        before = list(level.resources)
        target = int(override)
        if target < len(level.resources):
            level.resources = sorted(level.resources, key=lambda coord: (coord[1], coord[0]))[:target]
        elif target > len(level.resources):
            existing = set(level.resources)
            eligible: list[tuple[int, int]] = []
            for y in range(level.height):
                for x in range(level.width):
                    coord = (x, y)
                    if coord in existing or coord in {spec.spawn, spec.target}:
                        continue
                    terrain = level.cells[y][x]
                    if terrain in spec.resource_allowed_terrain and terrain not in spec.blocked_terrain:
                        eligible.append(coord)
            level.resources.extend(eligible[: target - len(level.resources)])
            level.resources = sorted(level.resources, key=lambda coord: (coord[1], coord[0]))
        changes.append(
            {
                "stage": "artifact_tamper",
                "resource_count": {
                    "before": len(before),
                    "requested": target,
                    "after": len(level.resources),
                },
            }
        )
    if changes:
        base["hashes"]["artifact_tamper_sha256"] = sha256_value(changes)
    return changes


def _compare_conclusions(
    generator_evidence: dict[str, Any], verification: dict[str, Any]
) -> dict[str, Any] | None:
    generator_status = {
        entry["constraint"]: entry["status"]
        for entry in generator_evidence.get("constraints", [])
    }
    verifier_status = {
        entry["constraint"]: entry["status"]
        for entry in verification.get("constraints", [])
    }
    names = sorted(set(generator_status) | set(verifier_status))
    differences = [
        {
            "constraint": name,
            "generator": generator_status.get(name, "missing"),
            "verifier": verifier_status.get(name, "missing"),
        }
        for name in names
        if generator_status.get(name, "missing") != verifier_status.get(name, "missing")
    ]
    if not differences:
        return None
    return {"differences": differences}


def _select_first_violation(
    error: dict[str, Any] | None, verification: dict[str, Any] | None
) -> dict[str, Any] | None:
    if verification is not None and verification.get("first_violation") is not None:
        return verification["first_violation"]
    if error is not None:
        details = error.get("details", {})
        if isinstance(details, dict) and details.get("first_violation") is not None:
            first = copy.deepcopy(details["first_violation"])
            if isinstance(first, dict):
                first.setdefault("code", error.get("code"))
                first.setdefault("input_fragment", details.get("conflict", first))
                return first
            return {
                "code": error.get("code"),
                "coord": None,
                "message": str(first),
                "input_fragment": details.get("conflict", first),
            }
        if isinstance(details, dict) and details.get("conflict") is not None:
            conflict = details["conflict"]
            return {
                "code": error.get("code"),
                "coord": conflict.get("coord"),
                "message": conflict.get("reason"),
                "input_fragment": conflict,
            }
    return None


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(value, sort_keys=True, allow_unicode=True, default_flow_style=False)
    path.write_text(text, encoding="utf-8", newline="\n")


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Deterministic Constraint Workbench Report",
        "",
        f"- Workbench version: `{report['workbench_version']}`",
        f"- Cases: {report['summary']['case_count']}",
        f"- Pass: {report['summary']['pass']}",
        f"- Expected failures: {report['summary']['fail']}",
        f"- In-process repeat byte-identical: `{str(report['determinism']['case_reports_byte_identical']).lower()}`",
        "",
        "## Case sequence",
        "",
    ]
    for case_id in report["sequence"]:
        case = report["cases"][case_id]
        first = case.get("first_violation")
        first_coord = first.get("coord") if isinstance(first, dict) else None
        lines.extend(
            [
                f"### `{case_id}`",
                "",
                f"- Status: `{case['status']}`",
                f"- Code: `{case['code']}`",
                f"- First violation coordinate: `{first_coord}`",
                f"- Grid SHA-256: `{case['hashes'].get('grid_sha256', '-')}`",
                f"- Level SHA-256: `{case['hashes'].get('level_sha256', '-')}`",
            ]
        )
        counterexample = case.get("minimal_counterexample")
        if counterexample is not None:
            lines.append(f"- Minimal counterexample: `{counterexample.get('violation_code')}`")
            lines.append(f"- Counterexample minimizable: `{counterexample.get('minimizable')}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _environment_metadata() -> dict[str, Any]:
    yaml_version = getattr(yaml, "__version__", "unknown")
    numpy_version: str | None = None
    if importlib.util.find_spec("numpy") is not None:
        import numpy

        numpy_version = numpy.__version__
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.system(),
        "pyyaml": yaml_version,
        "numpy": numpy_version,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


def _relative_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
