"""Repeatable end-to-end demonstration, including a real forced termination."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from .canonical import atomic_write_json, read_json, sha256_obj
from .errors import DomainError
from .injection import (
    injection_hash,
    run_parameter_injection_case,
    validate_injection_config,
)
from .persistence import PersistenceStore
from .reporting import build_report, compare_recovery_reports
from .synth import DEFAULT_SEED, demo_operations, generate_sample_scene, write_sample_scene
from .workspace import open_workspace


DEMO_SCHEMA = "spectral-workspace-demo-report/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _wait_for_file(path: Path, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        return_code = process.poll()
        if return_code is not None:
            stderr = b""
            if process.stderr is not None:
                stderr = process.stderr.read()
            raise RuntimeError(
                f"crash worker exited before staging: returncode={return_code}, "
                f"stderr={stderr.decode('utf-8', errors='replace')}"
            )
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for crash worker marker {path}")


def run_crash_worker(workspace: str, operations_path: str, ready_file: str) -> None:
    """Internal command: stage an uncommitted operation and wait to be killed."""

    operations = read_json(operations_path)
    if not isinstance(operations, list):
        raise RuntimeError("operations file must contain an array")
    engine = open_workspace(workspace)
    next_version = engine.revision + 1
    operation = next(
        (item for item in operations if item.get("version") == next_version), None
    )
    if operation is None:
        raise RuntimeError(f"no operation for next revision {next_version}")
    result = engine.apply(operation, commit=False)
    ready = {
        "ready": True,
        "revision": result["revision"],
        "snapshot_version_stable": engine.snapshot_version,
        "transaction_id": result["transaction_id"],
        "state_sha256": result["state_sha256"],
    }
    atomic_write_json(ready_file, ready)
    while True:
        time.sleep(1.0)


def _run_baseline(
    workspace: Path,
    operations: list[dict[str, Any]],
    *,
    stable_operation_count: int,
) -> dict[str, Any]:
    engine = open_workspace(workspace)
    for operation in operations[:stable_operation_count]:
        engine.apply(operation)
    report = build_report(engine, extra={"role": "baseline_before_forced_termination"})
    engine.close()
    return report


def _force_terminate_after_uncommitted_stage(
    workspace: Path,
    operations_path: Path,
    ready_file: Path,
) -> dict[str, Any]:
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not python_path
        else str(PROJECT_ROOT) + os.pathsep + python_path
    )
    command = [
        sys.executable,
        "-m",
        "spectral_workspace.cli",
        "crash-worker",
        "--workspace",
        str(workspace),
        "--operations",
        str(operations_path),
        "--ready-file",
        str(ready_file),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_file(ready_file, process, timeout=60.0)
        ready = read_json(ready_file)
        process.kill()
        process.communicate(timeout=20.0)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=20.0)
    return {
        "worker_terminated_by_os": process.returncode is not None,
        "uncommitted_revision_staged": int(ready["revision"]),
        "stable_snapshot_version_at_kill": int(ready["snapshot_version_stable"]),
        "staged_state_sha256": ready["state_sha256"],
        "ready_marker_written": True,
    }


def _stage_failure_case(
    root: Path,
    sample: Mapping[str, Any],
    operations: list[dict[str, Any]],
) -> dict[str, Any]:
    _reset_directory(root)
    stage_config = validate_injection_config(
        {
            "schema": "spectral-injection/v1",
            "name": "force-reused-adapted-stage-stale",
            "stage_failure": {
                "operation_version": 4,
                "stage_kind": "adapted_xyz",
                "material_id": "mat_oxide",
                "light_id": "light_day",
                "delta_xyz": [0.25, 0.0, 0.0],
            },
        }
    )
    engine = open_workspace(root, injection=stage_config)
    for operation in operations[:3]:
        engine.apply(operation)
    result: dict[str, Any] = {
        "injection": stage_config,
        "injection_hash": injection_hash(stage_config),
        "target_operation_version": 4,
    }
    try:
        engine.apply(operations[3])
        result.update({"ok": True, "unexpectedly_committed": True})
    except DomainError as error:
        result.update(
            {
                "ok": False,
                "error": error.to_dict()["error"],
                "stable_revision_unchanged": engine.revision == 3,
                "stable_snapshot_unchanged": engine.snapshot_version == 3,
            }
        )
    engine.close()
    return result


def _parameter_injection_cases(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = [
        (
            "reflectance_out_of_range",
            {
                "schema": "spectral-injection/v1",
                "name": "reflectance_out_of_range",
                "reflectance_out_of_range": {
                    "material_id": "mat_oxide",
                    "sample_index": 17,
                    "value": 1.25,
                },
            },
        ),
        (
            "white_point_zero",
            {
                "schema": "spectral-injection/v1",
                "name": "white_point_zero",
                "white_point_zero": {"value": [0.0, 1.0, 1.08883]},
            },
        ),
        (
            "observer_invalid",
            {
                "schema": "spectral-injection/v1",
                "name": "observer_invalid",
                "observer_invalid": {"value": "CIE2099_2"},
            },
        ),
    ]
    results: list[dict[str, Any]] = []
    for name, config in cases:
        outcome = run_parameter_injection_case(sample, config)
        outcome["case"] = name
        outcome["injection"] = config
        results.append(outcome)

    invalid_config = {
        "schema": "spectral-injection/v1",
        "unknown_injection_key": {"value": 1},
    }
    try:
        validate_injection_config(invalid_config)
        invalid_result: dict[str, Any] = {"ok": True, "case": "invalid_injection_config"}
    except DomainError as error:
        invalid_result = {
            "ok": False,
            "case": "invalid_injection_config",
            "injection": invalid_config,
            "error": error.to_dict()["error"],
        }
    results.append(invalid_result)
    return results


def run_demo(
    out_dir: str | os.PathLike[str],
    report_path: str | os.PathLike[str],
    *,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir)
    if out.exists():
        if not force:
            raise FileExistsError(
                f"output directory already exists: {out}; pass --force to replace it"
            )
    _reset_directory(out)

    sample = generate_sample_scene(seed)
    sample_path = out / "sample_scene.json"
    write_sample_scene(sample_path, seed=seed, replace=True)
    operations = demo_operations(sample, seed=seed)
    operations_path = out / "operations.json"
    atomic_write_json(operations_path, operations, pretty=True)

    workspace = out / "workspace"
    baseline = _run_baseline(workspace, operations, stable_operation_count=6)
    ready_file = out / "crash_ready.json"
    forced = _force_terminate_after_uncommitted_stage(
        workspace, operations_path, ready_file
    )

    recovered_engine = open_workspace(workspace)
    verification = recovered_engine.verify()
    recovered = build_report(
        recovered_engine,
        verification=verification,
        extra={
            "role": "recovered_after_forced_termination",
            "baseline_snapshot_version": baseline["workspace"]["snapshot_version"],
            "baseline_state_sha256": baseline["workspace"]["state_sha256"],
            "forced_termination": forced,
        },
    )
    comparison = compare_recovery_reports(baseline, recovered)
    injection_checks = {
        "parameter_cases": _parameter_injection_cases(sample),
        "stage_failure_case": _stage_failure_case(
            out / "injection_workspace", sample, operations
        ),
    }

    expected_errors = {
        item["case"]: item.get("error", {}).get("code")
        for item in injection_checks["parameter_cases"]
    }
    expected_errors["stage_failure"] = injection_checks["stage_failure_case"].get(
        "error", {}
    ).get("code")
    acceptance = {
        "forced_recovery_colors_match_baseline": comparison["colors_byte_equal"],
        "forced_recovery_stage_hashes_match_baseline": comparison["stages_byte_equal"],
        "forced_recovery_revision_is_last_committed": recovered["workspace"]["revision"]
        == baseline["workspace"]["revision"],
        "forced_recovery_snapshot_is_last_committed": recovered["workspace"][
            "snapshot_version"
        ]
        == baseline["workspace"]["snapshot_version"],
        "uncommitted_transaction_was_orphaned": bool(
            recovered["recovery"]["orphaned_transactions"]
        ),
        "incremental_vs_full_verification_ok": verification.get("status") == "ok",
        "injection_error_codes": expected_errors,
        "all_injection_cases_rejected": all(
            code is not None for code in expected_errors.values()
        ),
    }

    report = dict(recovered)
    report["schema"] = DEMO_SCHEMA
    report["sample"] = {
        "seed": seed,
        "sha256": sha256_obj(sample),
        "material_count": len(sample["materials"]),
        "light_count": len(sample["lights"]),
        "observation_count": 81,
        "wavelength_range_nm": [380, 780],
        "wavelength_step_nm": 5,
    }
    report["operations_sha256"] = sha256_obj(operations)
    report["recovery_comparison"] = comparison
    report["injection_checks"] = injection_checks
    report["acceptance"] = acceptance
    report["determinism"] = {
        "timestamps_omitted": True,
        "absolute_paths_omitted": True,
        "report_serialization": "canonical-sorted-json-pretty-v1",
    }

    destination = Path(report_path)
    atomic_write_json(destination, report, pretty=True)
    recovered_engine.close()
    return report


def load_operations(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    document = read_json(path)
    if not isinstance(document, list):
        raise ValueError("operations JSON must be an array")
    return document


def workspace_report(root: str | os.PathLike[str]) -> dict[str, Any]:
    engine = open_workspace(root)
    try:
        return build_report(engine)
    finally:
        engine.close()
