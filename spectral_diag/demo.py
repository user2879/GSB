"""Fixed local acceptance sequence with reproducible evidence capture."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import canonical_bytes, sha256_obj
from .closure import ConventionSession
from .config import RunConfig
from .matrix import solve_adaptation
from .pipeline import execute_pipeline
from .scene import generate_scene
from .storage import Workspace, atomic_write_bytes


def _utc_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _commit_run(workspace: Workspace, run_id: str, result: dict[str, Any]) -> dict[str, Any]:
    workspace.begin(run_id)
    result = dict(result)
    result["run_id"] = run_id
    return workspace.stage_and_commit(run_id, result)


def _matrix_roundtrip(scene: Any) -> dict[str, Any]:
    known = np.array(
        [
            [0.91, 0.08, -0.02],
            [0.03, 0.94, 0.02],
            [-0.01, 0.05, 0.88],
        ],
        dtype=float,
    )
    target = scene.source_xyz @ known
    solution = solve_adaptation(scene.source_xyz, target, cond_threshold=1e4)
    error = float(np.max(np.abs(solution.matrix - known))) if solution.matrix is not None else None
    return {
        "accepted": solution.accepted,
        "max_abs_matrix_error": error,
        "residual_rmse": solution.residual_rmse,
        "condition_number": solution.condition_number,
        "passed": bool(solution.accepted and error is not None and error < 1e-10),
    }


def _pruning_case(scene: Any) -> dict[str, Any]:
    source = scene.source_xyz.copy()
    target = scene.target_xyz.copy()
    source[3] = [1.0e7, 2.0e7, 5.0e6]
    target[3] = [1.1e7, 1.9e7, 4.8e6]
    solution = solve_adaptation(source, target, cond_threshold=1e4)
    trace = [step.to_dict() for step in solution.prune_trace]
    return {
        "status": solution.status,
        "accepted": solution.accepted,
        "condition_before": trace[0]["condition_before"] if trace else solution.condition_number,
        "condition_after": trace[-1]["condition_after"] if trace else solution.condition_number,
        "removed_indices": [step["removed_index"] for step in trace],
        "trace": trace,
        "passed": bool(
            solution.accepted
            and solution.status == "accepted_after_pruning"
            and trace
            and trace[0]["removed_index"] == 3
            and trace[-1]["condition_after"] <= 1e4
        ),
    }


def _anomaly_cases(config: RunConfig) -> list[dict[str, Any]]:
    cases = [
        ("collinear", {"kind": "collinear_sample", "params": {}}, "E_MATRIX_SINGULAR"),
        ("white-point", {"kind": "zero_white_point", "params": {"component": 1}}, "E_WHITE_POINT"),
        (
            "quant-range",
            {"kind": "quant_out_of_range", "params": {"sample_index": 7, "component": 1}},
            "E_QUANT_RANGE",
        ),
        ("observer", {"kind": "observer_invalid", "params": {"observer": "cie1955_2"}}, "E_OBSERVER_INVALID"),
        (
            "density",
            {"kind": "low_integration_density", "params": {"density_nm": 20.0}},
            "E_INTEGRATION_DENSITY",
        ),
    ]
    results: list[dict[str, Any]] = []
    for name, injection, expected in cases:
        case_config = replace(config, injections=(injection,))
        result = execute_pipeline(case_config)
        error_code = None if not result.get("error") else result["error"]["code"]
        first = result.get("first_affected")
        results.append(
            {
                "name": name,
                "config": case_config.to_dict(),
                "injection": injection,
                "expected_error_code": expected,
                "actual_error_code": error_code,
                "first_affected": first,
                "result_sha256": result.get("result_sha256"),
                "result": result,
                "passed": bool(result.get("status") == "rejected" and error_code == expected and first is not None),
            }
        )
    return results


def _closure_case(config: RunConfig) -> dict[str, Any]:
    session = ConventionSession(config)
    first = session.run()
    same = session.run()
    switched = session.run(observer="cie1964_10")
    restored = session.run(observer=config.observer, normalization=config.normalization)
    return {
        "initial_recomputed": first["report"]["recomputed"],
        "same_convention_recomputed": same["report"]["recomputed"],
        "switched_recomputed": switched["report"]["recomputed"],
        "restored_recomputed": restored["report"]["recomputed"],
        "restored_cache_hits": restored["report"]["cache_hits"],
        "initial_hashes": first["report"]["hashes"],
        "switched_hashes": switched["report"]["hashes"],
        "restored_hashes": restored["report"]["hashes"],
        "scene_hash_preserved": first["scene_sha256"] == switched["scene_sha256"],
        "sample_hash_changed_on_switch": first["sample_source_xyz_sha256"]
        != switched["sample_source_xyz_sha256"],
        "hashes_restored": first["report"]["hashes"] == restored["report"]["hashes"],
        "closure_reconciled": all(
            item["report"]["closure_reconciled"] for item in (first, same, switched, restored)
        ),
        "passed": bool(
            same["report"]["recomputed"] == []
            and restored["report"]["recomputed"] == []
            and first["report"]["hashes"] == restored["report"]["hashes"]
            and first["scene_sha256"] == switched["scene_sha256"]
            and all(item["report"]["closure_reconciled"] for item in (first, same, switched, restored))
        ),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    checks = report["checks"]
    lines = [
        "# Spectral diagnostics verification",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Workspace: `{report['workspace']}`",
        "",
        "| Check | Result | Evidence |",
        "|---|---|---|",
    ]
    evidence = report["evidence"]
    rows = [
        ("Deterministic scene bytes", checks["scene_deterministic"], evidence["scene"]["sha256"]),
        ("Deterministic pipeline result", checks["pipeline_deterministic"], evidence["pipeline"]["result_sha256"]),
        (
            "Exact matrix round-trip",
            checks["matrix_roundtrip"],
            f"max error {evidence['matrix_roundtrip']['max_abs_matrix_error']}",
        ),
        (
            "Ill-conditioned pruning",
            checks["pruning"],
            f"removed {evidence['pruning']['removed_indices']}",
        ),
        (
            "Attribution additivity",
            checks["attribution"],
            f"sum={evidence['attribution']['sum']} total={evidence['attribution']['total']}",
        ),
        (
            "Convention invalidation closure",
            checks["closure"],
            f"restored cache hits {len(evidence['closure']['restored_cache_hits'])}",
        ),
        (
            "Anomaly detection and first location",
            checks["anomalies"],
            ", ".join(item["actual_error_code"] for item in evidence["anomalies"]),
        ),
        (
            "Forced termination recovery",
            checks["crash_recovery"],
            f"orphans={evidence['crash_recovery']['orphan_count']}",
        ),
    ]
    for label, passed, detail in rows:
        lines.append(f"| {label} | {'PASS' if passed else 'FAIL'} | `{detail}` |")
    lines.extend(
        [
            "",
            "## Error codes",
            "",
            (
                "`E_MATRIX_SINGULAR`, `E_QUANT_RANGE`, `E_WHITE_POINT`, `E_OBSERVER_INVALID`, "
                "`E_INTEGRATION_DENSITY`, `E_ATTRIB_MISMATCH`, `E_SNAPSHOT_VERSION`."
            ),
            "",
            "## Attribution method",
            "",
            (
                "Three-factor Shapley allocation makes the quantisation, sample-distribution and "
                "integration contributions add exactly to `combined - baseline`. Standalone deltas and "
                "the interaction allocation are retained in the ledger for audit."
            ),
        ]
    )
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


def run_fixed_sequence(workspace_path: Path, *, reset: bool = False) -> dict[str, Any]:
    workspace_path = workspace_path.resolve()
    if reset and workspace_path.exists():
        shutil.rmtree(workspace_path)
    workspace = Workspace(workspace_path).ensure()
    if workspace.active_path.exists():
        workspace.recover()

    config = RunConfig()
    session = _utc_tag()
    scene_a = generate_scene(config)
    scene_b = generate_scene(config)
    pipeline_a = execute_pipeline(config)
    pipeline_b = execute_pipeline(config)

    matrix_roundtrip = _matrix_roundtrip(scene_a)
    pruning = _pruning_case(scene_a)
    closure = _closure_case(config)
    anomalies = _anomaly_cases(config)

    stable_run_id = f"stable-{session}"
    stable_commit = _commit_run(workspace, stable_run_id, pipeline_a)
    stable_result_sha = pipeline_a["result_sha256"]
    anomaly_commits = []
    for index, anomaly in enumerate(anomalies):
        anomaly_run_id = f"inject-{index}-{session}"
        anomaly_commits.append(_commit_run(workspace, anomaly_run_id, anomaly["result"]))

    # Re-commit a clean accepted state so the crash test starts from an
    # unambiguous normal-close snapshot.
    final_stable_run_id = f"stable-final-{session}"
    final_commit = _commit_run(workspace, final_stable_run_id, pipeline_b)
    final_result_sha = pipeline_b["result_sha256"]

    crash_run_id = f"crash-{session}"
    crash_config_path = workspace_path / "crash-config.json"
    atomic_write_bytes(crash_config_path, canonical_bytes(config.to_dict()))
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root) + os.pathsep + environment.get("PYTHONPATH", "")
    crash_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "spectral_diag",
            "run",
            "--workspace",
            str(workspace_path),
            "--run-id",
            crash_run_id,
            "--crash-after-stage",
            "--config-json",
            str(crash_config_path),
        ],
        cwd=str(project_root),
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    recovery = workspace.recover()
    current = workspace.load_current()
    current_snapshot = recovery["current"]["snapshot_id"] if recovery["current"] else None
    crash_passed = bool(
        crash_process.returncode == 86
        and current_snapshot == final_commit["snapshot_id"]
        and recovery["orphan_count"] >= 1
        and recovery["interrupted_run"] is not None
        and recovery["interrupted_run"]["run_id"] == crash_run_id
        and recovery["normal_exit_report_fabricated"] is False
        and recovery["current"]["result_sha256"] == final_result_sha
    )

    attribution = pipeline_a["attribution"]
    attribution_evidence = {
        "sum": attribution["additivity"]["sum_contributions"],
        "total": attribution["additivity"]["total_delta"],
        "baseline": attribution["residuals"]["baseline"],
        "combined": attribution["residuals"]["combined"],
        "reconciled": attribution["additivity"]["reconciled"],
    }

    checks = {
        "scene_deterministic": scene_a.canonical_bytes == scene_b.canonical_bytes,
        "pipeline_deterministic": pipeline_a["result_sha256"] == pipeline_b["result_sha256"],
        "matrix_roundtrip": matrix_roundtrip["passed"],
        "pruning": pruning["passed"],
        "attribution": bool(attribution["additivity"]["reconciled"]),
        "closure": closure["passed"],
        "anomalies": all(item["passed"] for item in anomalies),
        "crash_recovery": crash_passed,
        "stable_result_preserved": stable_result_sha == final_result_sha,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "workspace": str(workspace_path),
        "status": "passed" if all(checks.values()) else "failed",
        "all_passed": all(checks.values()),
        "checks": checks,
        "evidence": {
            "config": config.to_dict(),
            "scene": {
                "sha256": scene_a.sha256,
                "reference_truth": scene_a.payload["reference_truth"],
                "byte_length": len(scene_a.canonical_bytes),
            },
            "pipeline": {
                "status": pipeline_a["status"],
                "result_sha256": pipeline_a["result_sha256"],
                "matrix_condition_number": pipeline_a["matrix"]["condition_number"],
                "matrix_residual_rmse": pipeline_a["matrix"]["residual_rmse"],
            },
            "matrix_roundtrip": matrix_roundtrip,
            "pruning": pruning,
            "attribution": attribution_evidence,
            "closure": closure,
            "anomalies": [
                {
                    "name": item["name"],
                    "injection": item["injection"],
                    "expected_error_code": item["expected_error_code"],
                    "actual_error_code": item["actual_error_code"],
                    "first_affected": item["first_affected"],
                    "passed": item["passed"],
                    "result_sha256": item["result_sha256"],
                }
                for item in anomalies
            ],
            "persistence": {
                "stable_snapshot": stable_commit,
                "final_stable_snapshot": final_commit,
                "anomaly_snapshots": anomaly_commits,
                "crash_process_returncode": crash_process.returncode,
                "crash_stdout": crash_process.stdout,
                "crash_stderr": crash_process.stderr,
                "recovery": recovery,
                "current_snapshot": current_snapshot,
                "current_closed": None if current is None else current["manifest"]["closed"],
            },
            "crash_recovery": {
                "passed": crash_passed,
                "crash_process_returncode": crash_process.returncode,
                "orphan_count": recovery["orphan_count"],
                "recovered_snapshot_id": current_snapshot,
                "expected_snapshot_id": final_commit["snapshot_id"],
                "normal_exit_report_fabricated": recovery["normal_exit_report_fabricated"],
            },
        },
    }
    report_path = workspace_path / "demo-report.json"
    markdown_path = workspace_path / "VERIFICATION.md"
    report["report_path"] = str(report_path)
    report["markdown_path"] = str(markdown_path)
    report["report_sha256"] = sha256_obj({key: value for key, value in report.items() if key != "report_sha256"})
    atomic_write_bytes(report_path, canonical_bytes(report))
    _write_markdown(report, markdown_path)
    return report
