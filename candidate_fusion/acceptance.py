"""End-to-end local acceptance matrix."""

from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .diagnostics import sort_failures
from .edits import apply_edit_sequence
from .engine import run_workspace
from .io_utils import read_json, write_json
from .samples import generate_samples
from .snapshot import SimulatedCrash, SnapshotStore


CORE_REPORT_KEYS = (
    "config_echo",
    "failures",
    "fusion_summary",
    "injections_hit",
    "normalization_events",
    "observability",
    "quantization_evidence",
)


def _copy_workspace(source: Path, destination: Path) -> None:
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination)


def _report_diff(left: dict[str, Any], right: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    return [key for key in keys if left.get(key) != right.get(key)]


def _snapshot_projection(report: dict[str, Any]) -> dict[str, Any]:
    snapshot = report.get("snapshot") or {}
    return {
        "config_hash": snapshot.get("config_hash"),
        "stale": snapshot.get("stale"),
        "struct_hash": snapshot.get("struct_hash"),
    }


def _save_report(path: Path, report: dict[str, Any]) -> None:
    write_json(path, report)


def _truthfulness_pair(before: dict[str, Any], after: dict[str, Any], expected_label: str) -> dict[str, Any]:
    before_evidence = before["quantization_evidence"]
    after_evidence = after["quantization_evidence"]
    before_config = before["config_echo"]
    after_config = after["config_echo"]
    if expected_label == "normalization":
        label_changed = before_config["normalization"] != after_config["normalization"]
    else:
        label_changed = (
            before_config["quant_bits"] != after_config["quant_bits"]
            and before_evidence["quant_step"] != after_evidence["quant_step"]
        )
    result = {
        "after": {
            "fused_checksum": after_evidence["fused_checksum"],
            "kept_count": after["fusion_summary"]["kept_count"],
            "normalization": after_config["normalization"],
            "quant_bits": after_config["quant_bits"],
            "quant_step": after_evidence["quant_step"],
        },
        "before": {
            "fused_checksum": before_evidence["fused_checksum"],
            "kept_count": before["fusion_summary"]["kept_count"],
            "normalization": before_config["normalization"],
            "quant_bits": before_config["quant_bits"],
            "quant_step": before_evidence["quant_step"],
        },
        "fixed_integers_changed": before_evidence["fixed_checksum"] != after_evidence["fixed_checksum"],
        "fused_scores_changed": before_evidence["fused_checksum"] != after_evidence["fused_checksum"],
        "label_changed": label_changed,
        "truncation_result_changed": before["fusion_summary"]["kept"] != after["fusion_summary"]["kept"],
    }
    result["passed"] = all(
        result[key]
        for key in (
            "fixed_integers_changed",
            "fused_scores_changed",
            "label_changed",
            "truncation_result_changed",
        )
    )
    return result


def _check(report: dict[str, Any]) -> None:
    ordered = sort_failures(report.get("failures") or [])
    if report.get("failures") != ordered:
        raise AssertionError("failure list is not in deterministic order")


def _configure_budget_case(source_workspace: Path, destination: Path, quota: int, budget: int) -> None:
    _copy_workspace(source_workspace, destination)
    config_path = destination / "config.json"
    config = read_json(config_path)
    config["channel_quota"] = quota
    config["global_budget"] = budget
    write_json(config_path, config)


def run_acceptance(
    samples_root: Path | str,
    output_root: Path | str,
    *,
    force_fixtures: bool = False,
) -> dict[str, Any]:
    samples_root = Path(samples_root).resolve()
    output_root = Path(output_root).resolve()
    generate_samples(samples_root, force=force_fixtures)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    work_root = output_root / "work"
    reports_root = output_root / "reports"
    injections_output = output_root / "injections"
    budget_output = output_root / "budget"
    snapshot_output = output_root / "snapshot"
    for directory in (work_root, reports_root, injections_output, budget_output, snapshot_output):
        directory.mkdir(parents=True, exist_ok=True)

    base_workspace = samples_root / "base_workspace"
    sequence_path = samples_root / "edits" / "sequence.json"
    sequence = read_json(sequence_path)
    edit_ids = [str(edit["id"]) for edit in sequence["edits"]]

    checks: dict[str, Any] = {}
    sequence_workspace = work_root / "sequence_workspace"
    _copy_workspace(base_workspace, sequence_workspace)
    initial = run_workspace(sequence_workspace, report_path=reports_root / "00_initial_full.json").report
    _check(initial)
    reports_by_edit: dict[str, dict[str, Any]] = {}
    step_results: list[dict[str, Any]] = []
    for index, edit_id in enumerate(edit_ids, start=1):
        apply_edit_sequence(sequence_workspace, sequence_path, edit_ids={edit_id})
        incremental = run_workspace(
            sequence_workspace,
            report_path=reports_root / f"{index:02d}_{edit_id}_incremental.json",
        ).report
        _check(incremental)
        reports_by_edit[edit_id] = incremental

        full_workspace = work_root / f"full_{edit_id}"
        _copy_workspace(sequence_workspace, full_workspace)
        shutil.rmtree(full_workspace / "state", ignore_errors=True)
        full = run_workspace(
            full_workspace,
            report_path=reports_root / f"{index:02d}_{edit_id}_clean_full.json",
        ).report
        _check(full)
        core_diff = _report_diff(incremental, full, CORE_REPORT_KEYS)
        step_results.append(
            {
                "core_mismatches": core_diff,
                "edit_id": edit_id,
                "full_budget_consumed": full["budget"]["consumed"],
                "incremental_budget_consumed": incremental["budget"]["consumed"],
                "invalidation_counts": incremental["invalidation"]["counts"],
                "mode": incremental["mode"],
                "run_version": incremental["run_version"],
            }
        )
        if core_diff:
            raise AssertionError(f"incremental/full core mismatch after {edit_id}: {core_diff}")

    final_incremental = reports_by_edit[edit_ids[-1]]
    final_full = read_json(reports_root / f"{len(edit_ids):02d}_{edit_ids[-1]}_clean_full.json")
    final_core_diff = _report_diff(final_incremental, final_full, CORE_REPORT_KEYS)
    final_audit_equal = final_incremental["invalidation"] == final_full["invalidation"]
    final_budget_equal = final_incremental["budget"] == final_full["budget"]
    final_snapshot_equal = _snapshot_projection(final_incremental) == _snapshot_projection(final_full)
    checks["incremental_full_equivalence"] = {
        "all_intermediate_core_fields_equal": all(not item["core_mismatches"] for item in step_results),
        "final_audit_equal": final_audit_equal,
        "final_budget_equal": final_budget_equal,
        "final_core_mismatches": final_core_diff,
        "final_invalidation_equal": final_audit_equal,
        "final_snapshot_projection_equal": final_snapshot_equal,
        "passed": not final_core_diff and final_audit_equal and final_budget_equal and final_snapshot_equal,
    }
    if not checks["incremental_full_equivalence"]["passed"]:
        raise AssertionError("final incremental/full equivalence failed")

    final_snapshot_dir = output_root / "final_snapshot"
    final_snapshot_dir.mkdir()
    _copy_workspace(sequence_workspace / "state", final_snapshot_dir / "state")
    shutil.copy2(sequence_workspace / "config.json", final_snapshot_dir / "config.json")

    checks["truthfulness"] = {
        "normalization_switch": _truthfulness_pair(
            reports_by_edit["E04"],
            reports_by_edit["E05"],
            "normalization",
        ),
        "quantization_switch": _truthfulness_pair(
            reports_by_edit["E06"],
            reports_by_edit["E07"],
            "quantization",
        ),
    }
    checks["truthfulness"]["passed"] = all(
        checks["truthfulness"][key]["passed"] for key in ("normalization_switch", "quantization_switch")
    )
    if not checks["truthfulness"]["passed"]:
        raise AssertionError("quantization/normalization truthfulness check failed")

    # Full recomputation twice from an identical final workspace must produce
    # byte-for-byte equivalent fields, including fixed integers and residuals.
    determinism_workspace = work_root / "determinism_workspace"
    _copy_workspace(sequence_workspace, determinism_workspace)
    deterministic_reports: list[dict[str, Any]] = []
    deterministic_normalize: list[dict[str, Any]] = []
    for index in range(2):
        report = run_workspace(
            determinism_workspace,
            reset=True,
            report_path=reports_root / f"determinism_{index}.json",
        ).report
        deterministic_reports.append(report)
        deterministic_normalize.append(SnapshotStore(determinism_workspace).load().stages["normalize"])
    determinism_equal = deterministic_reports[0] == deterministic_reports[1]
    normalize_equal = deterministic_normalize[0] == deterministic_normalize[1]
    checks["determinism"] = {
        "fixed_and_residual_states_equal": normalize_equal,
        "reports_equal": determinism_equal,
        "passed": determinism_equal and normalize_equal,
    }
    if not checks["determinism"]["passed"]:
        raise AssertionError("determinism check failed")

    injection_results: list[dict[str, Any]] = []
    for index in range(1, 5):
        injection_id = f"INJ-{index:03d}"
        injection_path = samples_root / "injections" / f"{injection_id}.json"
        injection_workspace = work_root / f"injection_{injection_id}"
        _copy_workspace(base_workspace, injection_workspace)
        report = run_workspace(
            injection_workspace,
            injection_path=injection_path,
            report_path=injections_output / f"{injection_id}.json",
        ).report
        _check(report)
        injection_spec = read_json(injection_path)["injections"][0]
        expected_code = {
            "nonfinite_score": "E_NONFINITE_SCORE",
            "duplicate_candidate": "E_CONFIG_INVALID",
            "trunc_limit_out_of_range": "E_CONFIG_INVALID",
            "zero_capacity": "E_CONFIG_INVALID",
        }[injection_spec["type"]]
        hit = next((item for item in report["injections_hit"] if item["id"] == injection_id), None)
        mapped_failure = next(
            (
                item
                for item in report["failures"]
                if item["injection_id"] == injection_id and item["code"] == expected_code
            ),
            None,
        )
        isolated_processing = (
            report["mode"] == "aborted"
            if injection_spec["stage"] == "config"
            else report["mode"] in {"full", "incremental"}
            and report["fusion_summary"]["kept_count"] > 0
        )
        passed = bool(
            hit
            and hit.get("mapped_code") == expected_code
            and mapped_failure
            and isolated_processing
        )
        injection_results.append(
            {
                "code": expected_code,
                "failure": mapped_failure,
                "hit": hit,
                "id": injection_id,
                "isolated_processing": isolated_processing,
                "passed": passed,
                "type": injection_spec["type"],
            }
        )
        if not passed:
            raise AssertionError(f"injection {injection_id} did not map as expected")
    checks["injections"] = {
        "cases": injection_results,
        "passed": all(item["passed"] for item in injection_results),
    }

    budget_cases: list[dict[str, Any]] = []
    for name, quota, budget in (
        ("channel_quota_20", 20, 100),
        ("global_budget_50", 10000, 50),
        ("channel_quota_zero", 0, 10),
    ):
        reports: list[dict[str, Any]] = []
        for repeat in range(2):
            workspace = work_root / f"budget_{name}_{repeat}"
            _configure_budget_case(base_workspace, workspace, quota, budget)
            report = run_workspace(
                workspace,
                reset=True,
                report_path=budget_output / f"{name}_run_{repeat}.json",
            ).report
            _check(report)
            reports.append(report)
        budget_report = reports[0]["budget"]
        stable_fields = ("audit", "consumed", "pending_queue", "remaining")
        stable = all(reports[0]["budget"][field] == reports[1]["budget"][field] for field in stable_fields)
        audit = budget_report["audit"]
        channels_in_audit = [item["channel"] for item in audit]
        limits_respected = bool(
            budget_report["consumed"] <= budget
            and budget_report["remaining"] >= 0
            and all(item["granted"] <= quota for item in audit)
            and budget_report["concurrency"]["max_observed"] <= budget_report["max_workers"]
        )
        failure_codes = sorted({item["code"] for item in reports[0]["failures"]})
        if name == "channel_quota_20":
            expected_failures = ["E_CHANNEL_QUOTA_EXHAUSTED"]
        elif name == "global_budget_50":
            expected_failures = ["E_GLOBAL_BUDGET_EXHAUSTED"]
        else:
            expected_failures = ["E_CHANNEL_QUOTA_EXHAUSTED"]
        passed = (
            stable
            and limits_respected
            and channels_in_audit == sorted(channels_in_audit)
            and failure_codes == expected_failures
            and bool(budget_report["pending_queue"])
        )
        budget_cases.append(
            {
                "audit_stable": stable,
                "channel_quota": quota,
                "consumed": budget_report["consumed"],
                "failure_codes": failure_codes,
                "global_budget": budget,
                "max_observed": budget_report["concurrency"]["max_observed"],
                "max_workers": budget_report["max_workers"],
                "name": name,
                "passed": passed,
                "pending_queue_size": len(budget_report["pending_queue"]),
                "remaining": budget_report["remaining"],
            }
        )
        if not passed:
            raise AssertionError(f"budget case {name} failed")
    checks["budget"] = {
        "cases": budget_cases,
        "passed": all(item["passed"] for item in budget_cases),
    }

    snapshot_workspace = work_root / "snapshot_workspace"
    _copy_workspace(base_workspace, snapshot_workspace)
    snapshot_initial = run_workspace(
        snapshot_workspace,
        report_path=snapshot_output / "initial.json",
    ).report
    initial_version = SnapshotStore(snapshot_workspace).load().manifest_version
    noop = run_workspace(
        snapshot_workspace,
        report_path=snapshot_output / "noop_rerun.json",
    ).report
    noop_all_reused = (
        noop["invalidation"]["counts"]["invalidated_candidates"] == 0
        and noop["invalidation"]["counts"]["invalidated_fusions"] == 0
        and noop["invalidation"]["counts"]["reused_candidates"] > 0
        and noop["invalidation"]["counts"]["reused_fusions"] > 0
        and noop["budget"]["consumed"] == 0
    )
    crashed = False
    try:
        run_workspace(snapshot_workspace, simulate_crash_after="normalize")
    except SimulatedCrash:
        crashed = True
    after_crash = SnapshotStore(snapshot_workspace).load()
    crash_preserved = (
        crashed
        and after_crash.valid
        and after_crash.manifest_version == initial_version + 1
        and after_crash.ignored_temp_snapshots == 1
    )
    recovered = run_workspace(
        snapshot_workspace,
        report_path=snapshot_output / "after_crash_recovery.json",
    ).report
    recovery_ok = (
        recovered["mode"] == "incremental"
        and recovered["snapshot"]["resumed"] is True
        and recovered["snapshot"]["stale"] is False
        and recovered["invalidation"]["counts"]["invalidated_candidates"] == 0
    )

    current_name = (snapshot_workspace / "state" / "CURRENT").read_text(encoding="utf-8").strip()
    load_stage_path = snapshot_workspace / "state" / "snapshots" / current_name / "load.json"
    tampered = read_json(load_stage_path)
    tampered["tampered_by_acceptance"] = True
    write_json(load_stage_path, tampered)
    stale_report = run_workspace(
        snapshot_workspace,
        report_path=snapshot_output / "stale_recovery.json",
    ).report
    stale_detected = (
        stale_report["snapshot"]["stale"] is True
        and any(item["code"] == "E_SNAPSHOT_STALE" for item in stale_report["failures"])
        and stale_report["invalidation"]["counts"]["invalidated_candidates"] > 0
        and stale_report["budget"]["consumed"] > 0
    )
    expected_version_report = run_workspace(
        snapshot_workspace,
        expected_version=999,
        report_path=snapshot_output / "expected_version_mismatch.json",
    ).report
    expected_version_stale = (
        expected_version_report["snapshot"]["stale"] is True
        and any(item["code"] == "E_SNAPSHOT_STALE" for item in expected_version_report["failures"])
    )
    checks["snapshot"] = {
        "config_edit_remains_resumable": reports_by_edit["E03"]["snapshot"]["resumed"] is True
        and reports_by_edit["E03"]["snapshot"]["stale"] is False,
        "crash_preserved_last_commit": crash_preserved,
        "expected_version_mismatch_detected": expected_version_stale,
        "noop_all_reused": noop_all_reused,
        "recovery_from_committed_state": recovery_ok,
        "stale_detected_and_rebuilt": stale_detected,
    }
    checks["snapshot"]["passed"] = all(value for value in checks["snapshot"].values())
    if not checks["snapshot"]["passed"]:
        raise AssertionError("snapshot recovery checks failed")

    trunc_workspace = work_root / "trunc_zero_workspace"
    _copy_workspace(base_workspace, trunc_workspace)
    trunc_config_path = trunc_workspace / "config.json"
    trunc_config = read_json(trunc_config_path)
    trunc_config["trunc_k"] = 0
    write_json(trunc_config_path, trunc_config)
    trunc_report = run_workspace(
        trunc_workspace,
        report_path=reports_root / "trunc_zero.json",
    ).report
    trunc_zero_ok = (
        trunc_report["fusion_summary"]["kept"] == []
        and any(item["code"] == "E_TRUNC_ZERO" for item in trunc_report["failures"])
    )
    checks["trunc_zero"] = {"passed": trunc_zero_ok}
    if not trunc_zero_ok:
        raise AssertionError("trunc_k=0 behavior failed")

    all_passed = all(
        bool(value.get("passed")) if isinstance(value, dict) else bool(value)
        for value in checks.values()
    )
    summary = {
        "artifacts": {
            "final_snapshot": str(final_snapshot_dir),
            "reports": str(reports_root),
            "summary": str(output_root / "summary.json"),
        },
        "checks": checks,
        "environment": {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "fixture": read_json(samples_root / "fixture.json"),
        "sequence": {
            "edit_ids": edit_ids,
            "final": {
                "budget_consumed": final_incremental["budget"]["consumed"],
                "invalidation_counts": final_incremental["invalidation"]["counts"],
                "kept_count": final_incremental["fusion_summary"]["kept_count"],
                "run_version": final_incremental["run_version"],
                "total_candidates": final_incremental["fusion_summary"]["total_candidates"],
            },
            "initial": {
                "budget_consumed": initial["budget"]["consumed"],
                "kept_count": initial["fusion_summary"]["kept_count"],
                "total_candidates": initial["fusion_summary"]["total_candidates"],
            },
            "steps": step_results,
        },
        "status": "passed" if all_passed else "failed",
    }
    write_json(output_root / "summary.json", summary)
    shutil.rmtree(work_root, ignore_errors=True)
    return summary
