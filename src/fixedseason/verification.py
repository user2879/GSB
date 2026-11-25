"""Executable acceptance verification and evidence report."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .commands import CommandProcessor
from .engine import FitEngine
from .errors import E_CONFIG_INVALID, E_SERIES_INVALID
from .experiment import compare_reports, run_fresh_build, run_incremental_experiment
from .jsonutil import write_json_atomic, write_text_atomic
from .loader import load_config, load_series_jsonl
from .model import FitConfig, SeriesRecord
from .synthetic import generate_example_workspace
from .diagnostics import run_width_sign_check


def _series(series_id: str, length: int, base: float = 100.0) -> SeriesRecord:
    return SeriesRecord(
        series_id=series_id,
        timestamps=tuple(1000.0 + index for index in range(length)),
        values=tuple(
            base + 0.4 * index + (2.0 if index % 4 == 0 else -1.0)
            for index in range(length)
        ),
    )


def _coefficient_actions(report: dict[str, Any]) -> dict[str, str]:
    return {
        item["series_id"]: item["action"]
        for item in report["cache_ledger"]
        if item["stage"] == "coefficients"
    }


def _check_fields_equal(
    left: dict[str, Any], right: dict[str, Any], fields: Iterable[str]
) -> tuple[bool, list[dict[str, Any]]]:
    differences = []
    for field in fields:
        if left.get(field) != right.get(field):
            differences.append(
                {"field": field, "left": left.get(field), "right": right.get(field)}
            )
    return not differences, differences


def run_verification(
    workspace: str | Path,
    *,
    fixture_dir: str | Path | None = None,
    fresh: bool = True,
) -> dict[str, Any]:
    workspace = Path(workspace)
    if fresh and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    fixture_root = Path(fixture_dir) if fixture_dir is not None else workspace / "fixtures"
    fixture_paths = generate_example_workspace(fixture_root, force=True)
    series, series_issues = load_series_jsonl(fixture_paths["series"])
    config, config_issues = load_config(fixture_paths["config"])
    if config is None:
        raise RuntimeError(f"fixture config unexpectedly invalid: {config_issues}")
    checks: dict[str, Any] = {}

    invalid_path = workspace / "invalid-series.jsonl"
    write_text_atomic(
        invalid_path,
        '{"series_id":"good","timestamps":[0,1,2],"values":[1,2,3]}\n'
        '{"series_id":"bad","timestamps":[0,1,3],"values":[1,2,3]}\n',
    )
    valid_records, invalid_issues = load_series_jsonl(invalid_path)
    checks["failure_isolation"] = {
        "passed": len(valid_records) == 1
        and len(invalid_issues) == 1
        and invalid_issues[0].code == E_SERIES_INVALID,
        "valid_series_retained": [record.series_id for record in valid_records],
        "issue_codes": [issue.code for issue in invalid_issues],
    }

    bad_config_path = workspace / "bad-config.yaml"
    write_text_atomic(
        bad_config_path,
        "model:\n  harmonic_order: 99\nfixed_point:\n  bits: 31\nruntime:\n"
        f"  thread_pool_workers: {(os.cpu_count() or 1) + 1}\n",
    )
    _, bad_config_issues = load_config(bad_config_path)
    checks["config_validation"] = {
        "passed": bool(bad_config_issues)
        and all(issue.code == E_CONFIG_INVALID for issue in bad_config_issues),
        "issue_codes": [issue.code for issue in bad_config_issues],
        "messages": [issue.message for issue in bad_config_issues],
    }

    processor = CommandProcessor(series, config)
    invalid_command_skipped = not processor.apply({"op": "unknown"}, line=1)
    later_command_applied = processor.apply(
        {"op": "set_harmonic_order", "value": 2}, line=2
    )
    checks["command_validation"] = {
        "passed": invalid_command_skipped
        and later_command_applied
        and any(issue.code == "E_COMMAND_INVALID" for issue in processor.issues),
        "invalid_command_skipped": invalid_command_skipped,
        "later_command_applied": later_command_applied,
        "issue_codes": [issue.code for issue in processor.issues],
    }

    serial_workspace = workspace / "checks" / "serial"
    concurrent_workspace = workspace / "checks" / "concurrent"
    serial_report = FitEngine(
        serial_workspace, config, series, mode="serial"
    ).run(version=config.version, force_rerun=True)
    concurrent_report = FitEngine(
        concurrent_workspace, config, series, mode="concurrent"
    ).run(version=config.version, force_rerun=True)
    concurrency_fields = (
        "series",
        "cache_ledger",
        "budget",
        "pending",
        "failures",
        "reuse_ledger",
    )
    concurrency_equal, concurrency_differences = _check_fields_equal(
        serial_report, concurrent_report, concurrency_fields
    )
    checks["serial_concurrent_equivalence"] = {
        "passed": concurrency_equal,
        "compared_fields": list(concurrency_fields),
        "differences": concurrency_differences[:10],
        "serial_summary": serial_report["summary"],
        "concurrent_summary": concurrent_report["summary"],
        "reports": {
            "serial": str(serial_workspace / "snapshot" / "current" / "report.json"),
            "concurrent": str(
                concurrent_workspace / "snapshot" / "current" / "report.json"
            ),
        },
    }

    incremental = run_incremental_experiment(
        workspace / "checks" / "incremental",
        series,
        config,
        initial_issues=series_issues + config_issues,
        commands_path=fixture_paths["commands"],
        mode="concurrent",
        fresh=True,
    )
    fresh_final = run_fresh_build(
        workspace / "checks" / "full-rebuild",
        incremental.final_series,
        incremental.final_config,
        validation_issues=incremental.issues,
        mode="concurrent",
    )
    incremental_equal, incremental_differences = compare_reports(
        incremental.final_report, fresh_final
    )
    replace_actions = _coefficient_actions(incremental.reports[2])
    checks["incremental_vs_full_rebuild"] = {
        "passed": incremental_equal,
        "differences": incremental_differences[:10],
        "steps": len(incremental.reports),
        "final_version": incremental.final_config.version,
        "invalid_command_count": sum(
            1 for issue in incremental.issues if issue.code == "E_COMMAND_INVALID"
        ),
        "single_series_invalidation_actions": replace_actions,
        "single_series_invalidation_passed": (
            replace_actions.get("sku-alpha") == "miss"
            and replace_actions.get("sku-bravo") == "hit"
        ),
        "reports": {
            "incremental_final": str(
                workspace
                / "checks"
                / "incremental"
                / "reports"
                / "step-009-v9.json"
            ),
            "fresh_final": str(
                workspace / "checks" / "full-rebuild" / "snapshot" / "current" / "report.json"
            ),
        },
    }

    budget_series = [_series("a", 10), _series("b", 12), _series("c", 8)]
    quota_config = FitConfig(
        harmonic_order=1,
        global_fit_points=100,
        per_series_quota=10,
        batch_observation_points=20,
        thread_pool_workers=2,
    )
    pending_config = FitConfig(
        harmonic_order=1,
        global_fit_points=20,
        per_series_quota=20,
        batch_observation_points=20,
        thread_pool_workers=2,
    )
    quota_report = FitEngine(
        workspace / "checks" / "budget-quota", quota_config, budget_series
    ).run(version=1, force_rerun=True)
    pending_report = FitEngine(
        workspace / "checks" / "budget-pending", pending_config, budget_series
    ).run(version=1, force_rerun=True)
    quota_codes = [issue["code"] for issue in quota_report["fit_failures"]]
    pending_codes = [issue["code"] for issue in pending_report["pending"]]
    checks["budget_constraints"] = {
        "passed": quota_codes == ["E_FIT_BUDGET_EXCEEDED"]
        and pending_codes == ["E_POINT_BUDGET_PENDING"]
        and pending_report["budget"]["batches"] == [["a", "c"]],
        "per_series_quota": {
            "completed_series": [item["series_id"] for item in quota_report["series"]],
            "failure_codes": quota_codes,
        },
        "global_and_batch_budget": {
            "completed_series": [item["series_id"] for item in pending_report["series"]],
            "pending_codes": pending_codes,
            "pending_series": [item["series_id"] for item in pending_report["pending"]],
            "batches": pending_report["budget"]["batches"],
            "consumed_fit_points": pending_report["budget"]["consumed_fit_points"],
        },
    }

    resume_series = [_series(f"r{index}", 20, base=80 + index) for index in range(5)]
    resume_config = FitConfig(
        harmonic_order=2,
        global_fit_points=200,
        per_series_quota=50,
        batch_observation_points=40,
        thread_pool_workers=2,
        version=1,
    )
    interrupted_workspace = workspace / "checks" / "resume-interrupted"
    one_shot_workspace = workspace / "checks" / "resume-one-shot"
    partial = FitEngine(
        interrupted_workspace, resume_config, resume_series, mode="concurrent"
    ).run(version=1, stop_after=2, force_rerun=True)
    resumed = FitEngine(
        interrupted_workspace, resume_config, resume_series, mode="concurrent"
    ).run(version=1)
    one_shot = FitEngine(
        one_shot_workspace, resume_config, resume_series, mode="concurrent"
    ).run(version=1, force_rerun=True)
    resume_equal, resume_differences = _check_fields_equal(
        resumed,
        one_shot,
        ("series", "cache_ledger", "budget", "pending", "failures", "reuse_ledger"),
    )
    checks["interrupt_resume"] = {
        "passed": resume_equal
        and not partial["run"]["complete"]
        and resumed["run"]["resumed"],
        "partial_unfinished": partial["unfinished"],
        "resumed": resumed["run"]["resumed"],
        "differences": resume_differences[:10],
    }

    snapshot_dir = serial_workspace / "snapshot" / "current"
    required_stage_files = [
        snapshot_dir / "stages" / f"{stage}.json"
        for stage in (
            "loading",
            "interpolation",
            "basis",
            "normal_equations",
            "elimination",
            "back_substitution",
        )
    ]
    checks["snapshot_contract"] = {
        "passed": all(path.exists() for path in required_stage_files)
        and (snapshot_dir / "manifest.json").exists()
        and (snapshot_dir / "report.json").exists(),
        "snapshot_dir": str(snapshot_dir),
        "manifest": str(snapshot_dir / "manifest.json"),
    }

    width_report = run_width_sign_check(
        workspace / "checks" / "widths",
        series,
        config,
        config.comparison_bit_widths or (32, 48),
        mode="concurrent",
    )
    flip_series = SeriesRecord(
        "near-zero-trend",
        tuple(float(index) for index in range(24)),
        (
            94.602294,
            93.29754,
            90.240536,
            88.28638,
            88.876584,
            91.439266,
            92.611021,
            91.488051,
            88.609308,
            87.652029,
            89.292745,
            92.347576,
            93.860986,
            92.93034,
            89.204505,
            87.376713,
            88.709751,
            91.223733,
            92.437088,
            90.972189,
            88.278939,
            87.334068,
            88.908298,
            92.054929,
        ),
    )
    flip_config = FitConfig(
        harmonic_order=2,
        fixed_point_bits=48,
        quantization_shift=8,
        pivot_threshold=0,
        global_fit_points=1000,
        per_series_quota=100,
        batch_observation_points=100,
        thread_pool_workers=2,
        version=1,
    )
    flip_report = run_width_sign_check(
        workspace / "checks" / "width-sign-flip",
        [flip_series],
        flip_config,
        [16, 48],
        mode="concurrent",
    )
    checks["width_sign_diagnostics"] = {
        "passed": bool(width_report["series_comparisons"])
        and "near-zero-trend" in flip_report["flipped_series"]
        and flip_report["series_comparisons"][0]["attributed_cause"]
        in ("interpolation", "normal_equation_quantization", "elimination_pivot", "fixed_point_rounding"),
        "widths": width_report["widths"],
        "flipped_series": width_report["flipped_series"],
        "comparison_count": len(width_report["series_comparisons"]),
        "synthetic_flip": {
            "widths": flip_report["widths"],
            "flipped_series": flip_report["flipped_series"],
            "comparison": flip_report["series_comparisons"][0],
        },
        "classification_contract": width_report["classification_contract"],
    }
    write_json_atomic(workspace / "width-sign-report.json", width_report)
    write_json_atomic(workspace / "width-sign-flip-report.json", flip_report)

    fixed_actions = _coefficient_actions(serial_report)
    repeat = FitEngine(
        serial_workspace, config, series, mode="serial"
    ).run(version=config.version, force_rerun=True)
    repeat_actions = _coefficient_actions(repeat)
    checks["fixed_point_cache_replay"] = {
        "passed": repeat["series"] == serial_report["series"]
        and all(action == "hit" for action in repeat_actions.values()),
        "repeat_actions": repeat_actions,
        "initial_actions": fixed_actions,
    }

    overall = all(item.get("passed") is True for item in checks.values())
    report = {
        "schema_version": 1,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
        },
        "workspace": str(workspace),
        "fixtures": fixture_paths,
        "checks": checks,
        "overall_passed": overall,
    }
    write_json_atomic(workspace / "verification-report.json", report)
    return report
