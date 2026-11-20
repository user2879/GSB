"""Reproducible experiment workflows built on top of the pipeline."""

from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .canonical import (
    atomic_write_json,
    canonical_json,
    first_difference,
    materialized_projection,
    projections_equal,
)
from .errors import PipelineConfigMismatchError, SimulatedCrash
from .inputs import InputState
from .pipeline import PipelineResult, run_pipeline
from .samplegen import generate_sample


def _fresh_directory(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _copy_state(state: InputState, directory: Path) -> None:
    _fresh_directory(directory)
    state.save_to(directory)


def _task_projection(report: dict[str, Any]) -> dict[str, Any]:
    """Materialized task fields used for concurrency equivalence checks."""
    task_results = []
    for task in report.get("task_results", []):
        clean = dict(task)
        clean.pop("source", None)
        task_results.append(clean)
    base = report.get("base")
    if isinstance(base, dict):
        base = {
            "fingerprint": base.get("fingerprint"),
            "layers": base.get("layers"),
        }
    return {
        "base": base,
        "task_results": task_results,
        "adapter_summary": report.get("adapter_summary", []),
        "loss_trajectories": report.get("loss_trajectories", []),
        "validation_metrics": report.get("validation_metrics", []),
        "validation_summary": report.get("validation_summary", {}),
        "failures": report.get("failures", []),
        "injection_hits": report.get("injection_hits", []),
        "injection_summary": report.get("injection_summary", {}),
        "threshold_failures": report.get("threshold_failures", []),
    }


def compare_reports(incremental: dict[str, Any], full: dict[str, Any]) -> dict[str, Any]:
    left = materialized_projection(incremental)
    right = materialized_projection(full)
    difference = first_difference(left, right)
    return {
        "exact_materialization_match": difference is None,
        "first_difference": difference,
        "incremental_task_count": len(incremental.get("task_results", [])),
        "full_task_count": len(full.get("task_results", [])),
        "incremental_reused": [item["task_name"] for item in incremental.get("lifecycle", {}).get("reused", [])],
        "incremental_invalidated": [
            item["task_name"] for item in incremental.get("lifecycle", {}).get("invalidated", [])
        ],
        "incremental_retrained": [
            item["task_name"] for item in incremental.get("lifecycle", {}).get("retrained", [])
        ],
        "full_reused": [item["task_name"] for item in full.get("lifecycle", {}).get("reused", [])],
        "lifecycle_note": (
            "A blank rebuild has no prior cache by definition. The comparison therefore covers "
            "all materialized task/model/report fields; lifecycle provenance is checked separately."
        ),
    }


def _run_full_check(state: InputState, root: Path) -> PipelineResult:
    input_dir = root / "input"
    workspace = root / "workspace"
    _copy_state(state, input_dir)
    _fresh_directory(workspace)
    return run_pipeline(input_dir, workspace, mode="full", fresh=True)


def run_edit_sequence(
    input_dir: str | Path,
    workspace: str | Path,
    report_dir: str | Path,
    *,
    reset: bool = True,
    compare_full: bool = True,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    workspace = Path(workspace)
    report_dir = Path(report_dir)
    if reset:
        _fresh_directory(workspace)
        _fresh_directory(report_dir)
    else:
        workspace.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

    state_dir = workspace / "state" / "inputs"
    pipeline_workspace = workspace / "pipeline"
    if reset or not (state_dir / "tasks.json").is_file():
        source_state = InputState.from_directory(input_dir)
        _copy_state(source_state, state_dir)
    initial_state = InputState.from_directory(state_dir)

    batch_records: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []

    def run_stage(active_state: InputState, name: str, *, allow_migration: bool) -> tuple[PipelineResult, dict[str, Any] | None]:
        incremental_report_path = report_dir / f"{name}.incremental.json"
        incremental = run_pipeline(
            active_state,
            pipeline_workspace,
            mode="incremental",
            allow_config_migration=allow_migration,
            report_path=incremental_report_path,
        )
        comparison = None
        if compare_full:
            with tempfile.TemporaryDirectory(prefix="full-check-") as temporary:
                full = _run_full_check(active_state, Path(temporary))
                comparison = compare_reports(incremental.report, full.report)
                atomic_write_json(report_dir / f"{name}.full.json", full.report)
                atomic_write_json(report_dir / f"{name}.comparison.json", comparison)
                comparison["stage"] = name
                stage_records.append(comparison)
        return incremental, comparison

    initial_incremental, initial_comparison = run_stage(
        initial_state, "00-initial", allow_migration=False
    )
    initial_report = copy.deepcopy(initial_incremental.report)

    batches = list(initial_state.edit_batches)
    current_state = initial_state
    for index, batch in enumerate(batches, 1):
        current_state = InputState.from_directory(state_dir)
        applied = current_state.apply_batch(batch)
        _copy_state(current_state, state_dir)
        current_state = InputState.from_directory(state_dir)
        batch_id = str(batch.get("batch_id", f"batch-{index}"))
        stage_name = f"{index:02d}-{batch_id}"
        incremental, comparison = run_stage(
            current_state, stage_name, allow_migration=True
        )
        record = {
            "batch_index": index,
            "batch_id": batch_id,
            "operations": applied,
            "report": f"{stage_name}.incremental.json",
            "comparison": comparison,
            "reused": [
                item["task_name"] for item in incremental.report["lifecycle"]["reused"]
            ],
            "invalidated": [
                item["task_name"] for item in incremental.report["lifecycle"]["invalidated"]
            ],
            "retrained": [
                item["task_name"] for item in incremental.report["lifecycle"]["retrained"]
            ],
            "config_migration": incremental.report["recovery"].get("config_migration"),
            "final_snapshot": incremental.report.get("snapshot"),
        }
        batch_records.append(record)

    # Idempotent rerun with the final input state: every valid task should be
    # loaded directly from snapshots and no training should occur.
    current_state = InputState.from_directory(state_dir)
    idempotent_incremental, idempotent_comparison = run_stage(
        current_state, f"{len(batches) + 1:02d}-idempotent-rerun", allow_migration=False
    )
    idempotent_match = projections_equal(
        initial_incremental.report if not batch_records else incremental.report,
        idempotent_incremental.report,
    )
    idempotent_retrained = idempotent_incremental.report["lifecycle"]["retrained"]
    idempotent_reused = idempotent_incremental.report["lifecycle"]["reused"]

    disjoint_checks = []
    for record_report_path in sorted(report_dir.glob("*.incremental.json")):
        import json

        report = json.loads(record_report_path.read_text(encoding="utf-8"))
        reused = {item["task_name"] for item in report["lifecycle"]["reused"]}
        retrained = {item["task_name"] for item in report["lifecycle"]["retrained"]}
        invalidated = {item["task_name"] for item in report["lifecycle"]["invalidated"]}
        disjoint_checks.append(
            {
                "report": record_report_path.name,
                "reused_retrained_disjoint": not bool(reused & retrained),
                "invalidated_reused_disjoint": not bool(invalidated & reused),
            }
        )

    final_report = idempotent_incremental.report
    verification = {
        "schema_version": 1,
        "sample_input": str(input_dir),
        "workspace": str(workspace),
        "report_directory": str(report_dir),
        "initial_comparison": initial_comparison,
        "batches": batch_records,
        "stage_comparisons": stage_records,
        "idempotent_rerun": {
            "materialization_match": idempotent_match,
            "retrained": [item["task_name"] for item in idempotent_retrained],
            "reused": [item["task_name"] for item in idempotent_reused],
            "no_retraining": len(idempotent_retrained) == 0,
            "comparison": idempotent_comparison,
        },
        "lifecycle_disjoint_checks": disjoint_checks,
        "final_report": "final.incremental.json",
        "final_snapshot": final_report.get("snapshot"),
        "final_injection_summary": final_report.get("injection_summary"),
        "final_failures": final_report.get("failures"),
        "initial_report": initial_report,
    }
    atomic_write_json(report_dir / "final.incremental.json", final_report)
    atomic_write_json(report_dir / "verification.json", verification)
    return verification


def run_demo(
    root: str | Path,
    *,
    seed: int = 20261120,
    sample_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    _fresh_directory(root)
    if sample_dir is None:
        sample_path = root / "sample"
        sample_summary = generate_sample(sample_path, seed=seed)
    else:
        sample_path = Path(sample_dir)
        state = InputState.from_directory(sample_path)
        sample_summary = {
            "root": str(sample_path),
            "seed": seed,
            "sample_count": len(state.features),
            "task_count": len(state.tasks.get("tasks", [])),
            "source": "existing_local_sample",
        }
    run_root = root / "run"
    report_dir = root / "reports"
    verification = run_edit_sequence(sample_path, run_root, report_dir, reset=True)

    # Crash/resume: an exception is raised only after a complete snapshot
    # pointer has been atomically committed. The next invocation must recover
    # that stable prefix without relying on an exit report.
    crash_root = root / "crash-resume"
    crash_workspace = crash_root / "workspace"
    _fresh_directory(crash_root)
    crash_event: dict[str, Any] = {}
    try:
        run_pipeline(
            sample_path,
            crash_workspace,
            mode="incremental",
            crash_after_completed=2,
        )
        crash_event = {"crashed": False, "reason": "crash threshold was not reached"}
    except SimulatedCrash as exc:
        crash_event = {
            "crashed": True,
            "business_code": exc.business_code,
            "message": exc.message,
            **exc.details,
            "exit_report_guaranteed": False,
        }
    resumed = run_pipeline(
        sample_path,
        crash_workspace,
        mode="incremental",
        report_path=crash_root / "resume-report.json",
    )
    baseline_full = run_pipeline(sample_path, crash_root / "baseline", mode="full", fresh=True)
    crash_comparison = compare_reports(resumed.report, baseline_full.report)

    # Concurrency equivalence: same final inputs/model, different scheduler
    # capacity. Task materialization must be byte-for-byte identical.
    concurrency_results = {}
    with tempfile.TemporaryDirectory(prefix="concurrency-") as temporary:
        temporary_path = Path(temporary)
        for value in (1, 5):
            state = InputState.from_directory(sample_path)
            state.config["max_concurrency"] = value
            input_path = temporary_path / f"input-{value}"
            workspace_path = temporary_path / f"workspace-{value}"
            _copy_state(state, input_path)
            result = run_pipeline(input_path, workspace_path, mode="full", fresh=True)
            concurrency_results[str(value)] = result.report
    concurrency_equal = canonical_json(_task_projection(concurrency_results["1"])) == canonical_json(
        _task_projection(concurrency_results["5"])
    )

    # Direct resume with changed config must be refused, not silently applied.
    mismatch_workspace = root / "config-mismatch-workspace"
    _fresh_directory(mismatch_workspace)
    initial_mismatch_run = run_pipeline(
        sample_path, mismatch_workspace, mode="incremental"
    )
    changed_state = InputState.from_directory(sample_path)
    changed_state.config["validation_threshold"] = 0.91
    mismatch_input = root / "config-mismatch-input"
    _copy_state(changed_state, mismatch_input)
    mismatch_event: dict[str, Any]
    try:
        run_pipeline(mismatch_input, mismatch_workspace, mode="incremental")
        mismatch_event = {"rejected": False}
    except PipelineConfigMismatchError as exc:
        mismatch_event = {
            "rejected": True,
            "business_code": exc.business_code,
            "message": exc.message,
            **exc.details,
        }
    unchanged = run_pipeline(
        sample_path,
        mismatch_workspace,
        mode="incremental",
        expected_global_version=initial_mismatch_run.report["snapshot"]["global_version"],
    )

    # Repeated shared-forward runs must have exactly the same representative
    # fingerprint and must not mutate encoder weight arrays.
    repeat_a = run_pipeline(sample_path, root / "repeat-a", mode="full", fresh=True)
    repeat_b = run_pipeline(sample_path, root / "repeat-b", mode="full", fresh=True)
    forward_repeat_equal = (
        repeat_a.shared_forward["representation_fingerprint"]
        == repeat_b.shared_forward["representation_fingerprint"]
    )

    final_failures = verification["final_failures"]
    failure_codes = {failure["business_code"] for failure in final_failures}
    injection_summary = verification["final_injection_summary"]
    base_precise_reuse = (
        "inj_zero_rank" in verification["batches"][0]["reused"]
        if verification["batches"]
        else False
    )
    disjoint_ok = all(
        item["reused_retrained_disjoint"] and item["invalidated_reused_disjoint"]
        for item in verification["lifecycle_disjoint_checks"]
    )
    all_stage_matches = all(
        item["exact_materialization_match"] for item in verification["stage_comparisons"]
    )
    checks = {
        "initial_incremental_matches_full_rebuild": bool(
            verification["initial_comparison"]
            and verification["initial_comparison"]["exact_materialization_match"]
        ),
        "all_edit_batches_match_full_rebuild": bool(all_stage_matches),
        "lifecycle_lists_disjoint_from_retraining": bool(disjoint_ok),
        "idempotent_rerun_reuses_and_does_not_retrain": bool(
            verification["idempotent_rerun"]["materialization_match"]
            and verification["idempotent_rerun"]["no_retraining"]
        ),
        "concurrency_serial_equivalent": bool(concurrency_equal),
        "concurrency_respected": all(
            report["execution"]["observed_max_inflight"]
            <= report["execution"]["configured_max_concurrency"]
            for report in concurrency_results.values()
        ),
        "snapshot_recovery_after_forced_termination": bool(
            crash_event.get("crashed")
            and resumed.report["recovery"]["resumed"]
            and len(resumed.report["recovery"]["recovered_completed_tasks"]) >= 2
            and crash_comparison["exact_materialization_match"]
        ),
        "config_mismatch_resume_rejected": bool(mismatch_event.get("rejected")),
        "snapshot_untouched_after_config_mismatch": bool(
            unchanged.report["recovery"]["recovered_completed_tasks"]
        ),
        "injection_examples_all_hit": bool(injection_summary["all_hit"]),
        "runtime_failure_isolated": "E_TRAINING_EXCEPTION" in failure_codes,
        "invalid_loader_codes_present": bool(
            {"E_TASK_INVALID", "E_RANK_INVALID"} <= failure_codes
            and any(
                failure.get("reason_code") == "E_VALIDATION_EMPTY"
                for failure in final_failures
            )
        ),
        "zero_rank_pre_construction": all(
            failure["business_code"] != "E_RANK_INVALID"
            or failure["phase"] == "adapter_construction"
            for failure in final_failures
        ),
        "precise_base_layer_reuse": bool(base_precise_reuse),
        "repeat_shared_forward_fingerprint_equal": bool(forward_repeat_equal),
    }
    demo_report = {
        "schema_version": 1,
        "seed": seed,
        "sample": sample_summary,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "crash_event": crash_event,
        "resume_recovery": resumed.report["recovery"],
        "resume_comparison": crash_comparison,
        "concurrency": {
            "equal": concurrency_equal,
            "runs": {
                value: {
                    "configured_max_concurrency": report["execution"]["configured_max_concurrency"],
                    "observed_max_inflight": report["execution"]["observed_max_inflight"],
                }
                for value, report in concurrency_results.items()
            },
        },
        "config_mismatch": mismatch_event,
        "snapshot_after_rejected_resume": unchanged.report["snapshot"],
        "forward_repeat": {
            "equal": forward_repeat_equal,
            "fingerprint": repeat_a.shared_forward["representation_fingerprint"],
        },
        "final_report": str(report_dir / "final.incremental.json"),
        "verification": str(report_dir / "verification.json"),
    }
    atomic_write_json(root / "demo-verification.json", demo_report)
    return demo_report
