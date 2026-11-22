from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .cache import ContentCache
from .errors import Failure
from .loader import LoadedInputs, load_inputs
from .plan import ConstructionState, PlanApplyError, apply_batch, final_state, initial_state
from .pipeline import PipelineResult, run_pipeline
from .reporting import add_report_aliases, compare_reports, failure_report, write_report
from .snapshot import check_snapshot, commit_snapshot


def reset_output(output_dir: str | Path) -> None:
    output = Path(output_dir).resolve()
    for name in ("snapshot", ".snapshot-staging", ".snapshot-backup", "cache"):
        target = output / name
        if target.exists():
            shutil.rmtree(target)
    reports = output / "reports"
    if reports.exists():
        shutil.rmtree(reports)


def _prepare_report(
    *,
    result: PipelineResult,
    mode: str,
    inputs: LoadedInputs,
    completed_batches: int,
) -> dict[str, Any]:
    report = dict(result.report)
    report["run_mode"] = mode
    report["input_fingerprints"] = dict(inputs.fingerprints)
    report["input_files"] = inputs.paths.as_dict()
    report["completed_batches"] = completed_batches
    add_report_aliases(report)
    return report


def run_incremental(
    *,
    inputs: LoadedInputs,
    output_dir: str | Path,
    reset: bool = False,
) -> tuple[dict[str, Any], int]:
    output = Path(output_dir).resolve()
    if reset:
        reset_output(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = ContentCache(output / "cache")
    base_failures: list[Failure] = list(inputs.failures.items)

    snapshot_check = check_snapshot(output, inputs)
    if snapshot_check.stale:
        assert snapshot_check.rebuild_from is not None
        stale_failure = snapshot_check.stale_failure()
        if stale_failure is not None:
            base_failures.append(stale_failure)
        state = initial_state(inputs)
        completed_batches = 0
    elif snapshot_check.snapshot_exists and snapshot_check.state is not None:
        state = snapshot_check.state
        completed_batches = snapshot_check.completed_batches
    else:
        state = initial_state(inputs)
        completed_batches = 0

    batches = inputs.plan["batches"]
    interval = int(state.config["snapshot_commit_interval_batches"])
    last_result: PipelineResult | None = None
    last_report: dict[str, Any] | None = None
    latest_snapshot: dict[str, Any] | None = snapshot_check.manifest if not snapshot_check.stale else None

    if completed_batches < len(batches):
        for index in range(completed_batches, len(batches)):
            batch = batches[index]
            try:
                state = apply_batch(state, batch, inputs)
            except PlanApplyError as exc:
                base_failures.extend(exc.failures)
                report = failure_report(mode="incremental", inputs=inputs, failures=base_failures)
                report["completed_batches"] = completed_batches
                write_report(output / "reports" / "latest.json", report)
                return report, 2
            completed_batches = index + 1
            result = run_pipeline(state=state, inputs=inputs, cache=cache, base_failures=base_failures)
            report = _prepare_report(
                result=result,
                mode="incremental",
                inputs=inputs,
                completed_batches=completed_batches,
            )
            should_commit = completed_batches % interval == 0 or completed_batches == len(batches)
            if should_commit:
                manifest = commit_snapshot(
                    output_dir=output,
                    state=state,
                    inputs=inputs,
                    result=result,
                    report=report,
                    completed_batches=completed_batches,
                )
                latest_snapshot = manifest
                report["snapshot"] = manifest
            else:
                report["snapshot"] = {
                    "committed": False,
                    "reason": "snapshot_commit_interval_not_reached",
                    "phase_versions": result.phase_versions,
                }
            write_report(output / "reports" / f"batch_{completed_batches:04d}.json", report)
            last_result = result
            last_report = report

    if last_result is None or last_report is None:
        result = run_pipeline(state=state, inputs=inputs, cache=cache, base_failures=base_failures)
        report = _prepare_report(
            result=result,
            mode="incremental",
            inputs=inputs,
            completed_batches=completed_batches,
        )
        manifest = commit_snapshot(
            output_dir=output,
            state=state,
            inputs=inputs,
            result=result,
            report=report,
            completed_batches=completed_batches,
        )
        report["snapshot"] = manifest
        latest_snapshot = manifest
        last_result = result
        last_report = report
    elif latest_snapshot is None or not (
        last_report.get("snapshot", {}).get("global_version") == latest_snapshot.get("global_version")
        and last_report.get("snapshot", {}).get("completed_batches") == completed_batches
    ):
        manifest = commit_snapshot(
            output_dir=output,
            state=state,
            inputs=inputs,
            result=last_result,
            report=last_report,
            completed_batches=completed_batches,
        )
        last_report["snapshot"] = manifest

    assert last_report is not None
    add_report_aliases(last_report)
    write_report(output / "reports" / "latest.json", last_report)
    return last_report, 0


def run_full(
    *,
    inputs: LoadedInputs,
    output_dir: str | Path,
    reset: bool = False,
) -> tuple[dict[str, Any], int]:
    output = Path(output_dir).resolve()
    if reset:
        reset_output(output)
    output.mkdir(parents=True, exist_ok=True)
    state, plan_failures = final_state(inputs)
    base_failures: list[Failure] = list(inputs.failures.items) + list(plan_failures)
    if plan_failures:
        report = failure_report(mode="full", inputs=inputs, failures=base_failures)
        report["completed_batches"] = 0
        write_report(output / "reports" / "latest.json", report)
        return report, 2
    result = run_pipeline(state=state, inputs=inputs, cache=ContentCache(None), base_failures=base_failures)
    report = _prepare_report(
        result=result,
        mode="full",
        inputs=inputs,
        completed_batches=len(inputs.plan["batches"]),
    )
    manifest = commit_snapshot(
        output_dir=output,
        state=state,
        inputs=inputs,
        result=result,
        report=report,
        completed_batches=len(inputs.plan["batches"]),
    )
    report["snapshot"] = manifest
    add_report_aliases(report)
    write_report(output / "reports" / "latest.json", report)
    write_report(output / "reports" / "full.json", report)
    return report, 0


def run_verification(
    *,
    inputs: LoadedInputs,
    output_dir: str | Path,
) -> tuple[dict[str, Any], int]:
    output = Path(output_dir).resolve()
    reset_output(output)
    incremental_dir = output / "incremental"
    full_a_dir = output / "full_a"
    full_b_dir = output / "full_b"
    incremental_report, incremental_code = run_incremental(inputs=inputs, output_dir=incremental_dir, reset=True)
    full_a_report, full_a_code = run_full(inputs=inputs, output_dir=full_a_dir, reset=True)
    full_b_report, full_b_code = run_full(inputs=inputs, output_dir=full_b_dir, reset=True)
    incremental_vs_full = compare_reports(incremental_report, full_a_report)
    full_repeat = compare_reports(full_a_report, full_b_report)
    result = {
        "schema_version": 1,
        "incremental_vs_full": incremental_vs_full,
        "full_repeat_determinism": full_repeat,
        "exit_codes": {
            "incremental": incremental_code,
            "full_a": full_a_code,
            "full_b": full_b_code,
        },
        "paths": {
            "incremental_report": str(incremental_dir / "reports" / "latest.json"),
            "full_a_report": str(full_a_dir / "reports" / "latest.json"),
            "full_b_report": str(full_b_dir / "reports" / "latest.json"),
        },
        "ok": incremental_vs_full["equal"] and full_repeat["equal"] and incremental_code == 0 and full_a_code == 0 and full_b_code == 0,
    }
    write_report(output / "equivalence.json", result)
    return result, 0 if result["ok"] else 1


def load_and_validate(input_dir: str | Path) -> LoadedInputs:
    return load_inputs(input_dir)


def write_validation_report(inputs: LoadedInputs, output_dir: str | Path) -> dict[str, Any]:
    failures = list(inputs.failures.items)
    if not inputs.failures.fatal:
        _, plan_failures = final_state(inputs)
        failures.extend(plan_failures)
    report = failure_report(mode="validate", inputs=inputs, failures=failures)
    report["valid"] = not any(item.fatal for item in failures)
    write_report(Path(output_dir).resolve() / "reports" / "validation.json", report)
    return report
