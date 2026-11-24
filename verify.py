"""Run the local acceptance matrix and persist an auditable JSON summary."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from arbitration_workbench.codes import failure_sort_key
from arbitration_workbench.editing import apply_edit
from arbitration_workbench.engine import run_workbench
from arbitration_workbench.sample import generate_sample


ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
ARTIFACTS = ROOT / "artifacts" / "verification"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def copy_inputs(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in ("evidence.jsonl", "rules.json", "config.json", "input_meta.json", "edits.json", "injections.json"):
        shutil.copy2(source / name, target / name)


def projection(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "workorders": report["workorders"],
        "failures": report["failures"],
        "summary": {
            "workorder_count": summary["workorder_count"],
            "successful_count": summary["successful_count"],
            "failure_count": summary["failure_count"],
            "pending_count": summary["pending_count"],
            "boundary_workorders": summary["boundary_workorders"],
            "boundary_by_priority": summary["boundary_by_priority"],
            "priority_distribution": summary["priority_distribution"],
            "failure_code_distribution": summary["failure_code_distribution"],
        },
    }


def set_config(data_dir: Path, **updates: Any) -> None:
    path = data_dir / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(updates)
    path.write_text(json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def set_input_version(data_dir: Path, version: int) -> None:
    write_json(data_dir / "input_meta.json", {"version": version})


def main() -> int:
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    ARTIFACTS.mkdir(parents=True)
    generated = generate_sample(SAMPLES / "base")
    generated_summary = dict(generated)
    generated_summary["path"] = "samples/base"

    baseline_serial = run_workbench(SAMPLES / "base", ARTIFACTS / "state-baseline-serial", mode="serial")
    baseline_parallel = run_workbench(SAMPLES / "base", ARTIFACTS / "state-baseline-parallel", mode="parallel")
    write_json(ARTIFACTS / "baseline_serial.json", baseline_serial)
    write_json(ARTIFACTS / "baseline_parallel.json", baseline_parallel)
    repeat = run_workbench(SAMPLES / "base", ARTIFACTS / "state-baseline-parallel", mode="parallel")
    write_json(ARTIFACTS / "baseline_repeat.json", repeat)

    baseline_equal = projection(baseline_serial) == projection(baseline_parallel)
    repeat_reused = (
        repeat["ledger"]["recomputed_workorders"] == []
        and repeat["ledger"]["stage_reuse_counts"]
        == {
            "decay": len(repeat["workorders"]),
            "relations": len(repeat["workorders"]),
            "aggregation": len(repeat["workorders"]),
        }
        and projection(baseline_parallel) == projection(repeat)
    )

    incremental_data = ARTIFACTS / "incremental_data"
    copy_inputs(SAMPLES / "base", incremental_data)
    run_workbench(incremental_data, ARTIFACTS / "state-incremental", mode="parallel")
    commands = json.loads((incremental_data / "edits.json").read_text(encoding="utf-8"))["commands"]
    edit_steps: list[dict[str, Any]] = []
    first_edit_exact = False
    for index, command in enumerate(commands, start=1):
        applied = apply_edit(incremental_data, command)
        report = run_workbench(incremental_data, ARTIFACTS / "state-incremental", mode="parallel")
        write_json(ARTIFACTS / f"incremental_step_{index:02d}.json", report)
        reused = report["ledger"]["reused_workorders"]
        recomputed = report["ledger"]["recomputed_workorders"]
        ordered = reused == sorted(reused) and recomputed == sorted(recomputed)
        failures_ordered = [failure_sort_key(row) for row in report["failures"]] == sorted(
            failure_sort_key(row) for row in report["failures"]
        )
        if index == 1:
            first_edit_exact = recomputed == ["WO-0002"] and len(reused) == 7
        edit_steps.append(
            {
                "step": index,
                "command": command["op"],
                "input_version": applied["input_version"],
                "reused_workorders": reused,
                "recomputed_workorders": recomputed,
                "failure_count": len(report["failures"]),
                "ledger_order_valid": ordered,
                "failure_order_valid": failures_ordered,
            }
        )

    full_data = ARTIFACTS / "full_data"
    copy_inputs(incremental_data, full_data)
    full_report = run_workbench(full_data, ARTIFACTS / "state-full", mode="serial")
    write_json(ARTIFACTS / "full_final.json", full_report)
    incremental_final = report
    incremental_full_equal = projection(incremental_final) == projection(full_report)
    edit_ledger_ordered = all(step["ledger_order_valid"] and step["failure_order_valid"] for step in edit_steps)

    budget_data = ARTIFACTS / "budget_data"
    copy_inputs(SAMPLES / "base", budget_data)
    set_config(budget_data, batch_evidence_budget=7)
    budget_serial = run_workbench(budget_data, ARTIFACTS / "state-budget-serial", mode="serial")
    budget_parallel = run_workbench(budget_data, ARTIFACTS / "state-budget-parallel", mode="parallel")
    budget_second = run_workbench(budget_data, ARTIFACTS / "state-budget-serial", mode="serial")
    write_json(ARTIFACTS / "budget_serial.json", budget_serial)
    write_json(ARTIFACTS / "budget_parallel.json", budget_parallel)
    write_json(ARTIFACTS / "budget_second_batch.json", budget_second)
    expected_first_pending = ["WO-0003", "WO-0004", "WO-0005", "WO-0006", "WO-0007"]
    budget_equal = projection(budget_serial) == projection(budget_parallel)
    budget_pending_ok = (
        budget_serial["ledger"]["pending_workorders"] == expected_first_pending
        and budget_serial["ledger"]["budget"]["consumed_this_run"] == 6
        and budget_serial["ledger"]["budget"]["remaining_this_run"] == 1
    )
    next_batch_priority = (
        budget_second["ledger"]["scheduled_decay_order"][:5] == expected_first_pending
        and budget_second["ledger"]["reused_workorders"] == ["WO-0001", "WO-0002", "WO-0008"]
    )

    quota_data = ARTIFACTS / "quota_data"
    copy_inputs(SAMPLES / "base", quota_data)
    set_config(quota_data, batch_evidence_budget=100, workorder_evidence_quota=2)
    quota_report = run_workbench(quota_data, ARTIFACTS / "state-quota", mode="parallel")
    write_json(ARTIFACTS / "quota_report.json", quota_report)
    quota_rows = [(row["workorder_id"], row["code"]) for row in quota_report["failures"]]
    quota_order_ok = quota_rows == sorted(quota_rows) and all(
        code == "E_WORKORDER_BUDGET_EXHAUSTED" for _, code in quota_rows
    )

    injection_report = run_workbench(
        SAMPLES / "base",
        ARTIFACTS / "state-injection",
        mode="parallel",
        injections_path=SAMPLES / "base" / "injections.json",
    )
    write_json(ARTIFACTS / "injection_report.json", injection_report)
    expected_injection_codes = [
        "E_UNKNOWN_EVIDENCE_TYPE",
        "E_CONFIDENCE_RANGE",
        "E_EVENT_TIME_INVALID",
        "E_HALF_LIFE_ZERO",
        "E_RELATION_UNKNOWN_TYPE",
    ]
    injection_codes = [row["code"] for row in injection_report["failures"]]
    injection_ok = (
        injection_codes == expected_injection_codes
        and [row["injection_id"] for row in injection_report["injections"]["hits"]]
        == [f"inj-{index:03d}" for index in range(1, 6)]
        and len(injection_report["workorders"]) == 3
    )

    mismatch_data = ARTIFACTS / "mismatch_data"
    copy_inputs(SAMPLES / "base", mismatch_data)
    run_workbench(mismatch_data, ARTIFACTS / "state-mismatch", mode="serial")
    set_config(mismatch_data, concurrency=5)
    mismatch_report = run_workbench(mismatch_data, ARTIFACTS / "state-mismatch", mode="serial")
    write_json(ARTIFACTS / "snapshot_config_mismatch.json", mismatch_report)
    snapshot_mismatch_ok = (
        mismatch_report["fatal"]
        and mismatch_report["failures"][0]["code"] == "E_SNAPSHOT_CONFIG_MISMATCH"
    )

    version_data = ARTIFACTS / "version_data"
    copy_inputs(SAMPLES / "base", version_data)
    first_edit = json.loads((version_data / "edits.json").read_text(encoding="utf-8"))["commands"][0]
    apply_edit(version_data, first_edit)
    run_workbench(version_data, ARTIFACTS / "state-version", mode="serial")
    set_input_version(version_data, 0)
    version_report = run_workbench(version_data, ARTIFACTS / "state-version", mode="serial")
    write_json(ARTIFACTS / "snapshot_version_mismatch.json", version_report)
    version_mismatch_ok = (
        version_report["fatal"] and version_report["failures"][0]["code"] == "E_SNAPSHOT_VERSION_MISMATCH"
    )

    snapshot_pointer = json.loads(
        (ARTIFACTS / "state-baseline-parallel" / "current.json").read_text(encoding="utf-8")
    )
    snapshot_dir = ARTIFACTS / "state-baseline-parallel" / "snapshots" / snapshot_pointer["snapshot"]
    snapshot_files_ok = all(
        (snapshot_dir / name).is_file()
        for name in ("manifest.json", "load.jsonl", "decay.jsonl", "relations.jsonl", "aggregation.jsonl")
    ) and len(baseline_parallel["snapshot"]["commits"]) == 4
    snapshot_manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    snapshot_evidence = {
        "pointer": snapshot_pointer,
        "manifest": snapshot_manifest,
        "files": {
            name: (snapshot_dir / name).stat().st_size
            for name in ("manifest.json", "load.jsonl", "decay.jsonl", "relations.jsonl", "aggregation.jsonl", "records.jsonl")
        },
    }
    write_json(ARTIFACTS / "snapshot_evidence.json", snapshot_evidence)

    checks = {
        "serial_parallel_business_equal": baseline_equal,
        "repeat_run_reuses_all_workorders": repeat_reused,
        "first_edit_recomputes_only_affected_workorder": first_edit_exact,
        "edit_ledger_and_failure_lists_ordered": edit_ledger_ordered,
        "incremental_final_equals_full_rebuild": incremental_full_equal,
        "budget_serial_parallel_equal": budget_equal,
        "budget_pending_and_remaining_order_correct": budget_pending_ok,
        "next_batch_prioritizes_pending": next_batch_priority,
        "quota_failure_order_correct": quota_order_ok,
        "injection_codes_and_hits_correct": injection_ok,
        "snapshot_config_mismatch_requires_rebuild": snapshot_mismatch_ok,
        "snapshot_version_mismatch_requires_rebuild": version_mismatch_ok,
        "phase_snapshots_atomic_files_present": snapshot_files_ok,
    }
    summary = {
        "all_passed": all(checks.values()),
        "checks": checks,
        "actual_results": {
            "sample": generated_summary,
            "baseline": {
                "successful_count": baseline_serial["summary"]["successful_count"],
                "failure_count": baseline_serial["summary"]["failure_count"],
                "boundary_workorders": baseline_serial["summary"]["boundary_workorders"],
                "priority_distribution": baseline_serial["summary"]["priority_distribution"],
                "budget": baseline_serial["ledger"]["budget"],
            },
            "repeat_run": {
                "reused_workorders": repeat["ledger"]["reused_workorders"],
                "recomputed_workorders": repeat["ledger"]["recomputed_workorders"],
                "stage_reuse_counts": repeat["ledger"]["stage_reuse_counts"],
            },
            "edit_steps": edit_steps,
            "incremental_vs_full": {
                "incremental_recomputed": incremental_final["ledger"]["recomputed_workorders"],
                "full_recomputed": full_report["ledger"]["recomputed_workorders"],
                "business_equal": incremental_full_equal,
            },
            "budget": {
                "first_batch_pending": budget_serial["ledger"]["pending_workorders"],
                "first_batch_consumed": budget_serial["ledger"]["budget"]["consumed_this_run"],
                "first_batch_remaining": budget_serial["ledger"]["budget"]["remaining_this_run"],
                "second_batch_scheduled_order": budget_second["ledger"]["scheduled_decay_order"],
                "second_batch_pending": budget_second["ledger"]["pending_workorders"],
            },
            "quota_failures": quota_rows,
            "injection_failures": [
                {"workorder_id": row["workorder_id"], "code": row["code"], "injection_id": row.get("injection_id")}
                for row in injection_report["failures"]
            ],
            "snapshot_commit_versions": [row["version"] for row in baseline_parallel["snapshot"]["commits"]],
        },
        "artifact_files": sorted(str(path.relative_to(ROOT)) for path in ARTIFACTS.glob("*.json")),
    }
    write_json(ROOT / "artifacts" / "verification_report.json", summary)
    # Keep compact, auditable reports and one snapshot inspection record; the
    # multi-version runtime state and temporary data trees are reproducible.
    for path in list(ARTIFACTS.iterdir()):
        if path.is_dir():
            shutil.rmtree(path)
    print(json.dumps({"all_passed": summary["all_passed"], "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
