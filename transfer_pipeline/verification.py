"""Acceptance verification: incremental outputs versus blank full rebuilds."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .canonical import atomic_write_json, canonical_json
from .data import read_structured_file
from .edits import apply_edit
from .pipeline import run_pipeline


_OUTCOME_FIELDS = (
    "name",
    "stable_sort_key",
    "status",
    "adapter_type",
    "rank",
    "adapter_parameter_summary",
    "training_losses",
    "converged",
    "epochs_run",
    "metrics",
    "adapter_cache_key",
    "evaluation_cache_key",
)


def _outcomes(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tasks": [
            {field: copy.deepcopy(task.get(field)) for field in _OUTCOME_FIELDS}
            for task in report.get("tasks", [])
        ],
        "failures": copy.deepcopy(report.get("failures", [])),
        "sample_references": copy.deepcopy(report.get("sample_references", {})),
    }


def _expected_invalidated(
    edit: Mapping[str, Any], current_report: Mapping[str, Any]
) -> set[str]:
    operation = edit.get("op")
    if operation == "add_task":
        task = edit.get("task", {})
        return {task.get("name")} if isinstance(task, Mapping) and task.get("name") else set()
    if operation == "delete_task":
        return set()
    if operation == "set_classes":
        return {str(edit.get("name"))}
    if operation == "replace_sample":
        sample_id = str(edit.get("sample_id"))
        return set(current_report.get("sample_references", {}).get(sample_id, []))
    if operation == "set_adapter_params":
        name = str(edit.get("name"))
        if name == "*":
            return {str(task.get("name")) for task in current_report.get("tasks", [])}
        return {name}
    return set()


def run_acceptance_verification(
    *, examples_directory: str | Path, output_directory: str | Path
) -> dict[str, Any]:
    examples = Path(examples_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    documents = {
        "tasks": read_structured_file(examples / "tasks.json"),
        "features": read_structured_file(examples / "features.json"),
        "encoder": read_structured_file(examples / "encoder.json"),
        "config": read_structured_file(examples / "adapter_config.json"),
        "injections": None,
    }
    edits = read_structured_file(examples / "edits.json")
    steps: list[dict[str, Any]] = []
    passed = True
    with tempfile.TemporaryDirectory(prefix="transfer-verify-") as temporary:
        root = Path(temporary)
        incremental_state = root / "incremental-state"
        initial = run_pipeline(
            documents["tasks"],
            documents["features"],
            documents["encoder"],
            documents["config"],
            state_directory=incremental_state,
            report_path=output / "initial_report.json",
        )
        current_report = dict(initial.report)
        steps.append(
            {
                "step": 0,
                "edit": {"op": "initial"},
                "incremental_exit_code": initial.exit_code,
                "tasks": [task["name"] for task in current_report.get("tasks", [])],
                "cache": current_report.get("cache"),
            }
        )
        for step_index, edit in enumerate(edits, start=1):
            expected_invalidated = _expected_invalidated(edit, current_report)
            metadata = apply_edit(documents, edit)
            incremental = run_pipeline(
                documents["tasks"],
                documents["features"],
                documents["encoder"],
                documents["config"],
                state_directory=incremental_state,
                report_path=output / f"incremental_step_{step_index}.json",
            )
            full_state = root / f"full-state-{step_index}"
            full = run_pipeline(
                documents["tasks"],
                documents["features"],
                documents["encoder"],
                documents["config"],
                state_directory=full_state,
                report_path=output / f"full_step_{step_index}.json",
            )
            incremental_outcomes = _outcomes(incremental.report)
            full_outcomes = _outcomes(full.report)
            outcomes_equal = canonical_json(incremental_outcomes) == canonical_json(
                full_outcomes
            )
            actual_invalidated = set(
                incremental.report.get("incremental", {}).get("invalidated_tasks", [])
            )
            actual_reused = set(
                incremental.report.get("incremental", {}).get("reused_tasks", [])
            )
            all_current = {
                task["name"] for task in incremental.report.get("tasks", [])
            }
            invalidation_scope_ok = actual_invalidated == expected_invalidated
            reuse_ok = actual_reused == (all_current - expected_invalidated)
            step_passed = bool(
                incremental.exit_code == 0
                and full.exit_code == 0
                and outcomes_equal
                and invalidation_scope_ok
                and reuse_ok
            )
            passed = passed and step_passed
            step_result = {
                "step": step_index,
                "edit": metadata,
                "incremental_exit_code": incremental.exit_code,
                "full_exit_code": full.exit_code,
                "outcomes_equal": outcomes_equal,
                "expected_invalidated_tasks": sorted(expected_invalidated),
                "actual_invalidated_tasks": sorted(actual_invalidated),
                "reused_tasks": sorted(actual_reused),
                "trained_tasks": sorted(
                    incremental.report.get("incremental", {}).get("trained_tasks", [])
                ),
                "adapter_reused_tasks": sorted(
                    incremental.report.get("incremental", {}).get(
                        "adapter_reused_tasks", []
                    )
                ),
                "evaluation_recomputed_tasks": sorted(
                    incremental.report.get("incremental", {}).get(
                        "evaluation_recomputed_tasks", []
                    )
                ),
                "invalidation_scope_ok": invalidation_scope_ok,
                "reuse_ok": reuse_ok,
                "step_passed": step_passed,
                "incremental_cache": incremental.report.get("cache"),
                "full_cache": full.report.get("cache"),
                "incremental_snapshot_version": incremental.report.get(
                    "snapshot_version"
                ),
                "full_snapshot_version": full.report.get("snapshot_version"),
            }
            steps.append(step_result)
            if not outcomes_equal:
                atomic_write_json(
                    output / f"mismatch_incremental_step_{step_index}.json",
                    incremental_outcomes,
                )
                atomic_write_json(
                    output / f"mismatch_full_step_{step_index}.json",
                    full_outcomes,
                )
            current_report = dict(incremental.report)
            if not step_passed:
                break

    summary = {
        "passed": passed,
        "step_count": len(steps),
        "outcomes_compared": [
            "task adapter parameter summaries and digests",
            "training losses and convergence fields",
            "adapter and evaluation content cache keys",
            "evaluation metrics",
            "sorted failure list",
            "sample references",
        ],
    }
    document = {"summary": summary, "steps": steps}
    atomic_write_json(output / "verification.json", document)
    return document
