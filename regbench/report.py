"""JSON 诊断报告与并发一致性比对。"""

from __future__ import annotations

from typing import Any

from .config import ExperimentConfig
from .grid import canonical_completion_order
from .runner import RunOutcome
from .util import truncate_list


def build_report(
    config: ExperimentConfig,
    outcome: RunOutcome,
    concurrency_label: str,
    consistency: dict[str, Any] | None = None,
    command_trace: list[dict[str, Any]] | None = None,
    cache_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results = outcome.results
    scheme_rows: dict[str, list[dict[str, Any]]] = {}
    task_rows: list[dict[str, Any]] = []
    for planned in outcome.plan.ordered_tasks:
        task_key = planned.task.task_key
        result = results.get(task_key)
        row = {
            "task_key": task_key,
            "cache_key": planned.task.cache_key,
            "scheme": planned.task.scheme,
            "penalty": planned.task.penalty,
            "l1_ratio": planned.task.l1_ratio,
            "decision": planned.decision,
            "planned_epochs": planned.planned_epochs,
            "starts": planned.starts,
            "ready_position": planned.ready_position,
        }
        if result is not None:
            row.update(
                {
                    "status": result.status,
                    "epochs_executed": result.epochs_executed,
                    "convergence_epoch": result.convergence_epoch,
                    "converged": result.converged,
                    "quota_touched": result.quota_touched,
                    "loss_trace_summary": truncate_list(result.loss_history),
                    "validation": result.validation,
                    "final_weight_summary": result.final_weight_summary,
                    "started_monotonic": result.started_monotonic,
                    "ended_monotonic": result.ended_monotonic,
                }
            )
            scheme_rows.setdefault(result.scheme, []).append(
                {
                    "task_key": result.task_key,
                    "penalty": result.penalty,
                    "l1_ratio": result.l1_ratio,
                    "loss_trace_summary": truncate_list(result.loss_history),
                    "convergence_epoch": result.convergence_epoch,
                    "validation": result.validation,
                    "quota_touched": result.quota_touched,
                }
            )
        else:
            row.update(
                {
                    "status": (
                        "failed"
                        if any(
                            item.task_key == task_key
                            and item.code in ("E_TASK_FAILED", "E_STAGE_WRITE_FAILED")
                            for item in outcome.failures.records
                        )
                        else "not_started"
                        if planned.starts
                        else "budget_exhausted"
                    ),
                    "epochs_executed": 0,
                }
            )
        task_rows.append(row)

    canonical_order = canonical_completion_order(
        [item.task for item in outcome.plan.ordered_tasks]
    )
    report = {
        "experiment": config.name,
        "concurrency_label": concurrency_label,
        "config_hash": outcome.config_hash,
        "data_fingerprint": outcome.data.data_fingerprint,
        "snapshot_versions": outcome.snapshot_versions,
        "grid_completion": outcome.summary["grid_completion"],
        "schemes": {key: sorted(rows, key=lambda row: row["task_key"]) for key, rows in scheme_rows.items()},
        "tasks": task_rows,
        "canonical_completion_order": canonical_order,
        "scheduler": {
            "decision_log": outcome.plan.decision_log,
            "budget_consumed": outcome.plan.consumed_budget,
            "budget_remaining": outcome.plan.remaining_budget,
            "global_budget": config.scheduler.global_epoch_budget,
            "per_scheme_quota": config.scheduler.per_scheme_epoch_quota,
            "budget_exhausted_tasks": outcome.plan.exhausted_tasks,
            "quota_exceeded_tasks": outcome.plan.quota_tasks,
        },
        "failures": outcome.failures.to_list(),
        "graceful_stopped": outcome.graceful_stopped,
        "concurrency_consistency": consistency or {"status": "single_run"},
        "cache_stats": cache_stats or {"reused": [], "recomputed": [], "invalidated": []},
        "command_trace": command_trace or [],
    }
    return report


def comparable_payload(report: dict[str, Any]) -> dict[str, Any]:
    """剔除单调时钟/版本号等天然不稳定字段后的逐字段比较载荷。"""

    tasks = []
    for row in report["tasks"]:
        clean = {k: v for k, v in row.items() if not k.endswith("monotonic")}
        tasks.append(clean)
    return {
        "config_hash": report["config_hash"],
        "data_fingerprint": report["data_fingerprint"],
        "grid_completion": report["grid_completion"],
        "schemes": report["schemes"],
        "tasks": tasks,
        "canonical_completion_order": report["canonical_completion_order"],
        "scheduler": {
            k: v
            for k, v in report["scheduler"].items()
        },
        "failure_codes": sorted(item["code"] for item in report["failures"]),
        "failure_task_keys": sorted(
            item["task_key"] for item in report["failures"] if item["task_key"]
        ),
    }


def compare_reports(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """比较多份报告（如 serial / parallel / resumed）的可比较载荷。"""

    payloads = {label: comparable_payload(report) for label, report in reports.items()}
    baseline_label = next(iter(payloads))
    baseline = payloads[baseline_label]
    mismatches: list[dict[str, Any]] = []
    for label, payload in payloads.items():
        if label == baseline_label:
            continue
        for key in baseline:
            if payload.get(key) != baseline[key]:
                mismatches.append({"field": key, "left": baseline_label, "right": label})
    return {
        "status": "identical" if not mismatches else "mismatch",
        "baseline": baseline_label,
        "compared": list(reports.keys()),
        "mismatches": mismatches,
    }
