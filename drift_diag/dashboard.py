"""Stable, read-only rendering model shared by the GUI and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .ledger import LedgerStore
from .models import FIELD_ORDER


def build_dashboard_model(db_path: str | Path) -> dict[str, Any]:
    with LedgerStore(db_path, read_only=True) as ledger:
        snapshot = ledger.read_snapshot()
        batches = ledger.read_batches()
        integrity = ledger.verify_integrity()
        replacements = ledger.replacement_history()
        unknown = ledger.list_unknown_workloads()
    latest = batches[-1] if batches else None
    return {
        "snapshot": snapshot,
        "batches": batches,
        "latest": latest,
        "integrity": integrity,
        "replacements": replacements,
        "unknown_workloads": unknown,
        "model_digest": canonical_json(
            {
                "snapshot": snapshot["snapshot_digest"] if snapshot else None,
                "batch_digests": [batch["content_digest"] for batch in batches],
            }
        ),
    }


def render_dashboard_text(db_path: str | Path) -> str:
    model = build_dashboard_model(db_path)
    snapshot = model["snapshot"]
    latest = model["latest"]
    lines: list[str] = []
    lines.append("DECLARATIVE DRIFT DIAGNOSTIC - READ ONLY LEDGER VIEW")
    lines.append(f"model_digest={model['model_digest']}")
    if snapshot is None:
        lines.append("snapshot=<none>")
    else:
        lines.append(
            "snapshot="
            + canonical_json(
                {
                    "digest": snapshot["snapshot_digest"],
                    "format_versions": snapshot["declaration_format_versions"],
                    "declaration_digest": snapshot["declaration_digest"],
                    "target_digest": snapshot["target_digest"],
                    "defaults_version": snapshot["defaults_version"],
                    "scale_version": snapshot["scale_version"],
                    "scale_exp": snapshot["scale_exp"],
                    "tolerance": {
                        "numerator": snapshot["tolerance_num"],
                        "denominator": snapshot["tolerance_den"],
                    },
                }
            )
        )

    lines.append("")
    lines.append("[NORMALIZED TARGET STATE]")
    if snapshot is not None:
        for name, workload in sorted(snapshot["target"]["workloads"].items()):
            lines.append(f"workload={name} source_format=v{workload['source_format_version']}")
            for field in FIELD_ORDER:
                provenance = workload["provenance"][field]
                lines.append(
                    f"  {field}={canonical_json(workload['fields'][field])} "
                    f"source={provenance['source']}"
                    + (
                        f" rule={provenance['rule_id']}"
                        if provenance.get("rule_id")
                        else ""
                    )
                )
            lines.append(
                "  filled_fields=" + canonical_json(workload["filled_fields"])
            )

    lines.append("")
    lines.append("[LATEST FIELD DECISIONS]")
    if latest is None:
        lines.append("latest_batch=<none>")
    else:
        for decision in latest["decisions"]:
            metric = decision["fixed_metric"]
            lines.append(
                f"{decision['workload']}/{decision['field']} drift={decision['drift']} "
                f"target={canonical_json(decision['target'])} "
                f"observed={canonical_json(decision['observed'])} "
                f"reason={decision['reason']} basis={decision['basis']} "
                f"field_version={decision['field_version']} metric={canonical_json(metric)}"
            )

    lines.append("")
    lines.append("[WORKLOAD CONVERGENCE AND ERROR BOUNDS]")
    if latest is not None:
        metrics = latest["workload_metrics"]
        lines.append(
            f"overall_converged={metrics['overall_converged']} "
            f"latest_event_cursor={latest['event_cursor']}"
        )
        for name, result in sorted(metrics["workloads"].items()):
            lines.append(
                f"{name} converged={result['converged']} "
                f"drift_ratio={result['drift_ratio']['numerator']}/"
                f"{result['drift_ratio']['denominator']} "
                f"drift_fields={canonical_json(result['drift_fields'])} "
                f"metric_units={result['numeric_drift_metric_units']} "
                f"error_upper_units={result['workload_error_upper_units']} "
                f"boundary_sensitive={result['boundary_sensitive']}"
            )

    lines.append("")
    lines.append("[SCHEDULE TRACE]")
    if latest is not None:
        trace = latest["trace"]
        lines.append(
            f"global_limit={trace['global_limit']} quota={trace['per_workload_quota']} "
            f"global_peak={trace['global_active_peak']} "
            f"assignments_all_on_workers={trace['assignments_all_on_workers']} "
            f"quota_respected={trace['quota_respected']}"
        )
        for round_info in trace["rounds"]:
            for assignment in round_info["assignments"]:
                lines.append(
                    f"round={assignment['round']} task={assignment['task_id']} "
                    f"thread={assignment['worker_thread']} "
                    f"queue_wait_ms={assignment['queue_wait_ms']} "
                    f"quota_wait_ms={assignment['quota_wait_ms']} "
                    f"execution_ms={assignment['execution_ms']}"
                )

    lines.append("")
    lines.append("[LEDGER BATCHES]")
    for batch in model["batches"]:
        lines.append(
            f"batch_id={batch['batch_id']} key={batch['batch_key']} "
            f"cursor={batch['event_cursor']} kind={batch['batch_kind']} "
            f"converged={batch['workload_metrics']['overall_converged']} "
            f"digest={batch['content_digest']}"
        )
    lines.append(
        f"integrity_ok={model['integrity']['ok']} "
        f"batches_checked={model['integrity']['batches_checked']} "
        f"decisions_checked={model['integrity']['decisions_checked']}"
    )
    if model["unknown_workloads"]:
        lines.append(
            "unknown_workloads=" + canonical_json(model["unknown_workloads"])
        )

    lines.append("")
    lines.append("[SNAPSHOT REPLACEMENTS]")
    for replacement in model["replacements"]:
        lines.append(
            f"id={replacement['replacement_id']} reason={replacement['reason']} "
            f"old={replacement['old_snapshot_digest']} new={replacement['new_snapshot_digest']} "
            f"added={len(replacement['diff']['added'])} removed={len(replacement['diff']['removed'])} "
            f"upgraded={len(replacement['diff']['upgraded'])}"
        )
    return "\n".join(lines) + "\n"
