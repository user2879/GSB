from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import atomic_write_text, canonical_json_text, fingerprint
from .pipeline import RunResult, STAGE_ORDER


def _failure_codes(reports: list[dict[str, Any]]) -> list[str]:
    codes: set[str] = set()
    for report in reports:
        for failure in report.get("failures", []):
            codes.update(str(code) for code in failure.get("codes", []))
    return sorted(codes)


def _sum_timings(reports: list[dict[str, Any]]) -> dict[str, float]:
    totals = {stage: 0.0 for stage in STAGE_ORDER}
    for report in reports:
        for stage, value in report.get("stage_timings_ms", {}).items():
            totals[stage] = round(totals.get(stage, 0.0) + float(value), 3)
    return totals


def _stage_counts(reports: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = {stage: 0 for stage in STAGE_ORDER}
    for report in reports:
        for stage in report.get(field, []):
            counts[stage] = counts.get(stage, 0) + 1
    return counts


def build_run_report(
    result: RunResult,
    *,
    mode: str,
    plan_fingerprint: str | None,
) -> dict[str, Any]:
    reports = result.reports
    full_checks = [
        report.get("full_rebuild", {})
        for report in reports
        if "full_rebuild" in report
    ]
    full_equal = (
        all(check.get("equal") is True for check in full_checks)
        if full_checks
        else None
    )
    budget_consumed = sum(
        int(report.get("budget", {}).get("consumed", 0)) for report in reports
    )
    budget_pending = sum(len(report.get("pending", [])) for report in reports)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "plan_fingerprint": plan_fingerprint,
        "batch_count": len(reports),
        "unprocessed_batches": result.unprocessed_batches,
        "reports": reports,
        "summary": {
            "invalidations_by_stage": _stage_counts(reports, "invalidations"),
            "reuses_by_stage": _stage_counts(reports, "reuses"),
            "stage_timings_ms": _sum_timings(reports),
            "budget_consumed": budget_consumed,
            "budget_pending": budget_pending,
            "failure_codes": _failure_codes(reports),
            "pipeline_complete": (
                bool(reports[-1].get("pipeline_complete")) if reports else False
            ),
            "full_rebuild_verified": full_equal,
            "snapshot": reports[-1].get("snapshot") if reports else None,
        },
    }
    report["report_fingerprint"] = fingerprint(
        "run-report-deterministic:v1",
        {
            "mode": mode,
            "plan_fingerprint": plan_fingerprint,
            "reports": [
                {
                    key: value
                    for key, value in batch.items()
                    if key
                    in {
                        "batch_id",
                        "config_hash",
                        "data_fingerprint",
                        "raw_data_fingerprint",
                        "sample_count",
                        "pipeline_complete",
                        "invalidations",
                        "reuses",
                        "fragment_reuse",
                        "budget",
                        "pending",
                        "failures",
                        "stage_versions",
                    }
                }
                for batch in reports
            ],
        },
    )
    return report


def write_run_report(workspace: str | Path, report: dict[str, Any]) -> tuple[Path, Path]:
    reports_dir = Path(workspace) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    version = report.get("summary", {}).get("snapshot", {}) or {}
    global_version = version.get("global_version", 0)
    filename = f"run-v{int(global_version or 0):06d}.json"
    path = reports_dir / filename
    atomic_write_text(path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    latest = reports_dir / "latest.json"
    atomic_write_text(
        latest,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return path, latest


def write_stale_report(
    workspace: str | Path, failure: dict[str, Any], extra: dict[str, Any] | None = None
) -> Path:
    reports_dir = Path(workspace) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": False,
        "failure": failure,
    }
    if extra:
        payload.update(extra)
    path = reports_dir / "stale.json"
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path
