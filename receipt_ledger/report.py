from __future__ import annotations

from typing import Any


def build_report(
    ledger: dict[str, Any],
    *,
    elapsed_wall_ms: float,
    port_checks: list[dict[str, Any]] | None = None,
    snapshot_manifest: dict[str, Any] | None = None,
    recovery: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "diagnostic-report.v1",
        "end_to_end_elapsed_wall_ms": round(float(elapsed_wall_ms), 3),
        "deterministic_timing_statistics": ledger.get("latency_statistics", []),
        "receipt_counts_by_version": ledger.get("receipt_counts", {}),
        "worker_param_version_metrics": ledger.get("worker_version_metrics", []),
        "invalidation_count": ledger.get("summary", {}).get("invalidation_item_count", 0),
        "invalidation_items": ledger.get("invalidation_items", []),
        "reuse_count": ledger.get("summary", {}).get("reuse_hit_count", 0),
        "reuse_hits": ledger.get("reuse_hits", []),
        "pending_count": ledger.get("summary", {}).get("pending_count", 0),
        "pending_queue": ledger.get("pending_queue", []),
        "business_code_counts": ledger.get("business_code_counts", {}),
        "failure_count": ledger.get("summary", {}).get("failure_count", 0),
        "failures": ledger.get("failures", []),
        "injection_results": ledger.get("injection_results", []),
        "command_cursor": ledger.get("command_cursor", 0),
        "port_checks": port_checks or [],
        "snapshot_manifest": snapshot_manifest,
        "snapshot_recovery": recovery,
        "config_hash": ledger.get("config_hash"),
    }
