from __future__ import annotations

import copy
import time
from typing import Any, Callable

from .codes import E_VIEW_DIVERGENCE

VIEW_FIELDS = (
    "request_id",
    "authoritative_ref",
    "authoritative_batch_id",
    "authoritative_worker_id",
    "basis",
    "merge_rule",
    "coverage_ref",
    "rejected_refs",
    "is_duplicate",
    "invalidated",
)


def authoritative_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    invalid = set(report.get("invalidation", {}).get("invalid_decision_keys", []))
    rows = []
    for decision in report["decisions"]:
        rows.append(
            {
                "request_id": decision["request_id"],
                "authoritative_ref": decision["authoritative_ref"],
                "authoritative_batch_id": decision["authoritative_batch_id"],
                "authoritative_worker_id": decision["authoritative_worker_id"],
                "basis": decision["basis"],
                "merge_rule": decision["merge_rule"],
                "coverage_ref": decision["coverage_ref"],
                "rejected_refs": list(decision["rejected_refs"]),
                "is_duplicate": decision["is_duplicate"],
                "invalidated": decision["decision_key"] in invalid,
            }
        )
    return sorted(rows, key=lambda row: row["request_id"])


def progress_summary(rows: list[dict[str, Any]], total: int | None = None) -> dict[str, Any]:
    completed = len(rows)
    return {
        "completed_entries": completed,
        "total_entries": total if total is not None else completed,
        "progress_q": completed if total in (None, 0) else round(completed / total, 6),
        "attribution_summary": [
            {
                "request_id": row["request_id"],
                "batch_id": row["authoritative_batch_id"],
                "worker_id": row["authoritative_worker_id"],
                "basis": row["basis"],
            }
            for row in rows
        ],
        "duplicate_count": sum(1 for row in rows if row["is_duplicate"]),
        "invalid_entry_count": sum(1 for row in rows if row["invalidated"]),
    }


def fast_partial_view(
    report: dict[str, Any],
    completed_request_ids: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    all_rows = authoritative_rows(report)
    if completed_request_ids is None:
        rows = all_rows
    else:
        completed = set(completed_request_ids)
        order = {request_id: index for index, request_id in enumerate(sorted(completed))}
        rows = [row for row in all_rows if row["request_id"] in completed]
        rows.sort(key=lambda row: order[row["request_id"]])
    return {
        "view": "fast",
        "rows": rows,
        "summary": progress_summary(rows, total=len(all_rows)),
    }


def compare_common_scope(
    fast: dict[str, Any], authoritative: dict[str, Any]
) -> list[dict[str, Any]]:
    fast_by_id = {row["request_id"]: row for row in fast["rows"]}
    auth_by_id = {row["request_id"]: row for row in authoritative["rows"]}
    common = sorted(set(fast_by_id) & set(auth_by_id))
    diffs = []
    for request_id in common:
        left = fast_by_id[request_id]
        right = auth_by_id[request_id]
        for field in VIEW_FIELDS:
            if left.get(field) != right.get(field):
                diffs.append(
                    {
                        "code": E_VIEW_DIVERGENCE,
                        "request_id": request_id,
                        "field": field,
                        "fast_value": left.get(field),
                        "authoritative_value": right.get(field),
                    }
                )
    return diffs


def interactive_view(
    report: dict[str, Any],
    *,
    completed_request_ids: list[str] | None = None,
    budget_ms: int = 250,
    elapsed_ms: int | None = None,
    previous_summary: dict[str, Any] | None = None,
    mutate_fast_row: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    fast = fast_partial_view(report, completed_request_ids)
    if mutate_fast_row is not None and fast["rows"]:
        fast["rows"] = [mutate_fast_row(copy.deepcopy(fast["rows"][0]))] + fast["rows"][1:]
    authoritative = {"view": "authoritative", "rows": authoritative_rows(report)}
    scoped_auth = {
        "rows": [
            row
            for row in authoritative["rows"]
            if completed_request_ids is None or row["request_id"] in set(completed_request_ids)
        ]
    }
    diffs = compare_common_scope(fast, scoped_auth)
    measured_ms = elapsed_ms if elapsed_ms is not None else int((time.monotonic() - started) * 1000)

    if diffs:
        return {
            "status": E_VIEW_DIVERGENCE,
            "code": E_VIEW_DIVERGENCE,
            "latency_budget_ms": budget_ms,
            "elapsed_ms": measured_ms,
            "within_budget": measured_ms <= budget_ms,
            "divergences": diffs,
            "summary": previous_summary or fast["summary"],
            "preferred_last_completed_summary": previous_summary is not None,
        }

    if measured_ms > budget_ms:
        return {
            "status": "budget_exceeded_cached_summary",
            "latency_budget_ms": budget_ms,
            "elapsed_ms": measured_ms,
            "within_budget": False,
            "summary": previous_summary or fast["summary"],
            "preferred_last_completed_summary": previous_summary is not None,
        }

    return {
        "status": "consistent",
        "latency_budget_ms": budget_ms,
        "elapsed_ms": measured_ms,
        "within_budget": measured_ms <= budget_ms,
        "common_scope_request_ids": [row["request_id"] for row in fast["rows"]],
        "summary": fast["summary"],
        "field_diffs": [],
    }
